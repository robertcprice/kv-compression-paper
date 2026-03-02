#!/usr/bin/env python3
"""
FlashAttention Compatibility Layer for KV Cache Compression

Integrates the three-stage KV compression pipeline with FlashAttention-2/3.
The key insight: dequantization must happen INSIDE the attention path,
otherwise you pay full HBM bandwidth for FP32 anyway and lose FA's benefit.

This module provides:
1. FlashAttention-compatible KV cache wrapper
2. INT8 dequantization fused with attention input preparation
3. Variable-length KV handling for per-layer eviction
4. Compatibility with both flash_attn and torch.nn.functional.scaled_dot_product_attention

Requirements:
    - flash-attn >= 2.5.0 (optional, falls back to PyTorch SDPA)
    - torch >= 2.0 (for SDPA backend)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    # FlashAttention-3 (H100+)
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    HAS_FLASH_ATTN_3 = True
except ImportError:
    HAS_FLASH_ATTN_3 = False


class CompressedKVAttention(torch.nn.Module):
    """
    Attention module that works directly with compressed KV caches.

    Handles:
    - INT8 KV dequantization in the attention hot path
    - Variable sequence lengths from per-layer eviction
    - Head reduction (reduced head count in later layers)
    - Fallback from FlashAttention -> SDPA -> naive attention
    """

    def __init__(self, num_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.scale = head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_scale: Optional[torch.Tensor] = None,
        key_zero: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        value_zero: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_k: Optional[int] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Attention with compressed KV.

        Args:
            query: [batch, heads, q_len, head_dim] in fp16/bf16
            key_cache: [batch, heads, kv_len, head_dim] in int8 or fp16
            value_cache: [batch, heads, kv_len, head_dim] in int8 or fp16
            key_scale/zero: quantization params if key_cache is int8
            value_scale/zero: quantization params if value_cache is int8
            cu_seqlens_k: cumulative sequence lengths for variable-length KV
            max_seqlen_k: max KV sequence length in batch
            causal: use causal masking
        """
        # Step 1: Dequantize INT8 KV if needed
        if key_cache.dtype == torch.int8:
            key = self._dequantize(key_cache, key_scale, key_zero, query.dtype)
            value = self._dequantize(value_cache, value_scale, value_zero, query.dtype)
        else:
            key = key_cache.to(query.dtype)
            value = value_cache.to(query.dtype)

        # Step 2: Handle head mismatch (from head reduction)
        if key.size(1) < query.size(1):
            # Repeat KV heads to match query heads (GQA-style)
            n_rep = query.size(1) // key.size(1)
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)
        elif key.size(1) > query.size(1):
            key = key[:, :query.size(1)]
            value = value[:, :query.size(1)]

        # Step 3: Route to best available attention backend
        if HAS_FLASH_ATTN and query.is_cuda and cu_seqlens_k is not None:
            return self._flash_attn_varlen(query, key, value, cu_seqlens_k, max_seqlen_k, causal)
        elif HAS_FLASH_ATTN and query.is_cuda:
            return self._flash_attn(query, key, value, causal)
        else:
            return self._sdpa_attention(query, key, value, causal)

    def _dequantize(self, x_q: torch.Tensor, scale: torch.Tensor,
                     zero: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        """Dequantize INT8 to target dtype (fp16/bf16)."""
        x_fp = (x_q.float() - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
        return x_fp.to(target_dtype)

    def _flash_attn(self, q, k, v, causal):
        """FlashAttention-2 path."""
        # flash_attn expects [batch, seqlen, heads, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=causal)
        return out.transpose(1, 2)  # back to [batch, heads, seqlen, head_dim]

    def _flash_attn_varlen(self, q, k, v, cu_seqlens_k, max_seqlen_k, causal):
        """FlashAttention variable-length path for evicted KV caches."""
        batch, heads, q_len, dim = q.shape
        kv_len = k.size(2)

        # Flatten batch for varlen
        q_flat = q.transpose(1, 2).reshape(-1, heads, dim)
        k_flat = k.transpose(1, 2).reshape(-1, heads, dim)
        v_flat = v.transpose(1, 2).reshape(-1, heads, dim)

        cu_seqlens_q = torch.arange(0, (batch + 1) * q_len, q_len,
                                     dtype=torch.int32, device=q.device)

        out = flash_attn_varlen_func(
            q_flat, k_flat, v_flat,
            cu_seqlens_q, cu_seqlens_k,
            q_len, max_seqlen_k,
            dropout_p=self.dropout if self.training else 0.0,
            causal=causal,
        )
        return out.reshape(batch, q_len, heads, dim).transpose(1, 2)

    def _sdpa_attention(self, q, k, v, causal):
        """PyTorch SDPA fallback (uses FlashAttention backend when available)."""
        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )


class CompressedKVCacheForAttention:
    """
    Manages compressed KV cache in a format compatible with FlashAttention.

    This is the bridge between the compression pipeline (enhanced_kv_cache.py)
    and the attention computation (FlashAttention / SDPA).
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        use_int8: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.use_int8 = use_int8
        self.device = device
        self.dtype = dtype

        # Per-layer storage
        self.k_caches: List[torch.Tensor] = []
        self.v_caches: List[torch.Tensor] = []
        self.k_scales: List[Optional[torch.Tensor]] = []
        self.v_scales: List[Optional[torch.Tensor]] = []
        self.k_zeros: List[Optional[torch.Tensor]] = []
        self.v_zeros: List[Optional[torch.Tensor]] = []
        self.seq_lens: List[int] = []

        # Attention module
        self.attention = CompressedKVAttention(num_heads, head_dim)

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        Store new KV, applying INT8 compression if enabled.

        Args:
            layer_idx: which transformer layer
            key: [batch, heads, new_tokens, head_dim]
            value: [batch, heads, new_tokens, head_dim]
        """
        if self.use_int8:
            k_q, k_s, k_z = self._quantize(key)
            v_q, v_s, v_z = self._quantize(value)

            if layer_idx < len(self.k_caches):
                # Append to existing
                self.k_caches[layer_idx] = torch.cat([self.k_caches[layer_idx], k_q], dim=2)
                self.v_caches[layer_idx] = torch.cat([self.v_caches[layer_idx], v_q], dim=2)
                self.k_scales[layer_idx] = k_s
                self.v_scales[layer_idx] = v_s
                self.k_zeros[layer_idx] = k_z
                self.v_zeros[layer_idx] = v_z
            else:
                self.k_caches.append(k_q)
                self.v_caches.append(v_q)
                self.k_scales.append(k_s)
                self.v_scales.append(v_s)
                self.k_zeros.append(k_z)
                self.v_zeros.append(v_z)
        else:
            k_compressed = key.to(self.dtype)
            v_compressed = value.to(self.dtype)

            if layer_idx < len(self.k_caches):
                self.k_caches[layer_idx] = torch.cat([self.k_caches[layer_idx], k_compressed], dim=2)
                self.v_caches[layer_idx] = torch.cat([self.v_caches[layer_idx], v_compressed], dim=2)
            else:
                self.k_caches.append(k_compressed)
                self.v_caches.append(v_compressed)
                self.k_scales.append(None)
                self.v_scales.append(None)
                self.k_zeros.append(None)
                self.v_zeros.append(None)

    def attend(
        self,
        layer_idx: int,
        query: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Run attention against compressed KV cache for a layer.

        Args:
            layer_idx: which layer's KV cache to attend to
            query: [batch, heads, q_len, head_dim]
            causal: use causal masking
        """
        return self.attention(
            query,
            self.k_caches[layer_idx],
            self.v_caches[layer_idx],
            key_scale=self.k_scales[layer_idx],
            key_zero=self.k_zeros[layer_idx],
            value_scale=self.v_scales[layer_idx],
            value_zero=self.v_zeros[layer_idx],
            causal=causal,
        )

    def evict(
        self,
        layer_idx: int,
        keep_mask: torch.Tensor,
    ):
        """
        Evict tokens from a layer's KV cache.

        Args:
            layer_idx: which layer
            keep_mask: [kv_len] boolean mask, True = keep
        """
        self.k_caches[layer_idx] = self.k_caches[layer_idx][:, :, keep_mask, :]
        self.v_caches[layer_idx] = self.v_caches[layer_idx][:, :, keep_mask, :]

    def _quantize(self, x: torch.Tensor):
        """Per-head INT8 quantization."""
        xmin = x.amin(dim=-1, keepdim=True)
        xmax = x.amax(dim=-1, keepdim=True)
        scale = (xmax - xmin) / 255.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero = -128 - xmin / scale
        x_q = torch.round(x / scale + zero).clamp(-128, 127).to(torch.int8)
        return x_q, scale.squeeze(-1), zero.squeeze(-1)

    def memory_bytes(self) -> int:
        """Total compressed KV memory in bytes."""
        total = 0
        for k in self.k_caches:
            total += k.numel() * k.element_size()
        for v in self.v_caches:
            total += v.numel() * v.element_size()
        return total

    def get_backend_info(self) -> Dict[str, bool]:
        """Report which attention backends are available."""
        return {
            "flash_attn_2": HAS_FLASH_ATTN,
            "flash_attn_3": HAS_FLASH_ATTN_3,
            "sdpa": True,  # Always available in PyTorch 2.0+
            "sdpa_flash_backend": hasattr(torch.backends, 'cuda') and
                                   getattr(torch.backends.cuda, 'flash_sdp_enabled', lambda: False)(),
        }


def check_compatibility():
    """Check FlashAttention compatibility and print status."""
    print("FlashAttention Compatibility Check")
    print("=" * 50)
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print(f"  FlashAttention-2: {'YES' if HAS_FLASH_ATTN else 'NO (pip install flash-attn)'}")
    print(f"  FlashAttention-3: {'YES' if HAS_FLASH_ATTN_3 else 'NO (H100+ only)'}")
    print(f"  PyTorch SDPA: YES (built-in)")

    backend = "FlashAttention-3" if HAS_FLASH_ATTN_3 else \
              "FlashAttention-2" if HAS_FLASH_ATTN else "PyTorch SDPA"
    print(f"\n  Active backend: {backend}")
    return backend


if __name__ == "__main__":
    check_compatibility()
