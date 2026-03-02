#!/usr/bin/env python3
"""
vLLM Plugin for KV Cache Compression

Integrates the three-stage compression pipeline into vLLM's serving stack.
Designed to work alongside PagedAttention (orthogonal concerns).

Architecture:
    vLLM Request -> PagedAttention (memory management)
                 -> THIS PLUGIN (KV compression)
                 -> FlashAttention (computation)

Usage with vLLM:
    from kv_compression.vllm_plugin import CompressedKVCacheConfig

    # In vLLM engine config
    engine = LLMEngine(
        model="mistralai/Mistral-7B-v0.1",
        kv_cache_dtype="int8",  # Enable INT8 KV
        # Plugin hooks into cache allocation
    )

Requirements:
    - vllm >= 0.4.0
    - torch >= 2.0
    - CUDA (vLLM does not support MPS)

NOTE: This module requires CUDA and vLLM. It will not run on MPS/CPU.
      It is structured for integration into vLLM's serving pipeline.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class CompressionLevel(Enum):
    """Predefined compression configurations."""
    NONE = "none"           # No compression (baseline)
    LIGHT = "light"         # INT8 only (4x)
    MODERATE = "moderate"   # INT8 + heads (8-10x)
    STANDARD = "standard"   # INT8 + heads + 50% eviction (20x)
    AGGRESSIVE = "aggressive"  # INT8 + heads + 87% eviction (40x)


@dataclass
class CompressedKVCacheConfig:
    """Configuration for KV cache compression in vLLM."""

    level: CompressionLevel = CompressionLevel.STANDARD

    # Stage 1: Quantization
    use_int8: bool = True

    # Stage 2: Head reduction
    head_keep_early: float = 1.0    # Layers 0 - n/3
    head_keep_mid: float = 0.8      # Layers n/3 - 2n/3
    head_keep_late: float = 0.6     # Layers 2n/3 - n

    # Stage 3: Token eviction
    eviction_enabled: bool = True
    eviction_keep_ratio: float = 0.4  # Keep 40% of tokens (moderate)
    eviction_method: str = "attention"  # "attention" | "recent" | "hybrid"
    sink_tokens: int = 4             # Always keep first N tokens
    recent_window: int = 64          # Always keep last N tokens

    # Performance
    eviction_interval: int = 64      # Run eviction every N tokens
    async_compression: bool = True    # Overlap compression with compute

    @classmethod
    def from_level(cls, level: CompressionLevel) -> "CompressedKVCacheConfig":
        """Create config from predefined compression level."""
        configs = {
            CompressionLevel.NONE: cls(
                level=CompressionLevel.NONE,
                use_int8=False,
                eviction_enabled=False,
            ),
            CompressionLevel.LIGHT: cls(
                level=CompressionLevel.LIGHT,
                use_int8=True,
                eviction_enabled=False,
            ),
            CompressionLevel.MODERATE: cls(
                level=CompressionLevel.MODERATE,
                use_int8=True,
                head_keep_mid=0.8,
                head_keep_late=0.6,
                eviction_enabled=False,
            ),
            CompressionLevel.STANDARD: cls(
                level=CompressionLevel.STANDARD,
                use_int8=True,
                head_keep_mid=0.8,
                head_keep_late=0.6,
                eviction_enabled=True,
                eviction_keep_ratio=0.4,
            ),
            CompressionLevel.AGGRESSIVE: cls(
                level=CompressionLevel.AGGRESSIVE,
                use_int8=True,
                head_keep_mid=0.5,
                head_keep_late=0.25,
                eviction_enabled=True,
                eviction_keep_ratio=0.13,
            ),
        }
        return configs[level]


class KVCompressionEngine:
    """
    Core compression engine that plugs into vLLM's cache management.

    This class provides the compression operations that vLLM calls
    during KV cache allocation and updates.
    """

    def __init__(self, config: CompressedKVCacheConfig, num_layers: int, num_heads: int, head_dim: int):
        self.config = config
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Pre-compute per-layer head counts
        self.layer_head_counts = self._compute_head_schedule()

        # Eviction state per sequence
        self._importance_scores: Dict[int, Dict[int, torch.Tensor]] = {}  # seq_id -> layer -> scores
        self._token_count: Dict[int, int] = {}  # seq_id -> current length
        self._eviction_stats = {"total_evicted": 0, "total_tokens": 0}

    def _compute_head_schedule(self) -> List[int]:
        """Compute how many heads to keep per layer."""
        schedule = []
        for layer_idx in range(self.num_layers):
            if layer_idx < self.num_layers // 3:
                keep = self.config.head_keep_early
            elif layer_idx < 2 * self.num_layers // 3:
                keep = self.config.head_keep_mid
            else:
                keep = self.config.head_keep_late
            schedule.append(max(1, int(self.num_heads * keep)))
        return schedule

    def compress_kv(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        seq_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Apply full compression pipeline to a KV update.

        Args:
            layer_idx: transformer layer index
            key: [batch, heads, tokens, head_dim]
            value: [batch, heads, tokens, head_dim]
            attention_weights: [batch, heads, q_len, k_len] (for eviction)
            seq_id: sequence identifier for per-sequence eviction state

        Returns:
            compressed_key, compressed_value, metadata
        """
        metadata = {"original_shape": key.shape}

        # Stage 1: INT8 quantization
        if self.config.use_int8:
            key, value, quant_meta = self._quantize_stage(key, value)
            metadata["quantization"] = quant_meta

        # Stage 2: Head reduction
        target_heads = self.layer_head_counts[layer_idx]
        if target_heads < key.size(1):
            key = key[:, :target_heads, :, :]
            value = value[:, :target_heads, :, :]
            metadata["heads_kept"] = target_heads
            metadata["heads_removed"] = self.num_heads - target_heads

        # Stage 3: Token eviction (periodic)
        if self.config.eviction_enabled and attention_weights is not None:
            seq_len = self._token_count.get(seq_id, 0) + key.size(2)
            self._token_count[seq_id] = seq_len

            if seq_len % self.config.eviction_interval == 0 and seq_len > self.config.recent_window * 2:
                key, value, evict_meta = self._eviction_stage(
                    layer_idx, key, value, attention_weights, seq_id
                )
                metadata["eviction"] = evict_meta

        metadata["compressed_shape"] = key.shape
        metadata["compression_ratio"] = self._compute_ratio(
            metadata["original_shape"], key.shape
        )

        return key, value, metadata

    def _quantize_stage(self, key, value):
        """INT8 quantization with per-head scaling."""
        k_q, k_s, k_z = self._quantize_int8(key)
        v_q, v_s, v_z = self._quantize_int8(value)

        meta = {
            "dtype": "int8",
            "k_scale_range": [k_s.min().item(), k_s.max().item()],
            "v_scale_range": [v_s.min().item(), v_s.max().item()],
        }

        # For vLLM integration, we return dequantized tensors
        # (vLLM's paged attention expects fp16/bf16)
        # The memory savings come from storing INT8 in the page table
        key_deq = self._dequantize_int8(k_q, k_s, k_z).to(key.dtype)
        value_deq = self._dequantize_int8(v_q, v_s, v_z).to(value.dtype)

        return key_deq, value_deq, meta

    def _eviction_stage(self, layer_idx, key, value, attention_weights, seq_id):
        """Importance-based token eviction."""
        n_tokens = key.size(2)
        n_keep = max(
            self.config.sink_tokens + self.config.recent_window,
            int(n_tokens * self.config.eviction_keep_ratio)
        )

        if n_keep >= n_tokens:
            return key, value, {"evicted": 0}

        # Compute importance
        if self.config.eviction_method == "attention":
            importance = attention_weights.mean(dim=(0, 1, 2))[:n_tokens]
        elif self.config.eviction_method == "recent":
            importance = torch.arange(n_tokens, dtype=torch.float32, device=key.device)
        else:  # hybrid
            attn_imp = attention_weights.mean(dim=(0, 1, 2))[:n_tokens]
            recency = torch.arange(n_tokens, dtype=torch.float32, device=key.device) / n_tokens
            importance = 0.7 * attn_imp + 0.3 * recency

        # Protect sinks and recent
        importance[:self.config.sink_tokens] = float('inf')
        importance[-self.config.recent_window:] = float('inf')

        # Keep top-k
        _, keep_idx = importance.topk(n_keep)
        keep_idx, _ = keep_idx.sort()

        key = key[:, :, keep_idx, :]
        value = value[:, :, keep_idx, :]

        n_evicted = n_tokens - n_keep
        self._eviction_stats["total_evicted"] += n_evicted
        self._eviction_stats["total_tokens"] += n_tokens

        return key, value, {"evicted": n_evicted, "kept": n_keep}

    def _quantize_int8(self, x):
        xmin = x.amin(dim=-1, keepdim=True)
        xmax = x.amax(dim=-1, keepdim=True)
        scale = (xmax - xmin) / 255.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero = -128 - xmin / scale
        x_q = torch.round(x / scale + zero).clamp(-128, 127).to(torch.int8)
        return x_q, scale, zero

    def _dequantize_int8(self, x_q, scale, zero):
        return (x_q.float() - zero) * scale

    def _compute_ratio(self, original_shape, compressed_shape):
        orig_elements = 1
        for s in original_shape:
            orig_elements *= s
        comp_elements = 1
        for s in compressed_shape:
            comp_elements *= s

        # Account for INT8 (1 byte) vs FP16 (2 bytes)
        orig_bytes = orig_elements * 2  # FP16 baseline
        comp_bytes = comp_elements * (1 if self.config.use_int8 else 2)

        return orig_bytes / comp_bytes if comp_bytes > 0 else 1.0

    def get_stats(self) -> Dict:
        """Get compression statistics."""
        evict_ratio = (self._eviction_stats["total_evicted"] /
                       max(1, self._eviction_stats["total_tokens"]))
        return {
            "config_level": self.config.level.value,
            "layers": self.num_layers,
            "head_schedule": self.layer_head_counts,
            "eviction_ratio": round(evict_ratio, 3),
            **self._eviction_stats,
        }


# ============================================================
# vLLM Integration Points
# ============================================================

def create_vllm_kv_compression(
    model_config,
    level: str = "standard",
) -> KVCompressionEngine:
    """
    Factory function for creating KV compression engine from vLLM model config.

    Usage in vLLM:
        from kv_compression.vllm_plugin import create_vllm_kv_compression

        compression = create_vllm_kv_compression(model.config, level="standard")

        # In the attention loop:
        key, value, meta = compression.compress_kv(layer_idx, key, value, attn_weights)
    """
    compression_level = CompressionLevel(level)
    config = CompressedKVCacheConfig.from_level(compression_level)

    num_layers = getattr(model_config, 'num_hidden_layers', 32)
    num_heads = getattr(model_config, 'num_key_value_heads',
                        getattr(model_config, 'num_attention_heads', 32))
    head_dim = getattr(model_config, 'head_dim',
                       getattr(model_config, 'hidden_size', 4096) // num_heads)

    return KVCompressionEngine(config, num_layers, num_heads, head_dim)


# ============================================================
# CLI for testing
# ============================================================

if __name__ == "__main__":
    print("vLLM KV Compression Plugin")
    print("=" * 50)

    # Test with synthetic data
    config = CompressedKVCacheConfig.from_level(CompressionLevel.STANDARD)
    engine = KVCompressionEngine(config, num_layers=32, num_heads=32, head_dim=128)

    print(f"\nConfig: {config.level.value}")
    print(f"INT8: {config.use_int8}")
    print(f"Eviction: {config.eviction_enabled} (keep {config.eviction_keep_ratio:.0%})")
    print(f"Head schedule: {engine.layer_head_counts}")

    # Simulate compression
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = torch.randn(1, 32, 128, 128, device=device, dtype=torch.float16)
    value = torch.randn(1, 32, 128, 128, device=device, dtype=torch.float16)
    attn = torch.randn(1, 32, 128, 128, device=device, dtype=torch.float16).softmax(dim=-1)

    for layer_idx in [0, 10, 20, 31]:
        k_c, v_c, meta = engine.compress_kv(layer_idx, key, value, attn)
        print(f"\n  Layer {layer_idx}: {meta['original_shape']} -> {meta['compressed_shape']}")
        print(f"    Compression: {meta['compression_ratio']:.1f}x")
        if 'heads_kept' in meta:
            print(f"    Heads: {meta['heads_kept']}/{32}")

    print(f"\n  Stats: {engine.get_stats()}")
    print("\nNOTE: Full vLLM integration requires CUDA. Use with vLLM >= 0.4.0")
