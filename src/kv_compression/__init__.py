"""
KV Cache Compression Implementation

This module implements the three-stage KV cache compression pipeline:
1. INT8 quantization with per-head scaling
2. Layer-adaptive head reduction
3. Importance-based token eviction
"""

from .kv_cache_optimizer import KVCacheOptimizer
from .quantization import INT8Quantization
from .enhanced_kv_cache import EnhancedKVCache

__version__ = "1.0.0"
__all__ = ["KVCacheOptimizer", "INT8Quantization", "EnhancedKVCache"]
