"""
KV Cache Compression Pipeline

Three-stage composable compression:
1. INT8 quantization with per-head scaling (4x)
2. Layer-adaptive head reduction (1.3-2x)
3. Importance-based token eviction (2-8x)

Combined: 10-40x total compression.
"""

from .enhanced_kv_cache import (
    INT8QuantizedCache,
    ImportanceEvictionCache,
    EnhancedKVCacheManager,
)
from .kv_cache_optimizer import (
    QuantizedKVCache,
    ImportanceBasedKVCache,
    SpatialLocalityOptimizer,
    LayerAdaptiveCache,
    AdaptiveKVCacheManager,
)
from .flash_attention_compat import (
    CompressedKVAttention,
    CompressedKVCacheForAttention,
    check_compatibility,
)
from .vllm_plugin import (
    CompressedKVCacheConfig,
    CompressionLevel,
    KVCompressionEngine,
    create_vllm_kv_compression,
)

__version__ = "0.2.0"
