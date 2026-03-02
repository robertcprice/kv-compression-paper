# KV Cache Compression Paper

**"A Simple Composable Pipeline for High-Ratio KV Cache Compression"**

Robert Price | bobby@blackweb.ai | BlackWeb.ai

## Paper

The full paper is available in `paper/kv-compression-full.pdf`.

## Quick Start

```bash
pip install -r requirements.txt

# Run comprehensive GPU benchmark (6 models, 6 compression configs)
python benchmarks/comprehensive_gpu_benchmark.py

# Run with specific models
python benchmarks/comprehensive_gpu_benchmark.py --models "gpt2,microsoft/phi-2"

# Customize sample sizes
python benchmarks/comprehensive_gpu_benchmark.py --n-texts 200 --n-problems 100
```

## Repository Structure

```
kv-compression-paper/
├── paper/                        # Paper PDF
├── src/kv_compression/           # Core implementation
│   ├── enhanced_kv_cache.py      # Three-stage compression pipeline (heap-based eviction)
│   ├── kv_cache_optimizer.py     # Adaptive KV cache manager
│   ├── quantization.py           # INT8/INT4 quantization
│   ├── flash_attention_compat.py # FlashAttention-2/3 integration layer
│   └── vllm_plugin.py            # vLLM serving integration
├── benchmarks/                   # REAL benchmark scripts
│   └── comprehensive_gpu_benchmark.py  # Multi-model, multi-config GPU benchmark
├── results/                      # Experimental results
│   └── comprehensive_gpu_results_v5.json  # Full results from RTX 4090
├── requirements.txt
└── Makefile
```

## Real Benchmark Results (RTX 4090, March 2026)

All results are from **actual model inference** on NVIDIA GeForce RTX 4090 (24GB VRAM).
100 WikiText-2 texts, 50 GSM8K problems. PyTorch 2.10.0, CUDA 12.8, transformers 5.2.0.

### Three-Stage Compression Pipeline

1. **INT8 Quantization**: Per-channel quantize/dequantize round-trip (4x memory reduction)
2. **Layer-Adaptive Head Reduction**: Early layers keep all heads, mid/late layers progressively reduce
3. **Importance-Based Token Eviction**: Attention-weighted importance scoring with recency fallback

### Perplexity Impact (WikiText-2, prefix+continuation method)

| Model | Params | Baseline PPL | INT8 (4x) | 5x | 10x | 20x | 40x |
|-------|--------|-------------|-----------|-----|------|------|------|
| distilgpt2 | 82M | 52.51 | +21.9% | +37.4% | +77.2% | +161.8% | +299.2% |
| gpt2 | 124M | 35.28 | +8.8% | +19.9% | +69.6% | +171.6% | +376.7% |
| gpt2-medium | 355M | 25.49 | +10.9% | +18.7% | +54.8% | +128.9% | +465.7% |
| **phi-2** | **2.7B** | **12.44** | **+6.2%** | **+12.6%** | **+61.1%** | **+99.6%** | **+113.0%** |
| **Mistral-7B** | **7.2B** | **7.16** | **+3.1%** | **+6.5%** | **+44.8%** | **+57.1%** | **+58.0%** |
| Qwen2.5-7B | 7.6B | 7.77 | +33.3% | +41.6% | +147.3% | +170.0% | +364.3% |

### GSM8K Math Reasoning (50 problems)

| Model | Baseline | INT8 (4x) | 20x | 40x |
|-------|----------|-----------|-----|-----|
| phi-2 (2.7B) | 8.0% | 10.0% (+25%) | 4.0% (-50%) | 0.0% (-100%) |
| Mistral-7B | 2.0% | 10.0% (+400%) | 2.0% (0%) | 0.0% (-100%) |
| Qwen2.5-7B | 10.0% | 4.0% (-60%) | 2.0% (-80%) | 0.0% (-100%) |

*Note: GSM8K delta percentages for Mistral-7B are noisy due to very low baseline accuracy (1/50). Base models (non-instruct) have near-zero math reasoning ability.*

### Key Findings (Honest Assessment)

**What works:**
- **INT8 quantization on 7B models is viable**: Mistral-7B shows only +3.1% PPL increase at 4x compression. This is a genuine, useful result.
- **Larger models are more resilient**: INT8 PPL impact scales inversely with model size (22% at 82M vs 3% at 7.2B).
- **The compression pipeline is composable**: Each stage adds independently measurable degradation.

**What doesn't work (and the paper originally overclaimed):**
- **40x compression destroys quality**: PPL increases range from +58% (best case, Mistral-7B) to +465% (worst case, gpt2-medium). The original paper claimed "<0.5% PPL at 38.5x" — **this was fabricated**.
- **10x+ compression is lossy**: Even with the best model (Mistral-7B), 10x gives +44.8% PPL.
- **GSM8K accuracy collapses at high compression**: 40x compression kills all math reasoning across all models.
- **Architecture sensitivity is high**: Qwen2.5-7B is much more sensitive to INT8 (+33.3%) than Mistral-7B (+3.1%), despite similar parameter counts.

**Practical recommendation:**
INT8-only (4x compression) is the sweet spot for production use on 7B+ models. Beyond that, head reduction at 5x is acceptable (+6.5% PPL on Mistral-7B). Anything above 10x should only be used for latency-sensitive scenarios where quality degradation is acceptable.

### Previous (Fabricated) Results

The original benchmark scripts in this repo generated **simulated results** — synthetic accuracy
numbers that didn't come from actual model inference. Those files are prefixed with `SIMULATED_`
in the results directory for transparency. The current results above are from real forward passes
with real KV compression applied to real model weights.

## Integration

### FlashAttention Compatibility

```python
from kv_compression.flash_attention_compat import CompressedKVCacheForAttention

cache = CompressedKVCacheForAttention(
    num_layers=32, num_heads=32, head_dim=128,
    max_seq_len=4096, use_int8=True, device="cuda"
)
cache.update(layer_idx, key, value)
output = cache.attend(layer_idx, query)
```

Supports FlashAttention-2, FlashAttention-3 (H100+), and PyTorch SDPA fallback.

### vLLM Plugin

```python
from kv_compression.vllm_plugin import create_vllm_kv_compression, CompressionLevel

engine = create_vllm_kv_compression(model.config, level="standard")
key, value, meta = engine.compress_kv(layer_idx, key, value, attention_weights)
```

## Reproducing Results

```bash
# Full benchmark (requires CUDA GPU, ~45 min on RTX 4090)
python benchmarks/comprehensive_gpu_benchmark.py

# Quick validation (small models only, ~5 min)
python benchmarks/comprehensive_gpu_benchmark.py --models "distilgpt2,gpt2" --n-texts 50 --n-problems 0
```

## Citation

```bibtex
@article{price2026kv,
  title={A Simple Composable Pipeline for High-Ratio KV Cache Compression},
  author={Price, Robert},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details.
