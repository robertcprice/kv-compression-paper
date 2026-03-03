# KV Cache Compression Paper

**Benchmark Study: Multi-Scale Compression on Large Language Models**

Robert Price | bobby@blackweb.ai | BlackWeb.ai

## Paper

The full paper is available in `paper/kv-compression-benchmark-2026-03.pdf` (4 pages).

## TL;DR

We tested KV cache compression on Qwen3-14B and Qwen3-32B with multiple compression levels:

| Compression | Memory Saved | PPL Cost | Verdict |
|-------------|--------------|----------|---------|
| INT8 (4x) | 44-48% | **<1%** | ✅ Production-ready |
| 5x | 44-48% | ~19% | ⚠️ Marginal |
| 10x | 44-48% | >100% | ❌ Not viable |
| 20x | 44-48% | >273% | ❌ Not viable |
| 40x | 44-48% | >500% | ❌ Not viable |

**Key finding**: INT8 quantization is genuinely free on current Qwen models, but "40x compression with <0.5% PPL" claims are false.

## Benchmark Results

### Large Models (H100 NVL 94GB)

#### Qwen3-14B

| Config | PPL | ΔPPL | Memory |
|--------|-----|------|--------|
| baseline | 17.57 | -- | 29.5 GB |
| int8_4x | 17.73 | +0.95% | 16.4 GB |
| 5x | 21.07 | +20.0% | 16.4 GB |
| 10x | 37.99 | +116% | 16.4 GB |
| 20x | 68.51 | +290% | 16.4 GB |
| 40x | 112.10 | +538% | 16.4 GB |

#### Qwen3-32B

| Config | PPL | ΔPPL | Memory |
|--------|-----|------|--------|
| baseline | 16.14 | -- | 65.6 GB |
| int8_4x | 16.25 | +0.66% | 34.4 GB |
| 5x | 19.20 | +18.9% | 34.4 GB |
| 10x | 34.01 | +111% | 34.4 GB |
| 20x | 60.23 | +273% | 34.4 GB |
| 40x | 97.09 | +502% | 34.4 GB |

### Consumer GPU Validation (RTX 4090 24GB)

| Model | FP16 PPL | INT8 PPL | ΔPPL | Memory Saved |
|-------|----------|----------|------|--------------|
| Qwen3-8B | 18.83 | 18.79 | **-0.23%** | 42.3% |
| Qwen3-4B | 28.80 | 28.97 | +0.60% | 44.8% |

## Key Findings

1. **INT8 is production-ready**: <1% PPL cost at 44-48% memory savings
2. **5x compression is marginal**: ~19% PPL cost - use only for throughput-critical apps
3. **10x+ is not viable**: 100-500% PPL increase - model becomes unusable
4. **Larger models more resilient**: 32B shows better compression tolerance than 14B
5. **"Free compression" claims are false**: Real compression has real costs

## Repository Structure

```
kv-compression-paper/
├── paper/                                    # Paper PDF and LaTeX source
│   ├── kv-compression-benchmark-2026-03.pdf  # Current paper
│   └── src/kv-compression.tex                # LaTeX source
├── results/                                  # Benchmark results
│   ├── h100_large_models_2026-03-03.json     # H100 results (14B, 32B)
│   ├── current_models_benchmark_2026-03-02.json  # RTX 4090 results
│   └── comprehensive_gpu_results_v5.json     # Legacy results
├── benchmarks/                               # Benchmark scripts
└── src/kv_compression/                       # Compression implementation
```

## Quick Start

```bash
pip install -r requirements.txt

# Run full benchmark (requires 94GB+ GPU)
python benchmarks/full_bench.py

# Run consumer GPU validation (requires 24GB+ GPU)
python benchmarks/comprehensive_gpu_benchmark.py
```

## Practical Recommendations

### Model Selection

| GPU | Recommended Setup |
|-----|-------------------|
| 8GB | Qwen3-4B + INT8 |
| 16GB | Qwen3-8B + INT8 |
| 24GB | Qwen3-14B + INT8 |
| 48GB+ | Qwen3-32B + INT8 |

### Production Guidelines

- ✅ **INT8 (4x)**: Recommended for all deployments
- ⚠️ **5x**: Only for throughput-critical, quality-tolerant applications
- ❌ **10x+**: Not recommended - severe quality degradation

## Citation

```bibtex
@article{price2026kv,
  title={KV Cache Compression on Large Language Models: A Multi-Scale Benchmark Study},
  author={Price, Robert},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License
