# KV Cache Compression Paper

This repository contains the code, data, and scripts for reproducing the experiments in the paper:

**"A Simple Composable Pipeline for High-Ratio KV Cache Compression"**

## Paper

The full paper is available in `paper/kv-compression-full.pdf`.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
make reproduce-all

# Run specific experiments
make reproduce-compression    # KV compression quality experiments
make reproduce-ablation       # Ablation studies
make reproduce-h200           # H200 validation
```

## Repository Structure

```
kv-compression-paper/
├── paper/                   # Paper PDF
├── src/                     # Source code
│   └── kv_compression/      # KV compression implementation
├── benchmarks/              # Benchmark scripts
├── results/                 # Experimental results
├── scripts/                 # Reproduction scripts
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Experiments

### 1. KV Compression Quality
Measures perplexity impact of different compression configurations on WikiText-2.

**Results:** `results/compression_quality/`

### 2. Ablation Studies
Analyzes individual components (quantization, head reduction, eviction).

**Results:** `results/ablation/`

### 3. H200 Validation
Production-scale validation on NVIDIA H200 NVL with Mistral-7B and Qwen3-8B.

**Results:** `results/h200_validation/`

### 4. LongBench Evaluation
Evaluates compression impact on long-context tasks (document QA, summarization, etc.).

**Results:** `results/longbench/`

## Citation

If you use this code or find the paper helpful, please cite:

```bibtex
@article{price2025kv,
  title={A Simple Composable Pipeline for High-Ratio KV Cache Compression},
  author={Price, Robert},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

Robert Price - bobby@blackweb.ai
