.PHONY: help reproduce-all benchmark-gpu benchmark-quick clean test

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

reproduce-all: benchmark-gpu ## Run all experiments (REAL inference, no simulation)
	@echo "All experiments complete! Results in results/"

benchmark-gpu: ## Run comprehensive GPU benchmark (6 models, 6 configs, ~45 min on RTX 4090)
	@echo "Running comprehensive GPU benchmark..."
	python benchmarks/comprehensive_gpu_benchmark.py

benchmark-quick: ## Quick validation (small models only, ~5 min)
	@echo "Running quick validation benchmark..."
	python benchmarks/comprehensive_gpu_benchmark.py --models "distilgpt2,gpt2" --n-texts 50 --n-problems 0

test: ## Run unit tests
	python -m pytest tests/ -v

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	@echo "Clean complete!"
