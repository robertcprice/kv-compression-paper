.PHONY: help reproduce-all reproduce-compression reproduce-ablation reproduce-h200 clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

reproduce-all: ## Run all experiments
	@echo "Running all KV compression experiments..."
	python scripts/run_compression_experiments.py
	python scripts/run_ablation_experiments.py
	python scripts/run_h200_validation.py
	@echo "All experiments complete!"

reproduce-compression: ## Run compression quality experiments
	@echo "Running KV compression quality experiments..."
	python scripts/run_compression_experiments.py

reproduce-ablation: ## Run ablation studies
	@echo "Running ablation studies..."
	python scripts/run_ablation_experiments.py

reproduce-h200: ## Run H200 validation experiments
	@echo "Running H200 validation..."
	python scripts/run_h200_validation.py

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	@echo "Clean complete!"
