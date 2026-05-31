.PHONY: flagship flagship-ci reproduce test lint

flagship:
	python3 benchmarks/run_codesign_benchmarks.py --seeds 10

flagship-ci:
	python3 benchmarks/run_codesign_benchmarks.py --seeds 3

reproduce: ## One-key reproducibility (fixed seed)
	python3 benchmarks/run_codesign_benchmarks.py --seeds 3 --seed-start 42

test:
	python3 -m pytest tests/ -v --tb=short

lint:
	ruff check --fix diff_surrogate/ tests/ benchmarks/
	ruff format diff_surrogate/ tests/ benchmarks/
