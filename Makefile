.PHONY: flagship flagship-ci

flagship:
	python3 benchmarks/run_codesign_benchmarks.py --seeds 10

flagship-ci:
	python3 benchmarks/run_codesign_benchmarks.py --seeds 3
