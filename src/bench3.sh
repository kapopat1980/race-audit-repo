#!/bin/bash
# five independent benchmark runs; bench_median.py takes element-wise medians (run from the repository root)
for i in 1 2 3 4 5; do python3 src/bench.py > results/bench_run$i.log 2>&1 && cp results/bench.json results/bench_run$i.json; done
python3 src/bench_median.py
