#!/usr/bin/env bash

# This script is used to run the baseline experiments for the paper.
BENCHMARK_PREFIX="/tmp/iris" cargo nextest run iris --no-capture --release

./scripts/asset_generator.py --input "/tmp/iris" --output "/tmp/iris-tables" tables

mkdir -p "assets/experiments/baseline/figures"

for path in /tmp/iris-tables/*.csv; do
    echo "Processing $path"
    ./scripts/asset_generator.py --input "$path" --output "assets/experiments/baseline/figures" figures
    mv "$path" "assets/experiments/baseline/"
done