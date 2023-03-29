#!/usr/bin/env bash

mkdir -p "assets/experiments/baseline/figures"

# This script is used to run the baseline experiments for the paper.
BENCHMARK_PREFIX="/tmp/iris" cargo nextest run iris --no-capture --release

./scripts/asset_generator.py --input "/tmp/iris" --output "/tmp/iris-tables" tables

for path in /tmp/iris-tables/*.csv; do
    echo "Processing $path"
    mv "$path" "assets/experiments/baseline/"
done

./scripts/asset_generator.py --input "assets/experiments/baseline" --output "assets/experiments/baseline/figures" figures

ls -l "assets/experiments/baseline"