#!/usr/bin/env bash

mkdir -p "assets/experiments/baseline/figures"

BENCHMARK_PREFIX="assets/tmp" cargo nextest run iris --no-capture --release

./scripts/asset_generator.py --input "assets/tmp" --output "assets/experiments/baseline" tables
./scripts/asset_generator.py --input "assets/experiments/baseline" --output "assets/experiments/baseline/figures" figures

rm -rf "assets/tmp"
