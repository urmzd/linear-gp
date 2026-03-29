# lgp

Command-line interface for running Linear Genetic Programming experiments.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](../../LICENSE)

## Overview

`lgp` is a CLI for running LGP experiments using TOML-based configuration files. It supports listing available experiments, running them with config overrides, hyperparameter search, result analysis, and end-to-end experiment pipelines.

## Installation

```bash
cargo install lgp
```

Or build from source:

```bash
cargo install --path crates/lgp-cli

# With plot support (PNG chart generation)
cargo install --path crates/lgp-cli --features plot
```

## Usage

```bash
# List available experiments
lgp list

# Run an experiment with default config
lgp run iris_baseline

# Run with optimal config (after hyperparameter search)
lgp run iris_baseline --config optimal

# Run with parameter overrides
lgp run iris_baseline --override hyperparameters.program.max_instructions=50

# Preview resolved config without running
lgp run iris_baseline --dry-run

# Search hyperparameters
lgp search iris_baseline --n-trials 40 --n-threads 4

# Analyze experiment results
lgp analyze

# Run full pipeline (search -> run -> analyze)
lgp experiment iris_baseline --iterations 10

```

## Global Options

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Enable debug-level logging |
| `--log-format <FORMAT>` | Log output format: `pretty` (default), `compact`, `json` |
| `--log-file <PATH>` | Write logs to a file instead of stdout |

## Commands

### `lgp list`

Lists all available experiments discovered from the `configs/` directory.

### `lgp run <EXPERIMENT>`

Runs an experiment by name. Configuration is loaded from `configs/<experiment>/default.toml` (or `optimal.toml` with `--config optimal`).

| Option | Description |
|--------|-------------|
| `--config <NAME>` | Config variant to use (`default` or `optimal`) |
| `--override <KEY=VALUE>` | Override a config parameter (repeatable) |
| `--dry-run` | Print resolved config and exit |

### `lgp search [CONFIG]`

Search for optimal hyperparameters. If no config is specified, searches all configs.

| Option | Description |
|--------|-------------|
| `--n-trials <N>` | Number of trials (default: 40) |
| `--n-threads <N>` | Parallel threads (default: 4) |
| `--median-trials <N>` | Runs for median (default: 10) |

### `lgp analyze`

Generate statistics tables (CSV) and optional plots (PNG) from experiment results.

| Option | Description |
|--------|-------------|
| `--input <DIR>` | Input directory (default: `outputs`) |
| `--output <DIR>` | Output directory (default: `outputs`) |

### `lgp experiment [CONFIG]`

Run end-to-end pipeline: search -> run -> analyze.

| Option | Description |
|--------|-------------|
| `--iterations <N>` | Number of experiment iterations (default: 10) |
| `--skip-search` | Skip hyperparameter search phase |
| `--skip-analyze` | Skip analysis phase |
| `--n-trials <N>` | Search trials (default: 40) |
| `--n-threads <N>` | Search threads (default: 4) |
| `--median-trials <N>` | Runs for median (default: 10) |

## Available Experiments

| Experiment | Description |
|------------|-------------|
| `iris_baseline` | Iris baseline (no mutation, no crossover) |
| `iris_mutation` | Iris with mutation only |
| `iris_crossover` | Iris with crossover only |
| `iris_full` | Iris full (mutation + crossover) |
| `cart_pole_lgp` | CartPole with pure LGP |
| `cart_pole_with_q` | CartPole with Q-Learning |
| `mountain_car_lgp` | MountainCar with pure LGP |
| `mountain_car_with_q` | MountainCar with Q-Learning |

## Logging

The CLI integrates with the `tracing` ecosystem. Control log levels via `RUST_LOG`:

```bash
# Debug level for all LGP modules
RUST_LOG=lgp=debug lgp run iris_baseline

# JSON output for log aggregation
lgp --log-format json run iris_baseline
```
