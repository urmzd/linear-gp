# lgp-cli

Command-line interface for running Linear Genetic Programming experiments.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../../LICENSE)

## Overview

`lgp-cli` provides the `lgp` binary for running LGP experiments using TOML-based configuration files. It supports listing available experiments, running them with config overrides, and executing Rust API examples.

## Installation

```bash
cargo install lgp-cli
```

Or build from source:

```bash
cargo build --release -p lgp-cli
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

# Run a Rust API example
lgp example cart_pole

# List available examples
lgp example --list
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

### `lgp example <NAME>`

Runs a Rust API example from the `examples/` directory.

| Option | Description |
|--------|-------------|
| `--list` | List available examples |

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
