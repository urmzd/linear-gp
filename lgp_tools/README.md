# lgp-tools

Python CLI tools for hyperparameter search, experiment automation, and result analysis.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)

## Overview

`lgp-tools` provides a Python CLI built with [Typer](https://typer.tiangolo.com/) that wraps the Rust `lgp` binary to automate the full experimentation pipeline: hyperparameter optimization with [Optuna](https://optuna.org/), batch experiment execution, and statistical analysis with visualization.

## Prerequisites

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Runtime |
| UV | Latest | Package management |
| Docker | 20.10+ | PostgreSQL backend for Optuna |

## Installation

```bash
# Install Python dependencies
uv sync

# Start the PostgreSQL backend (required for hyperparameter search)
just start-db
```

## Usage

```bash
# Run hyperparameter search for all configs
uv run lgp-tools search

# Search a specific experiment
uv run lgp-tools search iris_baseline

# Search with custom options
uv run lgp-tools search -t 20 -j 8 -m 5

# Analyze results (generates tables + figures)
uv run lgp-tools analyze

# Run complete pipeline (search -> run -> analyze)
uv run lgp-tools experiment

# Run single experiment pipeline
uv run lgp-tools experiment iris_baseline

# Run with custom iterations, skip search
uv run lgp-tools experiment -n 20 --skip-search
```

## Commands

### `search [EXPERIMENT]`

Runs hyperparameter optimization using Optuna with a PostgreSQL backend.

| Option | Description |
|--------|-------------|
| `-t, --trials` | Number of Optuna trials |
| `-j, --jobs` | Number of parallel jobs |
| `-m, --min-trials` | Minimum trials before pruning |

**Parameters searched:**

| Parameter | Range |
|-----------|-------|
| `max_instructions` | 1-100 |
| `external_factor` | 0.0-100.0 |
| `alpha` (Q-Learning) | 0.0-1.0 |
| `gamma` (Q-Learning) | 0.0-1.0 |
| `epsilon` (Q-Learning) | 0.0-1.0 |
| `alpha_decay` (Q-Learning) | 0.0-1.0 |
| `epsilon_decay` (Q-Learning) | 0.0-1.0 |

Results are saved to `outputs/parameters/<config>.json` and `configs/<config>/optimal.toml`.

### `analyze`

Generates statistical tables (CSV) and visualizations (PNG) from experiment outputs.

| Option | Description |
|--------|-------------|
| `--input` | Input directory (default: `outputs/output`) |
| `--output` | Output directory (default: `outputs`) |

### `experiment [EXPERIMENT]`

Runs the complete pipeline: hyperparameter search, batch experiment execution, and analysis.

| Option | Description |
|--------|-------------|
| `-n, --iterations` | Number of experiment iterations |
| `--skip-search` | Skip search phase, use existing `optimal.toml` |

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `typer` | CLI framework |
| `optuna` | Hyperparameter optimization |
| `pandas`, `numpy` | Data processing |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn`, `scipy` | Statistical analysis |
| `pydantic` | Data validation |
| `SQLAlchemy`, `psycopg2-binary` | PostgreSQL backend |
