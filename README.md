# Linear Genetic Programming

A Rust framework for solving reinforcement learning and classification tasks using Linear Genetic Programming (LGP).

[![Build Status](https://github.com/urmzd/linear-gp/actions/workflows/experiments.yml/badge.svg)](https://github.com/urmzd/linear-gp/actions/workflows/experiments.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

Linear Genetic Programming (LGP) is a variant of genetic programming that evolves sequences of instructions operating on registers, similar to machine code. This framework provides:

- **Modular architecture** - Trait-based design for easy extension to new problem domains
- **Multiple problem types** - Built-in support for OpenAI Gym environments and classification tasks
- **Q-Learning integration** - Hybrid LGP + Q-Learning for enhanced reinforcement learning
- **Hyperparameter optimization** - Automated search using Optuna with PostgreSQL backend
- **Parallel evaluation** - Rayon-powered parallel fitness evaluation
- **Experiment automation** - Python CLI tools for batch experiments and visualization

### Supported Environments

| Environment | Type | Inputs | Actions | Description |
|-------------|------|--------|---------|-------------|
| CartPole | RL | 4 | 2 | Balance a pole on a moving cart |
| MountainCar | RL | 2 | 3 | Drive a car up a steep mountain |
| Iris | Classification | 4 | 3 | Classify iris flower species |

## Quick Start

### Prerequisites

| Dependency | Version | Installation |
|------------|---------|--------------|
| Rust | 1.70+ | [rustup.rs](https://rustup.rs/) |
| UV | Latest | [docs.astral.sh/uv](https://docs.astral.sh/uv/) |
| Docker | 20.10+ | [docker.com](https://www.docker.com/) |
| Docker Compose | 2.0+ | Included with Docker Desktop |
| just | Latest | `cargo install just` |

**macOS:**
```bash
brew install rust uv docker docker-compose
cargo install just
```

**Ubuntu:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get install docker.io docker-compose
cargo install just
```

### Installation

```bash
# Clone the repository
git clone https://github.com/urmzd/linear-gp.git
cd linear-gp

# Full setup (builds binary, installs Python deps, starts database)
just setup-full

# Or Python environment only
just setup
```

### First Experiment

```bash
# List available experiments
just list

# Run CartPole with pure LGP
just run cart_pole_lgp

# Run Iris classification
just run iris_baseline

# Run an example
just run-example cart_pole

# Run benchmarks
just bench
```

## Project Architecture

```
linear-gp/
├── crates/
│   ├── lgp/                    # Core LGP library
│   │   ├── src/
│   │   │   ├── lib.rs          # Library exports
│   │   │   ├── core/           # Core LGP implementation
│   │   │   ├── problems/       # Problem implementations
│   │   │   ├── extensions/     # Extended functionality
│   │   │   └── utils/          # Utility functions
│   │   ├── benches/            # Performance benchmarks
│   │   └── tests/              # Integration tests
│   └── lgp-cli/                # CLI application
│       └── src/
│           ├── main.rs         # CLI entry point
│           ├── commands/       # Subcommands (experiment)
│           ├── config_discovery.rs
│           └── config_override.rs
├── configs/                    # Experiment configurations
│   ├── iris_baseline/default.toml
│   ├── cart_pole_lgp/default.toml
│   └── ...
├── lgp_tools/                  # Python CLI package
│   ├── cli.py                  # Main CLI entry point
│   ├── commands/               # CLI subcommands
│   │   ├── analyze.py          # Results analysis
│   │   ├── experiment.py       # End-to-end pipeline
│   │   └── search.py           # Hyperparameter search
│   ├── config.py               # Config discovery
│   └── models.py               # Pydantic models
├── outputs/                    # Experiment outputs
│   ├── parameters/             # Optimized hyperparameters (JSON)
│   ├── <experiment>/           # Per-experiment outputs
│   │   └── <timestamp>/        # Timestamped runs
│   ├── tables/                 # CSV statistics
│   └── figures/                # PNG plots
└── docs/                       # Documentation
```

### Core Traits

The framework is built around these key traits:

- **`State`** - Represents an environment state with value access and action execution
- **`RlState`** - Extends State for RL environments with terminal state detection
- **`Core`** - Main trait defining the genetic algorithm components
- **`Fitness`** - Evaluates individual performance on states
- **`Breed`** - Two-point crossover for creating offspring
- **`Mutate`** - Mutation operators for genetic variation

See [docs/EXTENDING.md](docs/EXTENDING.md) for detailed trait documentation.

## Logging and Tracing

The framework includes comprehensive structured logging via the [tracing](https://docs.rs/tracing) ecosystem.

### Quick Start

```bash
# Default (info level, pretty format)
lgp run iris_baseline

# Verbose mode (debug level)
lgp -v run iris_baseline

# JSON format for log aggregation
lgp --log-format json run iris_baseline
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `RUST_LOG` | Control log level filtering | `lgp=debug`, `lgp=trace` |
| `LGP_LOG_FORMAT` | Override output format | `pretty`, `compact`, `json` |

### Log Level Guide

| Level | Use Case | Example Output |
|-------|----------|----------------|
| `error` | Fatal issues only | Panics, unrecoverable errors |
| `warn` | Potential problems | Deprecated features, suspicious values |
| `info` | Progress updates (default) | Generation stats, experiment start/complete |
| `debug` | Detailed diagnostics | Config loading, individual fitness scores |
| `trace` | Very verbose | Instruction execution, Q-table updates |

### Examples

```bash
# Debug level for LGP crate only
RUST_LOG=lgp=debug lgp run iris_baseline

# Trace level (very verbose - instruction-by-instruction)
RUST_LOG=lgp=trace lgp run iris_baseline

# Different levels for different modules
RUST_LOG=lgp::core=trace,lgp=info lgp run iris_baseline

# JSON output for log aggregation (ELK, Datadog, etc.)
lgp --log-format json run iris_baseline 2>&1 | jq .
```

## CLI Reference

### lgp (Rust CLI)

Run experiments with TOML-based configuration:

```bash
# List available experiments
lgp list

# Run experiment with default config
lgp run iris_baseline

# Run with optimal config (after search)
lgp run iris_baseline --config optimal

# Run with overrides
lgp run iris_baseline --override hyperparameters.program.max_instructions=50

# Q-learning parameter overrides
lgp run cart_pole_with_q --override operations.q_learning.alpha=0.5

# Preview config (dry-run)
lgp run iris_baseline --dry-run

# Run a Rust example
lgp example cart_pole

# List available examples
lgp example --list
```

**Available Experiments:**

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

### lgp-tools (Python)

Python CLI for hyperparameter search and analysis:

```bash
# Search hyperparameters (all configs)
uv run lgp-tools search

# Search specific config
uv run lgp-tools search iris_baseline

# Search with custom options
uv run lgp-tools search -t 20 -j 8 -m 5

# Analyze results (generates tables + figures)
uv run lgp-tools analyze

# Analyze with custom paths
uv run lgp-tools analyze --input outputs/output --output outputs

# Run complete experiment pipeline (search -> run -> analyze)
uv run lgp-tools experiment

# Run single config
uv run lgp-tools experiment iris_baseline

# Run with 20 iterations
uv run lgp-tools experiment -n 20

# Skip search phase (use existing optimal.toml)
uv run lgp-tools experiment --skip-search
```

## Examples

Run Rust API examples to see the library in action:

```bash
# List available examples
lgp example --list

# CartPole reinforcement learning example
lgp example cart_pole

# Iris classification example
lgp example iris_classification
```

## Hyperparameter Search

The framework includes automated hyperparameter optimization using [Optuna](https://optuna.org/).

### Setup

```bash
# Start PostgreSQL backend
just start-db

# Stop PostgreSQL
just stop-db

# Verify database is running
docker-compose ps
```

### Running Search

```bash
# Search for a specific config
uv run lgp-tools search cart_pole_lgp

# Search all configs (LGP first, then Q-Learning)
uv run lgp-tools search

# Search with custom options
uv run lgp-tools search cart_pole_with_q -t 100 -j 8 -m 15
```

### Available Configs

- `iris_baseline` - Iris classification (baseline)
- `iris_mutation` - Iris with mutation only
- `iris_crossover` - Iris with crossover only
- `iris_full` - Iris full (mutation + crossover)
- `cart_pole_lgp` - CartPole with pure LGP
- `cart_pole_with_q` - CartPole with Q-Learning
- `mountain_car_lgp` - MountainCar with pure LGP
- `mountain_car_with_q` - MountainCar with Q-Learning

### Parameters Searched

| Parameter | Range |
|-----------|-------|
| `max_instructions` | 1-100 |
| `external_factor` | 0.0-100.0 |
| `alpha` (Q-Learning) | 0.0-1.0 |
| `gamma` (Q-Learning) | 0.0-1.0 |
| `epsilon` (Q-Learning) | 0.0-1.0 |
| `alpha_decay` (Q-Learning) | 0.0-1.0 |
| `epsilon_decay` (Q-Learning) | 0.0-1.0 |

Results are saved to:
- `outputs/parameters/<config>.json`
- `configs/<config>/optimal.toml`

## Running Experiments

### Quick Start with Just

```bash
# List available experiments
just list

# Run individual experiments
just run cart_pole_lgp
just run cart_pole_with_q
just run mountain_car_lgp
just run iris_baseline

# Run with dry-run to preview config
just run iris_baseline --dry-run
```

### Running with lgp

```bash
# Run with default config
lgp run cart_pole_lgp

# Run with optimized config (after search)
lgp run cart_pole_lgp --config optimal

# Run with parameter overrides
lgp run cart_pole_lgp --override hyperparameters.n_generations=200
```

### Generating Visualizations

```bash
# Analyze results (generates tables + figures)
uv run lgp-tools analyze

# Analyze with custom paths
uv run lgp-tools analyze --input outputs/output --output outputs
```

### Output Structure

```
outputs/
├── parameters/                 # Optimized hyperparameters (JSON)
│   ├── cart_pole_lgp.json
│   └── ...
├── <experiment>/               # Experiment outputs (timestamped runs)
│   └── <timestamp>/            # e.g., 20260201_083623
│       ├── config/
│       │   └── config.toml     # Resolved config with seed/timestamp
│       ├── outputs/
│       │   ├── best.json       # Best individual from final generation
│       │   ├── median.json     # Median individual
│       │   ├── worst.json      # Worst individual
│       │   ├── population.json # Full population history
│       │   └── params.json     # Hyperparameters used
│       └── post_process/       # Post-processing outputs
├── tables/                     # Generated CSV statistics
│   └── <experiment>.csv
└── figures/                    # Generated PNG plots
    └── <experiment>.png

configs/
└── <experiment>/
    ├── default.toml            # Default configuration
    └── optimal.toml            # Generated by search
```

## Extending the Framework

The framework is designed to be extensible. You can add:

- New classification problems (e.g., XOR, MNIST)
- New RL environments (custom or gym-rs compatible)
- Custom genetic operators (mutation, crossover)
- Alternative fitness functions

See the [Quick Start](docs/EXTENDING.md#quick-start) for a minimal example, or [docs/EXTENDING.md](docs/EXTENDING.md) for the complete guide.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup instructions
- Code style guidelines
- Testing requirements
- Pull request process

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Brameier, M., & Banzhaf, W. (2007). *Linear Genetic Programming*. Springer.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
