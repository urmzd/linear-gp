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
# Run CartPole with pure LGP
just cartpole-lgp

# Run CartPole with Q-Learning
just cartpole-q

# Run Iris classification
just iris

# Run benchmarks
just bench
```

## Project Architecture

```
linear-gp/
├── src/                        # Core LGP library
│   ├── main.rs                 # CLI entry point
│   ├── lib.rs                  # Library exports
│   ├── core/                   # Core LGP implementation
│   │   ├── config.rs           # CLI configuration
│   │   ├── environment.rs      # State and RlState traits
│   │   ├── program.rs          # Program structure
│   │   ├── instruction.rs      # Instruction definition
│   │   ├── instructions.rs     # Instruction collection
│   │   ├── registers.rs        # Register management
│   │   └── engines/            # Genetic algorithm engines
│   ├── problems/               # Problem implementations
│   │   ├── gym.rs              # OpenAI Gym environments
│   │   └── iris.rs             # Iris classification
│   └── extensions/             # Extended functionality
│       ├── q_learning.rs       # Q-Learning integration
│       ├── classification.rs   # Classification fitness
│       └── interactive.rs      # RL fitness evaluation
├── experiments/                # Thesis experiments (workspace member)
│   ├── src/                    # lgp-experiments CLI
│   └── assets/                 # Parameters, results, figures
│       ├── parameters/         # Optimized hyperparameters (JSON)
│       ├── output/             # Raw experiment outputs
│       └── experiments/        # Aggregated results and figures
├── lgp_tools/                  # Python CLI package
│   ├── cli.py                  # Main CLI entry point
│   ├── search.py               # Hyperparameter optimization
│   ├── run.py                  # Experiment execution
│   ├── analyze.py              # Tables and figures generation
│   └── pipelines.py            # Composable workflows
├── examples/                   # Rust API examples
├── tests/                      # Integration smoke tests
├── benches/                    # Performance benchmarks
└── docs/                       # Additional documentation
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

## CLI Reference

### lgp-experiments (Rust)

Run individual experiments or batch jobs:

```bash
# Run a specific experiment
cargo run -p lgp-experiments --release -- run <EXPERIMENT> [OPTIONS]

# Run all experiments in batch
cargo run -p lgp-experiments --release -- batch [OPTIONS]
```

**Available Experiments:**

| Experiment | Description |
|------------|-------------|
| `iris-baseline` | Iris baseline (no mutation, no crossover) |
| `iris-mutation` | Iris with mutation only |
| `iris-crossover` | Iris with crossover only |
| `iris-full` | Iris full (mutation + crossover) |
| `cart-pole-lgp` | CartPole with pure LGP |
| `cart-pole-q` | CartPole with Q-Learning |
| `mountain-car-lgp` | MountainCar with pure LGP |
| `mountain-car-q` | MountainCar with Q-Learning |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--n-generations N` | Override number of generations | Config value |
| `--output-prefix PATH` | Output directory | `experiments/assets/output` |

**Examples:**

```bash
# Run CartPole with custom generations
cargo run -p lgp-experiments --release -- run cart-pole-lgp --n-generations 200

# Run all experiments
cargo run -p lgp-experiments --release -- batch

# Run specific experiments
cargo run -p lgp-experiments --release -- batch --experiments iris-full,cart-pole-lgp
```

### lgp-tools (Python)

Python CLI for hyperparameter search, experiment orchestration, and analysis:

```bash
# Hyperparameter search
uv run lgp-tools search single <ENV>    # Search for a single environment
uv run lgp-tools search all             # Search all environments

# Run experiments
uv run lgp-tools run baseline           # Run iris baseline experiments
uv run lgp-tools run experiments N      # Run N iterations with aggregation

# Generate analysis
uv run lgp-tools analyze tables         # Generate CSV tables from results
uv run lgp-tools analyze figures        # Generate PNG fitness plots

# Pipelines (composable workflows)
uv run lgp-tools pipeline full          # Search -> experiments -> analyze
uv run lgp-tools pipeline quick         # Experiments -> analyze (skip search)
uv run lgp-tools pipeline baseline      # Iris baseline with analysis
```

## Examples

Run Rust API examples to see the library in action:

```bash
# CartPole reinforcement learning example
cargo run --example cart_pole

# Iris classification example
cargo run --example iris_classification
```

## Hyperparameter Search

The framework includes automated hyperparameter optimization using [Optuna](https://optuna.org/).

### Setup

```bash
# Start PostgreSQL backend
just db-start

# Verify database is running
docker-compose ps
```

### Running Search

```bash
# Search for a specific environment
just search cart-pole-lgp 40 4 10
# Args: environment, n_trials, n_threads, median_trials

# Search all environments (LGP first, then Q-Learning)
just search-all

# Or use lgp-tools directly
uv run lgp-tools search single cart-pole-q --n-trials 100 --n-threads 8 --median-trials 15
```

### Available Environments

- `iris-lgp` - Iris classification
- `cart-pole-lgp` - CartPole with pure LGP
- `cart-pole-q` - CartPole with Q-Learning
- `mountain-car-lgp` - MountainCar with pure LGP
- `mountain-car-q` - MountainCar with Q-Learning

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

Results are saved to `experiments/assets/parameters/<env>.json`.

## Running Experiments

### Quick Start with Just

```bash
# Run individual environments
just cartpole-lgp
just cartpole-q
just mountaincar-lgp
just mountaincar-q
just iris

# Run batch experiments
just batch-experiments
```

### Batch Experiments with Aggregation

```bash
# Run 10 iterations and aggregate results
just experiments 10

# Or use lgp-tools directly
uv run lgp-tools run experiments 10 --keep-artifacts
```

### Generating Visualizations

```bash
# Generate CSV tables from experiment results
just tables

# Generate fitness evolution plots
just figures
```

### Pipelines

Run complete workflows with a single command:

```bash
# Full pipeline: search -> experiments -> analyze
just pipeline-full

# Quick pipeline: experiments -> analyze (uses existing parameters)
just pipeline-quick 10

# Baseline pipeline: iris experiments with analysis
just pipeline-baseline
```

### Output Structure

```
experiments/assets/
├── parameters/                 # Optimized hyperparameters
│   ├── cart-pole-lgp.json
│   ├── cart-pole-q.json
│   ├── mountain-car-lgp.json
│   └── mountain-car-q.json
├── output/                     # Raw experiment outputs
│   └── <experiment>/
│       ├── population.json
│       ├── best.json
│       ├── median.json
│       └── worst.json
└── experiments/
    ├── baseline/               # Iris baseline results
    │   ├── iris_baseline.csv
    │   ├── iris_mutation.csv
    │   ├── iris_crossover.csv
    │   ├── iris_full.csv
    │   └── figures/
    └── aggregate_results/      # Aggregated RL results
        ├── cart_pole_lgp.csv
        ├── cart_pole_q.csv
        ├── mountain_car_lgp.csv
        ├── mountain_car_q.csv
        └── figures/
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
