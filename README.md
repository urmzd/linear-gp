<p align="center">
  <h1 align="center">linear-gp</h1>
  <p align="center">
    A Rust framework for solving reinforcement learning and classification tasks using Linear Genetic Programming (LGP).
    <br /><br />
    <a href="https://github.com/urmzd/linear-gp/releases">Download</a>
    &middot;
    <a href="https://github.com/urmzd/linear-gp/issues">Report Bug</a>
    &middot;
    <a href="https://github.com/urmzd/linear-gp/tree/main/outputs">Experiments</a>
  </p>
</p>

<p align="center">
  <a href="https://github.com/urmzd/linear-gp/actions/workflows/ci.yml"><img src="https://github.com/urmzd/linear-gp/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

## Showcase

<p align="center">
  <img src="showcase/experiment-output.png" alt="LGP experiment output" width="600">
</p>

## Overview

Linear Genetic Programming (LGP) is a variant of genetic programming that evolves sequences of instructions operating on registers, similar to machine code. This framework provides:

- **Modular architecture** - Trait-based design for easy extension to new problem domains
- **Multiple problem types** - Built-in support for OpenAI Gym environments and classification tasks
- **Q-Learning integration** - Hybrid LGP + Q-Learning for enhanced reinforcement learning
- **Hyperparameter search** - Built-in random search with parallel evaluation
- **Parallel evaluation** - Rayon-powered parallel fitness evaluation
- **Experiment automation** - Full pipeline: search, run, and analyze from a single CLI
- **Optional plotting** - PNG chart generation via `plotters` (behind `--features plot`)

### Supported Environments

| Environment | Type | Inputs | Actions | Description |
|-------------|------|--------|---------|-------------|
| CartPole | RL | 4 | 2 | Balance a pole on a moving cart |
| MountainCar | RL | 2 | 3 | Drive a car up a steep mountain |
| Iris | Classification | 4 | 3 | Classify iris flower species |

## Install

**Prebuilt binary (recommended):**

```bash
curl -fsSL https://raw.githubusercontent.com/urmzd/linear-gp/main/install.sh | bash
```

You can pin a version or change the install directory:

```bash
LGP_VERSION=v1.0.0 LGP_INSTALL_DIR=~/.local/bin \
  curl -fsSL https://raw.githubusercontent.com/urmzd/linear-gp/main/install.sh | bash
```

**From source:**

```bash
git clone https://github.com/urmzd/linear-gp.git && cd linear-gp
cargo install --path crates/lgp-cli
```

## Quick Start

```bash
# List available experiments
lgp list

# Run CartPole with pure LGP
lgp run cart_pole_lgp

# Run Iris classification
lgp run iris_baseline

```

### Examples

Run the standalone Rust examples directly with cargo:

```bash
cargo run -p lgp-core --example cart_pole --features gym
cargo run -p lgp-core --example iris_classification
```

## CLI Reference

```bash
# Run experiment with default config
lgp run iris_baseline

# Run with optimal config (after search)
lgp run iris_baseline --config optimal

# Run with parameter overrides
lgp run iris_baseline --override hyperparameters.program.max_instructions=50

# Q-learning parameter overrides
lgp run cart_pole_with_q --override operations.q_learning.alpha=0.5

# Preview config (dry-run)
lgp run iris_baseline --dry-run

# Search hyperparameters
lgp search iris_baseline
lgp search iris_baseline --n-trials 20 --n-threads 8

# Analyze results (generates CSV tables + optional PNG plots)
lgp analyze
lgp analyze --input outputs --output outputs

# Run full experiment pipeline (search -> run -> analyze)
lgp experiment iris_baseline
lgp experiment iris_baseline --iterations 20
lgp experiment --skip-search
```

### Available Experiments

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

## Hyperparameter Search

The framework includes built-in hyperparameter search with parallel evaluation via `rayon`.

```bash
# Search for a specific config
lgp search cart_pole_lgp

# Search all configs (LGP first, then Q-Learning)
lgp search

# Search with custom options
lgp search cart_pole_with_q --n-trials 100 --n-threads 8 --median-trials 15
```

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

## Visualizations

```bash
# Analyze results (generates CSV tables)
lgp analyze

# Build with plot feature for PNG chart generation
cargo install --path crates/lgp-cli --features plot
lgp analyze
```

## Output Structure

```
outputs/
├── parameters/                 # Optimized hyperparameters (JSON)
│   ├── cart_pole_lgp.json
│   └── ...
├── <experiment>/               # Experiment outputs (timestamped runs)
│   └── <timestamp>/
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
└── figures/                    # Generated PNG plots (with --features plot)
    └── <experiment>.png
```

## Logging

```bash
# Default (info level, pretty format)
lgp run iris_baseline

# Verbose mode (debug level)
lgp -v run iris_baseline

# JSON format for log aggregation
lgp --log-format json run iris_baseline

# Fine-grained control via RUST_LOG
RUST_LOG=lgp=debug lgp run iris_baseline
RUST_LOG=lgp::core=trace,lgp=info lgp run iris_baseline
```

| Level | Use Case |
|-------|----------|
| `error` | Fatal issues only |
| `warn` | Potential problems |
| `info` | Progress updates (default) |
| `debug` | Detailed diagnostics |
| `trace` | Instruction-by-instruction execution |

## Packages

| Package | Description |
|---------|-------------|
| [lgp-core](crates/lgp/README.md) | Core library — traits, evolutionary engine, built-in problems |
| [lgp](crates/lgp-cli/README.md) | CLI binary for running experiments, search, and analysis |

## Extending the Framework

The framework is built around these key traits:

- **`State`** - Represents an environment state with value access and action execution
- **`RlState`** - Extends State for RL environments with terminal state detection
- **`Core`** - Main trait defining the genetic algorithm components
- **`Fitness`** - Evaluates individual performance on states
- **`Breed`** - Two-point crossover for creating offspring
- **`Mutate`** - Mutation operators for genetic variation

You can add new classification problems, RL environments, genetic operators, and fitness functions. See [skills/lgp-experiment/SKILL.md](skills/lgp-experiment/SKILL.md) for the complete guide.

## Thesis & References

The accompanying thesis, *Reinforced Linear Genetic Programming*, and full references are maintained in a [separate repository](https://github.com/urmzd/rlgp-thesis).
