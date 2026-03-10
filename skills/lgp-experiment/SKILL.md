---
name: lgp-experiment
description: Run Linear Genetic Programming experiments — train agents on CartPole, MountainCar, or Iris classification with optional Q-Learning and hyperparameter optimization. Use when running LGP experiments, tuning hyperparameters, or analyzing results.
argument-hint: [experiment-name]
---

# LGP Experiment Runner

Run and manage LGP experiments.

## Quick Start

```sh
# List available experiments
just list

# Run an experiment
just run cart_pole_lgp

# Run with optimized hyperparameters
just run cart_pole_lgp --config optimal
```

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

## Full Pipeline

```sh
# Search → Run → Analyze (end-to-end)
just experiment cart_pole_lgp

# Hyperparameter search only
just search cart_pole_lgp

# Analyze existing results
just analyze
```

## Logging

```sh
RUST_LOG=lgp=debug just run iris_baseline   # Debug output
RUST_LOG=lgp=trace just run iris_baseline   # Instruction-level trace
```

## Output Structure

Results are written to `outputs/` with timestamped runs containing config, best/median/worst individuals, and population history.
