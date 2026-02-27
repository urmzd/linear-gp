# lgp

Core Rust library for Linear Genetic Programming.

[![Crates.io](https://img.shields.io/crates/v/lgp.svg)](https://crates.io/crates/lgp)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../../LICENSE)

## Overview

`lgp` provides a trait-based framework for evolving sequences of register-based instructions to solve reinforcement learning and classification tasks. It includes built-in support for OpenAI Gym environments, Iris classification, and hybrid LGP + Q-Learning.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
lgp = "1.3"
```

To enable OpenAI Gym environment support:

```toml
[dependencies]
lgp = { version = "1.3", features = ["gym"] }
```

## Core Traits

The framework is built around these key traits:

- **`State`** - Represents an environment state with value access and action execution
- **`RlState`** - Extends `State` for RL environments with terminal state detection
- **`Core`** - Main trait defining the genetic algorithm components (registers, instructions, individuals)
- **`Fitness`** - Evaluates individual performance on a set of states
- **`Breed`** - Two-point crossover for creating offspring
- **`Mutate`** - Mutation operators for genetic variation

## Modules

| Module | Description |
|--------|-------------|
| `core` | Core LGP engine — registers, instructions, individuals, and evolutionary loop |
| `problems` | Built-in problem implementations (CartPole, MountainCar, Iris) |
| `extensions` | Extended functionality including Q-Learning integration |
| `utils` | Utility functions for tracing, serialization, and configuration |

## Quick Example

```rust
use lgp::core::engines::core_engine::CoreEngine;
use lgp::problems::iris::IrisLgp;

fn main() {
    // Configure and run an Iris classification experiment
    let engine = CoreEngine::<IrisLgp>::builder()
        .n_generations(100)
        .n_individuals(100)
        .build()
        .unwrap();

    engine.run();
}
```

See the [examples](../../examples/) directory and [extension guide](../../docs/EXTENDING.md) for more.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `gym` | No | Enables OpenAI Gym environment support via `gymnasia` |

## Supported Environments

| Environment | Type | Inputs | Actions |
|-------------|------|--------|---------|
| CartPole | RL | 4 | 2 |
| MountainCar | RL | 2 | 3 |
| Iris | Classification | 4 | 3 |
