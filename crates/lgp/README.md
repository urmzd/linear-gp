# lgp

Core Rust library for Linear Genetic Programming.

[![Crates.io](https://img.shields.io/crates/v/lgp-core.svg)](https://crates.io/crates/lgp-core)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](../../LICENSE)

## Overview

`lgp` provides a trait-based framework for evolving sequences of register-based instructions to solve reinforcement learning and classification tasks. It includes built-in support for OpenAI Gym environments, Iris classification, and hybrid LGP + Q-Learning.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
lgp-core = "1.6"
```

To enable OpenAI Gym environment support:

```toml
[dependencies]
lgp-core = { version = "1.6", features = ["gym"] }
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
use itertools::Itertools;
use lgp::core::engines::core_engine::HyperParametersBuilder;
use lgp::core::engines::status_engine::{Status, StatusEngine};
use lgp::core::instruction::InstructionGeneratorParametersBuilder;
use lgp::core::program::ProgramGeneratorParametersBuilder;
use lgp::problems::iris::IrisEngine;

fn main() {
    let instruction_parameters = InstructionGeneratorParametersBuilder::default()
        .n_actions(3)
        .n_inputs(4)
        .n_extras(2)
        .external_factor(1.0)
        .build()
        .unwrap();

    let program_parameters = ProgramGeneratorParametersBuilder::default()
        .max_instructions(50)
        .instruction_generator_parameters(instruction_parameters)
        .build()
        .unwrap();

    let parameters = HyperParametersBuilder::<IrisEngine>::default()
        .program_parameters(program_parameters)
        .population_size(100)
        .n_generations(50)
        .n_trials(1)
        .mutation_percent(0.5)
        .crossover_percent(0.5)
        .gap(0.5)
        .default_fitness(0.0)
        .build()
        .unwrap();

    let populations: Vec<_> = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    let best = populations.last().unwrap().first().unwrap();
    let accuracy = (StatusEngine::get_fitness(best) / 150.0) * 100.0;
    println!("Best accuracy: {accuracy:.1}%");
}
```

See the [examples](examples/) directory and [extension guide](../../skills/lgp-experiment/SKILL.md) for more.

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
