# AGENTS.md

## Identity

You are an agent working on **linear-gp** — a Rust framework for solving reinforcement learning and classification tasks using Linear Genetic Programming (LGP). Supports CartPole, MountainCar, and Iris classification with optional Q-Learning integration.

## Architecture

Pure Rust workspace:

| Package | Path | Role |
|---------|------|------|
| `lgp` | `crates/lgp` | Core library — traits, evolutionary engine, built-in problems |
| `lgp-cli` | `crates/lgp-cli` | CLI binary (`lgp`) for running experiments, hyperparameter search, and analysis |

### Core Traits

| Trait | Purpose |
|-------|---------|
| `State` | Environment state with value access and action execution |
| `RlState` | Extends State for RL environments with terminal detection |
| `Core` | Main trait defining genetic algorithm components |
| `Fitness` | Evaluates individual performance |
| `Breed` | Two-point crossover for offspring |
| `Mutate` | Mutation operators for genetic variation |

## Key Files

- `crates/lgp-cli/src/main.rs` — CLI entry point
- `crates/lgp/src/lib.rs` — Core library exports
- `crates/lgp/src/core/` — Core LGP implementation
- `crates/lgp/src/problems/` — Problem implementations (CartPole, MountainCar, Iris)
- `configs/` — TOML experiment configurations (`default.toml`, `optimal.toml`)
- `outputs/` — Experiment results (parameters, tables, figures)
- `skills/lgp-experiment/SKILL.md` — Guide for running experiments and adding new environments

## Commands

| Task | Command |
|------|---------|
| Install | `cargo install --path crates/lgp-cli` |
| Build | `cargo build` |
| Build (with plots) | `cargo build --features plot` |
| Test | `cargo test` |
| Bench | `cargo bench` |
| Lint | `cargo clippy -- -D warnings` |
| Format | `cargo fmt` |
| Run experiment | `lgp run <name>` (e.g., `lgp run cart_pole_lgp`) |
| List experiments | `lgp list` |
| Hyperparameter search | `lgp search <config>` |
| Analyze results | `lgp analyze` |
| Full pipeline | `lgp experiment <config>` (search -> run -> analyze) |

## Code Style

- Rust 2021 edition, Apache-2.0 license
- `cargo fmt` and `cargo clippy -- -D warnings`
- Structured logging via `tracing` (`RUST_LOG=lgp=debug`)
- Parallel evaluation via `rayon`

## Supported Environments

| Environment | Type | Inputs | Actions |
|-------------|------|--------|---------|
| CartPole | RL | 4 | 2 |
| MountainCar | RL | 2 | 3 |
| Iris | Classification | 4 | 3 |

## Adding a New Environment

See `skills/lgp-experiment/SKILL.md`. Implement the `State` (or `RlState`) trait, create a TOML config under `configs/`, and wire it into the CLI.
