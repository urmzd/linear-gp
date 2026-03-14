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
- `docs/EXTENDING.md` — Guide for adding new environments

## Commands

| Task | Command |
|------|---------|
| Build | `just build` or `cargo build --release` |
| Build (with plots) | `just build-plot` or `cargo build --release --features plot` |
| Test | `just test` or `cargo test --release` |
| Bench | `just bench` or `cargo bench` |
| Lint | `just lint` or `cargo clippy -- -D warnings` |
| Format | `just fmt` or `cargo fmt` |
| Run experiment | `just run <name>` (e.g., `just run cart_pole_lgp`) |
| List experiments | `just list` |
| Hyperparameter search | `just search <config>` or `lgp search <config>` |
| Analyze results | `just analyze` or `lgp analyze` |
| Full pipeline | `just experiment <config>` (search -> run -> analyze) |
| Setup | `just init` (Rust build + git hooks) |

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

See `docs/EXTENDING.md`. Implement the `State` (or `RlState`) trait, create a TOML config under `configs/`, and wire it into the CLI.
