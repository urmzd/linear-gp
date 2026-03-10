# AGENTS.md

## Identity

You are an agent working on **linear-gp** — a Rust framework for solving reinforcement learning and classification tasks using Linear Genetic Programming (LGP). Supports CartPole, MountainCar, and Iris classification with optional Q-Learning integration.

## Architecture

Rust workspace + Python CLI tooling:

| Package | Path | Role |
|---------|------|------|
| `lgp` | `crates/lgp` | Core library — traits, evolutionary engine, built-in problems |
| `lgp-cli` | `crates/lgp-cli` | Rust CLI binary for running experiments with TOML config |
| `lgp-tools` | `lgp_tools/` | Python CLI for hyperparameter search (Optuna), experiment automation, analysis |

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

- `crates/lgp-cli/src/main.rs` — Rust CLI entry point
- `crates/lgp/src/lib.rs` — Core library exports
- `crates/lgp/src/core/` — Core LGP implementation
- `crates/lgp/src/problems/` — Problem implementations (CartPole, MountainCar, Iris)
- `lgp_tools/cli.py` — Python CLI entry point
- `configs/` — TOML experiment configurations (`default.toml`, `optimal.toml`)
- `outputs/` — Experiment results (parameters, tables, figures)
- `docs/EXTENDING.md` — Guide for adding new environments

## Commands

| Task | Command |
|------|---------|
| Build | `just build` or `cargo build --release` |
| Test | `just test` or `cargo test --release` |
| Bench | `just bench` or `cargo bench` |
| Lint (Rust) | `just lint` or `cargo clippy -- -D warnings` |
| Lint (Python) | `just lint-py` or `uv run ruff check lgp_tools` |
| Format (Rust) | `just fmt` or `cargo fmt` |
| Format (Python) | `just fmt-py` or `uv run ruff format lgp_tools` |
| Run experiment | `just run <name>` (e.g., `just run cart_pole_lgp`) |
| List experiments | `just list` |
| Hyperparameter search | `just search <config>` or `uv run lgp-tools search <config>` |
| Analyze results | `just analyze` or `uv run lgp-tools analyze` |
| Full pipeline | `just experiment <config>` (search → run → analyze) |
| Setup | `just init` (Python deps + Rust build + git hooks) |
| Full setup | Build + Python deps + PostgreSQL for Optuna |

## Code Style

- Rust 2021 edition, MIT license
- `cargo fmt` and `cargo clippy -- -D warnings`
- Python: `ruff` for formatting and linting
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
