# Linear Genetic Programming Framework - Justfile
# Run `just --list` to see all available recipes

# Default recipe: show help
default:
    @just --list

# ============================================================================
# BUILD COMMANDS
# ============================================================================

# Build release binary
build:
    cargo build --release

# Build with debug symbols
build-dev:
    cargo build

# Clean build artifacts
clean:
    cargo clean

# ============================================================================
# TEST COMMANDS
# ============================================================================

# Run all tests
test:
    cargo test --release

# Run tests with verbose output
test-verbose:
    cargo test --release -- --nocapture

# Run tests with nextest (faster)
test-nextest:
    cargo nextest run --release

# Run a specific test suite
test-suite name:
    cargo test --release {{name}}

# Run benchmarks
bench:
    cargo bench

# ============================================================================
# EXPERIMENT COMMANDS
# ============================================================================

# Run CartPole with pure LGP
cartpole-lgp *args:
    cargo run -p lgp-experiments --release -- run cart-pole-lgp {{args}}

# Run CartPole with Q-Learning
cartpole-q *args:
    cargo run -p lgp-experiments --release -- run cart-pole-q {{args}}

# Run MountainCar with pure LGP
mountaincar-lgp *args:
    cargo run -p lgp-experiments --release -- run mountain-car-lgp {{args}}

# Run MountainCar with Q-Learning
mountaincar-q *args:
    cargo run -p lgp-experiments --release -- run mountain-car-q {{args}}

# Run Iris classification experiments
iris *args:
    cargo run -p lgp-experiments --release -- run iris-full {{args}}

# Run all experiments in batch
batch-experiments *args:
    cargo run -p lgp-experiments --release -- batch {{args}}

# ============================================================================
# PYTHON TOOLING (via UV)
# ============================================================================

# Setup Python environment with UV
setup:
    uv sync

# Hyperparameter search for a specific environment
search env n_trials="40" n_threads="4" median_trials="10":
    uv run python -m lgp_tools.cli search single {{env}} --n-trials {{n_trials}} --n-threads {{n_threads}} --median-trials {{median_trials}}

# Search hyperparameters for all environments
search-all:
    uv run python -m lgp_tools.cli search all

# Run baseline experiments (iris variants)
baseline:
    uv run python -m lgp_tools.cli run baseline

# Run N experiment iterations with aggregation
experiments n="10":
    uv run python -m lgp_tools.cli run experiments {{n}}

# Generate tables from experiment results
tables input="experiments/assets/output" output="experiments/assets/tables":
    uv run python -m lgp_tools.cli analyze tables --input {{input}} --output {{output}}

# Generate figures from tables
figures input="experiments/assets/tables" output="experiments/assets/figures":
    uv run python -m lgp_tools.cli analyze figures --input {{input}} --output {{output}}

# ============================================================================
# PIPELINES
# ============================================================================

# Full pipeline: search-all -> experiments -> analyze
pipeline-full:
    uv run python -m lgp_tools.cli pipeline full

# Quick pipeline: experiments -> analyze (skip search)
pipeline-quick n="10":
    uv run python -m lgp_tools.cli pipeline quick --iterations {{n}}

# Baseline pipeline: iris experiments with analysis
pipeline-baseline:
    uv run python -m lgp_tools.cli pipeline baseline

# ============================================================================
# DATABASE
# ============================================================================

# Start PostgreSQL database for Optuna
db-start:
    docker-compose up -d

# Stop PostgreSQL database
db-stop:
    docker-compose down

# ============================================================================
# DEVELOPMENT
# ============================================================================

# Format code
fmt:
    cargo fmt

# Run clippy linter
lint:
    cargo clippy -- -D warnings

# Run all checks (format, lint, test)
check: fmt lint test

# Generate and open documentation
docs:
    cargo doc --open

# Watch for changes and run tests
watch:
    cargo watch -x test

# ============================================================================
# FULL SETUP
# ============================================================================

# Full development setup (Rust + Python + Database)
setup-full:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Setting up development environment..."

    # Build release binary
    echo "Building release binary..."
    cargo build --release

    # Setup Python environment with UV
    echo "Setting up Python environment with UV..."
    uv sync

    # Start database
    echo "Starting PostgreSQL..."
    docker-compose up -d

    echo "Setup complete!"

# Verify all dependencies are installed
verify:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Verifying dependencies..."

    echo -n "Rust: "
    rustc --version || { echo "NOT FOUND"; exit 1; }

    echo -n "Cargo: "
    cargo --version || { echo "NOT FOUND"; exit 1; }

    echo -n "UV: "
    uv --version || { echo "NOT FOUND"; exit 1; }

    echo -n "Docker: "
    docker --version || { echo "NOT FOUND"; exit 1; }

    echo -n "Docker Compose: "
    docker-compose --version || { echo "NOT FOUND"; exit 1; }

    echo -n "Release binary: "
    if [ -f ./target/release/lgp ]; then
        echo "OK"
    else
        echo "NOT FOUND (run 'just build')"
    fi

    echo "All dependencies verified!"
