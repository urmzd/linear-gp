# Linear Genetic Programming Framework
# Run `just --list` to see all available recipes

default:
    @just --list

# === BUILD ===

# Build release binary
build:
    cargo build --release

# Build with debug symbols
build-dev:
    cargo build

# Clean build artifacts
clean:
    cargo clean

# === TEST ===

# Run all tests
test:
    cargo test --release

# Run tests with verbose output
test-verbose:
    cargo test --release -- --nocapture

# Run tests with nextest (faster)
test-nextest:
    cargo nextest run --release

# Run benchmarks
bench:
    cargo bench

# === RUN EXPERIMENTS ===

# Default log level for experiments (can be overridden)
export RUST_LOG := env_var_or_default("RUST_LOG", "lgp=info,lgp_cli=info")

# Run an experiment by name
run name *args:
    cargo run -p lgp-cli --release -- run {{name}} {{args}}

# Run an experiment with verbose (debug) logging (logs to file)
run-verbose name *args:
    RUST_LOG=lgp=debug,lgp_cli=debug cargo run -p lgp-cli --release -- --log-file debug-{{name}}.log run {{name}} {{args}}

# Run an experiment with trace logging (very verbose, logs to file)
run-trace name *args:
    RUST_LOG=lgp=trace,lgp_cli=trace cargo run -p lgp-cli --release -- --log-file trace-{{name}}.log run {{name}} {{args}}

# Run example by name
run-example name:
    cargo run -p lgp-cli --release -- example {{name}}

# List available experiments
list:
    cargo run -p lgp-cli --release -- list

# List available examples
list-examples:
    cargo run -p lgp-cli --release -- example --list

# === EXPERIMENT PIPELINE ===

# Run full experiment pipeline (search -> run -> analyze)
experiment config="" *args:
    uv run lgp-tools experiment {{config}} {{args}}

# Run experiments without search (use existing optimal.toml)
experiment-quick config="" n="10":
    uv run lgp-tools experiment {{config}} --skip-search -n {{n}}

# === HYPERPARAMETER SEARCH ===

# Search hyperparameters for a config
search config *args:
    uv run lgp-tools search {{config}} {{args}}

# Search all configs
search-all *args:
    uv run lgp-tools search {{args}}

# === ANALYSIS ===

# Analyze experiment results
analyze *args:
    uv run lgp-tools analyze {{args}}

# === DATABASE ===

# Start PostgreSQL database
start-db:
    docker-compose up -d

# Stop PostgreSQL database
stop-db:
    docker-compose down

# === DEVELOPMENT ===

# Format Rust code
fmt:
    cargo fmt

# Format Python code
fmt-py:
    uv run ruff format lgp_tools

# Format all code
fmt-all: fmt fmt-py

# Run clippy linter
lint:
    cargo clippy -- -D warnings

# Lint Python code
lint-py:
    uv run ruff check lgp_tools

# Lint all code
lint-all: lint lint-py

# Run all checks (format, lint, test)
check: fmt lint test

# Check all (Rust + Python)
check-all: fmt-all lint-all test

# Fix Python lint issues
fix-py:
    uv run ruff check --fix lgp_tools

# Install git hooks from scripts/
init_:
    cp scripts/pre-commit .git/hooks/pre-commit
    cp scripts/pre-push .git/hooks/pre-push
    chmod +x .git/hooks/pre-commit .git/hooks/pre-push
    @echo "Git hooks installed."

# Generate and open documentation
docs:
    cargo doc --open

# Watch for changes and run tests
watch:
    cargo watch -x test

# === RELEASE ===

# Preview the next release
release-plan:
    sr plan

# Dry-run a release (no side effects)
release-dry-run:
    sr release --dry-run

# Execute a full release
release:
    sr release

# Generate/update changelog
changelog:
    sr changelog

# Show next version
next-version:
    sr version

# === SETUP ===

# Initialize dev environment: sync Python deps, build Rust, install git hooks
init:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Syncing Python environment..."
    uv sync --extra dev
    echo "Building release binary..."
    cargo build --release
    echo "Installing git hooks..."
    just init_
    echo "Ready to go."

# Verify all dependencies are installed
verify:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Verifying dependencies..."
    rustc --version
    cargo --version
    uv --version
    docker --version
    echo "All dependencies verified!"
