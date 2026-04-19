default: check

# Initialize project
init:
    rustup component add clippy rustfmt
    sr init

# Install binary to PATH
install:
    cargo install --path crates/lgp-cli

# Build debug binary
build:
    cargo build --workspace

# Run with arguments
run *ARGS:
    cargo run -p lgp-cli -- {{ARGS}}

# Run all tests
test:
    cargo test --workspace

# Run clippy linter
lint:
    cargo clippy --workspace -- -D warnings

# Format code
fmt:
    cargo fmt --all

# Check formatting without modifying
check-fmt:
    cargo fmt --all -- --check

# Record showcase with teasr
record:
    teasr showme

# Quality gate: format + lint + test
check: check-fmt lint test

# Full CI gate: format + lint + build + test
ci: check-fmt lint build test
