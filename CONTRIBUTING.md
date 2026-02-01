# Contributing to Linear Genetic Programming

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Code Style](#code-style)
3. [Testing](#testing)
4. [Pull Request Process](#pull-request-process)
5. [Areas for Contribution](#areas-for-contribution)

## Development Setup

### Prerequisites

Ensure you have the following installed:

| Tool | Version | Purpose |
|------|---------|---------|
| Rust | 1.70+ | Core framework |
| Python | 3.11+ | Automation scripts |
| uv | Latest | Python package management |
| Docker | 20.10+ | PostgreSQL for Optuna |
| Docker Compose | 2.0+ | Container orchestration |
| just | Latest | Task runner |

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/urmzd/linear-gp.git
cd linear-gp

# Install just (task runner)
cargo install just

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run full setup (builds binary, Python environment, database)
just setup-full

# Verify everything is working
just verify
just test
```

### Manual Setup

If `just setup-full` doesn't work for your environment:

```bash
# Build the project
cargo build --release

# Set up Python environment with uv
uv sync

# Start PostgreSQL for hyperparameter search
docker-compose up -d
```

### IDE Setup

**VS Code:**
- Install the `rust-analyzer` extension
- Install the `Even Better TOML` extension
- The project includes standard Rust formatting settings

**IntelliJ/CLion:**
- Install the Rust plugin
- Enable rustfmt on save

## Code Style

### Rust

We follow standard Rust conventions with some project-specific guidelines:

**Formatting:**
```bash
# Format all code
just fmt

# Check formatting without modifying
cargo fmt -- --check
```

**Linting:**
```bash
# Run clippy with strict warnings
just lint

# Or directly
cargo clippy -- -D warnings
```

**Guidelines:**
- Use `rustfmt` defaults (no custom configuration)
- All public items should have documentation comments
- Use `#[derive(...)]` where possible
- Prefer iterators over manual loops
- Use `?` for error propagation
- Avoid `unwrap()` in library code; use `expect()` with descriptive messages or proper error handling

**Naming conventions:**
- Types: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Trait implementations: Match the trait's naming

### Python

For automation scripts in `lgp_tools/`:

- Follow PEP 8 style guidelines
- Use type hints where practical
- Use `pathlib` for file paths
- Document functions with docstrings

```bash
# Format Python code (if black is installed)
black lgp_tools/

# Check with flake8 (if installed)
flake8 lgp_tools/
```

## Testing

### Running Tests

```bash
# Run all tests
just test

# Run tests for specific crate
cargo test -p lgp
cargo test -p lgp-cli

# Run tests with output
just test-verbose

# Run specific test suite
cargo test -p lgp iris

# Run with nextest (faster)
just test-nextest

# Run benchmarks
just bench

# Test experiment CLI (dry-run)
lgp run iris_baseline --dry-run
```

### Writing Tests

**Unit tests:** Place in the same file as the code being tested:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() {
        // Arrange
        let input = ...;

        // Act
        let result = function_under_test(input);

        // Assert
        assert_eq!(result, expected);
    }
}
```

**Integration tests:** Place in `tests/` directory.

**Test naming:**
- Use descriptive names: `test_crossover_preserves_valid_instructions`
- Group related tests in modules

**Test guidelines:**
- Each test should be independent
- Use builders for complex setup (e.g., `HyperParametersBuilder`)
- Test edge cases and error conditions
- For stochastic tests, use fixed seeds or statistical assertions

### Experiment Verification

After making changes that affect evolution:

```bash
# Run baseline experiments
just run iris_baseline

# Generate analysis
just analyze
```

## Pull Request Process

### Before Starting

1. **Check existing issues** - See if there's an open issue for your change
2. **Open an issue** - For significant changes, discuss the approach first
3. **Fork the repository** - Create your own fork to work in

### Development Workflow

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes:**
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run checks locally:**
   ```bash
   just check  # Runs fmt, lint, and test
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add new classification problem"
   ```

   Commit message format:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `refactor:` - Code refactoring
   - `test:` - Test additions/modifications
   - `chore:` - Build/tooling changes

5. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```

### PR Requirements

- [ ] All tests pass (`just test`)
- [ ] Code is formatted (`just fmt`)
- [ ] No clippy warnings (`just lint`)
- [ ] Documentation updated if needed
- [ ] Commit messages follow convention
- [ ] PR description explains the change

### Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## Areas for Contribution

### Good First Issues

- Documentation improvements
- Adding examples to existing code
- Writing additional tests
- Fixing typos and clarifications

### Feature Ideas

**New Problem Domains:**
- Additional OpenAI Gym environments (Acrobot, Pendulum)
- Classic ML datasets (MNIST subset, Wine, Breast Cancer)
- Custom game environments

**Algorithm Improvements:**
- Alternative selection methods (tournament, roulette)
- Different crossover operators (uniform, single-point)
- Adaptive mutation rates
- Island model parallelism

**Tooling:**
- Visualization of evolved programs
- Program simplification/pruning
- Export to other formats

**Performance:**
- SIMD optimizations for register operations
- GPU acceleration for fitness evaluation
- Improved parallel evaluation strategies

### Documentation Needs

- API documentation improvements
- Tutorial for specific use cases
- Performance tuning guide
- Comparison with other GP frameworks

## Questions?

- Open an issue for questions about contributing
- Check existing issues and discussions
- Review the [README](README.md) and [Extension Guide](docs/EXTENDING.md)

Thank you for contributing!
