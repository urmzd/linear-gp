# Linear Genetic Programming

This repository contains a framework for solving tasks using linear genetic programming.

![build passing](https://github.com/urmzd/linear-genetic-programming/actions/workflows/experiments.yml/badge.svg)

## Prerequisites

To set up the environment and dependencies, follow the instructions below:

```bash
# Install required packages
sudo apt-get install docker
sudo apt-get install rust
sudo apt-get install docker-compose
sudo apt-get install python
...
```

## Usage

1. Setup the environment:

```bash
docker-compose up -d

python -m venv venv
pip install -r scripts/requirements.txt

cargo build --release
```

2. Execute the search script:
```bash
# Display help
./scripts/search.py -h # help

# Search for the best parameters for a specific environment
./scripts/search.py --env cart-pole-lgp --n-trials 40 --n-threads 4  

# Search for the best parameters for all environments
./scripts/search_all.sh
```

3. View search results:

```bash
# using optuna dashboard
docker run -it --rm -p 8080:8080 -v `pwd`:/app -w /app \
```

4. Run tests using the updated hyperparameters:

```bash
cargo nextest run --no-fail-fast --release --no-capture
```

5. Produce graphs and tables:

```bash
./scripts/produce_assets.py
```

6. Determine how well individuals perform after training:
```bash
cargo bench
```

## GitHub Actions Workflow

The repository includes a GitHub Actions workflow file that automates the process of running experiments, searching for optimal parameters, and benchmarking. The workflow is triggered manually and accepts the number of experiments as input. The workflow file can be found at `.github/workflows/experiments.yml`.

## Contributions
Contributions are welcome. Please refer to the guidelines in [CONTRIBUTING.md](./CONTRIBUTING.md) for more information.
