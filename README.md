# Linear Genetic Programming

A framework for solving tasks using linear genetic programming.

![build passing](https://github.com/urmzd/linear-genetic-programming/actions/workflows/build.yml/badge.svg)

## Prerequisities [WIP]

```bash
## These are mock reqs.
sudo apt-get install docker
sudo apt-get install rust
sudo apt-get install docker-compose
sudo apt-get install python
...
```

## Usage

Running Trials:

```bash
docker-compose up -d

python -m venv venv # optional
pip install -r scripts/requirements.txt

cargo build --release

./scripts/optimize.py -h # help
./scripts/optimize.py --env cart-pole-lgp --n-trials 40 --n-threads 4  
```

## Contributions

Contributions are welcomed. The guidelines can be found in [CONTRIBUTING.md](./CONTRIBUTING.md).
