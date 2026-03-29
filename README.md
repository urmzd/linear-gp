<p align="center">
  <h1 align="center">linear-gp</h1>
  <p align="center">
    A Rust framework for solving reinforcement learning and classification tasks using Linear Genetic Programming (LGP).
    <br /><br />
    <a href="https://github.com/urmzd/linear-gp/releases">Download</a>
    &middot;
    <a href="https://github.com/urmzd/linear-gp/issues">Report Bug</a>
    &middot;
    <a href="https://github.com/urmzd/linear-gp/tree/main/outputs">Experiments</a>
  </p>
</p>

<p align="center">
  <a href="https://github.com/urmzd/linear-gp/actions/workflows/ci.yml"><img src="https://github.com/urmzd/linear-gp/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

## Showcase

<p align="center">
  <img src="showcase/experiment-output.png" alt="LGP experiment output" width="600">
</p>

## Install

**Prebuilt binary (recommended):**

```bash
curl -fsSL https://raw.githubusercontent.com/urmzd/linear-gp/main/install.sh | bash
```

You can pin a version or change the install directory:

```bash
LGP_VERSION=v1.0.0 LGP_INSTALL_DIR=~/.local/bin \
  curl -fsSL https://raw.githubusercontent.com/urmzd/linear-gp/main/install.sh | bash
```

**From source:**

```bash
git clone https://github.com/urmzd/linear-gp.git && cd linear-gp
cargo install --path crates/lgp-cli
```

## Usage

```bash
# List available experiments
lgp list

# Run an experiment
lgp run iris_baseline
lgp run cart_pole_lgp

# Run a Rust example
lgp example cart_pole

# Search for optimal hyperparameters
lgp search iris_baseline

# Full pipeline: search -> run -> analyze
lgp experiment iris_baseline
```

### Available Experiments

| Experiment | Type | Description |
|------------|------|-------------|
| `iris_baseline` | Classification | Iris (no mutation, no crossover) |
| `iris_mutation` | Classification | Iris with mutation only |
| `iris_crossover` | Classification | Iris with crossover only |
| `iris_full` | Classification | Iris full (mutation + crossover) |
| `cart_pole_lgp` | RL | CartPole with pure LGP |
| `cart_pole_with_q` | RL | CartPole with Q-Learning |
| `mountain_car_lgp` | RL | MountainCar with pure LGP |
| `mountain_car_with_q` | RL | MountainCar with Q-Learning |

### Logging

```bash
# Verbose output
lgp -v run iris_baseline

# JSON format
lgp --log-format json run iris_baseline

# Fine-grained control
RUST_LOG=lgp=debug lgp run iris_baseline
```

## Packages

| Package | Description |
|---------|-------------|
| [lgp](crates/lgp/README.md) | Core library — traits, evolutionary engine, built-in problems |
| [lgp-cli](crates/lgp-cli/README.md) | CLI binary for running experiments, search, and analysis |

## Extending the Framework

The framework is trait-based and designed for extension. You can add new classification problems, RL environments, genetic operators, and fitness functions.

See [skills/lgp-experiment/SKILL.md](skills/lgp-experiment/SKILL.md) for the complete guide.

## Thesis

The accompanying thesis, *Reinforced Linear Genetic Programming*, is maintained in a [separate repository](https://github.com/urmzd/rlgp-thesis).

## References

### Genetic Programming

- Koza, J. R. (1993). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press.
- Poli, R., Langdon, W. B., & McPhee, N. F. (2008). *A Field Guide to Genetic Programming*. lulu.com. http://www.gp-field-guide.org.uk/
- Luke, S. (2009). *Essentials of Metaheuristics*. Lulu. https://cs.gmu.edu/~sean/book/metaheuristics/

### Linear Genetic Programming

- Brameier, M., & Banzhaf, W. (2001). A Comparison of Linear Genetic Programming and Neural Networks in Medical Data Mining. *IEEE Transactions on Evolutionary Computation*, 5(1), 17-26.
- Song, D., Heywood, M. I., & Zincir-Heywood, A. N. (2003). A Linear Genetic Programming Approach to Intrusion Detection. In *GECCO 2003*, LNCS 2724, pp. 2325-2336. Springer.
- Peeler, H., Li, S. S., Sloss, A. N., Reid, K. N., Yuan, Y., & Banzhaf, W. (2022). Optimizing LLVM Pass Sequences with Shackleton: A Linear Genetic Programming Framework. In *GECCO 2022 Companion*, pp. 578-581. ACM.

### Reinforcement Learning

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Downing, H. L. (1995). Reinforced Genetic Programming. In *Proceedings of the Sixth International Conference on Genetic Algorithms (ICGA95)*, pp. 276-283.
- Amaral, R., Ianta, A., Bayer, C., Smith, R. J., & Heywood, M. I. (2022). Benchmarking Genetic Programming in a Multi-Action Reinforcement Learning Locomotion Task. In *GECCO 2022 Companion*, pp. 522-525. ACM.

### Environments & Datasets

- Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). OpenAI Gym. arXiv:1606.01540.
- Fisher, R. A. (1936). The Use of Multiple Measurements in Taxonomic Problems. *Annals of Eugenics*, 7(2), 179-188.
