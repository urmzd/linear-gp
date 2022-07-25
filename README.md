# Linear Genetic Programming

A framework for solving tasks using linear genetic programming.

![build passing](https://github.com/urmzd/linear-genetic-programming/actions/workflows/develop.yml/badge.svg)

## Modules

- [Core](src/core/)
- [Extension](src/extensions/)
- [Utilities](src/utils/)

## Examples

All examples can be built and ran through Cargo:

```bash
cargo build --example <example_name>
cargo run --example <example_name>
```

### Classification

#### iris

```rust
//examples/iris/main.rs#L15-L35

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error>> {
    let ContentFilePair(_, file) = get_iris_content().await?;
    let inputs = IrisLgp::load_inputs(file.path());

    let mut hyper_params = HyperParameters {
        population_size: 100,
        max_generations: 100,
        gap: 0.5,
        n_mutations: 50,
        n_crossovers: 50,
        fitness_parameters: ClassificationParameters::new(inputs),
        program_parameters: ProgramGeneratorParameters::new(
            100,
            InstructionGeneratorParameters::from::<IrisInput>(1),
        ),
    };

    IrisLgp::execute(&mut hyper_params, EventHooks::default())?;
    Ok(())
}
```

### Reinforcement Learning

#### mountain_car

```rust
//examples/mountain_car/main.rs#L14-L34

};
use set_up::{MountainCarInput, MountainCarLgp};

mod set_up;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let environment = MountainCarEnv::new(RenderMode::Human, None);
    let input = MountainCarInput::new(environment);
    let initial_states = (vec![0; 5])
        .into_iter()
        .map(|_| MountainCarObservation::sample_between(&mut generator(), None))
        .collect_vec();

    let mut hyper_params = HyperParameters {
        population_size: 1,
        gap: 0.5,
        n_crossovers: 0,
        n_mutations: 0,
        max_generations: 1,
        fitness_parameters: ReinforcementLearningParameters::new(initial_states, 200, input),
        program_parameters: ProgramGeneratorParameters::new(
```

#### cart_pole

```rust
//examples/cart_pole/main.rs#L14-L34

};
use set_up::{CartPoleInput, CartPoleLgp};

mod set_up;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let environment = CartPoleEnv::new(RenderMode::Human);
    let input = CartPoleInput::new(environment);
    let initial_states = (vec![0; 5])
        .into_iter()
        .map(|_| CartPoleObservation::sample_between(&mut generator(), None))
        .collect_vec();

    let mut hyper_params = HyperParameters {
        population_size: 1,
        gap: 0.5,
        n_crossovers: 50,
        n_mutations: 50,
        max_generations: 1,
        fitness_parameters: ReinforcementLearningParameters::new(initial_states, 500, input),
        program_parameters: ProgramGeneratorParameters::new(
```

## Building

Requirements:

- Cargo
- Stable Rust

## Contributions

Contributions are welcomed. The guidelines can be found in [CONTRIBUTING.md](./CONTRIBUTING.md).
