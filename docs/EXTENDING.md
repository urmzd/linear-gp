# Extending the LGP Framework

This guide explains how to add new problem domains, custom genetic operators, and alternative fitness functions to the Linear Genetic Programming framework.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Core Traits](#core-traits)
4. [Adding a Classification Problem](#adding-a-classification-problem)
5. [Adding an RL Problem](#adding-an-rl-problem)
6. [Custom Genetic Operators](#custom-genetic-operators)
7. [Testing Extensions](#testing-extensions)

## Quick Start

Here's a minimal example - a classifier that predicts if a number is positive:

```rust
use lgp::core::{environment::State, engines::*};

// 1. Define your state (the data)
pub struct SignState {
    values: Vec<(f64, usize)>,  // (input, expected_class)
    idx: usize,
}

impl State for SignState {
    fn get_value(&self, _: usize) -> f64 { self.values[self.idx].0 }
    fn execute_action(&mut self, action: usize) -> f64 {
        let correct = action == self.values[self.idx].1;
        self.idx += 1;
        if correct { 1.0 } else { 0.0 }
    }
    fn get(&mut self) -> Option<&mut Self> {
        if self.idx < self.values.len() { Some(self) } else { None }
    }
}

// 2. Implement Reset and Generate for your state
impl Reset<SignState> for ResetEngine {
    fn reset(s: &mut SignState) { s.idx = 0; }
}

impl Generate<(), SignState> for GenerateEngine {
    fn generate(_: ()) -> SignState {
        SignState {
            values: vec![(-1.0, 0), (1.0, 1), (-5.0, 0), (3.0, 1)],
            idx: 0,
        }
    }
}

// 3. Define your engine using defaults for everything else
pub struct SignEngine;

impl Core for SignEngine {
    type State = SignState;
    type Individual = Program;
    type ProgramParameters = ProgramGeneratorParameters;
    type FitnessMarker = ();  // Uses default classification fitness
    type Generate = GenerateEngine;
    type Fitness = FitnessEngine;
    type Reset = ResetEngine;
    type Breed = BreedEngine;
    type Mutate = MutateEngine;
    type Status = StatusEngine;
    type Freeze = FreezeEngine;
}
```

That's it! The framework handles evolution, selection, and mutation automatically.
For a complete walkthrough including CLI integration, see [Adding a Classification Problem](#adding-a-classification-problem).

## Architecture Overview

The framework uses a trait-based plugin architecture. Each problem domain implements the `Core` trait, which ties together all the genetic algorithm components:

```
┌─────────────────────────────────────────────────────────────┐
│                         Core Trait                          │
├─────────────────────────────────────────────────────────────┤
│  Individual    - The evolving entity (Program, QProgram)   │
│  State         - Environment/dataset representation        │
│  FitnessMarker - Selects fitness evaluation strategy       │
├─────────────────────────────────────────────────────────────┤
│  Generate      - Creates new individuals and states        │
│  Fitness       - Evaluates individual performance          │
│  Reset         - Resets individuals and states             │
│  Breed         - Crossover operations                      │
│  Mutate        - Mutation operations                       │
│  Status        - Fitness management                        │
│  Freeze        - Optional freezing behavior                │
└─────────────────────────────────────────────────────────────┘
```

The genetic algorithm loop:

1. **Initialize** - Generate initial population
2. **Evaluate** - Run fitness evaluation on trials
3. **Rank** - Sort population by fitness (descending)
4. **Survive** - Select survivors based on gap parameter
5. **Variation** - Create offspring via mutation, crossover, cloning
6. **Repeat** from step 2

## Core Traits

### State Trait

The `State` trait represents the environment or dataset that programs are evaluated against:

```rust
// crates/lgp/src/core/environment.rs
pub trait State: Sized {
    /// Get the value at a specific input index
    fn get_value(&self, at_idx: usize) -> f64;

    /// Execute an action and return the reward/result
    fn execute_action(&mut self, action: usize) -> f64;

    /// Get the next state, or None if exhausted/terminal
    fn get(&mut self) -> Option<&mut Self>;
}
```

For RL problems, extend with `RlState`:

```rust
pub trait RlState: State {
    /// Returns true if the episode has terminated
    fn is_terminal(&mut self) -> bool;

    /// Returns the initial observation vector
    fn get_initial_state(&self) -> Vec<f64>;
}
```

### Core Trait

The `Core` trait defines all components for a problem domain:

```rust
// crates/lgp/src/core/engines/core_engine.rs
pub trait Core {
    type Individual: Ord + Clone + Send + Sync + Serialize + DeserializeOwned;
    type ProgramParameters: Copy + Send + Sync + Clone + Serialize + DeserializeOwned + Args;
    type State: State;
    type FitnessMarker;  // Empty type to select fitness implementation
    type Generate: Generate<Self::ProgramParameters, Self::Individual> + Generate<(), Self::State>;
    type Fitness: Fitness<Self::Individual, Self::State, Self::FitnessMarker>;
    type Reset: Reset<Self::Individual> + Reset<Self::State>;
    type Breed: Breed<Self::Individual>;
    type Mutate: Mutate<Self::ProgramParameters, Self::Individual>;
    type Status: Status<Self::Individual>;
    type Freeze: Freeze<Self::Individual>;
}
```

## Adding a Classification Problem

Let's walk through adding a complete XOR classification problem.

### Step 1: Define the State

```rust
// crates/lgp/src/problems/xor.rs
use serde::{Deserialize, Serialize};
use crate::core::environment::State;

/// A single XOR data point
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XorInput {
    pub x1: f64,
    pub x2: f64,
    pub expected: usize,  // 0 or 1
}

/// Iterator over XOR dataset
pub struct XorState {
    data: Vec<XorInput>,
    idx: usize,
}

impl State for XorState {
    fn get_value(&self, idx: usize) -> f64 {
        let item = &self.data[self.idx];
        match idx {
            0 => item.x1,
            1 => item.x2,
            _ => panic!("XOR only has 2 inputs"),
        }
    }

    fn execute_action(&mut self, action: usize) -> f64 {
        let item = &self.data[self.idx];
        self.idx += 1;

        // Return 1.0 if correct, 0.0 if wrong
        if action == item.expected { 1.0 } else { 0.0 }
    }

    fn get(&mut self) -> Option<&mut Self> {
        if self.idx >= self.data.len() {
            return None;
        }
        Some(self)
    }
}
```

### Step 2: Implement Reset and Generate

```rust
use crate::core::engines::reset_engine::{Reset, ResetEngine};
use crate::core::engines::generate_engine::{Generate, GenerateEngine};
use crate::utils::random::generator;
use rand::seq::SliceRandom;

impl Reset<XorState> for ResetEngine {
    fn reset(item: &mut XorState) {
        item.idx = 0;
    }
}

impl Generate<(), XorState> for GenerateEngine {
    fn generate(_using: ()) -> XorState {
        let mut data = vec![
            XorInput { x1: 0.0, x2: 0.0, expected: 0 },
            XorInput { x1: 0.0, x2: 1.0, expected: 1 },
            XorInput { x1: 1.0, x2: 0.0, expected: 1 },
            XorInput { x1: 1.0, x2: 1.0, expected: 0 },
        ];

        // Shuffle for variety across trials
        data.shuffle(&mut generator());

        XorState { data, idx: 0 }
    }
}
```

### Step 3: Define the Core Implementation

```rust
use crate::core::engines::{
    core_engine::Core,
    breed_engine::BreedEngine,
    fitness_engine::FitnessEngine,
    freeze_engine::FreezeEngine,
    mutate_engine::MutateEngine,
    status_engine::StatusEngine,
};
use crate::core::program::{Program, ProgramGeneratorParameters};

/// Marker type for XOR problem
#[derive(Clone)]
pub struct XorEngine;

impl Core for XorEngine {
    type State = XorState;
    type Individual = Program;
    type ProgramParameters = ProgramGeneratorParameters;
    type FitnessMarker = ();  // Uses default classification fitness
    type Generate = GenerateEngine;
    type Fitness = FitnessEngine;
    type Reset = ResetEngine;
    type Breed = BreedEngine;
    type Mutate = MutateEngine;
    type Status = StatusEngine;
    type Freeze = FreezeEngine;
}
```

### Step 4: Register as an Experiment

The config system uses TOML files in the `configs/` directory. To add a new experiment:

1. **Create config directory and `default.toml`:**
   ```bash
   mkdir -p configs/xor_lgp
   ```

   ```toml
   # configs/xor_lgp/default.toml
   [experiment]
   name = "xor_lgp"
   environment = "xor"

   [hyperparameters]
   population_size = 50
   n_generations = 100
   n_trials = 10
   gap = 0.5

   [hyperparameters.program]
   max_instructions = 20
   n_inputs = 2
   n_actions = 2

   [operations]
   mutation_rate = 0.5
   crossover_rate = 0.5
   ```

2. **Implement the State trait and Core trait** in `crates/lgp/src/problems/xor.rs`

3. **Register the environment in `experiment_runner.rs`:**

   Add a match arm in `crates/lgp-cli/src/experiment_runner.rs` (around line 81):

   ```rust
   // In run_experiment function's match statement:
   match (config.environment.as_str(), config.has_q_learning()) {
       // ... existing environments ...
       ("Xor" | "xor", _) => run_xor(config, seed, &output)?,
       _ => return Err(format!("Unknown environment: {}", config.environment).into()),
   }
   ```

4. **Implement the run function** following existing patterns (e.g., `run_iris`):

   ```rust
   fn run_xor(
       config: &ExperimentConfig,
       seed: u64,
       output: &ExperimentOutput,
   ) -> Result<(), Box<dyn std::error::Error>> {
       let parameters: HyperParameters<XorEngine> = HyperParameters {
           default_fitness: config.hyperparameters.default_fitness,
           population_size: config.hyperparameters.population_size,
           gap: config.hyperparameters.gap,
           mutation_percent: config.mutation_percent(),
           crossover_percent: config.crossover_percent(),
           n_generations: config.hyperparameters.n_generations,
           n_trials: config.hyperparameters.n_trials,
           seed: Some(seed),
           program_parameters: build_program_params(config),
       };

       run_and_save::<XorEngine>(&parameters, output)
   }
   ```

### Step 5: Add Module Export

In `crates/lgp/src/problems/mod.rs`:

```rust
pub mod gym;
pub mod iris;
pub mod xor;  // Add this line
```

## Adding an RL Problem

For RL problems, use the gym-rs adapter pattern or implement `RlState` directly.

### Using gym-rs

If your environment implements the `gym_rs::core::Env` trait, you can use the existing `GymRsEngine` adapter. Registration is done in `crates/lgp-cli/src/experiment_runner.rs`:

1. **Create TOML config:**
   ```toml
   # configs/your_env_lgp/default.toml
   [experiment]
   name = "your_env_lgp"
   environment = "your_env"

   [problem]
   n_inputs = 4   # YourEnv::OBSERVATION_SIZE
   n_actions = 2  # YourEnv::ACTION_COUNT

   [hyperparameters]
   population_size = 100
   n_generations = 500
   # ... other parameters
   ```

2. **Add match arm in `experiment_runner.rs`:**
   ```rust
   use gym_rs::envs::your_module::YourEnv;

   // In run_experiment function's match statement:
   match (config.environment.as_str(), config.has_q_learning()) {
       // ... existing environments ...
       ("YourEnv" | "your_env", false) => run_your_env_lgp(config, seed, &output)?,
       ("YourEnv" | "your_env", true) => {
           run_your_env_q(config, seed, &output, config.q_learning_params().unwrap())?
       }
       _ => return Err(format!("Unknown environment: {}", config.environment).into()),
   }
   ```

3. **Implement run functions** following the patterns for `run_cart_pole_lgp` and `run_cart_pole_q`.

### Custom RL Environment

For custom environments not using gym-rs:

```rust
use crate::core::environment::{State, RlState};

pub struct CustomRlState {
    observation: Vec<f64>,
    terminal: bool,
    episode_step: usize,
    max_steps: usize,
}

impl State for CustomRlState {
    fn get_value(&self, idx: usize) -> f64 {
        self.observation[idx]
    }

    fn execute_action(&mut self, action: usize) -> f64 {
        // Update state based on action
        let reward = self.step(action);
        self.episode_step += 1;
        reward
    }

    fn get(&mut self) -> Option<&mut Self> {
        if self.terminal || self.episode_step >= self.max_steps {
            return None;
        }
        Some(self)
    }
}

impl RlState for CustomRlState {
    fn is_terminal(&mut self) -> bool {
        self.terminal || self.episode_step >= self.max_steps
    }

    fn get_initial_state(&self) -> Vec<f64> {
        // Return the initial observation
        vec![0.0; self.observation.len()]
    }
}
```

Then use `UseRlFitness` as the `FitnessMarker`:

```rust
use crate::extensions::interactive::UseRlFitness;

impl Core for CustomRlEngine {
    // ...
    type FitnessMarker = UseRlFitness;  // Enables RL fitness evaluation
    // ...
}
```

## Custom Genetic Operators

### Custom Mutation Operator

Create a new mutate engine:

```rust
use crate::core::engines::mutate_engine::Mutate;
use crate::core::program::{Program, ProgramGeneratorParameters};

pub struct AggressiveMutateEngine;

impl Mutate<ProgramGeneratorParameters, Program> for AggressiveMutateEngine {
    fn mutate(item: &mut Program, using: ProgramGeneratorParameters) {
        // Mutate multiple instructions instead of just one
        let n_mutations = 3.min(item.instructions.len());

        for _ in 0..n_mutations {
            // Your mutation logic here
            // See MutateEngine in crates/lgp/src/core/engines/mutate_engine.rs for reference
        }
    }
}
```

Use it in your Core implementation:

```rust
impl Core for MyEngine {
    // ...
    type Mutate = AggressiveMutateEngine;
    // ...
}
```

### Custom Crossover Operator

```rust
use crate::core::engines::breed_engine::Breed;
use crate::core::program::Program;

pub struct UniformCrossoverEngine;

impl Breed<Program> for UniformCrossoverEngine {
    fn two_point_crossover(mate_1: &Program, mate_2: &Program) -> (Program, Program) {
        // Implement uniform crossover instead of two-point
        // Each instruction is randomly selected from either parent
        // ...
    }
}
```

### Custom Fitness Function

The `FitnessMarker` type parameter selects which fitness implementation to use. Create a new marker and implement `Fitness`:

```rust
use crate::core::engines::fitness_engine::{Fitness, FitnessEngine};

/// Marker for weighted accuracy fitness
pub struct WeightedAccuracyFitness;

impl<T: State> Fitness<Program, T, WeightedAccuracyFitness> for FitnessEngine {
    fn eval_fitness(program: &mut Program, states: &mut T) -> f64 {
        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;

        while let Some(state) = states.get() {
            program.run(state);

            // Apply class-specific weights
            let weight = get_class_weight(state);
            let correct = /* evaluation logic */;

            weighted_score += correct * weight;
            total_weight += weight;
        }

        weighted_score / total_weight
    }
}
```

Then use it:

```rust
impl Core for WeightedEngine {
    // ...
    type FitnessMarker = WeightedAccuracyFitness;
    // ...
}
```

## Testing Extensions

### Unit Test Pattern

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::engines::core_engine::HyperParametersBuilder;
    use crate::core::instruction::InstructionGeneratorParametersBuilder;
    use crate::core::program::ProgramGeneratorParametersBuilder;
    use itertools::Itertools;

    #[test]
    fn test_xor_evolution() {
        let instruction_params = InstructionGeneratorParametersBuilder::default()
            .n_actions(2)
            .n_inputs(2)
            .build()
            .unwrap();

        let program_params = ProgramGeneratorParametersBuilder::default()
            .max_instructions(20)
            .instruction_generator_parameters(instruction_params)
            .build()
            .unwrap();

        let hyperparams = HyperParametersBuilder::<XorEngine>::default()
            .program_parameters(program_params)
            .population_size(50)
            .n_generations(100)
            .n_trials(10)
            .build()
            .unwrap();

        let populations: Vec<_> = hyperparams
            .build_engine()
            .take(hyperparams.n_generations)
            .collect();

        // Verify evolution occurred
        let first_gen = &populations[0];
        let last_gen = populations.last().unwrap();

        let first_best = StatusEngine::get_fitness(first_gen.first().unwrap());
        let last_best = StatusEngine::get_fitness(last_gen.first().unwrap());

        // Expect improvement (XOR should achieve 1.0 fitness)
        assert!(last_best >= first_best, "Fitness should improve over generations");
    }
}
```

### Integration Test

Create a test file in `crates/lgp/tests/`:

```rust
// crates/lgp/tests/xor_integration.rs
use lgp::core::engines::core_engine::HyperParameters;
use lgp::problems::xor::XorEngine;

#[test]
fn xor_solves_problem() {
    // Load optimized parameters or use defaults
    let params = /* ... */;

    let populations: Vec<_> = params
        .build_engine()
        .take(params.n_generations)
        .collect();

    let best = populations.last().unwrap().first().unwrap();
    let fitness = StatusEngine::get_fitness(best);

    // XOR is simple - should achieve perfect accuracy
    assert!(fitness > 0.9, "Expected >90% accuracy, got {}", fitness);
}
```

### Benchmark Test

Add to `crates/lgp/benches/performance_after_training.rs`:

```rust
fn xor_benchmark(c: &mut Criterion) {
    // Load or create trained programs
    // Benchmark their execution time and fitness
}

criterion_group!(
    benches,
    // ... existing benchmarks ...
    xor_benchmark,
);
```

## Summary

To add a new problem:

1. **Define State** - Implement `State` (or `RlState` for RL)
2. **Implement Reset** - Reset state for re-evaluation
3. **Implement Generate** - Create initial state instances
4. **Define Engine** - Implement `Core` trait
5. **Create TOML config** - Add `configs/<name>/default.toml` with experiment configuration
6. **Register in experiment_runner** - Add match arm in `crates/lgp-cli/src/experiment_runner.rs`
7. **Test** - Write unit and integration tests

The framework's trait system allows mixing and matching components. You can use the default `BreedEngine`, `MutateEngine`, etc., or create custom implementations for specialized behavior.
