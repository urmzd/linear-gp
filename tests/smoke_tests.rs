//! Smoke tests for LGP environments
//!
//! These tests verify that each environment can run for a minimal number of generations
//! without crashing. They don't save results or require special setup.

use itertools::Itertools;

use lgp::core::engines::core_engine::HyperParametersBuilder;
use lgp::core::engines::status_engine::{Status, StatusEngine};
use lgp::core::instruction::InstructionGeneratorParametersBuilder;
use lgp::core::program::ProgramGeneratorParametersBuilder;
use lgp::problems::iris::IrisEngine;

/// Test that Iris classification works for 2 generations
#[test]
fn smoke_iris() {
    let instruction_parameters = InstructionGeneratorParametersBuilder::default()
        .n_actions(3)
        .n_inputs(4)
        .build()
        .expect("Failed to build instruction parameters");

    let program_parameters = ProgramGeneratorParametersBuilder::default()
        .max_instructions(10)
        .instruction_generator_parameters(instruction_parameters)
        .build()
        .expect("Failed to build program parameters");

    let parameters = HyperParametersBuilder::<IrisEngine>::default()
        .program_parameters(program_parameters)
        .n_trials(1)
        .n_generations(2)
        .population_size(10)
        .mutation_percent(0.5)
        .crossover_percent(0.5)
        .build()
        .expect("Failed to build hyperparameters");

    let populations: Vec<_> = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    assert_eq!(populations.len(), 2, "Should have 2 generations");
    assert!(
        !populations.last().unwrap().is_empty(),
        "Final population should not be empty"
    );

    // Verify fitness values are valid
    let best_fitness = StatusEngine::get_fitness(populations.last().unwrap().first().unwrap());
    assert!(
        best_fitness.is_finite(),
        "Best fitness should be a finite number"
    );
}

/// Test that CartPole LGP works for 2 generations
#[test]
fn smoke_cart_pole_lgp() {
    use gym_rs::envs::classical_control::cartpole::CartPoleEnv;
    use lgp::problems::gym::GymRsEngine;

    let instruction_parameters = InstructionGeneratorParametersBuilder::default()
        .n_actions(2)
        .n_inputs(4)
        .build()
        .expect("Failed to build instruction parameters");

    let program_parameters = ProgramGeneratorParametersBuilder::default()
        .max_instructions(10)
        .instruction_generator_parameters(instruction_parameters)
        .build()
        .expect("Failed to build program parameters");

    let parameters = HyperParametersBuilder::<GymRsEngine<CartPoleEnv>>::default()
        .program_parameters(program_parameters)
        .n_trials(1)
        .n_generations(2)
        .population_size(10)
        .mutation_percent(0.5)
        .crossover_percent(0.5)
        .default_fitness(500.0)
        .build()
        .expect("Failed to build hyperparameters");

    let populations: Vec<_> = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    assert_eq!(populations.len(), 2, "Should have 2 generations");
    assert!(
        !populations.last().unwrap().is_empty(),
        "Final population should not be empty"
    );
}

/// Test that CartPole Q-Learning works for 2 generations
#[test]
fn smoke_cart_pole_q() {
    use gym_rs::envs::classical_control::cartpole::CartPoleEnv;
    use lgp::extensions::q_learning::{QConsts, QProgramGeneratorParameters};
    use lgp::problems::gym::GymRsQEngine;

    let instruction_parameters = InstructionGeneratorParametersBuilder::default()
        .n_actions(2)
        .n_inputs(4)
        .build()
        .expect("Failed to build instruction parameters");

    let program_parameters = ProgramGeneratorParametersBuilder::default()
        .max_instructions(10)
        .instruction_generator_parameters(instruction_parameters)
        .build()
        .expect("Failed to build program parameters");

    let q_params = QProgramGeneratorParameters {
        program_parameters,
        consts: QConsts::new(0.1, 0.9, 0.05, 0.01, 0.001),
    };

    let parameters = HyperParametersBuilder::<GymRsQEngine<CartPoleEnv>>::default()
        .program_parameters(q_params)
        .n_trials(1)
        .n_generations(2)
        .population_size(10)
        .mutation_percent(0.5)
        .crossover_percent(0.5)
        .default_fitness(500.0)
        .build()
        .expect("Failed to build hyperparameters");

    let populations: Vec<_> = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    assert_eq!(populations.len(), 2, "Should have 2 generations");
    assert!(
        !populations.last().unwrap().is_empty(),
        "Final population should not be empty"
    );
}

/// Test that MountainCar LGP works for 2 generations
#[test]
fn smoke_mountain_car_lgp() {
    use gym_rs::envs::classical_control::mountain_car::MountainCarEnv;
    use lgp::problems::gym::GymRsEngine;

    let instruction_parameters = InstructionGeneratorParametersBuilder::default()
        .n_actions(3)
        .n_inputs(2)
        .build()
        .expect("Failed to build instruction parameters");

    let program_parameters = ProgramGeneratorParametersBuilder::default()
        .max_instructions(10)
        .instruction_generator_parameters(instruction_parameters)
        .build()
        .expect("Failed to build program parameters");

    let parameters = HyperParametersBuilder::<GymRsEngine<MountainCarEnv>>::default()
        .program_parameters(program_parameters)
        .n_trials(1)
        .n_generations(2)
        .population_size(10)
        .mutation_percent(0.5)
        .crossover_percent(0.5)
        .default_fitness(-200.0)
        .build()
        .expect("Failed to build hyperparameters");

    let populations: Vec<_> = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    assert_eq!(populations.len(), 2, "Should have 2 generations");
    assert!(
        !populations.last().unwrap().is_empty(),
        "Final population should not be empty"
    );
}

/// Test that MountainCar Q-Learning works for 2 generations
#[test]
fn smoke_mountain_car_q() {
    use gym_rs::envs::classical_control::mountain_car::MountainCarEnv;
    use lgp::extensions::q_learning::{QConsts, QProgramGeneratorParameters};
    use lgp::problems::gym::GymRsQEngine;

    let instruction_parameters = InstructionGeneratorParametersBuilder::default()
        .n_actions(3)
        .n_inputs(2)
        .build()
        .expect("Failed to build instruction parameters");

    let program_parameters = ProgramGeneratorParametersBuilder::default()
        .max_instructions(10)
        .instruction_generator_parameters(instruction_parameters)
        .build()
        .expect("Failed to build program parameters");

    let q_params = QProgramGeneratorParameters {
        program_parameters,
        consts: QConsts::new(0.1, 0.9, 0.05, 0.01, 0.001),
    };

    let parameters = HyperParametersBuilder::<GymRsQEngine<MountainCarEnv>>::default()
        .program_parameters(q_params)
        .n_trials(1)
        .n_generations(2)
        .population_size(10)
        .mutation_percent(0.5)
        .crossover_percent(0.5)
        .default_fitness(-200.0)
        .build()
        .expect("Failed to build hyperparameters");

    let populations: Vec<_> = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    assert_eq!(populations.len(), 2, "Should have 2 generations");
    assert!(
        !populations.last().unwrap().is_empty(),
        "Final population should not be empty"
    );
}
