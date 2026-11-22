//! A seeded run must repeat exactly: same programs, same ids, same fitness.
//!
//! Fitness evaluation and variation run on rayon worker threads, and gym
//! environments carry their own generators. These tests fail if any of them
//! draws randomness that the seed does not control.

#![cfg(feature = "gym")]

use gymnasia::envs::classical_control::cartpole::CartPoleEnv;
use itertools::Itertools;

use lgp::core::engines::core_engine::{Core, HyperParametersBuilder};
use lgp::core::instruction::InstructionGeneratorParametersBuilder;
use lgp::core::program::ProgramGeneratorParametersBuilder;
use lgp::problems::gym::GymRsEngine;

fn seeded_run(seed: u64) -> String {
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
        .n_trials(3)
        .n_generations(4)
        .population_size(32)
        .mutation_percent(0.5)
        .crossover_percent(0.5)
        .seed(Some(seed))
        .build()
        .expect("Failed to build hyperparameters");

    let populations: Vec<Vec<<GymRsEngine<CartPoleEnv> as Core>::Individual>> = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    serde_json::to_string(&populations).expect("Failed to serialize populations")
}

#[test]
fn same_seed_repeats_exactly() {
    assert_eq!(seeded_run(42), seeded_run(42));
}

#[test]
fn different_seeds_differ() {
    assert_ne!(seeded_run(1), seeded_run(2));
}
