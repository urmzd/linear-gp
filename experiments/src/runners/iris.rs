//! Iris classification experiment runners

use itertools::Itertools;

use lgp::core::engines::core_engine::HyperParametersBuilder;
use lgp::core::engines::status_engine::{Status, StatusEngine};
use lgp::core::instruction::InstructionGeneratorParametersBuilder;
use lgp::core::program::ProgramGeneratorParametersBuilder;
use lgp::problems::iris::IrisEngine;

use crate::benchmark_tools::{save_experiment, VoidResultAnyError};

/// Run baseline experiment (no mutation, no crossover)
pub fn run_baseline(n_generations_override: Option<usize>) -> VoidResultAnyError {
    let name = "iris_baseline";
    let instruction_parameters = InstructionGeneratorParametersBuilder::default()
        .n_actions(3)
        .n_inputs(4)
        .build()?;
    let program_parameters = ProgramGeneratorParametersBuilder::default()
        .max_instructions(100)
        .instruction_generator_parameters(instruction_parameters)
        .build()?;
    let parameters = HyperParametersBuilder::<IrisEngine>::default()
        .program_parameters(program_parameters)
        .n_trials(1)
        .n_generations(n_generations_override.unwrap_or(200))
        .mutation_percent(0.)
        .crossover_percent(0.)
        .build()?;

    let populations = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    save_experiment(&populations, &parameters, name)?;

    // In baseline (no mutation/crossover), fitness should converge after enough generations.
    // This is a sanity check, not a hard requirement.
    let last_population = populations.last().unwrap();
    let all_same_fitness = last_population.iter().all(|individual| {
        Some(StatusEngine::get_fitness(individual))
            == last_population.first().map(StatusEngine::get_fitness)
    });

    if !all_same_fitness {
        eprintln!(
            "Note: Baseline population has varying fitness. This is expected with few generations."
        );
    }

    Ok(())
}

/// Run mutation-only experiment
pub fn run_mutation(n_generations_override: Option<usize>) -> VoidResultAnyError {
    let name = "iris_mutation";
    let instruction_parameters = InstructionGeneratorParametersBuilder::default()
        .n_actions(3)
        .n_inputs(4)
        .build()?;
    let program_parameters = ProgramGeneratorParametersBuilder::default()
        .max_instructions(100)
        .instruction_generator_parameters(instruction_parameters)
        .build()?;
    let parameters = HyperParametersBuilder::<IrisEngine>::default()
        .program_parameters(program_parameters)
        .mutation_percent(1.0)
        .crossover_percent(0.)
        .n_trials(1)
        .n_generations(n_generations_override.unwrap_or(200))
        .build()?;

    let populations = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    save_experiment(&populations, &parameters, name)?;

    Ok(())
}

/// Run crossover-only experiment
pub fn run_crossover(n_generations_override: Option<usize>) -> VoidResultAnyError {
    let name = "iris_crossover";
    let instruction_parameters = InstructionGeneratorParametersBuilder::default()
        .n_actions(3)
        .n_inputs(4)
        .build()?;
    let program_parameters = ProgramGeneratorParametersBuilder::default()
        .max_instructions(100)
        .instruction_generator_parameters(instruction_parameters)
        .build()?;
    let parameters = HyperParametersBuilder::<IrisEngine>::default()
        .program_parameters(program_parameters)
        .mutation_percent(0.)
        .crossover_percent(1.0)
        .n_trials(1)
        .n_generations(n_generations_override.unwrap_or(200))
        .build()?;

    let populations = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    save_experiment(&populations, &parameters, name)?;

    Ok(())
}

/// Run full experiment (mutation + crossover)
pub fn run_full(n_generations_override: Option<usize>) -> VoidResultAnyError {
    let name = "iris_full";

    let instruction_parameters = InstructionGeneratorParametersBuilder::default()
        .n_actions(3)
        .n_inputs(4)
        .build()?;
    let program_parameters = ProgramGeneratorParametersBuilder::default()
        .max_instructions(100)
        .instruction_generator_parameters(instruction_parameters)
        .build()?;
    let parameters = HyperParametersBuilder::<IrisEngine>::default()
        .program_parameters(program_parameters)
        .mutation_percent(0.5)
        .crossover_percent(0.5)
        .n_trials(1)
        .n_generations(n_generations_override.unwrap_or(200))
        .build()?;

    let populations = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    save_experiment(&populations, &parameters, name)?;

    Ok(())
}
