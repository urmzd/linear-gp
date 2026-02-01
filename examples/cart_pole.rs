//! CartPole Example
//!
//! This example demonstrates how to use the LGP framework to evolve programs
//! that can balance a pole on a cart (the classic CartPole control problem).
//!
//! Run with: `cargo run --example cart_pole`

use itertools::Itertools;

use gym_rs::envs::classical_control::cartpole::CartPoleEnv;

use lgp::core::engines::core_engine::HyperParametersBuilder;
use lgp::core::engines::status_engine::{Status, StatusEngine};
use lgp::core::instruction::InstructionGeneratorParametersBuilder;
use lgp::core::program::ProgramGeneratorParametersBuilder;
use lgp::problems::gym::GymRsEngine;

fn main() {
    println!("=== CartPole LGP Example ===\n");

    // Step 1: Configure instruction parameters
    // CartPole has 4 inputs (cart position, velocity, pole angle, angular velocity)
    // and 2 actions (push left or right)
    let instruction_parameters = InstructionGeneratorParametersBuilder::default()
        .n_actions(2)
        .n_inputs(4)
        .n_extras(1) // Extra working registers
        .external_factor(10.0) // Scaling for input values
        .build()
        .expect("Failed to build instruction parameters");

    // Step 2: Configure program parameters
    let program_parameters = ProgramGeneratorParametersBuilder::default()
        .max_instructions(20) // Maximum instructions per program
        .instruction_generator_parameters(instruction_parameters)
        .build()
        .expect("Failed to build program parameters");

    // Step 3: Configure hyperparameters for the evolutionary algorithm
    let parameters = HyperParametersBuilder::<GymRsEngine<CartPoleEnv>>::default()
        .program_parameters(program_parameters)
        .population_size(50) // Number of programs per generation
        .n_generations(20) // Number of generations to evolve
        .n_trials(5) // Episodes per fitness evaluation
        .mutation_percent(0.5) // 50% of offspring via mutation
        .crossover_percent(0.5) // 50% of offspring via crossover
        .gap(0.5) // 50% of population survives each generation
        .default_fitness(500.0) // Max possible score for CartPole
        .build()
        .expect("Failed to build hyperparameters");

    println!("Configuration:");
    println!("  Population size: {}", parameters.population_size);
    println!("  Generations: {}", parameters.n_generations);
    println!("  Trials per evaluation: {}", parameters.n_trials);
    println!();

    // Step 4: Run evolution
    println!("Starting evolution...\n");

    let populations: Vec<_> = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    // Step 5: Report results
    println!("Generation | Best Fitness | Median Fitness | Worst Fitness");
    println!("-----------+--------------+----------------+--------------");

    for (gen, population) in populations.iter().enumerate() {
        let best = StatusEngine::get_fitness(population.first().unwrap());
        let worst = StatusEngine::get_fitness(population.last().unwrap());
        let median = StatusEngine::get_fitness(&population[population.len() / 2]);

        println!(
            "{:>10} | {:>12.2} | {:>14.2} | {:>13.2}",
            gen + 1,
            best,
            median,
            worst
        );
    }

    // Final summary
    let final_population = populations.last().unwrap();
    let best_program = final_population.first().unwrap();
    let best_fitness = StatusEngine::get_fitness(best_program);

    println!();
    println!("=== Evolution Complete ===");
    println!("Best fitness achieved: {:.2}", best_fitness);

    if best_fitness >= 475.0 {
        println!("Success! The evolved program can balance the pole effectively.");
    } else if best_fitness >= 200.0 {
        println!("Partial success. The program shows improvement but may need more generations.");
    } else {
        println!(
            "The program needs more evolution. Try increasing generations or population size."
        );
    }
}
