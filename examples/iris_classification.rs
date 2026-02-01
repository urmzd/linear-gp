//! Iris Classification Example
//!
//! This example demonstrates how to use the LGP framework for a classification task
//! using the classic Iris flower dataset.
//!
//! Run with: `cargo run --example iris_classification`

use itertools::Itertools;

use lgp::core::engines::core_engine::HyperParametersBuilder;
use lgp::core::engines::status_engine::{Status, StatusEngine};
use lgp::core::instruction::InstructionGeneratorParametersBuilder;
use lgp::core::program::ProgramGeneratorParametersBuilder;
use lgp::problems::iris::IrisEngine;

fn main() {
    println!("=== Iris Classification LGP Example ===\n");

    // Step 1: Configure instruction parameters
    // Iris has 4 input features (sepal length/width, petal length/width)
    // and 3 classes (Setosa, Versicolor, Virginica)
    let instruction_parameters = InstructionGeneratorParametersBuilder::default()
        .n_actions(3) // 3 classes to predict
        .n_inputs(4) // 4 input features
        .n_extras(2) // Extra working registers for intermediate calculations
        .external_factor(1.0) // Iris features are already normalized-ish
        .build()
        .expect("Failed to build instruction parameters");

    // Step 2: Configure program parameters
    let program_parameters = ProgramGeneratorParametersBuilder::default()
        .max_instructions(50) // Allow more complex programs for classification
        .instruction_generator_parameters(instruction_parameters)
        .build()
        .expect("Failed to build program parameters");

    // Step 3: Configure hyperparameters
    let parameters = HyperParametersBuilder::<IrisEngine>::default()
        .program_parameters(program_parameters)
        .population_size(100) // Larger population for classification
        .n_generations(50) // More generations for convergence
        .n_trials(1) // Single pass through dataset per evaluation
        .mutation_percent(0.5)
        .crossover_percent(0.5)
        .gap(0.5)
        .default_fitness(0.0) // Minimum accuracy
        .build()
        .expect("Failed to build hyperparameters");

    println!("Configuration:");
    println!("  Population size: {}", parameters.population_size);
    println!("  Generations: {}", parameters.n_generations);
    println!("  Max instructions: {}", parameters.program_parameters.max_instructions);
    println!();

    // Step 4: Run evolution
    println!("Starting evolution...\n");
    println!("(Note: Iris dataset is downloaded on first run)\n");

    let populations: Vec<_> = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    // Step 5: Report results (every 10 generations)
    println!("Generation | Best Accuracy | Median Accuracy");
    println!("-----------+---------------+----------------");

    for (gen, population) in populations.iter().enumerate() {
        if (gen + 1) % 10 == 0 || gen == 0 || gen == populations.len() - 1 {
            let best = StatusEngine::get_fitness(population.first().unwrap());
            let median = StatusEngine::get_fitness(&population[population.len() / 2]);

            // Convert to percentage (Iris has 150 samples)
            let best_pct = (best / 150.0) * 100.0;
            let median_pct = (median / 150.0) * 100.0;

            println!(
                "{:>10} | {:>12.1}% | {:>14.1}%",
                gen + 1,
                best_pct,
                median_pct
            );
        }
    }

    // Final summary
    let final_population = populations.last().unwrap();
    let best_program = final_population.first().unwrap();
    let best_fitness = StatusEngine::get_fitness(best_program);
    let accuracy = (best_fitness / 150.0) * 100.0;

    println!();
    println!("=== Evolution Complete ===");
    println!(
        "Best accuracy: {:.1}% ({:.0}/150 correct)",
        accuracy, best_fitness
    );

    if accuracy >= 95.0 {
        println!("Excellent! Near-perfect classification achieved.");
    } else if accuracy >= 90.0 {
        println!("Great result! Strong classification performance.");
    } else if accuracy >= 80.0 {
        println!("Good result. Consider more generations for improvement.");
    } else {
        println!("The classifier needs more evolution or parameter tuning.");
    }
}
