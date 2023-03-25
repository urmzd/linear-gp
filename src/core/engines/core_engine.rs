use std::iter::repeat_with;

use rand::seq::IteratorRandom;

use crate::{
    core::engines::{
        breed_engine::{Breed, BreedEngine},
        fitness_engine::{Fitness, FitnessEngine},
    },
    utils::random::generator,
};

use super::generate_engine::{Generate, GenerateEngine};

pub struct CoreEngine;

/// Possible Variations:
///
/// 1. Program, ProgramGeneratorParameters, (Environment,)
/// 2. Program, ProgramGeneratorParameters, (Environment, QTable)
pub trait Core<I, P, F> {
    fn init_pop(program_parameters: P, population_size: usize) -> Vec<I> {
        let population = repeat_with(GenerateEngine::generate(program_parameters))
            .take(population_size)
            .collect();

        population
    }

    fn eval_fitness(population: &mut Vec<I>, parameters: &mut F) {
        for individual in population.iter_mut() {
            FitnessEngine::eval_fitness(individual, parameters);
            debug_assert!(!individual.get_fitness().is_not_evaluated());
        }
    }

    fn rank(population: &mut Vec<I>) {
        population.sort();
        // Organize individuals by their fitness score.
        debug_assert!(population.worst() <= population.best());
    }

    fn survive(population: &mut Vec<I>, gap: f64) {
        let pop_len = population.len();

        let mut n_of_individuals_to_drop =
            (pop_len as isize) - ((1.0 - gap) * (pop_len as f64)).floor() as isize;

        // Drop invalid individuals.
        while let Some(true) = population.worst().map(|p| p.get_fitness().is_invalid()) {
            population.pop();
            n_of_individuals_to_drop -= 1;
        }

        // Drop remaining gap, if any...
        while n_of_individuals_to_drop > 0 {
            n_of_individuals_to_drop -= 1;
            population.pop();
        }
    }

    fn variation(
        population: &mut Vec<I>,
        crossover_percent: f64,
        mutation_percent: f64,
        program_parameters: P,
    ) {
        debug_assert!(population.len() > 0);

        let pop_cap = population.capacity();
        let pop_len = population.len();

        let mut remaining_pool_spots = pop_cap - pop_len;

        if remaining_pool_spots == 0 {
            return;
        }

        let mut n_mutations = (remaining_pool_spots as f64 * mutation_percent).floor() as usize;
        let mut n_crossovers = (remaining_pool_spots as f64 * crossover_percent).floor() as usize;

        debug_assert!(n_mutations + n_crossovers <= remaining_pool_spots);

        let mut offspring = vec![];

        // Crossover + Mutation
        while (n_crossovers + n_mutations) > 0 {
            // Step 1: Choose Parents
            let selected_a = population.iter().choose(&mut generator());
            let selected_b = population.iter().choose(&mut generator());

            // Step 2: Transform Children
            if let (Some(parent_a), Some(parent_b)) = (selected_a, selected_b) {
                // NOTE: This can be done in parallel.
                // Step 2A: Crossover
                if n_crossovers > 0 {
                    let child = BreedEngine::two_point_crossover(&parent_a, &parent_b)
                        .choose(&mut generator())
                        .unwrap()
                        .to_owned();

                    remaining_pool_spots -= 1;
                    n_crossovers -= 1;

                    offspring.push(child)
                }

                // Step 2B: Mutate
                if n_mutations > 0 {
                    let parents = [parent_a, parent_b];
                    let parent_to_mutate = parents.choose(&mut generator());

                    let child = parent_to_mutate
                        .map(|parent| program_parameters.mutate(parent))
                        .unwrap();

                    remaining_pool_spots -= 1;
                    n_mutations -= 1;

                    offspring.push(child)
                }
            } else {
                unreachable!()
            };
        }

        // Fill reset with clones
        for individual in population
            .iter()
            .choose_multiple(&mut generator(), remaining_pool_spots)
        {
            offspring.push(individual.reset_new())
        }

        population.extend(offspring);
    }
}
