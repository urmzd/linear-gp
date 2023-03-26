use std::iter::repeat_with;

use itertools::Itertools;
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
};

use rayon::join;

use crate::{
    core::engines::{breed_engine::Breed, reset_engine::Reset},
    utils::random::generator,
};

use super::{
    fitness_engine::Fitness, generate_engine::Generate, mutate_engine::Mutate, valid_engine::Valid,
};

pub struct CoreEngine;

/// init_population should using GenerateEngine::generate(ProgramParameters or QProgramParameters)
/// to generate a population of Programs or QPrograms respectively.
///
/// eval_fitness should use FitnessEngine::eval_fitness(Program or QProgram, Any State, Any Parameters) to evaluate the fitness of each individual.
/// It should also take the n_trials and generate n_states, taking the median fitness associated with each
/// n_trial. GenerateEngine::generate(State) should also exist to generate a new state.
///
/// rank should sort the population by fitness.
///
/// surive should drop the population by the given gap.
///
/// variation should use MutateEngine::mutate and BreedEngine::breed to fill the population with new indivudals.
///
/// The population should be a Vec of Programs or QPrograms.
pub trait Core<Individual, ProgramParameters, State, Marker>
where
    ProgramParameters: Copy + Send + Sync,
    Individual: Ord + Clone + Send + Sync,
{
    fn init_population<G: Generate<ProgramParameters, Individual>>(
        program_parameters: ProgramParameters,
        population_size: usize,
    ) -> Vec<Individual> {
        let population = repeat_with(|| G::generate(program_parameters))
            .take(population_size)
            .collect();

        population
    }

    fn eval_fitness<
        F: Fitness<Individual, State, Marker>,
        G: Generate<(), State>,
        R: Reset<Individual> + Reset<State>,
    >(
        population: &mut Vec<Individual>,
        n_trials: usize,
        _marker: Marker,
    ) {
        let mut trials = repeat_with(|| G::generate(())).take(n_trials).collect_vec();

        for individual in population.iter_mut() {
            for trial in trials.iter_mut() {
                R::reset(individual);
                R::reset(trial);
                F::eval_fitness(individual, trial);
            }
        }
    }

    fn rank(population: &mut Vec<Individual>) {
        population.sort();
    }

    fn survive<V: Valid<Individual>>(population: &mut Vec<Individual>, gap: f64) {
        let n_individuals = population.len();

        let mut n_of_individuals_to_drop =
            (n_individuals as isize) - ((1.0 - gap) * (n_individuals as f64)).floor() as isize;

        population.retain(V::valid);
        let n_individuals_dropped = n_individuals - population.len();
        n_of_individuals_to_drop -= n_individuals_dropped as isize;

        while n_of_individuals_to_drop > 0 {
            n_of_individuals_to_drop -= 1;
            population.pop();
        }
    }

    fn variation<B: Breed<Individual>, M: Mutate<ProgramParameters, Individual>>(
        population: &mut Vec<Individual>,
        crossover_percent: f64,
        mutation_percent: f64,
        program_parameters: ProgramParameters,
    ) {
        debug_assert!(population.len() > 0);

        let pop_cap = population.capacity();
        let pop_len = population.len();

        let remaining_pool_spots = pop_cap - pop_len;

        if remaining_pool_spots == 0 {
            return;
        }

        let n_mutations = (remaining_pool_spots as f64 * mutation_percent).floor() as usize;
        let n_crossovers = (remaining_pool_spots as f64 * crossover_percent).floor() as usize;

        debug_assert!(n_mutations + n_crossovers <= remaining_pool_spots);

        // Create separate vectors for offspring from crossover and mutation
        let (mut crossover_offspring, mut mutation_offspring) = join(
            || {
                (0..n_crossovers)
                    .filter_map(|_| {
                        let parent_a = population.iter().choose(&mut generator());
                        let parent_b = population.iter().choose(&mut generator());

                        if let (Some(parent_a), Some(parent_b)) = (parent_a, parent_b) {
                            let children = B::two_point_crossover(&parent_a, &parent_b);

                            match generator().gen_range(0..2) {
                                0 => Some(children.0),
                                1 => Some(children.1),
                                _ => unreachable!(),
                            }
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<Individual>>()
            },
            || {
                (0..n_mutations)
                    .filter_map(|_| {
                        let parent = population.iter().choose(&mut generator());

                        if let Some(parent) = parent {
                            let mut clone = parent.clone();
                            M::mutate(&mut clone, program_parameters);
                            Some(clone)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<Individual>>()
            },
        );

        // Step 3: Add Children to Population
        population.append(&mut crossover_offspring);
        population.append(&mut mutation_offspring);
    }
}
