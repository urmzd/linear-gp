use std::{iter::repeat_with, sync::Arc};

use itertools::Itertools;
use rand::{seq::IteratorRandom, Rng};

use crate::{
    core::engines::{breed_engine::Breed, reset_engine::Reset, status_engine::StatusEngine},
    utils::random::generator,
};

use super::{
    fitness_engine::Fitness, generate_engine::Generate, mutate_engine::Mutate,
    status_engine::Status,
};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Deserialize, Serialize, Builder)]
pub struct HyperParameters<P> {
    #[builder(default = "100")]
    pub population_size: usize,
    #[builder(default = "0.5")]
    pub gap: f64,
    #[builder(default = "0.5")]
    pub mutation_percent: f64,
    #[builder(default = "0.5")]
    pub crossover_percent: f64,
    #[builder(default = "100")]
    pub n_generations: usize,
    #[builder(default = "1")]
    pub n_trials: usize,
    pub program_parameters: P,
}

pub struct CoreIter<C>
where
    C: Core,
{
    generation: usize,
    next_population: Option<Vec<C::Individual>>,
    params: HyperParameters<C::ProgramParameters>,
    engine: C,
}

impl<C> CoreIter<C>
where
    C: Core,
{
    pub fn new(engine: C, hp: HyperParameters<C::ProgramParameters>) -> Self {
        let current_population = C::init_population(hp.program_parameters, hp.population_size);

        Self {
            generation: 0,
            next_population: Some(current_population),
            params: hp,
            engine,
        }
    }
}

impl<C> Iterator for CoreIter<C>
where
    C: Core,
{
    type Item = Vec<C::Individual>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.generation > self.params.n_generations {
            return None;
        }

        // Freeze population.
        let mut population = self.next_population.clone().unwrap();

        C::eval_fitness(&mut population, self.params.n_trials);
        C::rank(&mut population);

        assert!(population.iter().all(C::Status::evaluated));

        info!(
            best = serde_json::to_string(&population.last()).unwrap(),
            median = serde_json::to_string(&population.get(population.len() / 2)).unwrap(),
            worst = serde_json::to_string(&population.first()).unwrap(),
            generation = serde_json::to_string(&self.generation).unwrap()
        );

        let mut new_population = population.clone();

        C::survive(&mut new_population, self.params.gap);
        C::variation(
            &mut new_population,
            self.params.crossover_percent,
            self.params.mutation_percent,
            self.params.program_parameters,
        );

        self.next_population = Some(new_population);
        self.generation += 1;

        return Some(population.clone());
    }
}

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
pub trait Core {
    type Individual: Ord + Clone + Send + Sync + Serialize;
    type ProgramParameters: Copy + Send + Sync;
    type State;
    type Marker;
    type Generate: Generate<Self::ProgramParameters, Self::Individual> + Generate<(), Self::State>;
    type Fitness: Fitness<Self::Individual, Self::State, Self::Marker>;
    type Reset: Reset<Self::Individual> + Reset<Self::State>;
    type Breed: Breed<Self::Individual>;
    type Mutate: Mutate<Self::ProgramParameters, Self::Individual>;
    type Status: Status<Self::Individual>;

    fn init_population(
        program_parameters: Self::ProgramParameters,
        population_size: usize,
    ) -> Vec<Self::Individual> {
        let population = repeat_with(|| Self::Generate::generate(program_parameters))
            .take(population_size)
            .collect();

        population
    }

    fn eval_fitness(population: &mut Vec<Self::Individual>, n_trials: usize) {
        let mut trials: Vec<Self::State> = repeat_with(|| Self::Generate::generate(()))
            .take(n_trials)
            .collect_vec();

        for individual in population.iter_mut() {
            for trial in trials.iter_mut() {
                Self::Reset::reset(individual);
                Self::Reset::reset(trial);
                Self::Fitness::eval_fitness(individual, trial);
            }
        }
    }

    fn rank(population: &mut Vec<Self::Individual>) {
        population.sort();
    }

    fn survive(population: &mut Vec<Self::Individual>, gap: f64) {
        let n_individuals = population.len();

        let mut n_of_individuals_to_drop =
            (n_individuals as isize) - ((1.0 - gap) * (n_individuals as f64)).floor() as isize;

        population.retain(Self::Status::valid);
        let n_individuals_dropped = n_individuals - population.len();
        n_of_individuals_to_drop -= n_individuals_dropped as isize;

        while n_of_individuals_to_drop > 0 {
            n_of_individuals_to_drop -= 1;
            population.pop();
        }
    }

    fn variation(
        population: &mut Vec<Self::Individual>,
        crossover_percent: f64,
        mutation_percent: f64,
        program_parameters: Self::ProgramParameters,
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
        let n_clones = remaining_pool_spots - n_mutations - n_crossovers;

        let mut clone_offspring: Vec<Self::Individual> = Vec::with_capacity(n_clones);
        let mut mutation_offspring: Vec<Self::Individual> = Vec::with_capacity(n_mutations);
        let mut crossover_offspring: Vec<Self::Individual> = Vec::with_capacity(n_crossovers);

        debug_assert!(n_mutations + n_crossovers <= remaining_pool_spots);

        let rc_population = Arc::new(population.clone());
        rayon::scope(|s| {
            s.spawn(|_| {
                crossover_offspring.extend((0..n_crossovers).filter_map(|_| {
                    let population_to_read = rc_population.clone();
                    let parent_a = population_to_read.iter().choose(&mut generator());
                    let parent_b = population_to_read.iter().choose(&mut generator());

                    if let (Some(parent_a), Some(parent_b)) = (parent_a, parent_b) {
                        let children = Self::Breed::two_point_crossover(&parent_a, &parent_b);
                        match generator().gen_range(0..2) {
                            0 => Some(children.0),
                            1 => Some(children.1),
                            _ => unreachable!(),
                        }
                    } else {
                        None
                    }
                }));
            });

            s.spawn(|_| {
                mutation_offspring.extend((0..n_mutations).filter_map(|_| {
                    let population_to_read = rc_population.clone();
                    let parent = population_to_read.iter().choose(&mut generator());

                    if let Some(parent) = parent {
                        let mut clone = parent.clone();
                        Self::Mutate::mutate(&mut clone, program_parameters);
                        Some(clone)
                    } else {
                        None
                    }
                }))
            });

            s.spawn(|_| {
                clone_offspring.extend((0..n_clones).filter_map(|_| {
                    let population_to_read = rc_population.clone();
                    let parent = population_to_read.iter().choose(&mut generator());

                    if let Some(parent) = parent {
                        let mut clone = parent.clone();
                        Self::Reset::reset(&mut clone);
                        Some(clone)
                    } else {
                        None
                    }
                }))
            });
        });

        // Step 3: Add Children to Population
        population.append(&mut crossover_offspring);
        population.append(&mut mutation_offspring);
        population.append(&mut clone_offspring);
    }
}
