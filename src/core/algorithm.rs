use core::fmt;
use std::marker::PhantomData;
use std::path::PathBuf;

use csv::ReaderBuilder;
use more_asserts::assert_le;
use rand::prelude::{IteratorRandom, SliceRandom};
use serde::de::DeserializeOwned;
use tracing::field::valuable;
use tracing::trace;
use tracing_subscriber::EnvFilter;

use crate::core::characteristics::FitnessScore;
use crate::{
    core::characteristics::{Breed, Fitness, Generate},
    utils::random::generator,
};

use super::{
    characteristics::Mutate,
    inputs::{Inputs, ValidInput},
    population::Population,
};

#[derive(Debug, Clone)]
pub struct HyperParameters<T>
where
    T: Fitness + Mutate + Generate,
{
    pub population_size: usize,
    pub gap: f64,
    pub mutation_percent: f64,
    pub crossover_percent: f64,
    pub n_generations: usize,
    pub lazy_evaluate: bool,
    pub fitness_parameters: T::FitnessParameters,
    pub program_parameters: T::GeneratorParameters,
}

/// Defines a program capable of loading inputs from various sources.
pub trait Loader
where
    Self::InputType: ValidInput + DeserializeOwned,
{
    type InputType;

    /// Loads entities from a csv file found on the local file system.
    fn load_from_csv(file_path: impl Into<PathBuf>) -> Inputs<Self::InputType> {
        let mut csv_reader = ReaderBuilder::new()
            .has_headers(false)
            .from_path(file_path.into())
            .unwrap();

        let inputs: Result<Inputs<Self::InputType>, _> = csv_reader
            .deserialize()
            .into_iter()
            .map(|input| input)
            .collect();

        inputs.unwrap()
    }
}

pub struct GeneticAlgorithmIter<G>
where
    G: GeneticAlgorithm + ?Sized,
{
    generation: usize,
    current_population: Option<Population<G::O>>,
    marker: PhantomData<G>,
    params: HyperParameters<G::O>,
}

impl<G> GeneticAlgorithmIter<G>
where
    G: GeneticAlgorithm + ?Sized,
{
    pub fn new(params: HyperParameters<G::O>) -> Self {
        return GeneticAlgorithmIter {
            generation: 0,
            current_population: None,
            marker: PhantomData,
            params,
        };
    }
}

impl<G> Iterator for GeneticAlgorithmIter<G>
where
    G: GeneticAlgorithm,
{
    type Item = Population<G::O>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = if self.generation == 0 {
            let mut population = G::init_pop(&self.params);

            G::rank(
                &mut population,
                &mut self.params.fitness_parameters,
                self.params.lazy_evaluate,
            );

            Some(population)
        } else if self.generation < self.params.n_generations {
            let mut current_population = self.current_population.clone().unwrap();

            G::apply_selection(&mut current_population, self.params.gap);
            G::breed(
                &mut current_population,
                self.params.mutation_percent,
                self.params.crossover_percent,
                &self.params.program_parameters,
            );
            G::rank(
                &mut current_population,
                &mut self.params.fitness_parameters,
                self.params.lazy_evaluate,
            );

            Some(current_population)
        } else {
            return None;
        };

        self.current_population = item.clone();
        self.generation += 1;

        return item;
    }
}

pub trait GeneticAlgorithm
where
    Self::O: Fitness + Generate + PartialOrd + Sized + Clone + Mutate + Breed + fmt::Debug,
{
    type O;

    fn init_sys() {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(EnvFilter::from_default_env())
            .try_init()
            .unwrap_or(());
    }

    fn init_pop(hyperparams: &HyperParameters<Self::O>) -> Population<Self::O> {
        let mut population = Population::with_capacity(hyperparams.population_size);

        for _ in 0..hyperparams.population_size {
            let program = Self::O::generate(&hyperparams.program_parameters);
            population.push(program)
        }

        return population;
    }

    /// Evaluates the individuals found in the current population.
    fn rank(
        population: &mut Population<Self::O>,
        fitness_parameters: &mut <Self::O as Fitness>::FitnessParameters,
        lazy_evaluate: bool,
    ) {
        for individual in population.iter_mut() {
            // Only force evaluation when lazy evaluation is set off or in cases
            // where lazy evaluation is desired, evaluate individuals who haven't changed.
            let should_eval = if lazy_evaluate {
                individual.get_fitness().is_not_evaluated()
            } else {
                true
            };

            if should_eval {
                individual.eval_fitness(fitness_parameters)
            }
        }

        // Organize individuals by their fitness score.
        population.sort();
        assert_le!(population.worst(), population.best());

        // trace!(
        //     "{:?}",
        //     population = valuable(
        //         &population
        //             .iter()
        //             .map(|p| p.get_fitness())
        //             .collect::<Vec<FitnessScore>>()
        //     )
        // );

        Self::on_post_rank(population, fitness_parameters)
    }

    fn on_post_rank(
        _population: &mut Population<Self::O>,
        _fitness_params: &mut <Self::O as Fitness>::FitnessParameters,
    ) {
    }

    fn apply_selection(population: &mut Population<Self::O>, gap: f64) {
        assert!(gap >= 0. && gap <= 1.);

        let pop_len = population.len();

        let mut n_of_individuals_to_drop =
            pop_len - ((1.0 - gap) * (pop_len as f64)).floor() as usize;

        // Drop invalid individuals.
        loop {
            if population.worst().is_some() {
                if population
                    .worst()
                    .map(|v| v.get_fitness().is_invalid())
                    .unwrap()
                {
                    population.pop();
                    n_of_individuals_to_drop -= 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Drop remaining gap.
        while n_of_individuals_to_drop > 0 {
            n_of_individuals_to_drop -= 1;
            population.pop();
        }
    }

    fn breed(
        population: &mut Population<Self::O>,
        mutations_percent: f64,
        crossover_percent: f64,
        mutation_parameters: &<Self::O as Generate>::GeneratorParameters,
    ) {
        let pop_cap = population.capacity();
        let pop_len = population.len();

        let mut remaining_pool_spots = pop_cap - pop_len;

        if remaining_pool_spots == 0 {
            return;
        }

        let mut n_mutations = (remaining_pool_spots as f64 * mutations_percent).floor() as usize;
        let mut n_crossovers = (remaining_pool_spots as f64 * crossover_percent).floor() as usize;

        assert_le!(n_mutations + n_crossovers, remaining_pool_spots);

        let mut children = vec![];

        // Crossover + Mutation
        // TODO: Add a way to priortize mutations or crossovers.
        while (n_crossovers + n_mutations) > 0 {
            // Step 1: Choose Parents
            let selected_a = population.iter().choose(&mut generator());
            let selected_b = population.iter().choose(&mut generator());

            // Step 2: Transform Children
            if let (Some(parent_a), Some(parent_b)) = (selected_a, selected_b) {
                // NOTE: This can be done in parallel.
                // Step 2A: Crossover
                if n_crossovers > 0 {
                    let child = parent_a
                        .two_point_crossover(parent_b)
                        .choose(&mut generator())
                        .unwrap()
                        .to_owned();

                    remaining_pool_spots -= 1;
                    n_crossovers -= 1;

                    children.push(child)
                }

                // Step 2B: Mutate
                if n_mutations > 0 {
                    let parents = [parent_a, parent_b];
                    let parent_to_mutate = parents.choose(&mut generator());

                    let child = parent_to_mutate
                        .map(|parent| parent.mutate(mutation_parameters))
                        .unwrap();

                    remaining_pool_spots -= 1;
                    n_mutations -= 1;

                    children.push(child)
                }
            };
        }

        // Fill reset with clones
        for individual in population
            .iter()
            .cloned()
            .choose_multiple(&mut generator(), remaining_pool_spots)
        {
            population.push(individual)
        }

        population.extend(children)
    }

    fn execute<'b>(hyper_params: HyperParameters<Self::O>) -> GeneticAlgorithmIter<Self> {
        return GeneticAlgorithmIter::new(hyper_params);
    }
}
