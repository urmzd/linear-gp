use core::fmt;
use std::iter::repeat_with;
use std::marker::PhantomData;
use std::path::PathBuf;

use csv::ReaderBuilder;
use more_asserts::assert_le;
use rand::prelude::{IteratorRandom, SliceRandom};
use serde::de::DeserializeOwned;
use tracing::trace;
use tracing::{debug, field::valuable};
use tracing_subscriber::EnvFilter;

use crate::{
    core::characteristics::{Breed, Fitness, Generate},
    utils::random::generator,
};

use super::{
    characteristics::{DuplicateNew, Mutate},
    inputs::{Inputs, ValidInput},
    population::Population,
};

#[derive(Debug, Clone)]
pub struct HyperParameters<T>
where
    T: Fitness + Mutate + Generate + Clone,
{
    pub population_size: usize,
    pub gap: f64,
    pub mutation_percent: f64,
    pub crossover_percent: f64,
    pub n_generations: usize,
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
    next_population: Option<Population<G::O>>,
    marker: PhantomData<G>,
    params: HyperParameters<G::O>,
}

impl<G> GeneticAlgorithmIter<G>
where
    G: GeneticAlgorithm + ?Sized,
{
    pub fn new(params: HyperParameters<G::O>) -> Self {
        let (current_population, params) = G::init_pop(params.clone());
        Self {
            generation: 0,
            next_population: Some(current_population),
            marker: PhantomData,
            params,
        }
    }
}

impl<G> Iterator for GeneticAlgorithmIter<G>
where
    G: GeneticAlgorithm,
{
    type Item = Population<G::O>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.generation >= self.params.n_generations {
            return None;
        }

        debug!(generation = valuable(&self.generation));

        // Freeze population.
        let mut population = self.next_population.clone().unwrap();
        let mut params = self.params.clone();

        (population, params) = G::on_pre_eval_fitness(population, params);
        (population, params) = G::eval_fitness(population, params);
        (population, params) = G::rank(population, params);
        (population, params) = G::on_post_rank(population, params);

        assert!(population
            .iter()
            .all(|p| !p.get_fitness().is_not_evaluated()));

        trace!(
            best_score = valuable(
                &population
                    .best()
                    .map(|p| p.get_fitness().unwrap_or(f64::NAN))
            ),
            median_score = valuable(
                &population
                    .median()
                    .map(|p| p.get_fitness().unwrap_or(f64::NAN))
            ),
            worst_score = valuable(
                &population
                    .worst()
                    .map(|p| p.get_fitness().unwrap_or(f64::NAN))
            ),
            generation = valuable(&self.generation)
        );

        let (new_population, params) = G::survive(population.clone(), params);
        let (new_population, ..) = G::variation(new_population, params);

        self.next_population = Some(new_population.clone());
        self.generation += 1;

        return Some(population.clone());
    }
}

pub trait Organism:
    Fitness + Generate + DuplicateNew + PartialOrd + Sized + Clone + Mutate + Breed + fmt::Debug + Send
{
}

pub trait GeneticAlgorithm: Send
where
    Self::O: Organism,
{
    type O;

    fn init_sys() {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(EnvFilter::from_default_env())
            .try_init()
            .unwrap_or(());
    }

    fn init_pop(
        hyperparams: HyperParameters<Self::O>,
    ) -> (Population<Self::O>, HyperParameters<Self::O>) {
        let population = repeat_with(|| Self::O::generate(&hyperparams.program_parameters))
            .take(hyperparams.population_size)
            .collect();

        (population, hyperparams)
    }

    fn eval_fitness(
        mut population: Population<Self::O>,
        params: HyperParameters<Self::O>,
    ) -> (Population<Self::O>, HyperParameters<Self::O>) {
        for individual in population.iter_mut() {
            individual.eval_fitness(params.fitness_parameters.clone());
            assert!(!individual.get_fitness().is_not_evaluated());
        }

        (population, params)
    }

    /// Evaluates the individuals found in the current population.
    fn rank(
        mut population: Population<Self::O>,
        params: HyperParameters<Self::O>,
    ) -> (Population<Self::O>, HyperParameters<Self::O>) {
        population.sort();
        // Organize individuals by their fitness score.
        assert_le!(population.worst(), population.best());
        (population, params)
    }

    fn on_pre_eval_fitness(
        population: Population<Self::O>,
        params: HyperParameters<Self::O>,
    ) -> (Population<Self::O>, HyperParameters<Self::O>) {
        (population, params)
    }

    fn on_post_rank(
        population: Population<Self::O>,
        _parameters: HyperParameters<Self::O>,
    ) -> (Population<Self::O>, HyperParameters<Self::O>) {
        (population, _parameters)
    }

    fn survive(
        mut population: Population<Self::O>,
        parameters: HyperParameters<Self::O>,
    ) -> (Population<Self::O>, HyperParameters<Self::O>) {
        let pop_len = population.len();

        let mut n_of_individuals_to_drop =
            (pop_len as isize) - ((1.0 - parameters.gap) * (pop_len as f64)).floor() as isize;

        loop {
            if population.worst().is_some() {
                if population
                    .worst()
                    .map(|v| v.get_fitness().is_invalid())
                    .expect("At least one individual to exist.")
                {
                    population.pop();

                    n_of_individuals_to_drop -= 1;
                } else {
                    // We've encountered a valid individual. Stop deleting.
                    break;
                }
            } else {
                // If the population is empty, there's nothing to drop.
                break;
            }
        }

        // Drop remaining gap.
        while n_of_individuals_to_drop > 0 {
            n_of_individuals_to_drop -= 1;
            population.pop();
        }

        (population, parameters)
    }

    fn variation(
        mut population: Population<Self::O>,
        parameters: HyperParameters<Self::O>,
    ) -> (Population<Self::O>, HyperParameters<Self::O>) {
        let pop_cap = population.capacity();
        let pop_len = population.len();

        let mut remaining_pool_spots = pop_cap - pop_len;

        if remaining_pool_spots == 0 {
            return (population, parameters);
        }

        let mut n_mutations =
            (remaining_pool_spots as f64 * parameters.mutation_percent).floor() as usize;
        let mut n_crossovers =
            (remaining_pool_spots as f64 * parameters.crossover_percent).floor() as usize;

        assert_le!(n_mutations + n_crossovers, remaining_pool_spots);

        let mut offspring = vec![];

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

                    offspring.push(child)
                }

                // Step 2B: Mutate
                if n_mutations > 0 {
                    let parents = [parent_a, parent_b];
                    let parent_to_mutate = parents.choose(&mut generator());

                    let child = parent_to_mutate
                        .map(|parent| parent.mutate(&parameters.program_parameters))
                        .unwrap();

                    remaining_pool_spots -= 1;
                    n_mutations -= 1;

                    offspring.push(child)
                }
            } else {
                // Generate new children?
                // Or do we give up and panic?
                panic!("Woah, this should never happen. The whole population died out.")
            };
        }

        // Fill reset with clones
        for individual in population
            .iter()
            .choose_multiple(&mut generator(), remaining_pool_spots)
        {
            offspring.push(individual.duplicate_new())
        }

        population.extend(offspring);

        (population, parameters)
    }

    /// Build generator.
    fn build<'b>(hyper_params: HyperParameters<Self::O>) -> GeneticAlgorithmIter<Self> {
        Self::init_sys();
        return GeneticAlgorithmIter::new(hyper_params);
    }
}
