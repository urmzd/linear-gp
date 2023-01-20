use core::fmt;
use std::marker::PhantomData;
use std::path::PathBuf;

use csv::ReaderBuilder;
use more_asserts::assert_le;
use rand::prelude::{IteratorRandom, SliceRandom};
use serde::de::DeserializeOwned;
use tracing::{field::valuable, debug};
use tracing::trace;
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
        debug!(generation=valuable(&self.generation));

        let item = if self.generation == 0 {
            let mut population = G::init_pop(&self.params);

            G::on_pre_eval_fitness(&mut population, &mut self.params);
            G::eval_fitness(&mut population, &mut self.params);
            G::rank(&mut population);
            G::on_post_rank(&mut population, &mut self.params);

            population
        } else if self.generation < self.params.n_generations {
            // Freeze for now.
            let mut population = self.current_population.clone().unwrap();

            G::apply_selection(&mut population, self.params.gap);
            G::breed(
                &mut population,
                self.params.mutation_percent,
                self.params.crossover_percent,
                &self.params.program_parameters,
            );

            G::on_pre_eval_fitness(&mut population, &mut self.params);
            G::eval_fitness(&mut population, &mut self.params);
            G::rank(&mut population);
            G::on_post_rank(&mut population, &mut self.params);

            // Produce new populaiton.
            population
        } else {
            return None;
        };

        assert!(item.iter().all(|p| !p.get_fitness().is_not_evaluated()));

        trace!(
            best_score = valuable(&item.best().map(|p| p.get_fitness().unwrap_or(f64::NAN))),
            median_score = valuable(&item.median().map(|p| p.get_fitness().unwrap_or(f64::NAN))),
            worst_score = valuable(&item.worst().map(|p| p.get_fitness().unwrap_or(f64::NAN))),
            generation = valuable(&self.generation)
        );

        // Freeze population and store it.
        self.current_population = Some(item.clone());
        self.generation += 1;

        return Some(item);
    }
}

pub trait GeneticAlgorithm
where
    Self::O: Fitness
        + Generate
        + DuplicateNew
        + PartialOrd
        + Sized
        + Clone
        + Mutate
        + Breed
        + fmt::Debug,
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

    fn eval_fitness(p: &mut Population<Self::O>, params: &mut HyperParameters<Self::O>) {
        for individual in p.iter_mut() {
            // Only force evaluation when lazy evaluation is set off or in cases
            // where lazy evaluation is desired, evaluate individuals who haven't changed.
            let should_eval = if params.lazy_evaluate {
                individual.get_fitness().is_not_evaluated()
            } else {
                true
            };

            if should_eval {
                individual.eval_fitness(&mut params.fitness_parameters);
                assert!(!individual.get_fitness().is_not_evaluated());
            }
        }
    }

    /// Evaluates the individuals found in the current population.
    fn rank(population: &mut Population<Self::O>) {
        // Organize individuals by their fitness score.
        population.sort();
        assert_le!(population.worst(), population.best());
    }

    fn on_pre_eval_fitness(
        _population: &mut Population<Self::O>,
        _parameters: &mut HyperParameters<Self::O>,
    ) {
    }

    fn on_post_rank(
        _population: &mut Population<Self::O>,
        _parameters: &mut HyperParameters<Self::O>,
    ) {
    }

    fn apply_selection(population: &mut Population<Self::O>, gap: f64) {
        assert!(gap >= 0. && gap <= 1.);

        let pop_len = population.len();

        let mut n_of_individuals_to_drop =
            (pop_len as isize) - ((1.0 - gap) * (pop_len as f64)).floor() as isize;

        // Drop invalid individuals.
        // NOTE: what if we drop all individuals?
        loop {
            if population.worst().is_some() {
                if population
                    .worst()
                    .map(|v| v.get_fitness().is_invalid())
                    .expect("Atleast one individual to exist.")
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
            children.push(individual.duplicate_new())
        }

        population.extend(children)
    }

    fn execute<'b>(hyper_params: HyperParameters<Self::O>) -> GeneticAlgorithmIter<Self> {
        Self::init_sys();
        return GeneticAlgorithmIter::new(hyper_params);
    }
}
