use std::iter::repeat_with;
use std::marker::PhantomData;
use std::path::PathBuf;

use csv::ReaderBuilder;
use derive_builder::Builder;
use rand::prelude::{IteratorRandom, SliceRandom};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tracing::info;
use uuid::Uuid;

use crate::{
    core::characteristics::{Breed, Fitness, Generate},
    utils::random::generator,
};

use super::{
    characteristics::{Mutate, ResetNew},
    inputs::{Inputs, ValidInput},
    population::Population,
};

#[derive(Debug, Clone, Deserialize, Serialize, Builder)]
pub struct HyperParameters<F: Fitness, G: Generate> {
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
    pub fitness_parameters: F,
    pub program_parameters: G,
}

/// Defines a program capable of loading inputs from various sources.
pub trait SupportLoad
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

pub struct GeneticAlgorithmIter<S, T, P> {
    generation: usize,
    next_population: Option<Population<P>>,
    params: HyperParameters<S, T>,
}

impl<S, T, P> GeneticAlgorithmIter<S, T, P> {
    pub fn new(params: HyperParameters<S, T>) -> Self {
        let (current_population, params) = GeneticAlgorithm::init_pop::<S>(params.clone());

        Self {
            generation: 0,
            next_population: Some(current_population),
            params,
        }
    }
}

impl<S, T, P> Iterator for GeneticAlgorithmIter<S, T, P>
where
    G: GeneticAlgorithm,
{
    type Item = Population<G::O>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.generation > self.params.n_generations {
            return None;
        }

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

        info!(
            best = serde_json::to_string(&population.best()).unwrap(),
            median = serde_json::to_string(&population.median()).unwrap(),
            worst = serde_json::to_string(&population.worst()).unwrap(),
            generation = serde_json::to_string(&self.generation).unwrap()
        );

        let (new_population, params) = G::survive(population.clone(), params);
        let (new_population, ..) = G::variation(new_population, params);

        self.next_population = Some(new_population.clone());
        self.generation += 1;

        return Some(population.clone());
    }
}

pub trait GeneticAlgorithm {
    fn init_pop<G: Generate>(
        program_parameters: G,
        population_size: usize,
    ) -> Population<G::Output> {
        let population = repeat_with(program_parameters.generate)
            .take(population_size)
            .collect();

        population
    }

    fn eval_fitness<F: Fitness, T>(population: &mut Population<T>, evaluator: F) {
        for individual in population.iter_mut() {
            evaluator.eval_fitness(program, parameters);

            individual.eval_fitness(fitness_parameters);
            debug_assert!(!individual.get_fitness().is_not_evaluated());
        }
    }

    /// Evaluates the individuals found in the current population.
    fn rank<T>(population: &mut Population<T>) {
        population.sort();
        // Organize individuals by their fitness score.
        debug_assert!(population.worst() <= population.best());
    }

    fn on_pre_eval_fitness<F, O>(population: &mut Population<O>, fitness_parameters: F) {}

    fn on_post_rank<O>(population: &mut Population<O>) {}

    fn survive<O>(population: &mut Population<O>, gap: f64) {
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

    fn variation<O: Breed, P: Generate>(
        population: &mut Population<O>,
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
                        .map(|parent| program_parameters.mutate(parent))
                        .unwrap();

                    remaining_pool_spots -= 1;
                    n_mutations -= 1;

                    offspring.push(child)
                }
            } else {
                panic!("Woah, this should never happen. The whole population died out.")
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

    /// Build generator.
    fn build<S, T, P>(hyper_params: HyperParameters<S, T>) -> GeneticAlgorithmIter<S, T, P> {
        info!(run_id = &(Uuid::new_v4()).to_string());
        GeneticAlgorithmIter::new(hyper_params)
    }
}
