use core::fmt;
use std::path::PathBuf;

use csv::ReaderBuilder;
use more_asserts::assert_le;
use rand::prelude::{IteratorRandom, SliceRandom};
use serde::de::DeserializeOwned;
use tracing::debug;
use tracing::field::valuable;

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

pub trait GeneticAlgorithm
where
    Self::O: Fitness + Generate + PartialOrd + Sized + Clone + Mutate + Breed + fmt::Debug,
{
    type O;

    /// Generates a set of random individuals to undergo the evolutionary process.
    fn init_population(hyper_params: &HyperParameters<Self::O>) -> Population<Self::O> {
        let mut population = Population::with_capacity(hyper_params.population_size);

        for _ in 0..hyper_params.population_size {
            let program = Self::O::generate(&hyper_params.program_parameters);
            population.push(program)
        }

        population
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
    }

    fn apply_selection(population: &mut Population<Self::O>, gap: f64) {
        assert!(gap >= 0. && gap <= 1.);
        assert_le!(population.worst(), population.best());

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

    fn execute<'b>(
        mut hyper_params: &mut HyperParameters<Self::O>,
        mut hooks: EventHooks<'b, Self::O>,
    ) -> Result<Population<Self::O>, Box<dyn std::error::Error>> {
        let mut population = Self::init_population(hyper_params);

        if let Some(hook) = hooks.on_post_init {
            hook(&mut population, &mut hyper_params);
        }

        let mut rank_step =
            |rank_population: &mut Population<Self::O>,
             rank_hyper_params: &mut HyperParameters<Self::O>| {
                Self::rank(
                    rank_population,
                    &mut rank_hyper_params.fitness_parameters,
                    rank_hyper_params.lazy_evaluate,
                );

                if let Some(hook) = hooks.on_post_rank.as_mut() {
                    hook(rank_population, rank_hyper_params);
                }
            };

        for _generation in 0..hyper_params.n_generations {
            debug!(generation=valuable(&_generation));
            rank_step(&mut population, &mut hyper_params);

            Self::apply_selection(&mut population, hyper_params.gap);
            if let Some(hook) = hooks.on_post_selection.as_mut() {
                hook(&mut population, &mut hyper_params);
            }

            Self::breed(
                &mut population,
                hyper_params.mutation_percent,
                hyper_params.crossover_percent,
                &hyper_params.program_parameters,
            );
            if let Some(hook) = hooks.on_post_breed.as_mut() {
                hook(&mut population, &mut hyper_params);
            }
        }

        rank_step(&mut population, &mut hyper_params);

        Ok(population)
    }
}

pub type GpHook<'a, O> = &'a mut dyn FnMut(&mut Population<O>, &mut HyperParameters<O>);

pub struct EventHooks<'a, O>
where
    O: PartialOrd + Fitness + Mutate + Generate,
{
    pub on_post_init: Option<GpHook<'a, O>>,
    pub on_post_rank: Option<GpHook<'a, O>>,
    pub on_post_selection: Option<GpHook<'a, O>>,
    pub on_post_breed: Option<GpHook<'a, O>>,
    pub on_pre_rank: Option<GpHook<'a, O>>,
}

impl<'a, O> EventHooks<'a, O>
where
    O: PartialOrd + Clone + Fitness + Mutate + Generate,
{
    pub fn with_on_post_init(self, f: GpHook<'a, O>) -> Self {
        Self {
            on_post_init: Some(f),
            ..self
        }
    }

    pub fn with_on_post_selection(self, f: GpHook<'a, O>) -> Self {
        Self {
            on_post_selection: Some(f),
            ..self
        }
    }

    pub fn with_on_post_rank(self, f: GpHook<'a, O>) -> Self {
        Self {
            on_post_rank: Some(f),
            ..self
        }
    }

    pub fn with_on_post_breed(self, f: GpHook<'a, O>) -> Self {
        Self {
            on_post_breed: Some(f),
            ..self
        }
    }
}

impl<'a, O> fmt::Debug for EventHooks<'a, O>
where
    O: PartialOrd + Fitness + Mutate + Generate,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventHooks").finish()
    }
}

impl<'a, O> Default for EventHooks<'a, O>
where
    O: PartialOrd + Clone + Fitness + Mutate + Generate,
{
    fn default() -> Self {
        Self {
            on_post_init: None,
            on_post_rank: None,
            on_post_selection: None,
            on_post_breed: None,
            on_pre_rank: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use crate::{
        core::{instruction::InstructionGeneratorParameters, program::ProgramGeneratorParameters},
        extensions::classification::ClassificationParameters,
        utils::{
            random::generator,
            test::{TestInput, TestLgp},
        },
    };
    use rand::{distributions::Standard, Rng};

    use super::{EventHooks, GeneticAlgorithm, HyperParameters};

    #[test]
    fn given_lgp_instance_with_event_hooks_when_execute_then_closures_are_executed(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let inputs = [0; 5].map(|_| generator().sample(Standard)).to_vec();
        let received = Rc::new(RefCell::new(Vec::new()));
        let mut hyper_params = HyperParameters {
            population_size: 10,
            gap: 0.5,
            mutation_percent: 0.,
            crossover_percent: 0.,
            n_generations: 1,
            lazy_evaluate: true,
            fitness_parameters: ClassificationParameters::new(inputs),
            program_parameters: ProgramGeneratorParameters::new(
                10,
                InstructionGeneratorParameters::from::<TestInput>(1),
            ),
        };

        // TODO: Add prerank.
        TestLgp::execute(
            &mut hyper_params,
            EventHooks::default()
                .with_on_post_init(&mut |_, _| {
                    received.borrow_mut().push(1);
                })
                .with_on_post_rank(&mut |_, _| {
                    received.borrow_mut().push(2);
                })
                .with_on_post_selection(&mut |_, _| {
                    received.borrow_mut().push(3);
                })
                .with_on_post_breed(&mut |_, _| {
                    received.borrow_mut().push(4);
                }),
        )?;

        pretty_assertions::assert_eq!(received.borrow().as_slice(), &[1, 2, 3, 4, 2]);

        Ok(())
    }
}
