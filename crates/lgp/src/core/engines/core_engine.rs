use std::{iter::repeat_with, sync::Arc};

use clap::{Args, Parser};
use derivative::Derivative;
use itertools::Itertools;
use rand::{seq::IteratorRandom, Rng};

use crate::{
    core::{
        engines::{breed_engine::Breed, reset_engine::Reset},
        environment::State,
    },
    utils::random::{generator, update_seed},
};

use super::{
    fitness_engine::Fitness, freeze_engine::Freeze, generate_engine::Generate,
    mutate_engine::Mutate, status_engine::Status,
};
use derive_builder::Builder;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tracing::{debug, info, instrument, trace};

#[derive(Debug, Deserialize, Serialize, Builder, Copy, Derivative, Parser)]
#[command(author, version, about, long_about=None)]
#[command(propagate_version = true)]
#[derivative(Clone)]
pub struct HyperParameters<C>
where
    C: Core,
{
    #[builder(default = "0.")]
    #[arg(long, default_value = "0.")]
    pub default_fitness: f64,
    #[builder(default = "100")]
    #[arg(long, default_value = "100")]
    pub population_size: usize,
    #[builder(default = "0.5")]
    #[arg(long, default_value = "0.5")]
    pub gap: f64,
    #[builder(default = "0.5")]
    #[arg(long, default_value = "0.5")]
    pub mutation_percent: f64,
    #[builder(default = "0.5")]
    #[arg(long, default_value = "0.5")]
    pub crossover_percent: f64,
    #[builder(default = "100")]
    #[arg(long, default_value = "100")]
    pub n_generations: usize,
    #[builder(default = "100")]
    #[arg(long, default_value = "100")]
    pub n_trials: usize,
    #[builder(default = "None")]
    #[arg(long)]
    pub seed: Option<u64>,
    #[command(flatten)]
    pub program_parameters: C::ProgramParameters,
}

pub struct CoreIter<C>
where
    C: Core,
{
    generation: usize,
    next_population: Vec<C::Individual>,
    params: HyperParameters<C>,
    trials: Vec<C::State>,
}

impl<C> CoreIter<C>
where
    C: Core,
{
    #[instrument(skip_all, fields(
        population_size = hp.population_size,
        n_generations = hp.n_generations,
        n_trials = hp.n_trials,
        gap = hp.gap,
        mutation_percent = hp.mutation_percent,
        crossover_percent = hp.crossover_percent,
        seed = ?hp.seed
    ))]
    pub fn new(hp: HyperParameters<C>) -> Self {
        debug!("Initializing evolution engine");

        let current_population = C::init_population(hp.program_parameters, hp.population_size);
        trace!(
            individuals = current_population.len(),
            "Initial population generated"
        );

        let trials: Vec<C::State> = repeat_with(|| C::Generate::generate(()))
            .take(hp.n_trials)
            .collect_vec();
        trace!(trials = trials.len(), "Trial environments generated");

        Self {
            generation: 0,
            next_population: current_population,
            params: hp,
            trials,
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

        let mut population = self.next_population.clone();

        C::eval_fitness(
            &mut population,
            &mut self.trials,
            self.params.default_fitness,
        );
        C::rank(&mut population);

        assert!(population.iter().all(C::Status::evaluated));

        let best_fitness = population.first().map(C::Status::get_fitness);
        let median_fitness = population
            .get(population.len() / 2)
            .map(C::Status::get_fitness);
        let worst_fitness = population.last().map(C::Status::get_fitness);

        info!(
            generation = self.generation,
            best_fitness = ?best_fitness,
            median_fitness = ?median_fitness,
            worst_fitness = ?worst_fitness,
            population_size = population.len(),
            "Generation complete"
        );

        debug!(
            best = serde_json::to_string(&population.first()).ok(),
            median = serde_json::to_string(&population.get(population.len() / 2)).ok(),
            worst = serde_json::to_string(&population.last()).ok(),
            "Full individual details"
        );

        let mut new_population = population.clone();

        trace!(
            before_selection = new_population.len(),
            "Starting selection"
        );
        C::survive(&mut new_population, self.params.gap);
        trace!(after_selection = new_population.len(), "Selection complete");

        trace!(
            crossover_percent = self.params.crossover_percent,
            mutation_percent = self.params.mutation_percent,
            "Starting variation"
        );
        C::variation(
            &mut new_population,
            self.params.crossover_percent,
            self.params.mutation_percent,
            self.params.program_parameters,
        );
        trace!(after_variation = new_population.len(), "Variation complete");

        self.next_population = new_population;
        self.generation += 1;

        Some(population)
    }
}

impl<T> HyperParameters<T>
where
    T: Core,
{
    pub fn build_engine(&self) -> CoreIter<T> {
        update_seed(self.seed);
        CoreIter::new(self.clone())
    }
}

pub trait Core {
    type Individual: Ord + Clone + Send + Sync + Serialize + DeserializeOwned;
    type ProgramParameters: Copy + Send + Sync + Clone + Serialize + DeserializeOwned + Args;
    type State: State;
    type FitnessMarker;
    type Generate: Generate<Self::ProgramParameters, Self::Individual> + Generate<(), Self::State>;
    type Fitness: Fitness<Self::Individual, Self::State, Self::FitnessMarker>;
    type Reset: Reset<Self::Individual> + Reset<Self::State>;
    type Breed: Breed<Self::Individual>;
    type Mutate: Mutate<Self::ProgramParameters, Self::Individual>;
    type Status: Status<Self::Individual>;
    type Freeze: Freeze<Self::Individual>;

    fn init_population(
        program_parameters: Self::ProgramParameters,
        population_size: usize,
    ) -> Vec<Self::Individual> {
        repeat_with(|| Self::Generate::generate(program_parameters))
            .take(population_size)
            .collect()
    }

    fn eval_fitness(
        population: &mut Vec<Self::Individual>,
        trials: &mut Vec<Self::State>,
        default_fitness: f64,
    ) {
        for individual in population.iter_mut() {
            let mut scores = trials
                .iter_mut()
                .map(|trial| {
                    Self::Reset::reset(individual);
                    Self::Reset::reset(trial);
                    Self::Fitness::eval_fitness(individual, trial)
                })
                .collect_vec();

            let n_trials = scores.len();
            scores = scores
                .into_iter()
                .map(|s| if !s.is_finite() { default_fitness } else { s })
                .collect_vec();
            let average = scores.into_iter().sum::<f64>() / n_trials as f64;
            Self::Status::set_fitness(individual, average);
        }
    }

    fn rank(population: &mut Vec<Self::Individual>) {
        population.sort_by(|a, b| b.cmp(a));
        debug_assert!(population.windows(2).all(|w| {
            let a = &w[0];
            let b = &w[1];

            debug_assert!(a >= b);
            a >= b
        }));
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
        debug_assert!(!population.is_empty());

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
                        let children = Self::Breed::two_point_crossover(parent_a, parent_b);
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

                    if let Some(internal_parent) = parent {
                        let mut clone = internal_parent.clone();
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

                    if let Some(internal_parent) = parent {
                        let mut clone = internal_parent.clone();
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
