use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use tracing::info;

use super::engines::core_engine::CoreEngine;
use super::engines::fitness_engine::Fitness;

#[derive(Debug, Clone, Deserialize, Serialize, Builder)]
pub struct HyperParameters {
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
}

// /// Defines a program capable of loading inputs from various sources.
// pub trait SupportLoad
// where
//     Self::InputType: Input + DeserializeOwned,
// {
//     type InputType;

//     /// Loads entities from a csv file found on the local file system.
//     fn load_from_csv(file_path: impl Into<PathBuf>) -> Vec<Self::InputType> {
//         let mut csv_reader = ReaderBuilder::new()
//             .has_headers(false)
//             .from_path(file_path.into())
//             .unwrap();

//         let inputs: Result<Vec<Self::InputType>, _> = csv_reader
//             .deserialize()
//             .into_iter()
//             .map(|input| input)
//             .collect();

//         inputs.unwrap()
//     }
// }

pub struct GeneticAlgorithmIter<T> {
    generation: usize,
    next_population: Option<Vec<T>>,
    params: HyperParameters,
}

impl<T> GeneticAlgorithmIter<T> {
    pub fn new(params: HyperParameters) -> Self {
        let (current_population, params) = CoreEngine::init_pop(params.clone());

        Self {
            generation: 0,
            next_population: Some(current_population),
            params,
        }
    }
}

impl<T> Iterator for GeneticAlgorithmIter<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.generation > self.params.n_generations {
            return None;
        }

        // Freeze population.
        let mut population = self.next_population.clone().unwrap();
        let mut params = self.params.clone();

        CoreEngine::eval_fitness(population, params);
        CoreEngine::rank(population, params);

        assert!(population
            .iter()
            .all(|p| !p.get_fitness().is_not_evaluated()));

        info!(
            best = serde_json::to_string(&population.best()).unwrap(),
            median = serde_json::to_string(&population.median()).unwrap(),
            worst = serde_json::to_string(&population.worst()).unwrap(),
            generation = serde_json::to_string(&self.generation).unwrap()
        );

        let new_population = population.clone();

        CoreEngine::survive();
        CoreEngine::variation(new_population, params);

        self.next_population = Some(new_population);
        self.generation += 1;

        return Some(population.clone());
    }
}
