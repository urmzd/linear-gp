use derive_builder::Builder;
use serde::{Deserialize, Serialize};

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
