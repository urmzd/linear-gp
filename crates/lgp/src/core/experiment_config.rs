//! Experiment configuration types for TOML-based experiment definitions.
//!
//! This module provides the configuration structures for defining and running
//! LGP experiments in a reproducible, versioned manner.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::path::Path;

/// Complete experiment configuration loaded from a TOML file.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExperimentConfig {
    pub name: String,
    pub environment: String,
    pub metadata: Metadata,
    pub problem: Problem,
    pub hyperparameters: HyperParams,
    #[serde(default)]
    pub operations: Vec<Operation>,
}

/// Metadata about the experiment.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Metadata {
    pub version: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_timestamp: Option<String>,
}

/// Problem-specific configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Problem {
    pub n_inputs: usize,
    pub n_actions: usize,
}

/// Hyperparameters for the genetic algorithm.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HyperParams {
    pub population_size: usize,
    pub n_generations: usize,
    #[serde(default = "default_n_trials")]
    pub n_trials: usize,
    #[serde(default = "default_gap")]
    pub gap: f64,
    #[serde(default)]
    pub default_fitness: f64,
    /// Random seed. If None, a random seed will be generated.
    #[serde(default)]
    pub seed: Option<u64>,
    pub program: ProgramConfig,
}

fn default_n_trials() -> usize {
    1
}

fn default_gap() -> f64 {
    0.5
}

/// Program generation parameters.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProgramConfig {
    pub max_instructions: usize,
    #[serde(default = "default_n_extras")]
    pub n_extras: usize,
    #[serde(default = "default_external_factor")]
    pub external_factor: f64,
}

fn default_n_extras() -> usize {
    1
}

fn default_external_factor() -> f64 {
    10.0
}

/// An operation that can be applied to the evolutionary process.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum Operation {
    Mutation { parameters: MutationParams },
    Crossover { parameters: CrossoverParams },
    QLearning { parameters: QLearningParams },
}

/// Parameters for the mutation operation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MutationParams {
    pub percent: f64,
}

/// Parameters for the crossover operation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CrossoverParams {
    pub percent: f64,
}

/// Q-Learning specific parameters (for reinforcement learning with Q-Learning).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QLearningParams {
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
    #[serde(default = "default_alpha_decay")]
    pub alpha_decay: f64,
    #[serde(default = "default_epsilon_decay")]
    pub epsilon_decay: f64,
}

fn default_alpha() -> f64 {
    0.1
}

fn default_gamma() -> f64 {
    0.9
}

fn default_epsilon() -> f64 {
    0.05
}

fn default_alpha_decay() -> f64 {
    0.01
}

fn default_epsilon_decay() -> f64 {
    0.001
}

impl Default for QLearningParams {
    fn default() -> Self {
        Self {
            alpha: default_alpha(),
            gamma: default_gamma(),
            epsilon: default_epsilon(),
            alpha_decay: default_alpha_decay(),
            epsilon_decay: default_epsilon_decay(),
        }
    }
}

impl ExperimentConfig {
    /// Load an experiment configuration from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let content = fs::read_to_string(path)?;
        let config: ExperimentConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save the experiment configuration to a TOML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Create a copy of this config with resolved runtime values.
    pub fn with_runtime_values(&self, seed: u64, timestamp: &str) -> Self {
        let mut config = self.clone();
        config.metadata.run_timestamp = Some(timestamp.to_string());
        config.hyperparameters.seed = Some(seed);
        config
    }

    /// Extract mutation percent from operations, defaults to 0.0 if not found.
    pub fn mutation_percent(&self) -> f64 {
        self.operations
            .iter()
            .find_map(|op| match op {
                Operation::Mutation { parameters } => Some(parameters.percent),
                _ => None,
            })
            .unwrap_or(0.0)
    }

    /// Extract crossover percent from operations, defaults to 0.0 if not found.
    pub fn crossover_percent(&self) -> f64 {
        self.operations
            .iter()
            .find_map(|op| match op {
                Operation::Crossover { parameters } => Some(parameters.percent),
                _ => None,
            })
            .unwrap_or(0.0)
    }

    /// Extract Q-Learning parameters from operations if present.
    pub fn q_learning_params(&self) -> Option<QLearningParams> {
        self.operations.iter().find_map(|op| match op {
            Operation::QLearning { parameters } => Some(parameters.clone()),
            _ => None,
        })
    }

    /// Check if Q-Learning is enabled.
    pub fn has_q_learning(&self) -> bool {
        self.q_learning_params().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_baseline_config() {
        let toml_str = r#"
name = "iris_baseline"
environment = "Iris"

[metadata]
version = "1.0.0"
description = "Iris baseline - no genetic operators"

[problem]
n_inputs = 4
n_actions = 3

[hyperparameters]
population_size = 100
n_generations = 200
n_trials = 1
gap = 0.5
default_fitness = 0.0

[hyperparameters.program]
max_instructions = 100
n_extras = 1
external_factor = 10.0
"#;
        let config: ExperimentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.name, "iris_baseline");
        assert_eq!(config.environment, "Iris");
        assert_eq!(config.problem.n_inputs, 4);
        assert_eq!(config.problem.n_actions, 3);
        assert_eq!(config.hyperparameters.population_size, 100);
        assert_eq!(config.operations.len(), 0);
        assert_eq!(config.mutation_percent(), 0.0);
        assert_eq!(config.crossover_percent(), 0.0);
    }

    #[test]
    fn test_parse_mutation_only_config() {
        let toml_str = r#"
name = "iris_mutation"
environment = "Iris"

[metadata]
version = "1.0.0"

[problem]
n_inputs = 4
n_actions = 3

[hyperparameters]
population_size = 100
n_generations = 200

[hyperparameters.program]
max_instructions = 100

[[operations]]
name = "mutation"
parameters = { percent = 1.0 }
"#;
        let config: ExperimentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.name, "iris_mutation");
        assert_eq!(config.operations.len(), 1);
        assert_eq!(config.mutation_percent(), 1.0);
        assert_eq!(config.crossover_percent(), 0.0);
        assert!(!config.has_q_learning());
    }

    #[test]
    fn test_parse_full_lgp_config() {
        let toml_str = r#"
name = "cart_pole_lgp"
environment = "CartPole"

[metadata]
version = "1.0.0"
description = "CartPole with mutation and crossover"

[problem]
n_inputs = 4
n_actions = 2

[hyperparameters]
population_size = 100
n_generations = 100
n_trials = 100
gap = 0.5
default_fitness = 500.0

[hyperparameters.program]
max_instructions = 50
n_extras = 1
external_factor = 10.0

[[operations]]
name = "mutation"
parameters = { percent = 0.5 }

[[operations]]
name = "crossover"
parameters = { percent = 0.5 }
"#;
        let config: ExperimentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.name, "cart_pole_lgp");
        assert_eq!(config.environment, "CartPole");
        assert_eq!(config.operations.len(), 2);
        assert_eq!(config.mutation_percent(), 0.5);
        assert_eq!(config.crossover_percent(), 0.5);
        assert!(!config.has_q_learning());
    }

    #[test]
    fn test_parse_with_q_learning_config() {
        let toml_str = r#"
name = "cart_pole_with_q"
environment = "CartPole"

[metadata]
version = "1.0.0"
description = "CartPole with mutation, crossover, and Q-learning"

[problem]
n_inputs = 4
n_actions = 2

[hyperparameters]
population_size = 100
n_generations = 100
n_trials = 100
gap = 0.5
default_fitness = 500.0

[hyperparameters.program]
max_instructions = 50
n_extras = 1
external_factor = 10.0

[[operations]]
name = "mutation"
parameters = { percent = 0.5 }

[[operations]]
name = "crossover"
parameters = { percent = 0.5 }

[[operations]]
name = "q_learning"
parameters = { alpha = 0.1, gamma = 0.9, epsilon = 0.05, alpha_decay = 0.01, epsilon_decay = 0.001 }
"#;
        let config: ExperimentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.name, "cart_pole_with_q");
        assert!(config.has_q_learning());
        let q_params = config.q_learning_params().unwrap();
        assert_eq!(q_params.alpha, 0.1);
        assert_eq!(q_params.gamma, 0.9);
        assert_eq!(q_params.epsilon, 0.05);
    }
}
