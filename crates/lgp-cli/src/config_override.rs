//! Config override module
//!
//! Applies command-line overrides to experiment configurations using dot notation.

use lgp::core::experiment_config::{ExperimentConfig, Operation, QLearningParams};

/// Apply command-line overrides to a configuration.
///
/// Supports dot notation for nested fields:
/// - `hyperparameters.population_size=200`
/// - `hyperparameters.program.max_instructions=50`
/// - `operations.q_learning.alpha=0.1`
/// - `name=my_experiment`
pub fn apply_overrides(
    config: &mut ExperimentConfig,
    overrides: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    for override_str in overrides {
        let parts: Vec<&str> = override_str.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(format!(
                "Invalid override format: '{}'. Expected key=value",
                override_str
            )
            .into());
        }

        let key = parts[0].trim();
        let value = parts[1].trim();

        apply_single_override(config, key, value)?;
    }

    Ok(())
}

fn apply_single_override(
    config: &mut ExperimentConfig,
    key: &str,
    value: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let path: Vec<&str> = key.split('.').collect();

    match path.as_slice() {
        // Top-level fields
        ["name"] => config.name = value.to_string(),
        ["environment"] => config.environment = value.to_string(),

        // Metadata fields
        ["metadata", "version"] => config.metadata.version = value.to_string(),
        ["metadata", "description"] => config.metadata.description = Some(value.to_string()),

        // Problem fields
        ["problem", "n_inputs"] => config.problem.n_inputs = parse_value(value, key)?,
        ["problem", "n_actions"] => config.problem.n_actions = parse_value(value, key)?,

        // Hyperparameters
        ["hyperparameters", "population_size"] => {
            config.hyperparameters.population_size = parse_value(value, key)?
        }
        ["hyperparameters", "n_generations"] => {
            config.hyperparameters.n_generations = parse_value(value, key)?
        }
        ["hyperparameters", "n_trials"] => {
            config.hyperparameters.n_trials = parse_value(value, key)?
        }
        ["hyperparameters", "gap"] => config.hyperparameters.gap = parse_value(value, key)?,
        ["hyperparameters", "default_fitness"] => {
            config.hyperparameters.default_fitness = parse_value(value, key)?
        }
        ["hyperparameters", "seed"] => config.hyperparameters.seed = Some(parse_value(value, key)?),

        // Program parameters
        ["hyperparameters", "program", "max_instructions"] => {
            config.hyperparameters.program.max_instructions = parse_value(value, key)?
        }
        ["hyperparameters", "program", "n_extras"] => {
            config.hyperparameters.program.n_extras = parse_value(value, key)?
        }
        ["hyperparameters", "program", "external_factor"] => {
            config.hyperparameters.program.external_factor = parse_value(value, key)?
        }

        // Q-Learning parameters (in operations array)
        ["operations", "q_learning", "alpha"] => update_q_learning_param(config, |p| {
            p.alpha = parse_value(value, key)?;
            Ok(())
        })?,
        ["operations", "q_learning", "gamma"] => update_q_learning_param(config, |p| {
            p.gamma = parse_value(value, key)?;
            Ok(())
        })?,
        ["operations", "q_learning", "epsilon"] => update_q_learning_param(config, |p| {
            p.epsilon = parse_value(value, key)?;
            Ok(())
        })?,
        ["operations", "q_learning", "alpha_decay"] => update_q_learning_param(config, |p| {
            p.alpha_decay = parse_value(value, key)?;
            Ok(())
        })?,
        ["operations", "q_learning", "epsilon_decay"] => update_q_learning_param(config, |p| {
            p.epsilon_decay = parse_value(value, key)?;
            Ok(())
        })?,

        _ => return Err(format!("Unknown configuration key: '{}'", key).into()),
    }

    Ok(())
}

fn update_q_learning_param<F>(
    config: &mut ExperimentConfig,
    f: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnOnce(&mut QLearningParams) -> Result<(), Box<dyn std::error::Error>>,
{
    for op in &mut config.operations {
        if let Operation::QLearning { parameters } = op {
            return f(parameters);
        }
    }
    Err("No q_learning operation found in config".into())
}

fn parse_value<T: std::str::FromStr>(
    value: &str,
    key: &str,
) -> Result<T, Box<dyn std::error::Error>>
where
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|e: T::Err| format!("Invalid value for '{}': {}", key, e).into())
}
