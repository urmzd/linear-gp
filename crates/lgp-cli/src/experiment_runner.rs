//! Experiment runner module
//!
//! Runs experiments based on configuration and produces structured output.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use chrono::Utc;
use rand::Rng;
use tracing::{debug, info, instrument};

use gymnasia::envs::classical_control::cartpole::CartPoleEnv;
use gymnasia::envs::classical_control::mountain_car::MountainCarEnv;

use lgp::core::characteristics::Save;
use lgp::core::engines::core_engine::{evaluate, Core, HyperParameters};
use lgp::core::engines::freeze_engine::Freeze;
use lgp::core::experiment_config::{ExperimentConfig, QLearningParams};
use lgp::core::instruction::InstructionGeneratorParameters;
use lgp::core::program::ProgramGeneratorParameters;
use lgp::extensions::q_learning::{QConsts, QProgramGeneratorParameters};
use lgp::problems::gym::{GymRsEngine, GymRsQEngine};
use lgp::problems::iris::IrisEngine;

/// Output structure for an experiment run.
pub struct ExperimentOutput {
    /// Base directory for all outputs: outputs/<name>/<run_id>/
    pub base_dir: PathBuf,
}

impl ExperimentOutput {
    /// Create a new experiment output structure.
    pub fn new(output_base: &Path, name: &str, run_id: &str) -> Self {
        let base_dir = output_base.join(name).join(run_id);
        Self { base_dir }
    }

    /// Get the config directory path.
    pub fn config_dir(&self) -> PathBuf {
        self.base_dir.join("config")
    }

    /// Get the outputs directory path.
    pub fn outputs_dir(&self) -> PathBuf {
        self.base_dir.join("outputs")
    }

    /// Get the post_process directory path.
    pub fn post_process_dir(&self) -> PathBuf {
        self.base_dir.join("post_process")
    }

    /// Create all output directories.
    pub fn create_dirs(&self) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(self.config_dir())?;
        std::fs::create_dir_all(self.outputs_dir())?;
        std::fs::create_dir_all(self.post_process_dir())?;
        Ok(())
    }
}

/// Run an experiment based on its configuration.
#[instrument(skip_all, fields(
    experiment = %config.name,
    environment = %config.environment,
    q_learning = config.has_q_learning()
))]
pub fn run_experiment(
    config: &ExperimentConfig,
    output_base: &Path,
) -> Result<ExperimentOutput, Box<dyn std::error::Error>> {
    // Generate or use existing seed
    let seed = config
        .hyperparameters
        .seed
        .unwrap_or_else(|| rand::thread_rng().gen());

    debug!(seed = seed, "Using seed for experiment");

    // Generate run_id from timestamp
    let run_id = Utc::now().format("%Y%m%d_%H%M%S").to_string();
    debug!(run_id = %run_id, "Generated run ID");

    // Create output structure
    let output = ExperimentOutput::new(output_base, &config.name, &run_id);
    output.create_dirs()?;
    debug!(output_dir = %output.base_dir.display(), "Created output directories");

    info!(
        environment = %config.environment,
        q_learning = config.has_q_learning(),
        "Dispatching to environment runner"
    );

    // Run based on environment and operations
    match (config.environment.as_str(), config.has_q_learning()) {
        ("CartPole" | "cart_pole", false) => run_cart_pole_lgp(config, seed, &output)?,
        ("CartPole" | "cart_pole", true) => {
            run_cart_pole_q(config, seed, &output, config.q_learning_params().unwrap())?
        }
        ("MountainCar" | "mountain_car", false) => run_mountain_car_lgp(config, seed, &output)?,
        ("MountainCar" | "mountain_car", true) => {
            run_mountain_car_q(config, seed, &output, config.q_learning_params().unwrap())?
        }
        ("Iris" | "iris", _) => run_iris(config, seed, &output)?,
        _ => return Err(format!("Unknown environment: {}", config.environment).into()),
    }

    // Save the resolved config
    let resolved_config =
        config.with_runtime_values(seed, &Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string());
    let config_path = output.config_dir().join("config.toml");
    resolved_config.save(&config_path)?;
    debug!(config_path = %config_path.display(), "Saved resolved config");

    info!("Experiment run completed successfully");

    Ok(output)
}

/// Build instruction generator parameters from config.
fn build_instruction_params(config: &ExperimentConfig) -> InstructionGeneratorParameters {
    InstructionGeneratorParameters {
        n_extras: config.hyperparameters.program.n_extras,
        external_factor: config.hyperparameters.program.external_factor,
        n_actions: config.problem.n_actions,
        n_inputs: config.problem.n_inputs,
    }
}

/// Build program generator parameters from config.
fn build_program_params(config: &ExperimentConfig) -> ProgramGeneratorParameters {
    ProgramGeneratorParameters {
        max_instructions: config.hyperparameters.program.max_instructions,
        instruction_generator_parameters: build_instruction_params(config),
    }
}

/// Run Iris classification experiment.
fn run_iris(
    config: &ExperimentConfig,
    seed: u64,
    output: &ExperimentOutput,
) -> Result<(), Box<dyn std::error::Error>> {
    let parameters: HyperParameters<IrisEngine> = HyperParameters {
        default_fitness: config.hyperparameters.default_fitness,
        population_size: config.hyperparameters.population_size,
        gap: config.hyperparameters.gap,
        mutation_percent: config.mutation_percent(),
        crossover_percent: config.crossover_percent(),
        n_generations: config.hyperparameters.n_generations,
        n_trials: config.hyperparameters.n_trials,
        seed: Some(seed),
        n_threads: config.hyperparameters.n_threads,
        program_parameters: build_program_params(config),
    };

    // Iris trials reshuffle the same 150 rows, so fresh trials would not be
    // held out. Skip evaluation until the dataset has a test split.
    run_and_save::<IrisEngine>(&parameters, output, 0)
}

/// Run CartPole with pure LGP.
fn run_cart_pole_lgp(
    config: &ExperimentConfig,
    seed: u64,
    output: &ExperimentOutput,
) -> Result<(), Box<dyn std::error::Error>> {
    let parameters: HyperParameters<GymRsEngine<CartPoleEnv>> = HyperParameters {
        default_fitness: config.hyperparameters.default_fitness,
        population_size: config.hyperparameters.population_size,
        gap: config.hyperparameters.gap,
        mutation_percent: config.mutation_percent(),
        crossover_percent: config.crossover_percent(),
        n_generations: config.hyperparameters.n_generations,
        n_trials: config.hyperparameters.n_trials,
        seed: Some(seed),
        n_threads: config.hyperparameters.n_threads,
        program_parameters: build_program_params(config),
    };

    run_and_save::<GymRsEngine<CartPoleEnv>>(
        &parameters,
        output,
        config.hyperparameters.n_eval_trials,
    )
}

/// Run CartPole with Q-Learning.
fn run_cart_pole_q(
    config: &ExperimentConfig,
    seed: u64,
    output: &ExperimentOutput,
    q_params: QLearningParams,
) -> Result<(), Box<dyn std::error::Error>> {
    let q_consts = QConsts::new(
        q_params.alpha,
        q_params.gamma,
        q_params.epsilon,
        q_params.alpha_decay,
        q_params.epsilon_decay,
    );

    let q_program_params = QProgramGeneratorParameters {
        program_parameters: build_program_params(config),
        consts: q_consts,
    };

    let parameters: HyperParameters<GymRsQEngine<CartPoleEnv>> = HyperParameters {
        default_fitness: config.hyperparameters.default_fitness,
        population_size: config.hyperparameters.population_size,
        gap: config.hyperparameters.gap,
        mutation_percent: config.mutation_percent(),
        crossover_percent: config.crossover_percent(),
        n_generations: config.hyperparameters.n_generations,
        n_trials: config.hyperparameters.n_trials,
        seed: Some(seed),
        n_threads: config.hyperparameters.n_threads,
        program_parameters: q_program_params,
    };

    run_and_save::<GymRsQEngine<CartPoleEnv>>(
        &parameters,
        output,
        config.hyperparameters.n_eval_trials,
    )
}

/// Run MountainCar with pure LGP.
fn run_mountain_car_lgp(
    config: &ExperimentConfig,
    seed: u64,
    output: &ExperimentOutput,
) -> Result<(), Box<dyn std::error::Error>> {
    let parameters: HyperParameters<GymRsEngine<MountainCarEnv>> = HyperParameters {
        default_fitness: config.hyperparameters.default_fitness,
        population_size: config.hyperparameters.population_size,
        gap: config.hyperparameters.gap,
        mutation_percent: config.mutation_percent(),
        crossover_percent: config.crossover_percent(),
        n_generations: config.hyperparameters.n_generations,
        n_trials: config.hyperparameters.n_trials,
        seed: Some(seed),
        n_threads: config.hyperparameters.n_threads,
        program_parameters: build_program_params(config),
    };

    run_and_save::<GymRsEngine<MountainCarEnv>>(
        &parameters,
        output,
        config.hyperparameters.n_eval_trials,
    )
}

/// Run MountainCar with Q-Learning.
fn run_mountain_car_q(
    config: &ExperimentConfig,
    seed: u64,
    output: &ExperimentOutput,
    q_params: QLearningParams,
) -> Result<(), Box<dyn std::error::Error>> {
    let q_consts = QConsts::new(
        q_params.alpha,
        q_params.gamma,
        q_params.epsilon,
        q_params.alpha_decay,
        q_params.epsilon_decay,
    );

    let q_program_params = QProgramGeneratorParameters {
        program_parameters: build_program_params(config),
        consts: q_consts,
    };

    let parameters: HyperParameters<GymRsQEngine<MountainCarEnv>> = HyperParameters {
        default_fitness: config.hyperparameters.default_fitness,
        population_size: config.hyperparameters.population_size,
        gap: config.hyperparameters.gap,
        mutation_percent: config.mutation_percent(),
        crossover_percent: config.crossover_percent(),
        n_generations: config.hyperparameters.n_generations,
        n_trials: config.hyperparameters.n_trials,
        seed: Some(seed),
        n_threads: config.hyperparameters.n_threads,
        program_parameters: q_program_params,
    };

    run_and_save::<GymRsQEngine<MountainCarEnv>>(
        &parameters,
        output,
        config.hyperparameters.n_eval_trials,
    )
}

/// Run the experiment and save results.
///
/// Each generation is streamed to `population.json` as soon as the engine
/// produces it, so only the current generation is held in memory. The file
/// is still a single JSON array of generations (each an array of
/// individuals), written compactly.
fn run_and_save<C>(
    parameters: &HyperParameters<C>,
    output: &ExperimentOutput,
    n_eval_trials: usize,
) -> Result<(), Box<dyn std::error::Error>>
where
    C: Core,
{
    let outputs_dir = output.outputs_dir();
    std::fs::create_dir_all(&outputs_dir)?;

    let population_path = outputs_dir.join("population.json");
    debug!(path = %population_path.display(), "Streaming population.json");
    let mut writer = BufWriter::new(File::create(&population_path)?);

    writer.write_all(b"[")?;
    let mut n_written = 0usize;
    let mut last_population = None;
    for population in parameters.build_engine().take(parameters.n_generations) {
        if n_written > 0 {
            writer.write_all(b",")?;
        }
        serde_json::to_writer(&mut writer, &population)?;
        n_written += 1;
        last_population = Some(population);
    }
    writer.write_all(b"]")?;
    writer.flush()?;
    drop(writer);

    debug!(population_count = n_written, "Saved population.json");

    let last_population = last_population.ok_or("engine produced no generations")?;
    save_summary_outputs::<C>(&last_population, parameters, &outputs_dir, n_eval_trials)
}

/// Save best/median/worst individuals of the final generation and the
/// hyperparameters used.
fn save_summary_outputs<C>(
    last_population: &[C::Individual],
    params: &HyperParameters<C>,
    outputs_dir: &Path,
    n_eval_trials: usize,
) -> Result<(), Box<dyn std::error::Error>>
where
    C: Core,
{
    let best_path = outputs_dir.join("best.json");
    let median_path = outputs_dir.join("median.json");
    let worst_path = outputs_dir.join("worst.json");
    let params_path = outputs_dir.join("params.json");

    let mut worst = last_population
        .last()
        .cloned()
        .ok_or("final generation is empty")?;
    let mut median = last_population
        .get(last_population.len() / 2)
        .cloned()
        .ok_or("final generation is empty")?;
    let mut best = last_population
        .first()
        .cloned()
        .ok_or("final generation is empty")?;

    C::Freeze::freeze(&mut worst);
    C::Freeze::freeze(&mut median);
    C::Freeze::freeze(&mut best);

    debug!("Saving worst.json");
    worst.save(worst_path.to_str().unwrap())?;

    debug!("Saving median.json");
    median.save(median_path.to_str().unwrap())?;

    debug!("Saving best.json");
    best.save(best_path.to_str().unwrap())?;

    debug!(seed = ?params.seed, "Saving params.json");
    params.save(params_path.to_str().unwrap())?;

    if n_eval_trials > 0 {
        save_evaluation::<C>(&best, &median, params, outputs_dir, n_eval_trials)?;
    }

    Ok(())
}

/// Offset that separates the evaluation seed from the training seed.
const EVAL_SEED_OFFSET: u64 = 0x9E37_79B9_7F4A_7C15;

fn score_summary(scores: &[f64]) -> serde_json::Value {
    let mut sorted = scores.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.
    };
    serde_json::json!({
        "mean": scores.iter().sum::<f64>() / n as f64,
        "median": median,
        "min": sorted[0],
        "max": sorted[n - 1],
        "scores": scores,
    })
}

/// Score the frozen best and median programs on fresh trials and save
/// `evaluation.json`. A frozen RLGP program acts greedily and stops learning.
fn save_evaluation<C>(
    best: &C::Individual,
    median: &C::Individual,
    params: &HyperParameters<C>,
    outputs_dir: &Path,
    n_eval_trials: usize,
) -> Result<(), Box<dyn std::error::Error>>
where
    C: Core,
{
    let seed = params
        .seed
        .unwrap_or_default()
        .wrapping_add(EVAL_SEED_OFFSET);
    let best_scores = evaluate::<C>(best, n_eval_trials, seed, params.default_fitness);
    let median_scores = evaluate::<C>(median, n_eval_trials, seed, params.default_fitness);

    let evaluation = serde_json::json!({
        "n_trials": n_eval_trials,
        "seed": seed.to_string(),
        "best": score_summary(&best_scores),
        "median": score_summary(&median_scores),
    });

    let path = outputs_dir.join("evaluation.json");
    debug!(path = %path.display(), "Saving evaluation.json");
    let mut writer = BufWriter::new(File::create(&path)?);
    serde_json::to_writer_pretty(&mut writer, &evaluation)?;
    writer.flush()?;
    Ok(())
}
