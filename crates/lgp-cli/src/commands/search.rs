//! Search command: random hyperparameter search.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use clap::Args;
use rand::Rng;
use rayon::prelude::*;
use tracing::{debug, info, warn};

use lgp::core::experiment_config::ExperimentConfig;

use crate::config_discovery::{discover_configs, find_config, get_configs_dir};
use crate::config_override::apply_overrides;
use crate::experiment_runner::run_experiment;

#[derive(Args)]
pub struct SearchArgs {
    /// Config to optimize. If not specified, searches all configs.
    pub config: Option<String>,

    /// Number of trials per search
    #[arg(short = 't', long, default_value = "40")]
    pub n_trials: usize,

    /// Number of parallel threads
    #[arg(short = 'j', long, default_value = "4")]
    pub n_threads: usize,

    /// Number of runs to take median from
    #[arg(short = 'm', long, default_value = "10")]
    pub median_trials: usize,
}

/// Pruning thresholds by environment prefix.
fn prune_threshold(config_name: &str) -> f64 {
    let prefix = config_name.split('_').next().unwrap_or("");
    match prefix {
        "cart" => 400.0,
        "iris" => 0.9,
        "mountain" => -150.0,
        _ => 0.0,
    }
}

fn is_q_config(name: &str) -> bool {
    name.contains("with_q") || name.ends_with("_q")
}

fn lgp_config_for_q(name: &str) -> Option<String> {
    if name.contains("with_q") {
        Some(name.replace("with_q", "lgp"))
    } else {
        None
    }
}

/// Run a single trial: sample parameters, run experiment median_trials times, return median fitness.
fn run_trial(
    config_name: &str,
    median_trials: usize,
    lgp_params: Option<&serde_json::Value>,
) -> Result<(f64, serde_json::Value), Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();

    let discovered = find_config(config_name, "default")?;
    let mut config = ExperimentConfig::load(&discovered.config_path)?;

    // Sample hyperparameters
    let mut overrides = Vec::new();

    if lgp_params.is_none() {
        // Sample LGP parameters
        let max_instructions: usize = rng.gen_range(1..=100);
        let external_factor: f64 = rng.gen_range(0.0..=100.0);
        overrides.push(format!(
            "hyperparameters.program.max_instructions={}",
            max_instructions
        ));
        overrides.push(format!(
            "hyperparameters.program.external_factor={}",
            external_factor
        ));
    } else if let Some(params) = lgp_params {
        // Use pre-found LGP parameters, sample Q-learning parameters
        if let Some(prog) = params.get("program_parameters") {
            // Handle both nested (Q-learning) and flat (LGP) formats
            let inner = prog.get("program_parameters").unwrap_or(prog);

            if let Some(mi) = inner.get("max_instructions").and_then(|v| v.as_u64()) {
                overrides.push(format!("hyperparameters.program.max_instructions={}", mi));
            }
            if let Some(igp) = inner.get("instruction_generator_parameters") {
                if let Some(ef) = igp.get("external_factor").and_then(|v| v.as_f64()) {
                    overrides.push(format!("hyperparameters.program.external_factor={}", ef));
                }
            }
        }
    }

    // For Q-learning configs, sample Q parameters
    if config.has_q_learning() && lgp_params.is_some() {
        let alpha: f64 = rng.gen_range(0.0..=1.0);
        let gamma: f64 = rng.gen_range(0.0..=1.0);
        let epsilon: f64 = rng.gen_range(0.0..=1.0);
        let alpha_decay: f64 = rng.gen_range(0.0..=1.0);
        let epsilon_decay: f64 = rng.gen_range(0.0..=1.0);
        overrides.push(format!("operations.q_learning.alpha={}", alpha));
        overrides.push(format!("operations.q_learning.gamma={}", gamma));
        overrides.push(format!("operations.q_learning.epsilon={}", epsilon));
        overrides.push(format!("operations.q_learning.alpha_decay={}", alpha_decay));
        overrides.push(format!(
            "operations.q_learning.epsilon_decay={}",
            epsilon_decay
        ));
    }

    apply_overrides(&mut config, &overrides)?;

    // Run median_trials times and collect best fitness from each
    let mut scores = Vec::with_capacity(median_trials);
    let tmp_dir = tempdir_for_search()?;

    for _ in 0..median_trials {
        let output = run_experiment(&config, &tmp_dir)?;

        // Read the best.json to get champion fitness
        let best_path = output.outputs_dir().join("best.json");
        let best_data = std::fs::read_to_string(&best_path)?;
        let best: serde_json::Value = serde_json::from_str(&best_data)?;
        let fitness = if let Some(inner) = best.get("program") {
            inner.get("fitness").and_then(|v| v.as_f64()).unwrap_or(0.0)
        } else {
            best.get("fitness").and_then(|v| v.as_f64()).unwrap_or(0.0)
        };
        scores.push(fitness);
    }

    // Clean up temp directory
    let _ = std::fs::remove_dir_all(&tmp_dir);

    // Take median score
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_score = scores[scores.len() / 2];

    // Read the params.json from the last run to save as best params
    // We rebuild the params from the config instead
    let params_json = config_to_params_json(&config)?;

    Ok((median_score, params_json))
}

fn tempdir_for_search() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let tmp = std::env::temp_dir().join(format!("lgp_search_{}", rand::thread_rng().gen::<u64>()));
    std::fs::create_dir_all(&tmp)?;
    Ok(tmp)
}

/// Build a params JSON representation from the experiment config (matches the Rust HyperParameters format).
fn config_to_params_json(
    config: &ExperimentConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let igp = serde_json::json!({
        "n_extras": config.hyperparameters.program.n_extras,
        "external_factor": config.hyperparameters.program.external_factor,
        "n_actions": config.problem.n_actions,
        "n_inputs": config.problem.n_inputs,
    });

    let program_params = serde_json::json!({
        "max_instructions": config.hyperparameters.program.max_instructions,
        "instruction_generator_parameters": igp,
    });

    if config.has_q_learning() {
        let q = config.q_learning_params().unwrap();
        Ok(serde_json::json!({
            "default_fitness": config.hyperparameters.default_fitness,
            "population_size": config.hyperparameters.population_size,
            "gap": config.hyperparameters.gap,
            "mutation_percent": config.mutation_percent(),
            "crossover_percent": config.crossover_percent(),
            "n_generations": config.hyperparameters.n_generations,
            "n_trials": config.hyperparameters.n_trials,
            "seed": config.hyperparameters.seed,
            "program_parameters": {
                "program_parameters": program_params,
                "consts": {
                    "alpha": q.alpha,
                    "gamma": q.gamma,
                    "epsilon": q.epsilon,
                    "alpha_decay": q.alpha_decay,
                    "epsilon_decay": q.epsilon_decay,
                }
            }
        }))
    } else {
        Ok(serde_json::json!({
            "default_fitness": config.hyperparameters.default_fitness,
            "population_size": config.hyperparameters.population_size,
            "gap": config.hyperparameters.gap,
            "mutation_percent": config.mutation_percent(),
            "crossover_percent": config.crossover_percent(),
            "n_generations": config.hyperparameters.n_generations,
            "n_trials": config.hyperparameters.n_trials,
            "seed": config.hyperparameters.seed,
            "program_parameters": program_params,
        }))
    }
}

/// Generate optimal.toml from search results.
fn generate_optimal_config(
    config_name: &str,
    params: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let configs_dir = get_configs_dir();
    let default_path = configs_dir.join(config_name).join("default.toml");
    let optimal_path = configs_dir.join(config_name).join("optimal.toml");

    if !default_path.exists() {
        warn!(config = %config_name, "No default.toml found, skipping optimal.toml generation");
        return Ok(());
    }

    let mut config = ExperimentConfig::load(&default_path)?;

    // Apply parameters from the search result
    if let Some(prog) = params.get("program_parameters") {
        // Handle nested Q-learning format
        let inner = prog.get("program_parameters").unwrap_or(prog);

        if let Some(mi) = inner.get("max_instructions").and_then(|v| v.as_u64()) {
            config.hyperparameters.program.max_instructions = mi as usize;
        }
        if let Some(igp) = inner.get("instruction_generator_parameters") {
            if let Some(ef) = igp.get("external_factor").and_then(|v| v.as_f64()) {
                config.hyperparameters.program.external_factor = ef;
            }
        }

        // Q-learning consts
        if let Some(consts) = prog.get("consts") {
            for op in &mut config.operations {
                if let lgp::core::experiment_config::Operation::QLearning { parameters } = op {
                    if let Some(v) = consts.get("alpha").and_then(|v| v.as_f64()) {
                        parameters.alpha = v;
                    }
                    if let Some(v) = consts.get("gamma").and_then(|v| v.as_f64()) {
                        parameters.gamma = v;
                    }
                    if let Some(v) = consts.get("epsilon").and_then(|v| v.as_f64()) {
                        parameters.epsilon = v;
                    }
                    if let Some(v) = consts.get("alpha_decay").and_then(|v| v.as_f64()) {
                        parameters.alpha_decay = v;
                    }
                    if let Some(v) = consts.get("epsilon_decay").and_then(|v| v.as_f64()) {
                        parameters.epsilon_decay = v;
                    }
                    break;
                }
            }
        }
    }

    config.save(&optimal_path)?;
    info!(path = %optimal_path.display(), "Generated optimal config");
    Ok(())
}

/// Search hyperparameters for a single config.
pub fn search_single_config(
    config_name: &str,
    n_trials: usize,
    n_threads: usize,
    median_trials: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let configs: Vec<String> = discover_configs()?.iter().map(|c| c.name.clone()).collect();
    if !configs.contains(&config_name.to_string()) {
        return Err(format!(
            "Invalid config: {}. Valid configs: {}",
            config_name,
            configs.join(", ")
        )
        .into());
    }

    println!("Starting hyperparameter search for {}", config_name);
    println!("  Trials: {}", n_trials);
    println!("  Threads: {}", n_threads);
    println!("  Median trials: {}", median_trials);

    // Load LGP params for Q-learning configs
    let lgp_params: Option<serde_json::Value> = if is_q_config(config_name) {
        if let Some(lgp_name) = lgp_config_for_q(config_name) {
            let params_path = PathBuf::from(format!("outputs/parameters/{}.json", lgp_name));
            if !params_path.exists() {
                return Err(format!(
                    "LGP parameters not found: {}. Run LGP search first before Q-learning search.",
                    params_path.display()
                )
                .into());
            }
            let data = std::fs::read_to_string(&params_path)?;
            Some(serde_json::from_str(&data)?)
        } else {
            None
        }
    } else {
        None
    };

    let threshold = prune_threshold(config_name);
    let best: Arc<Mutex<Option<(f64, serde_json::Value)>>> = Arc::new(Mutex::new(None));

    // Configure rayon thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()?;

    pool.install(|| {
        (0..n_trials).into_par_iter().for_each(|trial_idx| {
            debug!(trial = trial_idx, "Starting trial");
            match run_trial(config_name, median_trials, lgp_params.as_ref()) {
                Ok((score, params)) => {
                    if score.is_nan() {
                        debug!(trial = trial_idx, "Trial returned NaN, skipping");
                        return;
                    }
                    if score < threshold {
                        debug!(trial = trial_idx, score, threshold, "Trial pruned");
                        return;
                    }
                    let mut guard = best.lock().unwrap();
                    if guard
                        .as_ref()
                        .is_none_or(|(best_score, _)| score > *best_score)
                    {
                        info!(trial = trial_idx, score, "New best score");
                        *guard = Some((score, params));
                    }
                }
                Err(e) => {
                    warn!(trial = trial_idx, error = %e, "Trial failed");
                }
            }
        });
    });

    let best_result = best.lock().unwrap().take();
    if let Some((score, params)) = best_result {
        // Save parameters
        let params_dir = Path::new("outputs/parameters");
        std::fs::create_dir_all(params_dir)?;
        let params_path = params_dir.join(format!("{}.json", config_name));
        std::fs::write(&params_path, serde_json::to_string_pretty(&params)?)?;
        println!("Saved parameters to {}", params_path.display());

        // Generate optimal.toml
        generate_optimal_config(config_name, &params)?;

        println!("Search complete! Best score: {}", score);
    } else {
        println!("Search complete. No valid trials found.");
    }

    Ok(())
}

/// Search all configs (LGP first, then Q-learning).
fn search_all_configs(
    n_trials: usize,
    n_threads: usize,
    median_trials: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let configs: Vec<String> = discover_configs()?.iter().map(|c| c.name.clone()).collect();
    let lgp_configs: Vec<&str> = configs
        .iter()
        .filter(|c| !is_q_config(c))
        .map(|s| s.as_str())
        .collect();
    let q_configs: Vec<&str> = configs
        .iter()
        .filter(|c| is_q_config(c))
        .map(|s| s.as_str())
        .collect();

    println!("Starting hyperparameter search for all configs");

    println!("\nPhase 1: LGP configs");
    for name in &lgp_configs {
        println!("\n{}", "=".repeat(50));
        search_single_config(name, n_trials, n_threads, median_trials)?;
    }

    println!("\nPhase 2: Q-learning configs");
    for name in &q_configs {
        println!("\n{}", "=".repeat(50));
        search_single_config(name, n_trials, n_threads, median_trials)?;
    }

    println!("\nAll configs completed!");
    Ok(())
}

pub fn execute(args: &SearchArgs) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ref config) = args.config {
        search_single_config(config, args.n_trials, args.n_threads, args.median_trials)
    } else {
        search_all_configs(args.n_trials, args.n_threads, args.median_trials)
    }
}
