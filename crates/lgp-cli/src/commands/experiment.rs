//! Experiment command: end-to-end pipeline (search -> run -> analyze).

use std::path::{Path, PathBuf};

use clap::Args;
use tracing::{info, warn};

use lgp::core::experiment_config::ExperimentConfig;

use crate::commands::analyze::AnalyzeArgs;
use crate::commands::search;
use crate::config_discovery::{discover_configs, get_configs_dir};
use crate::experiment_runner::run_experiment;

#[derive(Args)]
pub struct ExperimentArgs {
    /// Config to run. If not specified, runs all.
    pub config: Option<String>,

    /// Number of experiment iterations
    #[arg(short = 'n', long, default_value = "10")]
    pub iterations: usize,

    /// Skip hyperparameter search
    #[arg(long)]
    pub skip_search: bool,

    /// Skip analysis
    #[arg(long)]
    pub skip_analyze: bool,

    /// Search trials per thread
    #[arg(short = 't', long, default_value = "40")]
    pub n_trials: usize,

    /// Search threads
    #[arg(short = 'j', long, default_value = "4")]
    pub n_threads: usize,

    /// Runs for median
    #[arg(short = 'm', long, default_value = "10")]
    pub median_trials: usize,
}

fn run_single_experiment(
    config_name: &str,
    iteration: usize,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let configs_dir = get_configs_dir();
    let config_dir = configs_dir.join(config_name);

    let optimal_path = config_dir.join("optimal.toml");
    let default_path = config_dir.join("default.toml");

    let config_path = if optimal_path.exists() {
        optimal_path
    } else if default_path.exists() {
        default_path
    } else {
        return Err(format!("No config found for {}", config_name).into());
    };

    let config = ExperimentConfig::load(&config_path)?;
    info!(
        config = %config_name,
        iteration,
        path = %config_path.display(),
        "Running experiment iteration"
    );

    let output = run_experiment(&config, output_dir)?;
    info!(output_dir = %output.base_dir.display(), "Iteration complete");
    Ok(())
}

fn run_experiments(
    config_name: Option<&str>,
    iterations: usize,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let configs: Vec<String> = discover_configs()?.iter().map(|c| c.name.clone()).collect();

    let target_configs = if let Some(name) = config_name {
        if !configs.contains(&name.to_string()) {
            return Err(format!(
                "Invalid config: {}. Valid configs: {}",
                name,
                configs.join(", ")
            )
            .into());
        }
        vec![name.to_string()]
    } else {
        configs
    };

    println!(
        "Running {} iterations for {} config(s)",
        iterations,
        target_configs.len()
    );

    for cfg in &target_configs {
        println!("\n{}", "=".repeat(50));
        println!("Config: {}", cfg);

        for i in 1..=iterations {
            match run_single_experiment(cfg, i, output_dir) {
                Ok(()) => {}
                Err(e) => warn!(config = %cfg, iteration = i, error = %e, "Iteration failed"),
            }
        }

        println!("Completed {} iterations for {}", iterations, cfg);
    }

    Ok(())
}

pub fn execute(args: &ExperimentArgs) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = PathBuf::from("outputs");

    println!("Starting experiment pipeline");

    // Phase 1: Search
    if !args.skip_search {
        println!("\nPhase 1: Hyperparameter Search");
        if let Some(ref config) = args.config {
            search::search_single_config(
                config,
                args.n_trials,
                args.n_threads,
                args.median_trials,
            )?;
        } else {
            // Search all by calling execute with no config
            let search_args = search::SearchArgs {
                config: None,
                n_trials: args.n_trials,
                n_threads: args.n_threads,
                median_trials: args.median_trials,
            };
            search::execute(&search_args)?;
        }
    } else {
        println!("\nSkipping hyperparameter search");
    }

    // Phase 2: Run experiments
    println!("\nPhase 2: Running Experiments");
    run_experiments(args.config.as_deref(), args.iterations, &output_dir)?;

    // Phase 3: Analyze
    if !args.skip_analyze {
        println!("\nPhase 3: Analysis");
        let analyze_args = AnalyzeArgs {
            input: output_dir.clone(),
            output: output_dir,
        };
        crate::commands::analyze::execute(&analyze_args)?;
    } else {
        println!("\nSkipping analysis");
    }

    println!("\nPipeline complete!");
    Ok(())
}
