//! Run command: execute experiments from config

use clap::Args;
use std::path::PathBuf;

use crate::config_discovery::find_config;
use crate::config_override::apply_overrides;
use crate::experiment_runner::run_experiment;
use lgp::core::experiment_config::ExperimentConfig;

#[derive(Args)]
pub struct RunArgs {
    /// Config name (directory in configs/)
    pub name: String,

    /// Config variant to use (filename without .toml, default: "default")
    #[arg(short = 'c', long, default_value = "default")]
    pub config: String,

    /// Override values (key=value, dot notation supported)
    #[arg(long = "override")]
    pub overrides: Vec<String>,

    /// Output base directory
    #[arg(short, long, default_value = "outputs")]
    pub output_dir: PathBuf,

    /// Preview config without running
    #[arg(long)]
    pub dry_run: bool,
}

pub fn execute(args: &RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    let discovered = find_config(&args.name, &args.config)?;
    let mut config = ExperimentConfig::load(&discovered.config_path)?;

    if !args.overrides.is_empty() {
        apply_overrides(&mut config, &args.overrides)?;
    }

    if args.dry_run {
        println!("{}", toml::to_string_pretty(&config)?);
        return Ok(());
    }

    println!("Running experiment: {}", config.name);
    println!("  Config: {}/{}.toml", args.name, args.config);
    println!("  Environment: {}", config.environment);
    println!(
        "  Mutation: {:.0}%, Crossover: {:.0}%, Q-Learning: {}",
        config.mutation_percent() * 100.0,
        config.crossover_percent() * 100.0,
        if config.has_q_learning() { "yes" } else { "no" }
    );

    let output = run_experiment(&config, &args.output_dir)?;

    println!("Experiment complete!");
    println!("  Output: {}", output.base_dir.display());

    Ok(())
}
