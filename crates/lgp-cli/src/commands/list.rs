//! List command: show available experiments

use clap::Args;

use crate::config_discovery::discover_configs;
use lgp::core::experiment_config::ExperimentConfig;

#[derive(Args)]
pub struct ListArgs {
    /// Show detailed information including description
    #[arg(short, long)]
    pub verbose: bool,
}

pub fn execute(args: &ListArgs) -> Result<(), Box<dyn std::error::Error>> {
    let configs = discover_configs()?;

    if configs.is_empty() {
        println!("No experiments found in configs/");
        return Ok(());
    }

    println!("Available experiments:");
    for config in configs {
        if args.verbose {
            match ExperimentConfig::load(&config.config_path) {
                Ok(exp) => {
                    println!("  {} ({})", config.name, exp.environment);
                    if let Some(desc) = &exp.metadata.description {
                        println!("    {}", desc);
                    }
                }
                Err(e) => {
                    println!("  {} (error loading: {})", config.name, e);
                }
            }
        } else {
            println!("  {}", config.name);
        }
    }
    Ok(())
}
