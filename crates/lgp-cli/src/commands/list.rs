//! List command: show available experiments

use clap::Args;

use crate::config_discovery::discover_configs;
use crate::ui;
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
        ui::warn("No experiments found in configs/");
        return Ok(());
    }

    ui::header("Available experiments");
    for config in configs {
        if args.verbose {
            match ExperimentConfig::load(&config.config_path) {
                Ok(exp) => {
                    ui::line(&format!("{} ({})", config.name, exp.environment));
                    if let Some(desc) = &exp.metadata.description {
                        ui::info(desc);
                    }
                }
                Err(e) => {
                    ui::warn(&format!("{} (error loading: {})", config.name, e));
                }
            }
        } else {
            ui::line(&config.name);
        }
    }
    Ok(())
}
