//! List command: show available experiments

use clap::Args;

use crate::config_discovery::discover_configs;
use crate::ui;
use crate::OutputFormat;
use lgp::core::experiment_config::ExperimentConfig;

#[derive(Args)]
pub struct ListArgs {
    /// Show detailed information including description
    #[arg(short, long)]
    pub verbose: bool,
}

pub fn execute(args: &ListArgs, format: OutputFormat) -> Result<(), Box<dyn std::error::Error>> {
    let configs = discover_configs()?;

    if configs.is_empty() {
        ui::warn("No experiments found in configs/");
        return Ok(());
    }

    if format == OutputFormat::Json {
        let names: Vec<&str> = configs.iter().map(|c| c.name.as_str()).collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({"experiments": names}))?
        );
    } else {
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
    }
    Ok(())
}
