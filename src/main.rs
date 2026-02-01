use clap::Parser;
use lgp::core::config::{Cli, Commands};

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Experiment(params) => params.run(),
    }
}
