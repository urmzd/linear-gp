//! LGP CLI - Command-line interface for Linear Genetic Programming experiments

use clap::{Parser, Subcommand};

mod commands;
mod config_discovery;
mod config_override;
mod experiment_runner;

#[derive(Parser)]
#[command(name = "lgp", about = "Linear Genetic Programming Framework")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List available experiments
    List(commands::list::ListArgs),

    /// Run an experiment from config
    Run(commands::run::RunArgs),

    /// Run a Rust example
    Example(commands::example::ExampleArgs),
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::List(args) => commands::list::execute(&args),
        Commands::Run(args) => commands::run::execute(&args),
        Commands::Example(args) => commands::example::execute(&args),
    };
    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
