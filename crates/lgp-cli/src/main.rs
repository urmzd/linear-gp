//! LGP CLI - Command-line interface for Linear Genetic Programming experiments

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use lgp::utils::tracing::{init_tracing, TracingConfig, TracingFormat};
use tracing::info;

mod commands;
mod config_discovery;
mod config_override;
mod experiment_runner;

/// Output format for log messages.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum LogFormat {
    /// Human-readable, colorized output (default)
    #[default]
    Pretty,
    /// Condensed single-line output
    Compact,
    /// JSON-structured output for log aggregation
    Json,
}

impl From<LogFormat> for TracingFormat {
    fn from(format: LogFormat) -> Self {
        match format {
            LogFormat::Pretty => TracingFormat::Pretty,
            LogFormat::Compact => TracingFormat::Compact,
            LogFormat::Json => TracingFormat::Json,
        }
    }
}

#[derive(Parser)]
#[command(name = "lgp", about = "Linear Genetic Programming Framework")]
struct Cli {
    /// Enable verbose output (debug level logging)
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Set the log output format
    #[arg(long, global = true, value_enum, default_value = "pretty")]
    log_format: LogFormat,

    /// Write logs to file instead of stdout
    #[arg(long, global = true)]
    log_file: Option<PathBuf>,

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

    // Initialize tracing with CLI configuration
    let default_filter = if cli.verbose {
        "lgp=debug,lgp_cli=debug"
    } else {
        "lgp=info,lgp_cli=info"
    };

    let config = TracingConfig::new()
        .with_format(cli.log_format.into())
        .with_default_filter(default_filter)
        .with_span_events(cli.verbose);

    let config = if let Some(path) = &cli.log_file {
        config.with_log_file(path).with_stdout(false)
    } else {
        config
    };

    // Hold the guard for the program lifetime to ensure logs are flushed
    let _guard = init_tracing(config);

    info!(verbose = cli.verbose, "Starting LGP CLI");

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
