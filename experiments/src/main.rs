use clap::{Parser, Subcommand, ValueEnum};
use lgp_experiments::runners::{gym, iris};

/// LGP Experiments CLI - Run thesis experiments
#[derive(Parser)]
#[command(name = "lgp-exp", author, version, about = "LGP Experiment Runner")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a specific experiment
    Run {
        /// Experiment to run
        #[arg(value_enum)]
        experiment: Experiment,

        /// Number of generations to run (overrides config)
        #[arg(long)]
        n_generations: Option<usize>,

        /// Output prefix for saving results
        #[arg(long, default_value = "experiments/assets/output")]
        output_prefix: String,
    },
    /// Run batch experiments
    Batch {
        /// Experiments to run (comma-separated or 'all')
        #[arg(long, default_value = "all")]
        experiments: String,

        /// Output prefix for saving results
        #[arg(long, default_value = "experiments/assets/output")]
        output_prefix: String,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Experiment {
    /// Iris baseline (no mutation, no crossover)
    IrisBaseline,
    /// Iris with mutation only
    IrisMutation,
    /// Iris with crossover only
    IrisCrossover,
    /// Iris full (mutation + crossover)
    IrisFull,
    /// CartPole with Q-Learning
    CartPoleQ,
    /// CartPole with pure LGP
    CartPoleLgp,
    /// MountainCar with Q-Learning
    MountainCarQ,
    /// MountainCar with pure LGP
    MountainCarLgp,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            experiment,
            n_generations,
            output_prefix,
        } => {
            std::env::set_var("BENCHMARK_PREFIX", &output_prefix);

            let result = match experiment {
                Experiment::IrisBaseline => iris::run_baseline(n_generations),
                Experiment::IrisMutation => iris::run_mutation(n_generations),
                Experiment::IrisCrossover => iris::run_crossover(n_generations),
                Experiment::IrisFull => iris::run_full(n_generations),
                Experiment::CartPoleQ => gym::run_cart_pole_q(n_generations),
                Experiment::CartPoleLgp => gym::run_cart_pole_lgp(n_generations),
                Experiment::MountainCarQ => gym::run_mountain_car_q(n_generations),
                Experiment::MountainCarLgp => gym::run_mountain_car_lgp(n_generations),
            };

            if let Err(e) = result {
                eprintln!("Experiment failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Batch {
            experiments,
            output_prefix,
        } => {
            std::env::set_var("BENCHMARK_PREFIX", &output_prefix);

            let all_experiments = vec![
                Experiment::IrisBaseline,
                Experiment::IrisMutation,
                Experiment::IrisCrossover,
                Experiment::IrisFull,
                Experiment::CartPoleQ,
                Experiment::CartPoleLgp,
                Experiment::MountainCarQ,
                Experiment::MountainCarLgp,
            ];

            let to_run: Vec<Experiment> = if experiments == "all" {
                all_experiments
            } else {
                experiments
                    .split(',')
                    .filter_map(|s| match s.trim().to_lowercase().as_str() {
                        "iris-baseline" => Some(Experiment::IrisBaseline),
                        "iris-mutation" => Some(Experiment::IrisMutation),
                        "iris-crossover" => Some(Experiment::IrisCrossover),
                        "iris-full" => Some(Experiment::IrisFull),
                        "cart-pole-q" => Some(Experiment::CartPoleQ),
                        "cart-pole-lgp" => Some(Experiment::CartPoleLgp),
                        "mountain-car-q" => Some(Experiment::MountainCarQ),
                        "mountain-car-lgp" => Some(Experiment::MountainCarLgp),
                        _ => {
                            eprintln!("Unknown experiment: {}", s);
                            None
                        }
                    })
                    .collect()
            };

            for experiment in to_run {
                println!("Running {:?}...", experiment);
                let result = match experiment {
                    Experiment::IrisBaseline => iris::run_baseline(None),
                    Experiment::IrisMutation => iris::run_mutation(None),
                    Experiment::IrisCrossover => iris::run_crossover(None),
                    Experiment::IrisFull => iris::run_full(None),
                    Experiment::CartPoleQ => gym::run_cart_pole_q(None),
                    Experiment::CartPoleLgp => gym::run_cart_pole_lgp(None),
                    Experiment::MountainCarQ => gym::run_mountain_car_q(None),
                    Experiment::MountainCarLgp => gym::run_mountain_car_lgp(None),
                };

                if let Err(e) = result {
                    eprintln!("Experiment {:?} failed: {}", experiment, e);
                }
            }
        }
    }
}
