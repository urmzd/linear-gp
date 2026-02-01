use crate::core::engines::reset_engine::{Reset, ResetEngine};
use crate::core::engines::status_engine::{Status, StatusEngine};
use crate::{
    core::engines::core_engine::HyperParameters,
    problems::{
        gym::{GymRsEngine, GymRsQEngine},
        iris::IrisEngine,
    },
};
use clap::{Parser, Subcommand, ValueEnum};
use config::{Config, Environment, File};
use gym_rs::envs::classical_control::{cartpole::CartPoleEnv, mountain_car::MountainCarEnv};
use serde::{Deserialize, Serialize};

use super::engines::core_engine::Core;
use super::instruction::InstructionGeneratorParameters;
use super::program::ProgramGeneratorParameters;
use crate::extensions::q_learning::{QConsts, QProgramGeneratorParameters};

/// Environment types supported by the framework
#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EnvironmentType {
    /// CartPole with pure Linear Genetic Programming
    CartPoleLgp,
    /// CartPole with LGP + Q-Learning
    CartPoleQ,
    /// MountainCar with pure Linear Genetic Programming
    MountainCarLgp,
    /// MountainCar with LGP + Q-Learning
    MountainCarQ,
    /// Iris classification with Linear Genetic Programming
    IrisLgp,
}

/// Experiment parameters for running LGP experiments
#[derive(Debug, Parser, Serialize, Deserialize)]
pub struct ExperimentParams {
    /// Environment to run
    #[arg(value_enum)]
    pub env: EnvironmentType,

    // === GA Parameters ===
    /// Number of individuals per generation
    #[arg(long, default_value = "100")]
    pub population_size: usize,

    /// Number of generations to evolve
    #[arg(long, default_value = "100")]
    pub n_generations: usize,

    /// Proportion of offspring created by mutation
    #[arg(long, default_value = "0.5")]
    pub mutation_percent: f64,

    /// Proportion of offspring created by crossover
    #[arg(long, default_value = "0.5")]
    pub crossover_percent: f64,

    /// Survival rate (fraction of population that survives)
    #[arg(long, default_value = "0.5")]
    pub gap: f64,

    /// Number of trial episodes for fitness evaluation
    #[arg(long, default_value = "100")]
    pub n_trials: usize,

    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,

    /// Fitness assigned to invalid programs (overridden per environment if not set)
    #[arg(long)]
    pub default_fitness: Option<f64>,

    // === Program Parameters ===
    /// Maximum instructions per program
    #[arg(long, default_value = "12")]
    pub max_instructions: usize,

    /// Number of extra working registers
    #[arg(long, default_value = "1")]
    pub n_extras: usize,

    /// Scaling factor for external inputs
    #[arg(long, default_value = "10.0")]
    pub external_factor: f64,

    // === Q-Learning Parameters (only for Q variants) ===
    /// Learning rate (Q-Learning only)
    #[arg(long, default_value = "0.1")]
    pub alpha: f64,

    /// Discount factor (Q-Learning only)
    #[arg(long, default_value = "0.9")]
    pub gamma: f64,

    /// Exploration rate (Q-Learning only)
    #[arg(long, default_value = "0.05")]
    pub epsilon: f64,

    /// Learning rate decay per trial (Q-Learning only)
    #[arg(long, default_value = "0.01")]
    pub alpha_decay: f64,

    /// Exploration rate decay per trial (Q-Learning only)
    #[arg(long, default_value = "0.001")]
    pub epsilon_decay: f64,
}

/// CLI structure for the LGP framework
#[derive(Parser)]
#[command(
    name = "lgp",
    author,
    version,
    about = "Linear Genetic Programming Framework"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Available CLI commands
#[derive(Subcommand)]
pub enum Commands {
    /// Run an experiment with the specified environment
    Experiment(ExperimentParams),
}

// Generate a macro which takes hyperparameters, builds the necessary engine and runs it,
// outputting the best score for each generation
macro_rules! run_experiment {
    ($hyperparameters:ident) => {
        for population in $hyperparameters
            .build_engine()
            .take($hyperparameters.n_generations)
        {
            println!("{}", StatusEngine::get_fitness(population.first().unwrap()));
        }
        println!("{}", serde_json::to_string(&$hyperparameters).unwrap());
    };
}

impl ExperimentParams {
    /// Get the number of inputs for the environment
    fn n_inputs(&self) -> usize {
        match self.env {
            EnvironmentType::CartPoleLgp | EnvironmentType::CartPoleQ => 4,
            EnvironmentType::MountainCarLgp | EnvironmentType::MountainCarQ => 2,
            EnvironmentType::IrisLgp => 4,
        }
    }

    /// Get the number of actions for the environment
    fn n_actions(&self) -> usize {
        match self.env {
            EnvironmentType::CartPoleLgp | EnvironmentType::CartPoleQ => 2,
            EnvironmentType::MountainCarLgp | EnvironmentType::MountainCarQ => 3,
            EnvironmentType::IrisLgp => 3,
        }
    }

    /// Get the default fitness for the environment
    fn env_default_fitness(&self) -> f64 {
        match self.env {
            EnvironmentType::CartPoleLgp | EnvironmentType::CartPoleQ => 500.0,
            EnvironmentType::MountainCarLgp | EnvironmentType::MountainCarQ => -200.0,
            EnvironmentType::IrisLgp => 0.0,
        }
    }

    /// Build instruction generator parameters
    fn build_instruction_params(&self) -> InstructionGeneratorParameters {
        InstructionGeneratorParameters {
            n_extras: self.n_extras,
            external_factor: self.external_factor,
            n_actions: self.n_actions(),
            n_inputs: self.n_inputs(),
        }
    }

    /// Build program generator parameters
    fn build_program_params(&self) -> ProgramGeneratorParameters {
        ProgramGeneratorParameters {
            max_instructions: self.max_instructions,
            instruction_generator_parameters: self.build_instruction_params(),
        }
    }

    /// Build Q-Learning constants
    fn build_q_consts(&self) -> QConsts {
        QConsts::new(
            self.alpha,
            self.gamma,
            self.epsilon,
            self.alpha_decay,
            self.epsilon_decay,
        )
    }

    /// Build Q-Program generator parameters
    fn build_q_program_params(&self) -> QProgramGeneratorParameters {
        QProgramGeneratorParameters {
            program_parameters: self.build_program_params(),
            consts: self.build_q_consts(),
        }
    }

    /// Run the experiment based on the selected environment
    pub fn run(&self) {
        let default_fitness = self
            .default_fitness
            .unwrap_or_else(|| self.env_default_fitness());

        match self.env {
            EnvironmentType::CartPoleLgp => {
                let hyperparameters: HyperParameters<GymRsEngine<CartPoleEnv>> = HyperParameters {
                    default_fitness,
                    population_size: self.population_size,
                    gap: self.gap,
                    mutation_percent: self.mutation_percent,
                    crossover_percent: self.crossover_percent,
                    n_generations: self.n_generations,
                    n_trials: self.n_trials,
                    seed: self.seed,
                    program_parameters: self.build_program_params(),
                };
                run_experiment!(hyperparameters);
            }
            EnvironmentType::CartPoleQ => {
                let mut hyperparameters: HyperParameters<GymRsQEngine<CartPoleEnv>> =
                    HyperParameters {
                        default_fitness,
                        population_size: self.population_size,
                        gap: self.gap,
                        mutation_percent: self.mutation_percent,
                        crossover_percent: self.crossover_percent,
                        n_generations: self.n_generations,
                        n_trials: self.n_trials,
                        seed: self.seed,
                        program_parameters: self.build_q_program_params(),
                    };
                ResetEngine::reset(&mut hyperparameters.program_parameters.consts);
                run_experiment!(hyperparameters);
            }
            EnvironmentType::MountainCarLgp => {
                let hyperparameters: HyperParameters<GymRsEngine<MountainCarEnv>> =
                    HyperParameters {
                        default_fitness,
                        population_size: self.population_size,
                        gap: self.gap,
                        mutation_percent: self.mutation_percent,
                        crossover_percent: self.crossover_percent,
                        n_generations: self.n_generations,
                        n_trials: self.n_trials,
                        seed: self.seed,
                        program_parameters: self.build_program_params(),
                    };
                run_experiment!(hyperparameters);
            }
            EnvironmentType::MountainCarQ => {
                let mut hyperparameters: HyperParameters<GymRsQEngine<MountainCarEnv>> =
                    HyperParameters {
                        default_fitness,
                        population_size: self.population_size,
                        gap: self.gap,
                        mutation_percent: self.mutation_percent,
                        crossover_percent: self.crossover_percent,
                        n_generations: self.n_generations,
                        n_trials: self.n_trials,
                        seed: self.seed,
                        program_parameters: self.build_q_program_params(),
                    };
                ResetEngine::reset(&mut hyperparameters.program_parameters.consts);
                run_experiment!(hyperparameters);
            }
            EnvironmentType::IrisLgp => {
                let hyperparameters: HyperParameters<IrisEngine> = HyperParameters {
                    default_fitness,
                    population_size: self.population_size,
                    gap: self.gap,
                    mutation_percent: self.mutation_percent,
                    crossover_percent: self.crossover_percent,
                    n_generations: self.n_generations,
                    n_trials: self.n_trials,
                    seed: self.seed,
                    program_parameters: self.build_program_params(),
                };
                run_experiment!(hyperparameters);
            }
        }
    }
}

pub fn load_hyper_parameters<C>(
    filename: &str,
) -> Result<HyperParameters<C>, Box<dyn std::error::Error>>
where
    C: Core,
{
    let settings = Config::builder()
        .add_source(File::with_name(filename))
        .add_source(Environment::default())
        .build()?;

    let parameters: HyperParameters<C> = settings.try_deserialize()?;
    Ok(parameters)
}
