use crate::core::engines::reset_engine::{Reset, ResetEngine};
use crate::core::engines::status_engine::{Status, StatusEngine};
use crate::{
    core::engines::core_engine::HyperParameters,
    problems::{
        gym::{GymRsEngine, GymRsQEngine},
        iris::IrisEngine,
    },
};
use clap::Parser;
use config::{Config, Environment, File};
use gym_rs::envs::classical_control::{cartpole::CartPoleEnv, mountain_car::MountainCarEnv};
use serde::{Deserialize, Serialize};

use super::engines::core_engine::Core;

// Generate a macro which takes hyperparameters, builds the necessary engine and run its
// outputting the best score for each generation
macro_rules! run_accuator {
    ($engine:ident, $hyperparameters:ident) => {
        for population in $hyperparameters
            .build_engine()
            .take($hyperparameters.population_size)
        {
            println!("{}", StatusEngine::get_fitness(population.first().unwrap()));
        }
        println!("{}", serde_json::to_string(&$hyperparameters).unwrap());
    };
}

#[derive(Parser, Deserialize, Serialize)]
pub enum Accuator {
    MountainCarQ(HyperParameters<GymRsQEngine<MountainCarEnv>>),
    MountainCarLGP(HyperParameters<GymRsEngine<MountainCarEnv>>),
    CartPoleQ(HyperParameters<GymRsQEngine<CartPoleEnv>>),
    CartPoleLGP(HyperParameters<GymRsEngine<CartPoleEnv>>),
    IrisLgp(HyperParameters<IrisEngine>),
}

impl Accuator {
    pub fn run(&mut self) {
        // Use the run engine macro for each branch of the enum
        match self {
            Accuator::MountainCarQ(hyperparameters) => {
                ResetEngine::reset(&mut hyperparameters.program_parameters.consts);

                hyperparameters
                    .program_parameters
                    .program_parameters
                    .instruction_generator_parameters
                    .n_actions = 3;
                hyperparameters
                    .program_parameters
                    .program_parameters
                    .instruction_generator_parameters
                    .n_inputs = 2;
                hyperparameters.default_fitness = -200.0;

                run_accuator!(GymRsQEngine, hyperparameters);
            }
            Accuator::MountainCarLGP(hyperparameters) => {
                hyperparameters
                    .program_parameters
                    .instruction_generator_parameters
                    .n_actions = 3;
                hyperparameters
                    .program_parameters
                    .instruction_generator_parameters
                    .n_inputs = 2;
                hyperparameters.default_fitness = -200.0;

                run_accuator!(GymRsEngine, hyperparameters);
            }
            Accuator::IrisLgp(hyperparameters) => {
                hyperparameters
                    .program_parameters
                    .instruction_generator_parameters
                    .n_actions = 3;
                hyperparameters
                    .program_parameters
                    .instruction_generator_parameters
                    .n_inputs = 4;

                run_accuator!(IrisEngine, hyperparameters);
            }
            Accuator::CartPoleQ(hyperparameters) => {
                ResetEngine::reset(&mut hyperparameters.program_parameters.consts);
                hyperparameters
                    .program_parameters
                    .program_parameters
                    .instruction_generator_parameters
                    .n_actions = 2;
                hyperparameters
                    .program_parameters
                    .program_parameters
                    .instruction_generator_parameters
                    .n_inputs = 4;
                hyperparameters.default_fitness = 500.0;

                run_accuator!(GymRsQEngine, hyperparameters);
            }
            Accuator::CartPoleLGP(hyperparameters) => {
                hyperparameters
                    .program_parameters
                    .instruction_generator_parameters
                    .n_actions = 2;
                hyperparameters
                    .program_parameters
                    .instruction_generator_parameters
                    .n_inputs = 4;
                hyperparameters.default_fitness = 500.0;

                run_accuator!(GymRsEngine, hyperparameters);
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
