use crate::core::engines::status_engine::{Status, StatusEngine};
use crate::{
    core::engines::core_engine::HyperParameters,
    problems::{
        gym::{GymRsEngine, GymRsQEngine},
        iris::IrisEngine,
    },
};
use clap::Parser;
use config::{Config, Environment, File, FileSourceFile};
use gym_rs::envs::classical_control::{cartpole::CartPoleEnv, mountain_car::MountainCarEnv};
use serde::{Deserialize, Serialize};

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
    };
}

#[derive(Parser, Deserialize, Serialize)]
pub enum Accuator {
    MountainCarQ(HyperParameters<GymRsQEngine<MountainCarEnv, 3, 2>>),
    MountainCarLGP(HyperParameters<GymRsEngine<MountainCarEnv, 3, 2>>),
    Iris(HyperParameters<IrisEngine>),
    CartPoleQ(HyperParameters<GymRsQEngine<CartPoleEnv, 4, 2>>),
    CartPoleLGP(HyperParameters<GymRsEngine<CartPoleEnv, 4, 2>>),
}

impl Accuator {
    pub fn run(&self) {
        // Use the run engine macro for each branch of the enum
        match self {
            Accuator::MountainCarQ(hyperparameters) => {
                run_accuator!(GymRsQEngine, hyperparameters)
            }
            Accuator::MountainCarLGP(hyperparameters) => {
                run_accuator!(GymRsEngine, hyperparameters)
            }
            Accuator::Iris(hyperparameters) => {
                run_accuator!(IrisEngine, hyperparameters)
            }
            Accuator::CartPoleQ(hyperparameters) => {
                run_accuator!(GymRsQEngine, hyperparameters)
            }
            Accuator::CartPoleLGP(hyperparameters) => {
                run_accuator!(GymRsEngine, hyperparameters)
            }
        }
    }
}

pub fn load_configuration(filename: &str) -> Result<Accuator, Box<dyn std::error::Error>> {
    let settings = Config::builder()
        .add_source(File::with_name(filename))
        .add_source(Environment::default())
        .build()?;

    let accuator: Accuator = settings.try_deserialize()?;
    Ok(accuator)
}
