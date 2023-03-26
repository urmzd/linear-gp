use std::fmt::Error;

use clap::{FromArgMatches, Parser, Subcommand};
use gym_rs::envs::classical_control::{cartpole::CartPoleEnv, mountain_car::MountainCarEnv};
use lgp::{
    core::engines::core_engine::HyperParameters,
    problems::{
        gym::{GymRsEngine, GymRsQEngine},
        iris::IrisEngine,
    },
};

#[derive(Parser)]
enum Engine {
    MountainCarQ(HyperParameters<GymRsQEngine<MountainCarEnv, 3, 2>>),
    MountainCarLGP(HyperParameters<GymRsEngine<MountainCarEnv, 3, 2>>),
    Iris(HyperParameters<IrisEngine>),
    CartPoleQ(HyperParameters<GymRsQEngine<CartPoleEnv, 4, 2>>),
    CartPoleLGP(HyperParameters<GymRsEngine<CartPoleEnv, 4, 2>>),
}

fn main() {
    let cli = Engine::parse();
}
