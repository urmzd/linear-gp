//! Gym environment experiment runners (CartPole, MountainCar)

use itertools::Itertools;

use gym_rs::envs::classical_control::cartpole::CartPoleEnv;
use gym_rs::envs::classical_control::mountain_car::MountainCarEnv;

use lgp::core::config::load_hyper_parameters;
use lgp::core::engines::core_engine::HyperParameters;
use lgp::problems::gym::{GymRsEngine, GymRsQEngine};

use crate::benchmark_tools::{save_experiment, VoidResultAnyError};

/// Run CartPole with Q-Learning
pub fn run_cart_pole_q(n_generations_override: Option<usize>) -> VoidResultAnyError {
    let name = "cart_pole_q";

    let mut parameters: HyperParameters<GymRsQEngine<CartPoleEnv>> =
        load_hyper_parameters("experiments/assets/parameters/cart-pole-q.json")?;

    if let Some(n) = n_generations_override {
        parameters.n_generations = n;
    }

    let populations = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    save_experiment(&populations, &parameters, name)?;

    Ok(())
}

/// Run CartPole with pure LGP
pub fn run_cart_pole_lgp(n_generations_override: Option<usize>) -> VoidResultAnyError {
    let name = "cart_pole_lgp";

    let mut parameters: HyperParameters<GymRsEngine<CartPoleEnv>> =
        load_hyper_parameters("experiments/assets/parameters/cart-pole-lgp.json")?;

    if let Some(n) = n_generations_override {
        parameters.n_generations = n;
    }

    let populations = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    save_experiment(&populations, &parameters, name)?;

    Ok(())
}

/// Run MountainCar with pure LGP
pub fn run_mountain_car_lgp(n_generations_override: Option<usize>) -> VoidResultAnyError {
    let name = "mountain_car_lgp";

    let mut parameters: HyperParameters<GymRsEngine<MountainCarEnv>> =
        load_hyper_parameters("experiments/assets/parameters/mountain-car-lgp.json")?;

    if let Some(n) = n_generations_override {
        parameters.n_generations = n;
    }

    let populations = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    save_experiment(&populations, &parameters, name)?;

    Ok(())
}

/// Run MountainCar with Q-Learning
pub fn run_mountain_car_q(n_generations_override: Option<usize>) -> VoidResultAnyError {
    let name = "mountain_car_q";

    let mut parameters: HyperParameters<GymRsQEngine<MountainCarEnv>> =
        load_hyper_parameters("experiments/assets/parameters/mountain-car-q.json")?;

    if let Some(n) = n_generations_override {
        parameters.n_generations = n;
    }

    let populations = parameters
        .build_engine()
        .take(parameters.n_generations)
        .collect_vec();

    save_experiment(&populations, &parameters, name)?;

    Ok(())
}
