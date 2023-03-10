use crate::{core::inputs::ValidInput, extensions::interactive::InteractiveLearningInput};
use gym_rs::{core::Env, envs::classical_control::cartpole::CartPoleEnv};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct CartPoleInput {
    environment: CartPoleEnv,
}

impl ValidInput for CartPoleInput {
    const N_INPUT_REGISTERS: usize = 4;
    const N_ACTION_REGISTERS: usize = 2;

    fn flat(&self) -> Vec<f64> {
        self.environment.state.into()
    }
}

impl InteractiveLearningInput for CartPoleInput {
    type Environment = CartPoleEnv;

    const MAX_EPISODE_LENGTH: usize = 500;

    fn get_env(&mut self) -> &mut Self::Environment {
        &mut self.environment
    }

    fn new() -> Self {
        Self {
            environment: CartPoleEnv::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error;

    use crate::{
        core::{
            algorithm::{GeneticAlgorithm, HyperParameters},
            instruction::InstructionGeneratorParameters,
            program::ProgramGeneratorParameters,
        },
        extensions::{
            interactive::{ILgp, InteractiveLearningInput, InteractiveLearningParameters},
            q_learning::{QConsts, QLgp, QProgramGeneratorParameters},
        },
        utils::plots::plot_benchmarks,
    };
    use itertools::Itertools;

    use super::*;

    #[test]
    fn solve_cart_pole_default() -> Result<(), Box<dyn error::Error>> {
        let input = CartPoleInput::new();
        let n_generations = 100;
        let n_trials = 5;
        let initial_states = CartPoleInput::get_initial_states(n_generations, n_trials);

        let hyper_params = HyperParameters {
            population_size: 100,
            gap: 0.5,
            crossover_percent: 0.5,
            mutation_percent: 0.5,
            n_generations,
            fitness_parameters: InteractiveLearningParameters::new(initial_states, input),
            program_parameters: ProgramGeneratorParameters::new(
                8,
                InstructionGeneratorParameters::from::<CartPoleInput>(1, 10.),
            ),
        };

        let populations = ILgp::build(hyper_params).collect_vec();

        const PLOT_FILE_NAME: &'static str = "assets/plots/tests/cart_pole/smoke/default.png";
        let range = (0.)..(CartPoleInput::MAX_EPISODE_LENGTH as f64);
        plot_benchmarks(populations, PLOT_FILE_NAME, range)?;
        Ok(())
    }

    #[test]
    fn solve_cart_pole_with_q_learning() -> Result<(), Box<dyn error::Error>> {
        let input = CartPoleInput::new();
        let n_generations = 100;
        let n_trials = 5;
        let initial_states = CartPoleInput::get_initial_states(n_generations, n_trials);

        let hyper_params = HyperParameters {
            population_size: 100,
            gap: 0.5,
            crossover_percent: 0.5,
            mutation_percent: 0.5,
            n_generations,
            fitness_parameters: InteractiveLearningParameters::new(initial_states, input),
            program_parameters: QProgramGeneratorParameters::new(
                ProgramGeneratorParameters::new(
                    8,
                    InstructionGeneratorParameters::from::<CartPoleInput>(1, 10.),
                ),
                QConsts::default(),
            ),
        };

        let populations = QLgp::build(hyper_params).collect_vec();

        const PLOT_FILE_NAME: &'static str = "assets/plots/tests/cart_pole/smoke/q.png";
        let range = (0.)..(CartPoleInput::MAX_EPISODE_LENGTH as f64);
        plot_benchmarks(populations, PLOT_FILE_NAME, range)?;
        Ok(())
    }
}