use derive_new::new;
use gym_rs::{core::Env, envs::classical_control::cartpole::CartPoleEnv};
use lgp::{core::inputs::ValidInput, extensions::gym_rs::ExtendedGymRsEnvironment};
use serde::Serialize;

#[derive(Clone, Debug, Serialize, new)]
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

impl ExtendedGymRsEnvironment for CartPoleInput {
    type Environment = CartPoleEnv;

    const EPISODE_LENGTH: usize = 500;

    fn get_state(&self) -> <Self::Environment as Env>::Observation {
        self.environment.state
    }

    fn update_state(&mut self, new_state: <Self::Environment as Env>::Observation) {
        self.environment.state = new_state;
    }

    fn get_env(&mut self) -> &mut Self::Environment {
        &mut self.environment
    }

    fn new() -> Self {
        Self {
            environment: CartPoleEnv::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error;

    use gym_rs::envs::classical_control::cartpole::CartPoleEnv;

    use itertools::Itertools;
    use lgp::{
        core::{
            algorithm::{GeneticAlgorithm, HyperParameters},
            instruction::InstructionGeneratorParameters,
            program::ProgramGeneratorParameters,
        },
        extensions::{
            gym_rs::ExtendedGymRsEnvironment,
            interactive::{ILgp, InteractiveLearningInput, InteractiveLearningParameters},
            q_learning::{QConsts, QLgp, QProgramGeneratorParameters},
        },
        utils::plots::plot_benchmarks,
    };

    use crate::config::CartPoleInput;

    #[test]
    fn solve_cart_pole_default() -> Result<(), Box<dyn error::Error>> {
        let environment = CartPoleEnv::new();
        let input = CartPoleInput::new(environment);
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
                InstructionGeneratorParameters::from::<CartPoleInput>(1),
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
        let environment = CartPoleEnv::new();
        let input = CartPoleInput::new(environment);
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
                    InstructionGeneratorParameters::from::<CartPoleInput>(1),
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
