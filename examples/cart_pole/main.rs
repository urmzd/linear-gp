use gym_rs::{envs::classical_control::cartpole::CartPoleEnv, utils::renderer::RenderMode};

use config::CartPoleInput;
use lgp::{
    core::{
        algorithm::{GeneticAlgorithm, HyperParameters},
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    },
    extensions::{
        gym_rs::ExtendedGymRsEnvironment,
        q_learning::{QConsts, QLgp, QProgramGeneratorParameters},
        reinforcement_learning::ReinforcementLearningParameters,
    },
    utils::types::VoidResultAnyError,
};

mod config;

fn main() -> VoidResultAnyError {
    let environment = CartPoleEnv::new(RenderMode::Human);
    let input = CartPoleInput::new(environment);
    let n_generations = 1;
    let n_trials = 5;
    let initial_states = CartPoleInput::get_initial_states(n_generations, n_trials);

    let hyper_params = HyperParameters {
        population_size: 1,
        gap: 0.5,
        crossover_percent: 0.5,
        mutation_percent: 0.5,
        n_generations: 1,
        lazy_evaluate: false,
        fitness_parameters: ReinforcementLearningParameters::new(initial_states, 500, input),
        program_parameters: QProgramGeneratorParameters::new(
            ProgramGeneratorParameters::new(
                32,
                InstructionGeneratorParameters::from::<CartPoleInput>(1),
            ),
            QConsts::new(0.25, 0.125, 0.05),
        ),
    };

    QLgp::execute(hyper_params).last();

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error;

    use gym_rs::{envs::classical_control::cartpole::CartPoleEnv, utils::renderer::RenderMode};

    use itertools::Itertools;
    use lgp::{
        core::{
            algorithm::{GeneticAlgorithm, HyperParameters},
            instruction::InstructionGeneratorParameters,
            program::ProgramGeneratorParameters,
        },
        extensions::{
            gym_rs::ExtendedGymRsEnvironment,
            q_learning::{QConsts, QLgp, QProgramGeneratorParameters},
            reinforcement_learning::{RLgp, ReinforcementLearningParameters},
        },
        utils::plots::plot_benchmarks,
    };

    use crate::config::CartPoleInput;

    #[test]
    fn solve_cart_pole_default() -> Result<(), Box<dyn error::Error>> {
        let environment = CartPoleEnv::new(RenderMode::None);
        let input = CartPoleInput::new(environment);
        let n_generations = 100;
        let n_trials = 5;
        let initial_states = CartPoleInput::get_initial_states(n_generations, n_trials);

        let max_episode_length = 500;

        let hyper_params = HyperParameters {
            population_size: 100,
            gap: 0.5,
            crossover_percent: 0.5,
            mutation_percent: 0.5,
            lazy_evaluate: false,
            n_generations,
            fitness_parameters: ReinforcementLearningParameters::new(
                initial_states,
                max_episode_length,
                input,
            ),
            program_parameters: ProgramGeneratorParameters::new(
                32,
                InstructionGeneratorParameters::from::<CartPoleInput>(1),
            ),
        };

        let populations = RLgp::execute(hyper_params).collect_vec();

        const PLOT_FILE_NAME: &'static str = "assets/plots/tests/cart_pole/smoke/default.png";
        let range = (0.)..(max_episode_length as f64);
        plot_benchmarks(populations, PLOT_FILE_NAME, range)?;
        Ok(())
    }

    #[test]
    fn solve_cart_pole_with_q_learning() -> Result<(), Box<dyn error::Error>> {
        let environment = CartPoleEnv::new(RenderMode::None);
        let input = CartPoleInput::new(environment);
        let n_generations = 100;
        let n_trials = 5;
        let initial_states = CartPoleInput::get_initial_states(n_generations, n_trials);

        let max_episode_length = 500;

        let hyper_params = HyperParameters {
            population_size: 100,
            gap: 0.5,
            crossover_percent: 0.5,
            mutation_percent: 0.5,
            lazy_evaluate: false,
            n_generations,
            fitness_parameters: ReinforcementLearningParameters::new(
                initial_states,
                max_episode_length,
                input,
            ),
            program_parameters: QProgramGeneratorParameters::new(
                ProgramGeneratorParameters::new(
                    16,
                    InstructionGeneratorParameters::from::<CartPoleInput>(1),
                ),
                QConsts::new(0.25, 0.5, 0.05),
            ),
        };

        let populations = QLgp::execute(hyper_params).collect_vec();

        const PLOT_FILE_NAME: &'static str = "assets/plots/tests/cart_pole/smoke/q.png";
        let range = (0.)..(max_episode_length as f64);
        plot_benchmarks(populations, PLOT_FILE_NAME, range)?;
        Ok(())
    }
}
