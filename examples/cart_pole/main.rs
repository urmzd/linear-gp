use gym_rs::{envs::classical_control::cartpole::CartPoleEnv, utils::renderer::RenderMode};

use lgp::{
    core::{
        algorithm::{EventHooks, GeneticAlgorithm, HyperParameters},
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    },
    extensions::{
        gym_rs::ExtendedGymRsEnvironment,
        q_learning::{QConsts, QProgramGeneratorParameters},
        reinforcement_learning::ReinforcementLearningParameters,
    },
};
use set_up::{CartPoleInput, QCartPoleLgp};

mod set_up;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let environment = CartPoleEnv::new(RenderMode::Human);
    let input = CartPoleInput::new(environment);
    let n_generations = 1;
    let n_trials = 5;
    let initial_states = CartPoleInput::get_initial_states(n_generations, n_trials);

    let mut hyper_params = HyperParameters {
        population_size: 1,
        gap: 0.5,
        crossover_percent: 0.5,
        mutation_percent: 0.5,
        n_generations: 1,
        lazy_evaluate: true,
        fitness_parameters: ReinforcementLearningParameters::new(initial_states, 500, input),
        program_parameters: QProgramGeneratorParameters::new(
            ProgramGeneratorParameters::new(
                32,
                InstructionGeneratorParameters::from::<CartPoleInput>(1),
            ),
            QConsts::new(0.25, 0.125, 0.05),
        ),
    };

    QCartPoleLgp::execute(&mut hyper_params, EventHooks::default())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error;

    use gym_rs::{envs::classical_control::cartpole::CartPoleEnv, utils::renderer::RenderMode};

    use lgp::{
        core::{
            algorithm::{EventHooks, GeneticAlgorithm, HyperParameters},
            instruction::InstructionGeneratorParameters,
            program::ProgramGeneratorParameters,
        },
        extensions::{
            gym_rs::ExtendedGymRsEnvironment,
            q_learning::{QConsts, QProgramGeneratorParameters},
            reinforcement_learning::ReinforcementLearningParameters,
        },
        utils::plots::plot_population_benchmarks,
    };

    use crate::set_up::{CartPoleInput, CartPoleLgp, QCartPoleLgp};

    #[test]
    fn given_cart_pole_when_lgp_executed_then_task_is_solved() -> Result<(), Box<dyn error::Error>>
    {
        let environment = CartPoleEnv::new(RenderMode::None);
        let input = CartPoleInput::new(environment);
        let n_generations = 100;
        let n_trials = 5;
        let initial_states = CartPoleInput::get_initial_states(n_generations, n_trials);

        let mut hyper_params = HyperParameters {
            population_size: 10,
            gap: 0.5,
            crossover_percent: 0.,
            mutation_percent: 1.,
            lazy_evaluate: true,
            n_generations,
            fitness_parameters: ReinforcementLearningParameters::new(initial_states, 500, input),
            program_parameters: ProgramGeneratorParameters::new(
                32,
                InstructionGeneratorParameters::from::<CartPoleInput>(1),
            ),
        };

        let mut populations = vec![];

        CartPoleLgp::execute(
            &mut hyper_params,
            EventHooks::default().with_on_post_rank(&mut |population, params| {
                populations.push(population.clone());
                params.fitness_parameters.next_generation();
            }),
        )?;

        const PLOT_FILE_NAME: &'static str = "plots/tests/cart_pole.png";
        let range = (0.)..(hyper_params.fitness_parameters.max_episode_length as f64);
        plot_population_benchmarks(populations, PLOT_FILE_NAME, range)?;
        Ok(())
    }

    #[test]
    fn given_cart_pole_when_lgp_with_q_learning_executed_then_task_is_solved(
    ) -> Result<(), Box<dyn error::Error>> {
        let environment = CartPoleEnv::new(RenderMode::None);
        let input = CartPoleInput::new(environment);
        let n_generations = 1;
        let n_trials = 5;
        let initial_states = CartPoleInput::get_initial_states(n_generations, n_trials);

        let mut hyper_params = HyperParameters {
            population_size: 10,
            gap: 0.5,
            crossover_percent: 0.,
            mutation_percent: 1.,
            lazy_evaluate: true,
            n_generations,
            fitness_parameters: ReinforcementLearningParameters::new(initial_states, 500, input),
            program_parameters: QProgramGeneratorParameters::new(
                ProgramGeneratorParameters::new(
                    32,
                    InstructionGeneratorParameters::from::<CartPoleInput>(1),
                ),
                QConsts::new(0.25, 0.125, 0.05),
            ),
        };

        let mut populations = vec![];

        QCartPoleLgp::execute(
            &mut hyper_params,
            EventHooks::default().with_on_post_rank(&mut |population, params| {
                populations.push(population.clone());
                params.fitness_parameters.next_generation()
            }),
        )?;

        const PLOT_FILE_NAME: &'static str = "plots/tests/q_cart_pole.png";
        let range = (0.)..(hyper_params.fitness_parameters.max_episode_length as f64);
        plot_population_benchmarks(populations, PLOT_FILE_NAME, range)?;
        Ok(())
    }
}
