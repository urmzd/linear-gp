use gym_rs::{envs::classical_control::cartpole::CartPoleEnv, utils::renderer::RenderMode};
use itertools::Itertools;
use lgp::{
    core::{
        algorithm::{GeneticAlgorithm, HyperParameters},
        instruction::InstructionGeneratorParameters,
        program::{Program, ProgramGeneratorParameters},
    },
    extensions::{
        gym_rs::ExtendedGymRsEnvironment,
        interactive::{ILgp, InteractiveLearningInput, InteractiveLearningParameters},
        q_learning::{QConsts, QLgp, QProgram, QProgramGeneratorParameters},
    },
    utils::{plots::plot_benchmarks, types::VoidResultAnyError},
};
mod config;
use config::CartPoleInput;

fn main() -> VoidResultAnyError {
    let environment = CartPoleEnv::new(RenderMode::None);
    let input = CartPoleInput::new(environment);
    let n_generations = 100;
    let n_trials = 5;
    let initial_states = CartPoleInput::get_initial_states(n_generations, n_trials);
    let fitness_parameters = InteractiveLearningParameters::new(initial_states, input);
    let program_parameters = ProgramGeneratorParameters::new(
        8,
        InstructionGeneratorParameters::from::<CartPoleInput>(1),
    );

    let lgp_hp: HyperParameters<Program<InteractiveLearningParameters<CartPoleInput>>> =
        HyperParameters {
            population_size: 100,
            gap: 0.5,
            crossover_percent: 0.5,
            mutation_percent: 0.5,
            n_generations,
            fitness_parameters,
            program_parameters,
        };

    let lgpq_hp: HyperParameters<QProgram<CartPoleInput>> = HyperParameters {
        population_size: lgp_hp.population_size,
        gap: lgp_hp.gap,
        crossover_percent: lgp_hp.crossover_percent,
        mutation_percent: lgp_hp.mutation_percent,
        n_generations: lgp_hp.n_generations,
        fitness_parameters: lgp_hp.fitness_parameters.clone(),
        program_parameters: QProgramGeneratorParameters::new(
            lgp_hp.program_parameters.clone(),
            QConsts::default(),
        ),
    };

    let lgp_pops = ILgp::build(lgp_hp).collect_vec();
    let q_pops = QLgp::build(lgpq_hp).collect_vec();

    const PLOT_FILE_NAME: &'static str = "assets/plots/examples/cart_pole/default.png";
    const Q_PLOT_FILE_NAME: &'static str = "assets/plots/examples/cart_pole/q.png";

    plot_benchmarks(
        lgp_pops,
        PLOT_FILE_NAME,
        0.0..(CartPoleInput::MAX_EPISODE_LENGTH as f64),
    )?;
    plot_benchmarks(
        q_pops,
        Q_PLOT_FILE_NAME,
        0.0..(CartPoleInput::MAX_EPISODE_LENGTH as f64),
    )?;

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
            interactive::{ILgp, InteractiveLearningInput, InteractiveLearningParameters},
            q_learning::{QConsts, QLgp, QProgramGeneratorParameters},
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
        let environment = CartPoleEnv::new(RenderMode::None);
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
