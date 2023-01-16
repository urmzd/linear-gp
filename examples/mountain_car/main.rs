use gym_rs::{envs::classical_control::mountain_car::MountainCarEnv, utils::renderer::RenderMode};

use itertools::Itertools;
use lgp::{
    core::{
        algorithm::{GeneticAlgorithm, HyperParameters},
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    },
    extensions::{
        gym_rs::ExtendedGymRsEnvironment,
        q_learning::{QConsts, QLgp, QProgram, QProgramGeneratorParameters},
        reinforcement_learning::ReinforcementLearningParameters,
    },
    utils::types::VoidResultAnyError,
};
use config::MountainCarInput;
mod config;

fn main() -> VoidResultAnyError {
    let game = MountainCarEnv::new(RenderMode::None);
    let environment = MountainCarInput::new(game);
    let n_generations = 1;
    let n_trials = 5;
    let initial_states = MountainCarInput::get_initial_states(n_generations, n_trials);
    let parameters = ReinforcementLearningParameters::new(initial_states, 200, environment);

    let hyper_params: HyperParameters<QProgram<MountainCarInput>> = HyperParameters {
        population_size: 100,
        gap: 0.5,
        mutation_percent: 0.5,
        crossover_percent: 0.5,
        n_generations,
        lazy_evaluate: false,
        fitness_parameters: parameters,
        program_parameters: QProgramGeneratorParameters::new(
            ProgramGeneratorParameters::new(
                32,
                InstructionGeneratorParameters::from::<MountainCarInput>(1),
            ),
            QConsts::new(0.48, 0.25, 0.035),
        ),
    };

    QLgp::execute(hyper_params).collect_vec();

    Ok(())
}

#[cfg(test)]
mod tests {

    use gym_rs::{
        envs::classical_control::mountain_car::MountainCarEnv, utils::renderer::RenderMode,
    };

    use itertools::Itertools;
    use lgp::{
        core::{
            algorithm::{GeneticAlgorithm, HyperParameters},
            instruction::InstructionGeneratorParameters,
            program::ProgramGeneratorParameters,
        },
        extensions::{
            gym_rs::ExtendedGymRsEnvironment,
            q_learning::{QConsts, QLgp, QProgram, QProgramGeneratorParameters},
            reinforcement_learning::{RLgp, ReinforcementLearningParameters},
        },
        utils::{plots::plot_benchmarks, types::VoidResultAnyError},
    };

    use crate::config::MountainCarInput;

    #[test]
    fn given_mountain_car_example_when_lgp_executed_then_task_is_solved(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let environment = MountainCarEnv::new(RenderMode::None);
        let input = MountainCarInput::new(environment);
        let n_generations = 100;
        let n_trials = 5;
        let initial_states = MountainCarInput::get_initial_states(n_generations, n_trials);

        let hyper_params = HyperParameters {
            population_size: 100,
            gap: 0.5,
            crossover_percent: 0.5,
            mutation_percent: 0.5,
            n_generations,
            lazy_evaluate: false,
            fitness_parameters: ReinforcementLearningParameters::new(initial_states, 200, input),
            program_parameters: ProgramGeneratorParameters::new(
                32,
                InstructionGeneratorParameters::from::<MountainCarInput>(1),
            ),
        };

        let populations = RLgp::execute(hyper_params).collect_vec();

        const PLOT_FILE_NAME: &'static str = "assets/plots/tests/mountain_car/smoke/default.png";
        plot_benchmarks(populations, PLOT_FILE_NAME, -200.0..0.0)?;
        Ok(())
    }

    #[test]
    fn given_mountain_car_task_when_q_learning_lgp_is_used_then_task_is_solved(
    ) -> VoidResultAnyError {
        let game = MountainCarEnv::new(RenderMode::None);
        let environment = MountainCarInput::new(game);
        let n_generations = 100;
        let n_trials = 5;
        let initial_states = MountainCarInput::get_initial_states(n_generations, n_trials);
        let parameters = ReinforcementLearningParameters::new(initial_states, 200, environment);

        let hyper_params: HyperParameters<QProgram<MountainCarInput>> = HyperParameters {
            population_size: 100,
            gap: 0.5,
            mutation_percent: 0.5,
            crossover_percent: 0.5,
            n_generations,
            lazy_evaluate: false,
            fitness_parameters: parameters,
            program_parameters: QProgramGeneratorParameters::new(
                ProgramGeneratorParameters::new(
                    16,
                    InstructionGeneratorParameters::from::<MountainCarInput>(1),
                ),
                QConsts::new(0.48, 0.25, 0.035),
            ),
        };

        let pops = QLgp::execute(hyper_params).collect_vec();

        const PLOT_FILE_NAME: &'static str = "assets/plots/tests/mountain_car/smoke/q.png";
        plot_benchmarks(pops, PLOT_FILE_NAME, -200.0..0.0)?;
        Ok(())
    }
}
