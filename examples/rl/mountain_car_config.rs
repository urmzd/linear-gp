use derive_new::new;
use gym_rs::{core::Env, envs::classical_control::mountain_car::MountainCarEnv};
use lgp::core::inputs::ValidInput;
use lgp::extensions::gym_rs::ExtendedGymRsEnvironment;
use serde::Serialize;

#[derive(Debug, Serialize, new, Clone)]
pub struct MountainCarInput {
    environment: MountainCarEnv,
}

impl ValidInput for MountainCarInput {
    const N_INPUT_REGISTERS: usize = 2;
    const N_ACTION_REGISTERS: usize = 3;

    fn flat(&self) -> Vec<f64> {
        self.environment.state.into()
    }
}

impl ExtendedGymRsEnvironment for MountainCarInput {
    type Environment = MountainCarEnv;

    const EPISODE_LENGTH: usize = 200;

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
            environment: MountainCarEnv::new()
        }
    }
}

#[cfg(test)]
mod tests {

    use gym_rs::envs::classical_control::mountain_car::MountainCarEnv;

    use itertools::Itertools;
    use lgp::{
        core::{
            algorithm::{GeneticAlgorithm, HyperParameters},
            instruction::InstructionGeneratorParameters,
            program::ProgramGeneratorParameters,
        },
        extensions::{
            gym_rs::ExtendedGymRsEnvironment,
            interactive::{ILgp, InteractiveLearningParameters},
            q_learning::{QConsts, QLgp, QProgram, QProgramGeneratorParameters},
        },
        utils::{plots::plot_benchmarks, types::VoidResultAnyError},
    };

    use crate::config::MountainCarInput;

    #[test]
    fn given_mountain_car_example_when_lgp_executed_then_task_is_solved(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let environment = MountainCarEnv::new();
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
            fitness_parameters: InteractiveLearningParameters::new(initial_states, input),
            program_parameters: ProgramGeneratorParameters::new(
                12,
                InstructionGeneratorParameters::from::<MountainCarInput>(1),
            ),
        };

        let populations = ILgp::build(hyper_params).collect_vec();

        const PLOT_FILE_NAME: &'static str = "assets/plots/tests/mountain_car/smoke/default.png";
        plot_benchmarks(populations, PLOT_FILE_NAME, -200.0..0.0)?;
        Ok(())
    }

    #[test]
    fn given_mountain_car_task_when_q_learning_lgp_is_used_then_task_is_solved(
    ) -> VoidResultAnyError {
        let game = MountainCarEnv::new();
        let environment = MountainCarInput::new(game);
        let n_generations = 100;
        let n_trials = 5;
        let initial_states = MountainCarInput::get_initial_states(n_generations, n_trials);
        let parameters = InteractiveLearningParameters::new(initial_states, environment);

        let hyper_params: HyperParameters<QProgram<MountainCarInput>> = HyperParameters {
            population_size: 100,
            gap: 0.5,
            mutation_percent: 0.5,
            crossover_percent: 0.5,
            n_generations,
            fitness_parameters: parameters,
            program_parameters: QProgramGeneratorParameters::new(
                ProgramGeneratorParameters::new(
                    12,
                    InstructionGeneratorParameters::from::<MountainCarInput>(1),
                ),
                QConsts::default(),
            ),
        };

        let pops = QLgp::build(hyper_params).collect_vec();

        const PLOT_FILE_NAME: &'static str = "assets/plots/tests/mountain_car/smoke/q.png";
        plot_benchmarks(pops, PLOT_FILE_NAME, -200.0..0.0)?;
        Ok(())
    }
}
