use crate::{core::inputs::ValidInput, extensions::interactive::InteractiveLearningInput};
use gym_rs::{core::Env, envs::classical_control::mountain_car::MountainCarEnv};

#[derive(Debug, Clone)]
pub struct MountainCarInput {
    environment: MountainCarEnv,
}

impl ValidInput for MountainCarInput {
    const N_INPUTS: usize = 2;
    const N_ACTIONS: usize = 3;

    fn flat(&self) -> Vec<f64> {
        self.environment.state.into()
    }
}

impl InteractiveLearningInput for MountainCarInput {
    type Environment = MountainCarEnv;

    const MAX_EPISODE_LENGTH: usize = 200;

    fn get_env(&mut self) -> &mut Self::Environment {
        &mut self.environment
    }

    fn new() -> Self {
        Self {
            environment: MountainCarEnv::new(),
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        core::{
            algorithm::{GeneticAlgorithm, HyperParameters},
            instruction::InstructionGeneratorParameters,
            program::ProgramGeneratorParameters,
        },
        extensions::{
            interactive::{ILgp, InteractiveLearningParameters, InteractiveLearningParametersArgs},
            q_learning::{QConsts, QLgp, QProgram, QProgramGeneratorParameters},
        },
        utils::{
            benchmark_tools::{log_benchmarks, output_benchmarks, with_named_logger},
            types::VoidResultAnyError,
        },
    };
    use itertools::Itertools;

    use crate::problems::mountain_car::MountainCarInput;

    #[test]
    fn given_mountain_car_example_when_lgp_executed_then_task_is_solved(
    ) -> Result<(), Box<dyn std::error::Error>> {
        with_named_logger!("mountain-car-default", {
            let n_generations = 100;
            let n_trials = 5;

            let hyper_params = HyperParameters {
                population_size: 100,
                gap: 0.5,
                crossover_percent: 0.5,
                mutation_percent: 0.5,
                n_generations,
                fitness_parameters: InteractiveLearningParameters::<MountainCarInput>::new(
                    InteractiveLearningParametersArgs::new(n_generations, n_trials),
                ),
                program_parameters: ProgramGeneratorParameters::new(
                    12,
                    InstructionGeneratorParameters::new(1, 10.),
                ),
            };
            let populations = ILgp::build(hyper_params).collect_vec();
            output_benchmarks(&populations, NAME)?;
            log_benchmarks(&populations, NAME)?;
            Ok(())
        })
    }

    #[test]
    fn given_mountain_car_task_when_q_learning_lgp_is_used_then_task_is_solved(
    ) -> VoidResultAnyError {
        with_named_logger!("mountain-car-q", {
            let n_generations = 100;
            let n_trials = 5;

            let hyper_params: HyperParameters<QProgram<MountainCarInput>> = HyperParameters {
                population_size: 100,
                gap: 0.5,
                mutation_percent: 0.5,
                crossover_percent: 0.5,
                n_generations,
                fitness_parameters: InteractiveLearningParameters::<MountainCarInput>::new(
                    InteractiveLearningParametersArgs::new(n_generations, n_trials),
                ),
                program_parameters: QProgramGeneratorParameters::new(
                    ProgramGeneratorParameters::new(
                        12,
                        InstructionGeneratorParameters::new(1, 10.),
                    ),
                    QConsts::default(),
                ),
            };

            let populations = QLgp::build(hyper_params).collect_vec();

            output_benchmarks(&populations, NAME)?;
            log_benchmarks(&populations, NAME)?;
            Ok(())
        })
    }
}
