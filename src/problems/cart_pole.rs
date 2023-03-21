use crate::{core::inputs::ValidInput, extensions::interactive::InteractiveLearningInput};
use gym_rs::{core::Env, envs::classical_control::cartpole::CartPoleEnv};

#[derive(Clone, Debug)]
pub struct CartPoleInput {
    environment: CartPoleEnv,
}

impl ValidInput for CartPoleInput {
    const N_INPUTS: usize = 4;
    const N_ACTIONS: usize = 2;

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
            interactive::{ILgp, InteractiveLearningParameters, InteractiveLearningParametersArgs},
            q_learning::{QConsts, QProgram, QProgramGeneratorParameters},
        },
        utils::benchmark_tools::{log_benchmarks, output_benchmarks, with_named_logger},
    };
    use itertools::Itertools;

    use super::*;

    #[test]
    fn solve_cart_pole_default() -> Result<(), Box<dyn error::Error>> {
        with_named_logger!("cart-pole-default", {
            let n_generations = 100;
            let n_trials = 5;

            let hyper_params = HyperParameters {
                population_size: 100,
                gap: 0.5,
                crossover_percent: 0.5,
                mutation_percent: 0.5,
                n_generations,
                fitness_parameters: InteractiveLearningParameters::<CartPoleInput>::new(
                    InteractiveLearningParametersArgs::new(n_generations, n_trials),
                ),
                program_parameters: ProgramGeneratorParameters::new(
                    8,
                    InstructionGeneratorParameters::new(1, 10.),
                ),
            };

            let populations = ILgp::<CartPoleInput>::build(hyper_params).collect_vec();

            output_benchmarks(&populations, NAME)?;
            log_benchmarks(&populations, NAME)?;
            Ok(())
        })
    }

    #[test]
    fn solve_cart_pole_with_q_learning() -> Result<(), Box<dyn error::Error>> {
        with_named_logger!("cart-pole-q", {
            let n_generations = 100;
            let n_trials = 5;

            let hyper_params = HyperParameters {
                population_size: 100,
                gap: 0.5,
                crossover_percent: 0.5,
                mutation_percent: 0.5,
                n_generations,
                fitness_parameters: InteractiveLearningParameters::<CartPoleInput>::new(
                    InteractiveLearningParametersArgs::new(n_generations, n_trials),
                ),
                program_parameters: QProgramGeneratorParameters::new(
                    ProgramGeneratorParameters::new(8, InstructionGeneratorParameters::new(1, 10.)),
                    QConsts::default(),
                ),
            };

            let populations = ILgp::<QProgram<CartPoleInput>>::build(hyper_params).collect_vec();

            output_benchmarks(&populations, NAME)?;
            log_benchmarks(&populations, NAME)?;
            Ok(())
        })
    }
}
