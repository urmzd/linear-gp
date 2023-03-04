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
        q_learning::{QConsts, QLgp, QProgram, QProgramGeneratorParameters},
    },
    utils::{plots::plot_benchmarks, types::VoidResultAnyError},
};
mod config;
use config::CartPoleInput;

fn main() -> VoidResultAnyError {
    let environment = CartPoleEnv::new(RenderMode::None);
    let input = CartPoleInput::new(environment.clone());
    let n_generations = 100;
    let n_trials = 5;
    let initial_states = CartPoleInput::get_initial_states(n_generations, n_trials);

    let lgp_hyper_params = HyperParameters {
        population_size: 100,
        gap: 0.5,
        crossover_percent: 0.5,
        mutation_percent: 0.5,
        n_generations,
        lazy_evaluate: false,
        fitness_parameters: InteractiveLearningParameters::new(
            initial_states.clone(),
            input.clone(),
        ),
        program_parameters: ProgramGeneratorParameters::new(
            8,
            InstructionGeneratorParameters::from::<CartPoleInput>(1),
        ),
    };

    let q_params: HyperParameters<QProgram<CartPoleInput>> = HyperParameters {
        population_size: lgp_hyper_params.population_size,
        gap: lgp_hyper_params.gap,
        crossover_percent: lgp_hyper_params.crossover_percent,
        mutation_percent: lgp_hyper_params.mutation_percent,
        n_generations: lgp_hyper_params.n_generations,
        lazy_evaluate: lgp_hyper_params.lazy_evaluate,
        fitness_parameters: InteractiveLearningParameters::new(
            initial_states.clone(),
            input.clone(),
        ),
        program_parameters: QProgramGeneratorParameters::new(
            ProgramGeneratorParameters::new(
                8,
                InstructionGeneratorParameters::from::<CartPoleInput>(1),
            ),
            QConsts::default(),
        ),
    };

    let lgp_pops = ILgp::build(lgp_hyper_params).collect_vec();
    let q_pops = QLgp::build(q_params).collect_vec();

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
