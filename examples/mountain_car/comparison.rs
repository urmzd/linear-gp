use gym_rs::{envs::classical_control::mountain_car::MountainCarEnv, utils::renderer::RenderMode};
use lgp::{
    core::{
        algorithm::{EventHooks, GeneticAlgorithm, HyperParameters},
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    },
    extensions::{
        gym_rs::ExtendedGymRsEnvironment,
        q_learning::{QConsts, QProgram, QProgramGeneratorParameters},
        reinforcement_learning::ReinforcementLearningParameters,
    },
    utils::{plots::plot_benchmarks, types::VoidResultAnyError},
};
mod set_up;
use set_up::{MountainCarInput, MountainCarLgp, QMountainCarLgp};

fn main() -> VoidResultAnyError {
    let environment = MountainCarEnv::new(RenderMode::None, None);
    let input = MountainCarInput::new(environment.clone());
    let n_generations = 100;
    let n_trials = 5;
    let initial_states = MountainCarInput::get_initial_states(n_generations, n_trials);

    let mut lgp_hyper_params = HyperParameters {
        population_size: 100,
        gap: 0.5,
        crossover_percent: 0.5,
        mutation_percent: 0.5,
        n_generations,
        lazy_evaluate: false,
        fitness_parameters: ReinforcementLearningParameters::new(
            initial_states.clone(),
            200,
            input.clone(),
        ),
        program_parameters: ProgramGeneratorParameters::new(
            32,
            InstructionGeneratorParameters::from::<MountainCarInput>(1),
        ),
    };

    let mut q_params: HyperParameters<QProgram<MountainCarInput>> = HyperParameters {
        population_size: lgp_hyper_params.population_size,
        gap: lgp_hyper_params.gap,
        crossover_percent: lgp_hyper_params.crossover_percent,
        mutation_percent: lgp_hyper_params.mutation_percent,
        n_generations: lgp_hyper_params.n_generations,
        lazy_evaluate: lgp_hyper_params.lazy_evaluate,
        fitness_parameters: ReinforcementLearningParameters::new(
            initial_states.clone(),
            200,
            input.clone(),
        ),
        program_parameters: QProgramGeneratorParameters::new(
            ProgramGeneratorParameters::new(
                32,
                InstructionGeneratorParameters::from::<MountainCarInput>(1),
            ),
            QConsts::new(0.48, 0.25, 0.035),
        ),
    };

    let mut lgp_pops = vec![];
    let mut q_pops = vec![];

    MountainCarLgp::execute(
        &mut lgp_hyper_params,
        EventHooks::default().with_on_post_rank(&mut |population, params| {
            lgp_pops.push(population.clone());
            params.fitness_parameters.next_generation()
        }),
    )?;

    QMountainCarLgp::execute(
        &mut q_params,
        EventHooks::default().with_on_post_rank(&mut |population, params| {
            q_pops.push(population.clone());
            params.fitness_parameters.next_generation()
        }),
    )?;

    const PLOT_FILE_NAME: &'static str = "plots/comparisons/mountain_car.png";
    plot_benchmarks(lgp_pops, PLOT_FILE_NAME, -200.0..0.0)?;

    const Q_PLOT_FILE_NAME: &'static str = "plots/comparisons/q_mountain_car.png";
    plot_benchmarks(q_pops, Q_PLOT_FILE_NAME, -200.0..0.0)?;
    Ok(())
}
