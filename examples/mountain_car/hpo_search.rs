use gym_rs::{envs::classical_control::mountain_car::MountainCarEnv, utils::renderer::RenderMode};

use lgp::{
    core::{
        algorithm::{EventHooks, GeneticAlgorithm, HyperParameters},
        characteristics::Fitness,
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    },
    extensions::{
        gym_rs::ExtendedGymRsEnvironment,
        q_learning::{QConsts, QProgram, QProgramGeneratorParameters},
        reinforcement_learning::ReinforcementLearningParameters,
    },
    utils::{random::generator, types::VoidResultAnyError},
};
use log::debug;
use set_up::{MountainCarInput, QMountainCarLgp};
mod set_up;

fn main() -> VoidResultAnyError {
    pretty_env_logger::init();

    let mut alpha_optim = tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(0., 1.)?);
    let mut gamma_optim = tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(0., 1.)?);
    let mut epsilon_optim = tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(0., 1.)?);

    let game = MountainCarEnv::new(RenderMode::None, None);
    let environment = MountainCarInput::new(game);

    let n_generations = 100;
    let n_trials = 5;
    let initial_states = MountainCarInput::get_initial_states(n_generations, n_trials);

    let mut best_alpha = 0.25;
    let mut best_gamma = 0.5;
    let mut best_epsilon = 0.05;
    let mut best_result = -200.;

    for _ in 0..1000 {
        let alpha = alpha_optim.ask(&mut generator())?;
        let gamma = gamma_optim.ask(&mut generator())?;
        let epsilon = epsilon_optim.ask(&mut generator())?;

        let parameters =
            ReinforcementLearningParameters::new(initial_states.clone(), 200, environment.clone());

        let mut hyper_params: HyperParameters<QProgram<MountainCarInput>> = HyperParameters {
            population_size: 10,
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
                QConsts::new(alpha, gamma, epsilon),
            ),
        };

        let population = QMountainCarLgp::execute(
            &mut hyper_params,
            EventHooks::default().with_on_post_rank(&mut |_, params| {
                params.fitness_parameters.next_generation();
            }),
        )?;
        let result = population.first().unwrap().get_fitness().unwrap_or(-200.);

        alpha_optim.tell(alpha, result)?;
        gamma_optim.tell(gamma, result)?;
        epsilon_optim.tell(epsilon, result)?;

        debug!(
            "Fitness: {}, Alpha: {}, Gamma: {}, Epsilon: {}",
            result, alpha, gamma, epsilon
        );
        if result > best_result {
            best_alpha = alpha;
            best_gamma = gamma;
            best_epsilon = epsilon;
            best_result = result;
        }
    }
    debug!(
        "Best -- Fitness: {}, Alpha: {}, Gamma: {}, Epsilon: {}",
        best_result, best_alpha, best_gamma, best_epsilon
    );

    Ok(())
}
