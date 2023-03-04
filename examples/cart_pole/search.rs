use gym_rs::{envs::classical_control::cartpole::CartPoleEnv, utils::renderer::RenderMode};

use lgp::{
    core::{
        algorithm::{GeneticAlgorithm, HyperParameters},
        characteristics::Fitness,
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    },
    extensions::{
        gym_rs::ExtendedGymRsEnvironment,
        q_learning::{QConsts, QLgp, QProgramGeneratorParameters},
        reinforcement_learning::ReinforcementLearningParameters,
    },
    utils::{random::generator, types::VoidResultAnyError},
};
use tracing::field::valuable;
use tracing::info;

mod config;
use config::CartPoleInput;

fn main() -> VoidResultAnyError {
    let mut alpha_optim = tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(0., 1.)?);
    let mut gamma_optim = tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(0., 1.)?);
    let mut epsilon_optim = tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(0., 1.)?);

    let game = CartPoleEnv::new(RenderMode::None);
    let environment = CartPoleInput::new(game);

    let n_generations = 100;
    let n_trials = 5;
    let initial_states = CartPoleInput::get_initial_states(n_generations, n_trials);

    let mut best_alpha = 0.25;
    let mut best_gamma = 0.5;
    let mut best_epsilon = 0.05;
    let mut best_result = 0.;

    for _ in 0..1000 {
        let alpha = alpha_optim.ask(&mut generator())?;
        let gamma = gamma_optim.ask(&mut generator())?;
        let epsilon = epsilon_optim.ask(&mut generator())?;

        let parameters =
            ReinforcementLearningParameters::new(initial_states.clone(), 500, environment.clone());

        let hyper_params = HyperParameters {
            population_size: 10,
            gap: 0.5,
            mutation_percent: 0.5,
            crossover_percent: 0.5,
            n_generations,
            lazy_evaluate: false,
            fitness_parameters: parameters,
            program_parameters: QProgramGeneratorParameters::new(
                ProgramGeneratorParameters::new(
                    64,
                    InstructionGeneratorParameters::from::<CartPoleInput>(1),
                ),
                QConsts::new(alpha, gamma, epsilon, 0.01, 0.01),
            ),
        };

        let population = QLgp::build(hyper_params).last().unwrap();
        let result = population.best().unwrap().get_fitness().unwrap_or(0.);

        alpha_optim.tell(alpha, result)?;
        gamma_optim.tell(gamma, result)?;
        epsilon_optim.tell(epsilon, result)?;

        info!(
            fitness = valuable(&result),
            alpha = valuable(&alpha),
            gamma = valuable(&gamma),
            epsilon = valuable(&epsilon)
        );

        if result > best_result {
            best_alpha = alpha;
            best_gamma = gamma;
            best_epsilon = epsilon;
            best_result = result;
        }
    }
    info!(
        best_result = valuable(&best_result),
        best_alpha = valuable(&best_alpha),
        best_gamma = valuable(&best_gamma),
        best_epsilon = valuable(&best_epsilon)
    );

    Ok(())
}
