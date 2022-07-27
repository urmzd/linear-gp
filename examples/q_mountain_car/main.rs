use gym_rs::{
    envs::classical_control::mountain_car::{MountainCarEnv, MountainCarObservation},
    utils::{custom::Sample, renderer::RenderMode},
};
use itertools::Itertools;
use lgp::{
    core::{
        algorithm::{EventHooks, GeneticAlgorithm, HyperParameters},
        characteristics::Fitness,
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    },
    extensions::{
        q_learning::{QConsts, QProgram, QProgramGeneratorParameters},
        reinforcement_learning::ReinforcementLearningParameters,
    },
    utils::{random::generator, types::VoidResultAnyError},
};
use log::debug;
use set_up::{MountainCarInput, QMountainCarLgp};
mod set_up;

fn main() -> VoidResultAnyError {
    let mut alpha_optim = tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(0., 1.)?);
    let mut gamma_optim = tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(0., 1.)?);
    let mut epsilon_optim = tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(0., 1.)?);

    let game = MountainCarEnv::new(RenderMode::None, None);
    let environment = MountainCarInput::new(game);

    let n_generations = 100;
    let initial_states = (vec![0..n_generations])
        .iter()
        .map(|_| {
            (vec![0; 5])
                .into_iter()
                .map(|_| MountainCarObservation::sample_between(&mut generator(), None))
                .collect_vec()
        })
        .collect_vec();

    let mut parameter_cycle = initial_states.iter().cycle();

    let mut best_alpha = 0.25;
    let mut best_gamma = 0.5;
    let mut best_epsilon = 0.05;
    let mut best_result = -200.;

    for _ in 0..1000 {
        let alpha = alpha_optim.ask(&mut generator())?;
        let gamma = gamma_optim.ask(&mut generator())?;
        let epsilon = epsilon_optim.ask(&mut generator())?;

        let parameters = ReinforcementLearningParameters::new(
            parameter_cycle.next().unwrap().clone(),
            200,
            environment.clone(),
        );

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
            EventHooks::default().with_on_post_fitness_params(
                &mut &mut |params: &mut ReinforcementLearningParameters<MountainCarInput>| {
                    params.update(parameter_cycle.next().unwrap().clone());
                    Ok(())
                },
            ),
        )?;
        let result = population
            .first()
            .and_then(|r| r.get_fitness())
            .unwrap_or(-200.);

        alpha_optim.tell(alpha, result)?;
        gamma_optim.tell(gamma, result)?;
        epsilon_optim.tell(epsilon, result)?;

        debug!(
            "Current - Fitness: {:.32}, Alpha: {:.32}, Gamma: {:.32}, Epsilon: {:.32}",
            result, alpha, gamma, epsilon
        );
        debug!(
            "Best So Far - Fitness: {:.32}, Alpha: {:.32}, Gamma: {:.32}, Epsilon: {:.32}",
            best_result, best_alpha, best_gamma, best_epsilon
        );
        if result > best_result {
            best_alpha = alpha;
            best_gamma = gamma;
            best_epsilon = epsilon;
            best_result = result;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use gym_rs::{
        envs::classical_control::mountain_car::{MountainCarEnv, MountainCarObservation},
        utils::{custom::Sample, renderer::RenderMode},
    };
    use itertools::Itertools;
    use lgp::{
        core::{
            algorithm::{EventHooks, GeneticAlgorithm, HyperParameters},
            instruction::InstructionGeneratorParameters,
            program::ProgramGeneratorParameters,
        },
        extensions::{
            q_learning::{QConsts, QProgram, QProgramGeneratorParameters},
            reinforcement_learning::ReinforcementLearningParameters,
        },
        utils::{plots::plot_population_benchmarks, random::generator, types::VoidResultAnyError},
    };

    use crate::set_up::{MountainCarInput, QMountainCarLgp};

    #[test]
    fn given_mountain_car_task_when_q_learning_lgp_is_used_then_task_is_solved(
    ) -> VoidResultAnyError {
        let game = MountainCarEnv::new(RenderMode::None, None);
        let environment = MountainCarInput::new(game);
        let parameters = ReinforcementLearningParameters::new(
            (vec![0; 5])
                .into_iter()
                .map(|_| MountainCarObservation::sample_between(&mut generator(), None))
                .collect_vec(),
            200,
            environment,
        );

        let mut hyper_params: HyperParameters<QProgram<MountainCarInput>> = HyperParameters {
            population_size: 100,
            gap: 0.5,
            mutation_percent: 0.5,
            crossover_percent: 0.5,
            n_generations: 100,
            lazy_evaluate: false,
            fitness_parameters: parameters,
            program_parameters: QProgramGeneratorParameters::new(
                ProgramGeneratorParameters::new(
                    32,
                    InstructionGeneratorParameters::from::<MountainCarInput>(1),
                ),
                QConsts::new(0.05, 0.125, 0.05),
            ),
        };

        let mut pops = vec![];

        QMountainCarLgp::execute(
            &mut hyper_params,
            EventHooks::default().with_on_after_rank(&mut |population| {
                pops.push(population.clone());
                Ok(())
            }),
        )?;

        const PLOT_FILE_NAME: &'static str = "./assets/tests/plots/q_mountain_car.png";
        plot_population_benchmarks(pops, PLOT_FILE_NAME, -200.0..0.0)?;
        Ok(())
    }
}
