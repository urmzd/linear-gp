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
    utils::{random::generator, types::VoidResultAnyError},
};
use set_up::{MountainCarInput, QMountainCarLgp};
mod set_up;

fn main() -> VoidResultAnyError {
    let game = MountainCarEnv::new(RenderMode::Human, None);
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
        population_size: 1,
        gap: 0.,
        mutation_percent: 0.,
        crossover_percent: 0.,
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
        EventHooks::default()
            .with_on_after_rank(&mut |population| {
                pops.push(population.clone());
                Ok(())
            })
            .with_on_post_fitness_params(
                &mut &mut |params: &mut ReinforcementLearningParameters<MountainCarInput>| {
                    params.update(
                        (vec![0; 5])
                            .into_iter()
                            .map(|_| MountainCarObservation::sample_between(&mut generator(), None))
                            .collect_vec(),
                    );
                    Ok(())
                },
            ),
    )?;

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
            EventHooks::default()
                .with_on_after_rank(&mut |population| {
                    pops.push(population.clone());
                    Ok(())
                })
                .with_on_post_fitness_params(
                    &mut &mut |params: &mut ReinforcementLearningParameters<MountainCarInput>| {
                        params.update(
                            (vec![0; 5])
                                .into_iter()
                                .map(|_| {
                                    MountainCarObservation::sample_between(&mut generator(), None)
                                })
                                .collect_vec(),
                        );
                        Ok(())
                    },
                ),
        )?;

        const PLOT_FILE_NAME: &'static str = "./assets/tests/plots/q_mountain_car.png";
        plot_population_benchmarks(pops, PLOT_FILE_NAME, -200.0..0.0)?;
        Ok(())
    }
}
