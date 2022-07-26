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
    utils::random::generator,
};
use set_up::{MountainCarInput, QMountainCarLgp};
mod set_up;

fn main() {
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
            QConsts::new(0.75, 0.25, 0.05),
        ),
    };

    let mut pops = vec![];

    let population = QMountainCarLgp::execute(
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
    )
    .expect("Example to have been ran successfully.");

    println!(
        "{:?}",
        population
            .into_iter()
            .map(|p| p.get_fitness())
            .collect_vec()
    );
}
