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
    extensions::q_learning::{QConsts, QLearningParameters, QProgramGeneratorParameters},
    utils::random::generator,
};
use set_up::{MountainCarInput, QMountainCarLgp};
mod set_up;

fn main() {
    let game = MountainCarEnv::new(RenderMode::None, None);
    let environment = MountainCarInput::new(game);
    let parameters = QLearningParameters::new(
        environment,
        200,
        [MountainCarObservation::sample_between(&mut generator(), None); 5].to_vec(),
    );

    let mut hyper_params = HyperParameters {
        population_size: 10,
        gap: 0.5,
        n_mutations: 0.5,
        n_crossovers: 0.5,
        max_generations: 100,
        fitness_parameters: parameters,
        program_parameters: QProgramGeneratorParameters::new(
            ProgramGeneratorParameters::new(
                100,
                InstructionGeneratorParameters::from::<MountainCarInput>(1),
            ),
            QConsts::new(0.01, 0.01, 0.05),
        ),
    };

    let mut pops = vec![];

    let population = QMountainCarLgp::execute(
        &mut hyper_params,
        EventHooks::default().with_after_rank(&mut |population| {
            pops.push(population.clone());
            Ok(())
        }),
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
