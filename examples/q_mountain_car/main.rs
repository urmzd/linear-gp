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
        q_learning::{QConsts, QLearningParameters, QProgramGeneratorParameters},
        reinforcement_learning::ReinforcementLearningInput,
    },
};
use rand::thread_rng;
use set_up::{MountainCarInput, QMountainCarLgp};
mod set_up;

fn main() {
    let game = MountainCarEnv::new(RenderMode::None, None);
    let mut environment = MountainCarInput::new(game);
    environment.init();
    let parameters = QLearningParameters::new(
        environment,
        200,
        [MountainCarObservation::sample_between(&mut thread_rng(), None); 5].to_vec(),
    );

    let mut hyper_params = HyperParameters {
        population_size: 2,
        gap: 0.5,
        n_mutations: 1,
        n_crossovers: 0,
        max_generations: 1000,
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
