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
        q_learning::{QConsts, QProgramGeneratorParameters},
        reinforcement_learning::{ReinforcementLearningInput, ReinforcementLearningParameters},
    },
    utils::random::generator,
};
use set_up::{MountainCarInput, QMountainCarLgp};
mod set_up;

fn main() {
    let game = MountainCarEnv::new(RenderMode::None, None);
    let mut environment = MountainCarInput::new(game);
    environment.init();
    let parameters = ReinforcementLearningParameters::new(
        (vec![0; 5])
            .into_iter()
            .map(|_| MountainCarObservation::sample_between(&mut generator(), None))
            .collect_vec(),
        200,
        environment,
    );

    let mut hyper_params = HyperParameters {
        population_size: 10,
        gap: 0.5,
        mutation_percent: 1.,
        crossover_percent: 0.,
        n_generations: 100,
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
