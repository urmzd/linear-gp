use gym_rs::{
    envs::classical_control::mountain_car::{MountainCarEnv, MountainCarObservation},
    utils::{custom::Sample, renderer::RenderMode},
};
use lgp::{
    core::{
        algorithm::{EventHooks, GeneticAlgorithm, HyperParameters},
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    },
    extensions::q_learning::{QConsts, QLearningParameters, QProgramGeneratorParameters},
    utils::random::generator,
};
use set_up::{MountainCarInput, QMountainCarLgp};
mod set_up;

fn main() {
    let game = MountainCarEnv::new(RenderMode::Human, None);
    let environment = MountainCarInput::new(game);
    let mut parameters = QLearningParameters::new(
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

    QMountainCarLgp::execute(&mut hyper_params, EventHooks::default())
        .expect("Example to have been ran successfully.");
}
