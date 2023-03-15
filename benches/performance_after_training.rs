use lgp::{
    core::{characteristics::Load, program::Program},
    extensions::{
        interactive::{InteractiveLearningInput, InteractiveLearningParameters},
        q_learning::QProgram,
    },
    problems::mountain_car::MountainCarInput,
    utils::types::VoidResultAnyError,
};

fn main() -> VoidResultAnyError {
    // Run prog to test how well programs do afterwards.

    // LGP - Mountain Car
    // Q Learning - Mountain Car
    //

    let mut lgp_mountain_car_program: Program<InteractiveLearningParameters<MountainCarInput>> =
        Load::load("./assets/benchmarks/")?;
    let _q_mountain_car_program: QProgram<MountainCarInput> = Load::load("./assets/benchmarks/")?;

    let mut mountain_car_env = MountainCarInput::new();

    for _step in 0..MountainCarInput::MAX_EPISODE_LENGTH {
        // Run.
        lgp_mountain_car_program.run(&mut mountain_car_env);
    }

    // LGP - Cart Pole
    // Q Learning - Cart Pole

    return Ok(());
}
