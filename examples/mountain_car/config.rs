use derive_new::new;
use gym_rs::{core::Env, envs::classical_control::mountain_car::MountainCarEnv};
use lgp::core::inputs::ValidInput;
use lgp::extensions::gym_rs::ExtendedGymRsEnvironment;
use serde::Serialize;

#[derive(Debug, Serialize, new, Clone)]
pub struct MountainCarInput {
    environment: MountainCarEnv,
}

impl ValidInput for MountainCarInput {
    const N_INPUT_REGISTERS: usize = 2;
    const N_ACTION_REGISTERS: usize = 3;

    fn flat(&self) -> Vec<f64> {
        self.environment.state.into()
    }
}

impl ExtendedGymRsEnvironment for MountainCarInput {
    type Environment = MountainCarEnv;

    fn get_state(&self) -> <Self::Environment as Env>::Observation {
        self.environment.state
    }

    fn update_state(&mut self, new_state: <Self::Environment as Env>::Observation) {
        self.environment.state = new_state;
    }

    fn get_env(&mut self) -> &mut Self::Environment {
        &mut self.environment
    }
}
