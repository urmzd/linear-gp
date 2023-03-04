use derive_new::new;
use gym_rs::{core::Env, envs::classical_control::cartpole::CartPoleEnv};
use lgp::{core::inputs::ValidInput, extensions::gym_rs::ExtendedGymRsEnvironment};
use serde::Serialize;

#[derive(Clone, Debug, Serialize, new)]
pub struct CartPoleInput {
    environment: CartPoleEnv,
}

impl ValidInput for CartPoleInput {
    const N_INPUT_REGISTERS: usize = 4;
    const N_ACTION_REGISTERS: usize = 2;

    fn flat(&self) -> Vec<f64> {
        self.environment.state.into()
    }
}

impl ExtendedGymRsEnvironment for CartPoleInput {
    type Environment = CartPoleEnv;
    const EPISODE_LENGTH: usize = 500;

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
