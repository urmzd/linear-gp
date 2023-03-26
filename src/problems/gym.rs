use gym_rs::core::Env;
use gym_rs::envs::classical_control::cartpole::CartPoleEnv;
use gym_rs::envs::classical_control::mountain_car::MountainCarEnv;

use crate::core::engines::generate_engine::Generate;
use crate::core::engines::generate_engine::GenerateEngine;
use crate::core::engines::reset_engine::Reset;
use crate::core::engines::reset_engine::ResetEngine;
use crate::core::environment::State;

#[derive(Clone, Debug)]
pub struct GymRsInput<E: Env, const N_INPUTS: usize, const N_ACTIONS: usize> {
    environment: E,
    terminated: bool,
    episode_idx: usize,
    episode_length: usize,
    initial_state: E::Observation,
}

impl<T, const N_PUTS: usize, const N_ACTS: usize> State for GymRsInput<T, N_PUTS, N_ACTS>
where
    T: Env,
{
    const N_INPUTS: usize = N_PUTS;
    const N_ACTIONS: usize = N_ACTS;

    fn get_value(&self, idx: usize) -> f64 {
        self.environment.get_observation_property(idx)
    }

    fn execute_action(&mut self, action: usize) -> f64 {
        let action_reward = self.environment.step(action);
        self.episode_idx += 1;
        self.terminated = self.episode_idx >= self.episode_length || action_reward.done;
        action_reward.reward
    }

    fn get(&mut self) -> Option<&mut Self> {
        if self.terminated {
            return None;
        }

        Some(self)
    }
}

impl<T, const N_PUTS: usize, const N_ACTS: usize> Reset<GymRsInput<T, N_PUTS, N_ACTS>>
    for ResetEngine
where
    T: Env,
{
    fn reset(item: &mut GymRsInput<T, N_PUTS, N_ACTS>) {
        item.environment.reset(None, false, None);
        item.environment.set_observation(item.initial_state);
    }
}

impl<const N_PUTS: usize, const N_ACTS: usize>
    Generate<(), GymRsInput<MountainCarEnv, N_PUTS, N_ACTS>> for GenerateEngine
{
    fn generate(from: ()) -> GymRsInput<MountainCarEnv, N_PUTS, N_ACTS> {
        let mut environment = MountainCarEnv::new();
        let (initial_state, _) = environment.reset(None, false, None);

        GymRsInput {
            environment,
            terminated: false,
            episode_idx: 0,
            episode_length: 200,
            initial_state,
        }
    }
}

impl<const N_PUTS: usize, const N_ACTS: usize> Generate<(), GymRsInput<CartPoleEnv, N_PUTS, N_ACTS>>
    for GenerateEngine
{
    fn generate(from: ()) -> GymRsInput<CartPoleEnv, N_PUTS, N_ACTS> {
        let mut environment = CartPoleEnv::new();
        let (initial_state, _) = environment.reset(None, false, None);

        GymRsInput {
            environment,
            terminated: false,
            episode_idx: 0,
            episode_length: 200,
            initial_state,
        }
    }
}
