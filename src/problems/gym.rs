use gym_rs::core::Env;

use crate::core::{characteristics::Reset, input_engine::State};

#[derive(Clone, Debug)]
pub struct GymRsInput<E: Env, const N_INPUTS: usize, const N_ACTIONS: usize> {
    environment: E,
    terminated: bool,
    n_steps: usize,
    episode_length: usize,
}

impl<T, const N_PUTS: usize, const N_ACTS: usize> State for GymRsInput<T, N_PUTS, N_ACTS>
where
    T: Env,
{
    const N_INPUTS: usize = N_PUTS;
    const N_ACTIONS: usize = N_ACTS;

    fn get_value(&self, idx: usize) -> f64 {
        self.environment.get_state_at(idx)
    }

    fn execute_action(&mut self, action: usize) -> f64 {
        let action_reward = self.environment.step(action);
        self.n_steps += 1;
        self.terminated = self.n_steps >= self.episode_length || action_reward.done;
        action_reward.reward
    }
}

impl<T, const N_PUTS: usize, const N_ACTS: usize> Reset for GymRsInput<T, N_PUTS, N_ACTS>
where
    T: Env,
{
    fn reset(&mut self) {}
}

impl<T, const N_PUTS: usize, const N_ACTS: usize> Iterator for GymRsInput<T, N_PUTS, N_ACTS>
where
    T: Env,
{
    type Item = Self;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
