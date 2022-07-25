use std::usize;

use gym_rs::core::{ActionReward, Env};

use crate::core::inputs::ValidInput;

use super::reinforcement_learning::{ReinforcementLearningInput, Reward, StateRewardPair};

pub trait ExtendedGymRsEnvironment
where
    Self::Environment: Env,
{
    type Environment;

    fn get_state(&self) -> <Self::Environment as Env>::Observation;
    fn update_state(&mut self, new_state: <Self::Environment as Env>::Observation);
    fn get_env(&mut self) -> &mut Self::Environment;
}

impl<T> ReinforcementLearningInput for T
where
    Self: ValidInput,
    T: ExtendedGymRsEnvironment,
    <T::Environment as Env>::Action: From<usize>,
{
    type State = <T::Environment as Env>::Observation;

    fn init(&mut self) {
        self.get_env().reset(Some(0), false, None);
    }

    fn sim(&mut self, action: usize) -> StateRewardPair {
        let ActionReward { reward, done, .. } = self.get_env().step(action.into());
        let reward = reward.into_inner();
        let wrapped_reward = match done {
            true => Reward::Terminal(reward),
            false => Reward::Continue(reward),
        };
        let state = self.flat_input();

        StateRewardPair {
            state,
            reward: wrapped_reward,
        }
    }

    fn finish(&mut self) {
        self.get_env().close();
    }

    fn reset(&mut self) {
        self.get_env().reset(None, false, None);
    }

    fn update_state(&mut self, state: Self::State) {
        self.update_state(state)
    }
}
