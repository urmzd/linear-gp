use std::usize;

use gym_rs::core::{ActionReward, Env};
use gym_rs::utils::custom::traits::Sample;
use itertools::Itertools;

use crate::core::inputs::ValidInput;
use crate::utils::random::generator;

use super::interactive::{InteractiveLearningInput, Reward, StateRewardPair};

pub trait ExtendedGymRsEnvironment
where
    Self::Environment: Env,
{
    type Environment;

    const EPISODE_LENGTH: usize;

    fn get_state(&self) -> <Self::Environment as Env>::Observation;
    fn update_state(&mut self, new_state: <Self::Environment as Env>::Observation);
    fn get_env(&mut self) -> &mut Self::Environment;

    fn get_initial_states(
        number_of_generations: usize,
        n_trials: usize,
    ) -> Vec<Vec<<Self::Environment as Env>::Observation>> {
        itertools::repeat_n((), number_of_generations)
            .map(|_| {
                itertools::repeat_n((), n_trials)
                    .map(|_| {
                        <<Self::Environment as Env>::Observation>::sample_between(
                            &mut generator(),
                            None,
                        )
                    })
                    .collect_vec()
            })
            .collect_vec()
    }
}

impl<T> InteractiveLearningInput for T
where
    Self: ValidInput,
    T: ExtendedGymRsEnvironment,
    <T::Environment as Env>::Action: From<usize>,
{
    type State = <T::Environment as Env>::Observation;
    const MAX_EPISODE_LENGTH: usize = T::EPISODE_LENGTH;

    fn sim(&mut self, action: usize) -> StateRewardPair {
        let ActionReward { reward, done, .. } = self.get_env().step(action.into());

        let reward = reward.into_inner();

        let wrapped_reward = match done {
            true => Reward::Terminal(reward),
            false => Reward::Continue(reward),
        };

        let state = self.flat();

        StateRewardPair {
            state,
            reward: wrapped_reward,
        }
    }

    fn reset(&mut self) {
        self.get_env().reset(None, false, None);
    }

    fn set_state(&mut self, state: Self::State) {
        self.update_state(state)
    }
}
