use derive_new::new;
use gym_rs::core::ActionReward;
use gym_rs::{core::Env, envs::classical_control::mountain_car::MountainCarEnv};
use lgp::extensions::reinforcement_learning::{Reward, StateRewardPair};
use lgp::{
    core::{
        algorithm::GeneticAlgorithm, inputs::ValidInput, program::Program, registers::RegisterValue,
    },
    extensions::reinforcement_learning::{
        ReinforcementLearningInput, ReinforcementLearningParameters,
    },
};
use noisy_float::prelude::r64;
use serde::Serialize;

pub struct MountainCarLgp;

impl GeneticAlgorithm for MountainCarLgp {
    type O = Program<ReinforcementLearningParameters<MountainCarInput>>;
}

#[derive(Debug, Serialize, new, Clone)]
pub struct MountainCarInput {
    environment: MountainCarEnv,
}

impl ValidInput for MountainCarInput {
    const N_INPUT_REGISTERS: usize = 2;
    const N_ACTION_REGISTERS: usize = 3;

    fn flat(&self) -> Vec<RegisterValue> {
        let state = self.get_state();
        state
    }
}

impl ReinforcementLearningInput for MountainCarInput {
    fn init(&mut self) {
        self.environment.reset(Some(0), false, None);
    }

    fn sim(&mut self, action: usize) -> StateRewardPair {
        let ActionReward { reward, done, .. } = self.environment.step(action);
        let reward_ = reward.into_inner();

        StateRewardPair {
            state: self.get_state(),
            reward: match done {
                true => Reward::Terminal(reward_),
                false => Reward::Continue(reward_),
            },
        }
    }

    fn get_state(&self) -> Vec<RegisterValue> {
        let state = &self.environment.state;
        [state.position, state.velocity]
            .map(|v| r64(v.into_inner()))
            .to_vec()
    }

    fn finish(&mut self) {
        self.environment.close();
    }

    fn reset(&mut self) {
        self.environment.reset(None, false, None);
    }
}
