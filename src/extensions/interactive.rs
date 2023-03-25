use core::fmt::Debug;
use std::iter::repeat_with;

use gym_rs::core::ActionReward;
use gym_rs::core::Env;
use gym_rs::utils::custom::traits::Sample;
use serde::Serialize;

use crate::core::engines::fitness_engine::Fitness;
use crate::core::engines::fitness_engine::FitnessEngine;
use crate::core::engines::fitness_engine::FitnessScore;
use crate::core::input_engine::Environment;
use crate::core::input_engine::RlState;
use crate::core::input_engine::State;
use crate::core::registers::ActionRegister;
use crate::core::registers::ArgmaxInput;

#[derive(Debug, Serialize, Clone, Copy)]
pub enum Reward {
    Continue(f64),
    Terminal(f64),
}

impl Reward {
    pub fn get_value(&self) -> f64 {
        match *self {
            Reward::Continue(reward) => reward,
            Reward::Terminal(reward) => reward,
        }
    }

    pub fn is_terminal(&self) -> bool {
        match self {
            Reward::Continue(_) => false,
            Reward::Terminal(_) => true,
        }
    }
}

// pub trait InteractiveLearningInput: ValidInput
// where
//     Self::Environment: Env,
// {
//     type Environment;

//     const MAX_EPISODE_LENGTH: usize;

//     fn new() -> Self;
//     fn get_env(&mut self) -> &mut Self::Environment;

//     fn set_state(&mut self, state: <Self::Environment as Env>::Observation) {
//         self.get_env().reset(None, false, None);
//         self.get_env().set_state(state)
//     }

//     fn get_new_initial_state() -> <Self::Environment as Env>::Observation {
//         <<Self::Environment as Env>::Observation>::sample_between(&mut generator(), None)
//     }

//     fn get_initial_states(
//         number_of_generations: usize,
//         n_trials: usize,
//     ) -> Vec<Vec<<Self::Environment as Env>::Observation>> {
//         repeat_with(|| {
//             repeat_with(Self::get_new_initial_state)
//                 .take(n_trials)
//                 .collect_vec()
//         })
//         .take(number_of_generations)
//         .collect_vec()
//     }

//     fn execute_action(&mut self, action: usize) -> Reward {
//         let ActionReward { reward, done, .. } = self.get_env().step(action.into());

//         let reward = reward.into_inner();

//         let wrapped_reward = match done {
//             true => Reward::Terminal(reward),
//             false => Reward::Continue(reward),
//         };

//         wrapped_reward
//     }
// }

/// simply a marker
pub struct Rl;

impl<T> Fitness<T, Rl> for FitnessEngine
where
    T: RlState,
{
    fn eval_fitness(
        program: &mut crate::core::program::Program,
        states: &mut T,
        parameters: &mut Rl,
    ) -> crate::core::engines::fitness_engine::FitnessScore {
        let mut score = 0.;

        while let Some(state) = states.next_state() {
            // Run program.
            program.run(&state);

            // Eval
            let reward = match program.registers.argmax(ArgmaxInput::To(T::N_ACTIONS)).any() {
                ActionRegister::Value(action) => state.execute_action(action),
                ActionRegister::Overflow => return FitnessScore::OutOfBounds,
            };

            score += reward;
        }

        FitnessScore::Valid(score)
    }
}
