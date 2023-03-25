use core::fmt::Debug;
use std::iter::repeat_with;

use gym_rs::core::ActionReward;
use gym_rs::core::Env;
use gym_rs::utils::custom::traits::Sample;
use itertools::Itertools;
use serde::ser::SerializeStruct;
use serde::Deserialize;
use serde::Serialize;

use crate::core::algorithm::HyperParameters;
use crate::core::characteristics::Reset;
use crate::core::engines::fitness_engine::Fitness;
use crate::core::engines::fitness_engine::FitnessEngine;
use crate::core::program::ProgramParameters;
use crate::core::registers::ArgmaxInput;
use crate::core::registers::AR;
use crate::{
    core::{inputs::ValidInput, program::Program},
    utils::random::generator,
};

#[derive(Debug, Serialize, Clone, Copy)]
pub enum Reward {
    Continue(f64),
    Terminal(f64),
}

impl Reward {
    pub fn get_value(&self) -> f64 {
        match self.reward {
            Reward::Continue(reward) => reward,
            Reward::Terminal(reward) => reward,
        }
    }

    pub fn is_terminal(&self) -> bool {
        match self.reward {
            Reward::Continue(_) => false,
            Reward::Terminal(_) => true,
        }
    }
}

pub trait InteractiveLearningInput: ValidInput
where
    Self::Environment: Env,
{
    type Environment;

    const MAX_EPISODE_LENGTH: usize;

    fn new() -> Self;
    fn get_env(&mut self) -> &mut Self::Environment;

    fn set_state(&mut self, state: <Self::Environment as Env>::Observation) {
        self.get_env().reset(None, false, None);
        self.get_env().set_state(state)
    }

    fn get_new_initial_state() -> <Self::Environment as Env>::Observation {
        <<Self::Environment as Env>::Observation>::sample_between(&mut generator(), None)
    }

    fn get_initial_states(
        number_of_generations: usize,
        n_trials: usize,
    ) -> Vec<Vec<<Self::Environment as Env>::Observation>> {
        repeat_with(|| {
            repeat_with(Self::get_new_initial_state)
                .take(n_trials)
                .collect_vec()
        })
        .take(number_of_generations)
        .collect_vec()
    }

    fn execute_action(&mut self, action: usize) -> Reward {
        let ActionReward { reward, done, .. } = self.get_env().step(action.into());

        let reward = reward.into_inner();

        let wrapped_reward = match done {
            true => Reward::Terminal(reward),
            false => Reward::Continue(reward),
        };

        wrapped_reward
    }
}


impl<T> Fitness<Program, InteractiveLearningParameters<T>> for FitnessEngine {
    fn eval_fitness(item: &mut Program, parameters: InteractiveLearningParameters<T>) {
        self.registers.reset();

        let mut score = 0.;

        for _ in 0..T::MAX_EPISODE_LENGTH {
            // Run program.
            self.run(&parameters.environment);

            // Eval
            let state_reward = match self.registers.argmax(ArgmaxInput::To(T::N_ACTIONS)).any() {
                AR::Value(action) => parameters.environment.execute_action(action),
                AR::Overflow => return self.fitness = FitnessScore::OutOfBounds,
            };

            score += state_reward.get_value();

            if state_reward.is_terminal() {
                break;
            }
        }

        self.fitness = FitnessScore::Valid(*median);
    }
}
