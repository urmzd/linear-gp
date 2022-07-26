use std::fmt::Debug;

use derive_new::new;
use rand::prelude::SliceRandom;
use serde::Serialize;

use crate::{
    core::{characteristics::Fitness, inputs::ValidInput, program::Program},
    utils::random::generator,
};

#[derive(Debug, Clone, new)]
pub struct ReinforcementLearningParameters<T>
where
    T: ReinforcementLearningInput,
{
    pub initial_states: Vec<T::State>,
    pub max_episode_length: usize,
    pub environment: T,
}

impl<T> ReinforcementLearningParameters<T>
where
    T: ReinforcementLearningInput,
{
    pub fn update(&mut self, states: Vec<T::State>) {
        self.initial_states = states;
    }
}

#[derive(Debug, Serialize, Clone, Copy)]
pub enum Reward {
    Continue(f64),
    Terminal(f64),
}

#[derive(Debug, Clone)]
pub struct StateRewardPair {
    pub state: Vec<f64>,
    pub reward: Reward,
}

impl StateRewardPair {
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

pub trait ReinforcementLearningInput: ValidInput + Sized {
    type State;

    fn init(&mut self);
    fn sim(&mut self, action: usize) -> StateRewardPair;
    fn reset(&mut self);
    fn update_state(&mut self, state: Self::State);
    fn finish(&mut self);
}

impl<T> Fitness for Program<ReinforcementLearningParameters<T>>
where
    T: ReinforcementLearningInput,
    T::State: Clone,
{
    type FitnessParameters = ReinforcementLearningParameters<T>;

    fn eval_fitness(&mut self, parameters: &mut Self::FitnessParameters) {
        let mut scores = vec![];

        parameters.environment.init();

        for initial_state in parameters.initial_states.clone() {
            let mut score = 0.;

            parameters.environment.update_state(initial_state);

            for _ in 0..parameters.max_episode_length {
                // Run program.
                self.exec(&parameters.environment);
                // Eval
                let winning_registers =
                    match self.registers.all_argmax(Some(0..T::N_ACTION_REGISTERS)) {
                        None => {
                            return {
                                self.fitness = None;
                            }
                        }
                        Some(registers) => registers,
                    };
                let picked_action = winning_registers
                    .choose(&mut generator())
                    .map(|v| *v)
                    .expect("Register to have been chosen.");
                let state_reward = parameters.environment.sim(picked_action as usize);

                score += state_reward.get_value();

                if state_reward.is_terminal() {
                    break;
                }
            }

            scores.push(score);
            self.registers.reset();
            parameters.environment.reset();
        }

        parameters.environment.finish();

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = scores.swap_remove(scores.len() / 2);

        self.fitness = Some(median);
    }

    fn get_fitness(&self) -> Option<f64> {
        self.fitness
    }
}
