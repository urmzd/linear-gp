use std::{
    fmt::{self, Debug},
    iter::repeat_with,
    marker::PhantomData,
};

use derive_new::new;
use gym_rs::core::ActionReward;
use gym_rs::core::Env;
use gym_rs::utils::custom::traits::Sample;
use itertools::Itertools;
use rand::prelude::SliceRandom;
use serde::Serialize;
use tracing::field::valuable;
use tracing::trace;

use crate::{
    core::{
        algorithm::{GeneticAlgorithm, Organism},
        characteristics::{Fitness, FitnessScore},
        inputs::ValidInput,
        program::Program,
    },
    utils::random::generator,
};

#[derive(Debug, Clone, new)]
pub struct InteractiveLearningParameters<T>
where
    T: InteractiveLearningInput,
{
    // Collection of X intial states per generation.
    initial_states: Vec<Vec<<T::Environment as Env>::Observation>>,
    pub environment: T,
    #[new(value = "0")]
    generations: usize,
}

impl<T> InteractiveLearningParameters<T>
where
    T: InteractiveLearningInput,
{
    pub fn get_states(&self) -> Vec<<T::Environment as Env>::Observation> {
        self.initial_states.get(self.generations).cloned().unwrap()
    }

    pub fn next_generation(&mut self) {
        self.generations += 1;
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

pub trait InteractiveLearningInput: ValidInput + Sized
where
    Self::Environment: Env,
{
    type Environment;

    const MAX_EPISODE_LENGTH: usize;

    fn reset(&mut self) {
        self.get_env().reset(None, false, None);
    }

    fn set_state(&mut self, state: <Self::Environment as Env>::Observation) {
        self.get_env().set_state(state)
    }

    fn get_initial_states(
        number_of_generations: usize,
        n_trials: usize,
    ) -> Vec<Vec<<Self::Environment as Env>::Observation>> {
        repeat_with(|| {
            repeat_with(|| {
                <<Self::Environment as Env>::Observation>::sample_between(&mut generator(), None)
            })
            .take(n_trials)
            .collect_vec()
        })
        .take(number_of_generations)
        .collect_vec()
    }

    fn get_env(&mut self) -> &mut Self::Environment;

    fn new() -> Self;

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
}

impl<T> Organism for Program<InteractiveLearningParameters<T>> where
    T: InteractiveLearningInput + fmt::Debug
{
}

impl<T> Fitness for Program<InteractiveLearningParameters<T>>
where
    T: InteractiveLearningInput,
{
    type FitnessParameters = InteractiveLearningParameters<T>;

    fn eval_fitness(&mut self, mut parameters: Self::FitnessParameters) {
        let mut scores = vec![];

        for initial_state in parameters.get_states() {
            parameters.environment.reset();
            parameters.environment.set_state(initial_state.clone());

            let mut score = 0.;

            for _ in 0..T::MAX_EPISODE_LENGTH {
                // Run program.
                self.exec(&parameters.environment);
                // Eval
                let winning_registers =
                    match self.registers.all_argmax(Some(0..T::N_ACTION_REGISTERS)) {
                        None => {
                            return {
                                self.fitness = FitnessScore::OutOfBounds;
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
        }

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        trace!(scores = valuable(&scores));
        let median = scores.get(scores.len() / 2).take().unwrap();

        self.fitness = FitnessScore::Valid(*median);
    }

    fn get_fitness(&self) -> FitnessScore {
        self.fitness
    }
}

pub struct ILgp<T>(PhantomData<T>);

impl<T> GeneticAlgorithm for ILgp<T>
where
    T: InteractiveLearningInput + fmt::Debug,
{
    type O = Program<InteractiveLearningParameters<T>>;
}
