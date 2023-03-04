use std::{
    fmt::{self, Debug},
    marker::PhantomData,
};

use derive_new::new;
use rand::prelude::SliceRandom;
use serde::Serialize;

use crate::{
    core::{
        algorithm::GeneticAlgorithm,
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
    initial_states: Vec<Vec<T::State>>,
    pub max_episode_length: usize,
    pub environment: T,
    #[new(value = "0")]
    generations: usize,
}

impl<T> InteractiveLearningParameters<T>
where
    T: InteractiveLearningInput,
{
    pub fn get_state(&self) -> &Vec<T::State> {
        self.initial_states.get(self.generations).unwrap()
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
    Self::State: Into<Vec<f64>>,
{
    type State;

    fn init(&mut self);
    fn sim(&mut self, action: usize) -> StateRewardPair;
    fn reset(&mut self);
    fn update_state(&mut self, state: Self::State);
    fn finish(&mut self);
}

impl<T> Fitness for Program<InteractiveLearningParameters<T>>
where
    T: InteractiveLearningInput,
    T::State: Clone,
{
    type FitnessParameters = InteractiveLearningParameters<T>;

    fn eval_fitness(&mut self, parameters: &mut Self::FitnessParameters) {
        let mut scores = vec![];

        for initial_state in parameters.get_state().clone() {
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
            parameters.environment.reset();
        }

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = scores.swap_remove(scores.len() / 2);

        self.fitness = FitnessScore::Valid(median);
    }

    fn get_fitness(&self) -> FitnessScore {
        self.fitness
    }
}

pub struct ILgp<T>(PhantomData<T>);

impl<T> GeneticAlgorithm for ILgp<T>
where
    T: InteractiveLearningInput + fmt::Debug,
    T::State: Clone + fmt::Debug,
{
    type O = Program<InteractiveLearningParameters<T>>;

    fn on_post_rank(
        _population: &mut crate::core::population::Population<Self::O>,
        parameters: &mut crate::core::algorithm::HyperParameters<Self::O>,
    ) {
        parameters.fitness_parameters.environment.finish();
    }

    fn on_pre_eval_fitness(
        _population: &mut crate::core::population::Population<Self::O>,
        parameters: &mut crate::core::algorithm::HyperParameters<Self::O>,
    ) {
        parameters.fitness_parameters.environment.init();
    }
}