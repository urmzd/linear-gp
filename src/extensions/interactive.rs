use core::fmt::Debug;
use std::{iter::repeat_with, marker::PhantomData};

use derive_new::new;
use gym_rs::core::ActionReward;
use gym_rs::core::Env;
use gym_rs::utils::custom::traits::Sample;
use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;

use crate::core::algorithm::HyperParameters;
use crate::core::population::Population;
use crate::core::program::ProgramParameters;
use crate::core::registers::ArgmaxInput;
use crate::core::registers::AR;
use crate::{
    core::{
        algorithm::GeneticAlgorithm,
        characteristics::{Fitness, FitnessScore},
        inputs::ValidInput,
        program::Program,
    },
    utils::random::generator,
};

#[derive(Debug, Clone, Deserialize, new)]
pub struct InteractiveLearningParametersArgs<T>
where
    T: InteractiveLearningInput,
{
    n_generations: usize,
    n_trials: usize,
    marker: PhantomData<T>,
}

impl<T> From<InteractiveLearningParametersArgs<T>> for InteractiveLearningParameters<T>
where
    T: InteractiveLearningInput,
{
    fn from(value: InteractiveLearningParametersArgs<T>) -> Self {
        InteractiveLearningParameters::new(value)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(from = "InteractiveLearningParametersArgs<T>")]
pub struct InteractiveLearningParameters<T>
where
    T: InteractiveLearningInput,
{
    // Collection of X intial states per generation.
    initial_states: Vec<Vec<<T::Environment as Env>::Observation>>,
    pub environment: T,
    generations: usize,
}

impl<T> InteractiveLearningParameters<T>
where
    T: InteractiveLearningInput,
{
    pub fn new(args: InteractiveLearningParametersArgs<T>) -> Self {
        Self {
            initial_states: T::get_initial_states(args.n_generations, args.n_trials),
            environment: T::new(),
            generations: 0,
        }
    }

    pub fn get_states(&mut self) -> Vec<<T::Environment as Env>::Observation> {
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

    fn execute_action(&mut self, action: usize) -> StateRewardPair {
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

impl<T> ProgramParameters for InteractiveLearningParameters<T>
where
    T: InteractiveLearningInput,
{
    type InputType = T;
}

impl<T> Fitness for Program<InteractiveLearningParameters<T>>
where
    T: InteractiveLearningInput,
{
    type FitnessParameters = InteractiveLearningParameters<T>;

    fn eval_fitness(&mut self, mut parameters: Self::FitnessParameters) {
        let mut scores = vec![];

        for initial_state in parameters.get_states() {
            parameters.environment.set_state(initial_state.clone());

            let mut score = 0.;

            for _ in 0..T::MAX_EPISODE_LENGTH {
                // Run program.
                self.run(&parameters.environment);

                // Eval
                let state_reward = match self.registers.argmax(ArgmaxInput::To(T::N_ACTIONS)).any()
                {
                    AR::Value(action) => parameters.environment.execute_action(action),
                    AR::Overflow => return self.fitness = FitnessScore::OutOfBounds,
                };

                score += state_reward.get_value();

                if state_reward.is_terminal() {
                    break;
                }
            }

            scores.push(score);
            self.registers.reset();
        }

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
    T: InteractiveLearningInput,
{
    type O = Program<InteractiveLearningParameters<T>>;

    fn on_post_rank(
        population: Population<Self::O>,
        mut parameters: HyperParameters<Self::O>,
    ) -> (Population<Self::O>, HyperParameters<Self::O>) {
        parameters.fitness_parameters.next_generation();

        return (population, parameters);
    }
}
