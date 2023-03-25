use core::fmt::Debug;
use std::{iter::repeat_with, marker::PhantomData};

use gym_rs::core::ActionReward;
use gym_rs::core::Env;
use gym_rs::utils::custom::traits::Sample;
use itertools::Itertools;
use serde::ser::SerializeStruct;
use serde::Deserialize;
use serde::Serialize;

use crate::core::algorithm::HyperParameters;
use crate::core::characteristics::Reset;
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

#[derive(Debug, Clone, Deserialize)]
pub struct InteractiveLearningParametersArgs {
    n_generations: usize,
    n_trials: usize,
}

impl<T> From<InteractiveLearningParametersArgs> for InteractiveLearningParameters<T>
where
    T: InteractiveLearningInput,
{
    fn from(value: InteractiveLearningParametersArgs) -> Self {
        InteractiveLearningParameters::new(value)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(from = "InteractiveLearningParametersArgs", bound = "T: Sized")]
pub struct InteractiveLearningParameters<T>
where
    T: InteractiveLearningInput,
{
    // Collection of X intial states per generation.
    initial_states: Vec<Vec<<T::Environment as Env>::Observation>>,
    pub environment: T,
    current_generation: usize,
    n_generations: usize,
    n_trials: usize,
}

impl<T> Serialize for InteractiveLearningParameters<T>
where
    T: InteractiveLearningInput,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("InteractiveLearningParameters", 2)?;
        state.serialize_field("n_generations", &self.n_generations)?;
        state.serialize_field("n_trials", &self.n_trials)?;
        state.end()
    }
}

impl<T> InteractiveLearningParameters<T>
where
    T: InteractiveLearningInput,
{
    pub fn new(args: InteractiveLearningParametersArgs) -> Self {
        let InteractiveLearningParametersArgs {
            n_generations,
            n_trials,
            ..
        } = args;
        Self {
            initial_states: T::get_initial_states(n_generations, n_trials),
            environment: T::new(),
            current_generation: 0,
            n_generations,
            n_trials,
        }
    }

    pub fn get_states(&mut self) -> Vec<<T::Environment as Env>::Observation> {
        self.initial_states
            .get(self.current_generation)
            .cloned()
            .unwrap()
    }

    pub fn next_generation(&mut self) {
        self.current_generation += 1;
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

pub struct InteractiveLearner<T>(T);

impl<T> Fitness for InteractiveLearner<T>
where
    T: InteractiveLearningInput,
{
    fn eval_fitness(
        &mut self,
        program: &mut Program,
        parameters: Self::Parameters,
    ) -> (FitnessScore, Self::Parameters) {
    }
}

impl<T> Fitness<InteractiveLearningParameters<T>> for Program
where
    T: InteractiveLearningInput,
{
    fn eval_fitness(&mut self, mut parameters: InteractiveLearningParameters<T>) {
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
}

pub struct ILgp<T>(PhantomData<T>);

impl<T> GeneticAlgorithm for ILgp<T>
where
    T: InteractiveLearningInput,
{
    type O = Program;

    fn on_post_rank(
        population: Population<Self::O>,
        mut parameters: HyperParameters<Self::O>,
    ) -> (Population<Self::O>, HyperParameters<Self::O>) {
        parameters.evaluator.next_generation();

        return (population, parameters);
    }
}
