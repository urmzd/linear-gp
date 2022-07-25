use derivative::Derivative;
use derive_new::new;
use itertools::Itertools;
use noisy_float::prelude::r64;
use rand::prelude::SliceRandom;
use serde::Serialize;

use crate::{
    core::{
        characteristics::Fitness,
        inputs::ValidInput,
        program::Program,
        registers::{RegisterValue, Registers},
    },
    utils::random::generator,
};

use super::core::ExtensionParameters;

#[derive(Debug, Serialize, Derivative, new)]
#[derivative(PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct ReinforcementLearningParameters<T>
where
    T: ReinforcementLearningInput,
{
    pub n_runs: usize,
    pub max_episode_length: usize,
    #[derivative(PartialEq = "ignore", PartialOrd = "ignore", Ord = "ignore")]
    pub environment: T,
}

#[derive(Debug, Serialize, Clone, Copy)]
pub enum Reward {
    Continue(f64),
    Terminal(f64),
}

#[derive(Debug, Clone)]
pub struct StateRewardPair {
    pub state: Vec<RegisterValue>,
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
    fn init(&mut self);
    fn sim(&mut self, action: usize) -> StateRewardPair;
    fn reset(&mut self);
    fn get_state(&self) -> Vec<RegisterValue>;
    fn finish(&mut self);
}

impl<T> ExtensionParameters for ReinforcementLearningParameters<T>
where
    T: ReinforcementLearningInput,
{
    fn argmax(registers: &Registers) -> i32 {
        let action_registers = &registers[0..T::N_ACTION_REGISTERS];
        let max_value = action_registers.into_iter().sorted().last().unwrap();

        let indices = action_registers
            .into_iter()
            .enumerate()
            .filter(|(_, value)| *value == max_value)
            .map(|(index, _)| index)
            .collect_vec();

        let index_chosen = indices
            .choose(&mut generator())
            .map(|v| *v as i32)
            .expect("Index to exist.");

        index_chosen
    }
}

impl<T> Fitness for Program<ReinforcementLearningParameters<T>>
where
    T: ReinforcementLearningInput,
{
    type FitnessParameters = ReinforcementLearningParameters<T>;

    fn eval_fitness(
        &mut self,
        parameters: &mut Self::FitnessParameters,
    ) -> crate::core::characteristics::FitnessScore {
        let mut scores = vec![];

        parameters.environment.init();

        for _ in 0..parameters.n_runs {
            let mut score = 0.;

            for _ in 0..parameters.max_episode_length {
                // Run program.
                self.exec(&parameters.environment);
                // Eval
                let picked_action = ReinforcementLearningParameters::<T>::argmax(&self.registers);
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
        let median = r64(scores.swap_remove(parameters.n_runs / 2));

        self.fitness = Some(median);

        median
    }

    fn get_fitness(&self) -> Option<crate::core::characteristics::FitnessScore> {
        self.fitness
    }
}
