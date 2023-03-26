use core::fmt::Debug;

use serde::Serialize;

use crate::core::engines::fitness_engine::Fitness;
use crate::core::engines::fitness_engine::FitnessEngine;
use crate::core::engines::fitness_engine::FitnessScore;

use crate::core::environment::RlState;
use crate::core::program::Program;
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

/// simply a marker
struct Rl;

impl<T> Fitness<Program, T, Rl> for FitnessEngine
where
    T: RlState,
{
    fn eval_fitness(
        program: &mut crate::core::program::Program,
        states: &mut T,
    ) -> crate::core::engines::fitness_engine::FitnessScore {
        let mut score = 0.;

        while let Some(state) = states.get() {
            // Run program.
            program.run(state);

            // Eval
            let reward = match program
                .registers
                .argmax(ArgmaxInput::To(T::N_ACTIONS))
                .any()
            {
                ActionRegister::Value(action) => state.execute_action(action),
                ActionRegister::Overflow => return FitnessScore::OutOfBounds,
            };

            score += reward;
        }

        FitnessScore::Valid(score)
    }
}
