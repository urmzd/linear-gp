use core::fmt::Debug;

use serde::Serialize;

use crate::core::engines::fitness_engine::Fitness;
use crate::core::engines::fitness_engine::FitnessEngine;

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

pub struct UseRlFitness;

impl<T> Fitness<Program, T, UseRlFitness> for FitnessEngine
where
    T: RlState,
{
    fn eval_fitness(program: &mut crate::core::program::Program, states: &mut T) -> f64 {
        let mut score = 0.;

        while let Some(state) = states.get() {
            // Run program.
            program.run(state);

            // Eval
            let reward = match program.registers.argmax(ArgmaxInput::ActionRegisters).any() {
                ActionRegister::Value(action) => state.execute_action(action),
                ActionRegister::Overflow => {
                    return f64::NEG_INFINITY;
                }
            };

            score += reward;
        }

        score
    }
}
