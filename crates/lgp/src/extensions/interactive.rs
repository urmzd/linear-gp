use core::fmt::Debug;

use serde::Serialize;
use tracing::{instrument, trace};

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
    #[instrument(skip_all, fields(program_id = %program.id), level = "trace")]
    fn eval_fitness(program: &mut crate::core::program::Program, states: &mut T) -> f64 {
        let mut score = 0.;
        let mut step = 0;

        while let Some(state) = states.get() {
            // Run program.
            program.run(state);

            // Eval
            let reward = match program.registers.argmax(ArgmaxInput::ActionRegisters).any() {
                ActionRegister::Value(action) => {
                    let r = state.execute_action(action);
                    trace!(step = step, action = action, reward = r, "Step executed");
                    r
                }
                ActionRegister::Overflow => {
                    trace!(step = step, "Register overflow - returning NEG_INFINITY");
                    return f64::NEG_INFINITY;
                }
            };

            score += reward;
            step += 1;
        }

        trace!(total_steps = step, final_score = score, "Episode complete");
        score
    }
}
