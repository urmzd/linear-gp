use core::fmt::Debug;
use std::cmp::Ordering;

use derive_more::Display;
use serde::{Deserialize, Serialize};

use super::reset_engine::{Reset, ResetEngine};

#[derive(Clone, Debug, Copy, PartialEq, Display, Serialize, Deserialize)]
pub enum FitnessScore {
    #[display(fmt = "valid: {}", _0)]
    Valid(f64),
    #[display(format = "out-of-bounds")]
    OutOfBounds,
    #[display(format = "not-evaluated")]
    NotEvaluated,
}

impl Reset<FitnessScore> for ResetEngine {
    fn reset(item: &mut FitnessScore) {
        *item = FitnessScore::NotEvaluated
    }
}

impl PartialOrd for FitnessScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::Valid(a), Self::Valid(b)) => a.partial_cmp(b),
            (Self::Valid(_), _) => Some(Ordering::Greater),
            (_, Self::Valid(_)) => Some(Ordering::Less),
            _ => Some(Ordering::Equal),
        }
    }
}

impl FitnessScore {
    pub fn is_evaluated(&self) -> bool {
        match self {
            FitnessScore::NotEvaluated => false,
            _ => true,
        }
    }

    pub fn is_valid(&self) -> bool {
        match self {
            FitnessScore::Valid(_) => true,
            _ => false,
        }
    }

    pub fn unwrap_or(&self, value: f64) -> f64 {
        match self {
            FitnessScore::Valid(fitness_score) => *fitness_score,
            _ => value,
        }
    }

    pub fn unwrap(&self) -> f64 {
        match self {
            FitnessScore::Valid(fitness_score) => *fitness_score,
            _ => unreachable!(),
        }
    }
}

pub trait Fitness<I, S, P> {
    fn eval_fitness(program: &mut I, states: &mut S) -> FitnessScore;
}

pub struct FitnessEngine;
