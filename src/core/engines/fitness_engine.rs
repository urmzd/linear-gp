use core::fmt::Debug;
use std::cmp::Ordering;

use derive_more::Display;
use serde::{Deserialize, Serialize};

use super::reset_engine::{Reset, ResetEngine};

#[derive(Clone, Debug, Copy, PartialEq, Display, Serialize, Deserialize)]
pub enum FitnessScore {
    #[display(fmt = "{}", _0)]
    Valid(f64),
    #[display(format = "nan")]
    OutOfBounds,
    #[display(format = "nan")]
    NotEvaluated,
}

impl Reset<FitnessScore> for ResetEngine {
    fn reset(item: &mut FitnessScore) {
        *item = FitnessScore::NotEvaluated
    }
}

impl Ord for FitnessScore {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Valid(a), Self::Valid(b)) => a.partial_cmp(b).unwrap(),
            (Self::Valid(_), _) => Ordering::Greater,
            (_, Self::Valid(_)) => Ordering::Less,
            _ => Ordering::Equal,
        }
    }
}

impl Eq for FitnessScore {}

impl PartialOrd for FitnessScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
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
}

pub trait Fitness<I, S, P> {
    fn eval_fitness(program: &mut I, states: &mut S) -> FitnessScore;
}

pub struct FitnessEngine;
