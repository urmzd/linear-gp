use core::fmt::Debug;
use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Sub},
};

use derive_more::{Display};
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

impl Add for FitnessScore {
    type Output = f64;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Valid(a), Self::Valid(b)) => a + b,
            (Self::Valid(a), _) => a,
            (_, Self::Valid(b)) => b,
            _ => 0.0,
        }
    }
}

impl Sub for FitnessScore {
    type Output = f64;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Valid(a), Self::Valid(b)) => a - b,
            (Self::Valid(a), _) => a,
            (_, Self::Valid(b)) => -b,
            _ => 0.0,
        }
    }
}

impl Mul for FitnessScore {
    type Output = f64;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Valid(a), Self::Valid(b)) => a * b,
            (Self::Valid(_a), _) => 0.,
            (_, Self::Valid(_b)) => 0.,
            _ => 0.0,
        }
    }
}

impl Div for FitnessScore {
    type Output = f64;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Valid(a), Self::Valid(b)) => a / b,
            (Self::Valid(a), _) if a >= 0. => f64::INFINITY,
            (Self::Valid(a), _) if a < 0. => f64::NEG_INFINITY,
            (_, Self::Valid(_b)) => 0.0,
            _ => 0.0,
        }
    }
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
