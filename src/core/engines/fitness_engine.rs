use core::fmt::Debug;
use std::{
    cmp::Ordering,
    iter::Sum,
    ops::{Add, Div, Mul, Sub},
};

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

impl FitnessScore {
    pub fn unwrap(self) -> f64 {
        match self {
            Self::Valid(f) => f,
            _ => panic!("Fitness is not valid"),
        }
    }

    pub fn unwrap_or(self, default: f64) -> f64 {
        match self {
            Self::Valid(f) => f,
            _ => default,
        }
    }
}

impl Add for FitnessScore {
    type Output = FitnessScore;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Valid(a), Self::Valid(b)) => Self::Valid(a + b),
            (Self::Valid(a), _) => Self::Valid(a),
            (_, Self::Valid(b)) => Self::Valid(b),
            _ => Self::Valid(0.0),
        }
    }
}

impl Sum for FitnessScore {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = FitnessScore>,
    {
        iter.fold(FitnessScore::NotEvaluated, |acc, score| acc + score)
    }
}

impl Sub for FitnessScore {
    type Output = FitnessScore;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Valid(a), Self::Valid(b)) => Self::Valid(a - b),
            (Self::Valid(a), _) => Self::Valid(a),
            (_, Self::Valid(b)) => Self::Valid(-b),
            _ => Self::Valid(0.0),
        }
    }
}

impl Mul for FitnessScore {
    type Output = FitnessScore;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Valid(a), Self::Valid(b)) => Self::Valid(a * b),
            (Self::Valid(_a), _) => Self::Valid(0.),
            (_, Self::Valid(_b)) => Self::Valid(0.),
            _ => Self::Valid(0.0),
        }
    }
}

impl Div for FitnessScore {
    type Output = FitnessScore;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Valid(a), Self::Valid(b)) if b != 0. => Self::Valid(a / b),
            (Self::Valid(_), _) => Self::OutOfBounds,
            (_, Self::Valid(b)) if b != 0. => Self::Valid(0.),
            _ => Self::OutOfBounds,
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
