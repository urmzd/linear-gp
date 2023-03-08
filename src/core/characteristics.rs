use std::cmp::Ordering;

use derive_more::Display;
use valuable::Valuable;

#[derive(Clone, Debug, Copy, PartialEq, Valuable, Display)]
pub enum FitnessScore {
    #[display(fmt = "valid: {}", _0)]
    Valid(f64),
    #[display(format = "out-of-bounds")]
    OutOfBounds,
    #[display(format = "not-evaluated")]
    NotEvaluated,
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
    pub fn is_not_evaluated(&self) -> bool {
        match self {
            Self::NotEvaluated => true,
            _ => false,
        }
    }

    pub fn is_invalid(&self) -> bool {
        match self {
            FitnessScore::Valid(_) | FitnessScore::NotEvaluated => false,
            _ => true,
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
            _ => panic!("Tried to unwrap a value from an invalid FitnessScore."),
        }
    }
}

pub trait Fitness: Send
where
    Self::FitnessParameters: Send,
{
    type FitnessParameters;

    fn eval_fitness(&mut self, parameters: &mut Self::FitnessParameters);
    fn get_fitness(&self) -> FitnessScore;
}

pub trait Breed: Clone {
    fn two_point_crossover(&self, mate: &Self) -> [Self; 2];
}

pub trait Mutate: Generate + Clone {
    fn mutate(&self, parameters: &Self::GeneratorParameters) -> Self;
}

pub trait Generate
where
    Self::GeneratorParameters: Send,
{
    type GeneratorParameters;

    fn generate(parameters: &Self::GeneratorParameters) -> Self;
}

pub trait DuplicateNew {
    fn duplicate_new(&self) -> Self;
}
