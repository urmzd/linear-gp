use std::fmt::Display;

use valuable::Valuable;

#[derive(Clone, Debug, Copy, PartialEq, PartialOrd, Valuable)]
pub enum FitnessScore {
    OutOfBounds,
    NotEvaluated,
    Valid(f64),
}

impl Display for FitnessScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitnessScore::OutOfBounds => write!(f, "Out of Bounds!"),
            FitnessScore::NotEvaluated => write!(f, "Not Evaluated"),
            FitnessScore::Valid(value) => write!(f, "{}", value),
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

pub trait Fitness {
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

pub trait Generate {
    type GeneratorParameters;

    fn generate(parameters: &Self::GeneratorParameters) -> Self;
}

pub trait DuplicateNew {
    fn duplicate_new(&self) -> Self;
}
