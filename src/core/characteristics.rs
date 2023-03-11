use crate::utils::benchmark_tools::create_path;
use core::fmt;
use std::error::Error;
use std::path::Path;
use std::{cmp::Ordering, path::PathBuf};

use derive_more::Display;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use valuable::Valuable;

#[derive(Clone, Debug, Copy, PartialEq, Valuable, Display, Serialize, Deserialize)]
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

pub trait Fitness
where
    Self::FitnessParameters: Send + Clone,
{
    type FitnessParameters;

    fn eval_fitness(&mut self, parameters: Self::FitnessParameters);
    fn get_fitness(&self) -> FitnessScore;
}

pub trait Reproducible: Serialize + DeserializeOwned + Sized {
    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        create_path(path)?;

        // Serialize the object to a JSON string
        let serialized = serde_json::to_string(&self)?;

        // Open the file for writing
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(Path::new(path))?;

        // Write the serialized data to the file
        file.write_all(serialized.as_bytes())?;

        Ok(())
    }

    fn load(path: impl Into<PathBuf>) -> Result<Self, Box<dyn Error>> {
        // Open the file for reading
        let file = File::open(path.into())?;

        // Deserialize the data from the file
        let deserialized: Self = serde_json::from_reader(file)?;

        Ok(deserialized)
    }
}

pub trait Organism:
    Fitness
    + Generate
    + DuplicateNew
    + PartialOrd
    + Sized
    + Clone
    + Mutate
    + Breed
    + Reproducible
    + fmt::Debug
    + Send
{
}

pub trait Breed: Clone {
    fn two_point_crossover(&self, mate: &Self) -> [Self; 2];
}

pub trait Mutate: Generate + Clone {
    fn mutate(&self, parameters: Self::GeneratorParameters) -> Self;
}

pub trait Generate
where
    Self::GeneratorParameters: Send + Clone,
{
    type GeneratorParameters;

    fn generate(parameters: Self::GeneratorParameters) -> Self;
}

pub trait DuplicateNew {
    fn duplicate_new(&self) -> Self;
}
