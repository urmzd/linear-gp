use crate::utils::benchmark_tools::create_path;
use core::fmt::Debug;
use std::error::Error;
use std::path::Path;
use std::{cmp::Ordering, path::PathBuf};

use derive_more::Display;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs::{read_to_string, File, OpenOptions};
use std::io::prelude::*;

#[derive(Clone, Debug, Copy, PartialEq, Display, Serialize, Deserialize)]
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
    Self::FitnessParameters: Send + Clone + Serialize + DeserializeOwned,
{
    type FitnessParameters;

    fn eval_fitness(&mut self, parameters: Self::FitnessParameters);
    fn get_fitness(&self) -> FitnessScore;
}

pub trait Load
where
    Self: Sized + DeserializeOwned,
{
    fn load(path: impl Into<PathBuf>) -> Result<Self, Box<dyn Error>> {
        // Read the file contents into a string for debugging purposes
        let contents = read_to_string(&path.into())?;

        // Deserialize the data from the file
        let deserialized: Self = serde_json::from_str(&contents).unwrap();

        Ok(deserialized)
    }
}

pub trait Save
where
    Self: Serialize,
{
    fn save(&self, path: &str) -> Result<String, Box<dyn Error>> {
        create_path(path, true)?;

        // Serialize the object to a json string
        let serialized = serde_json::to_string_pretty(&self)?;

        // Open the file for writing
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(Path::new(path))?;

        // Write the serialized data to the file
        file.write_all(serialized.as_bytes())?;

        Ok(serialized)
    }
}

pub trait Reproduce: Load + Save {}

pub trait Organism:
    Fitness
    + Generate
    + DuplicateNew
    + PartialOrd
    + Sized
    + Clone
    + Mutate
    + Breed
    + Reproduce
    + Debug
    + Send
{
}

impl<T> Organism for T where
    T: Fitness
        + Generate
        + DuplicateNew
        + PartialOrd
        + Sized
        + Clone
        + Mutate
        + Breed
        + Reproduce
        + Debug
        + Send
{
}

impl<T> Load for T where T: Sized + DeserializeOwned {}
impl<T> Save for T where T: Serialize {}
impl<T> Reproduce for T where T: Load + Save {}

pub trait Breed: Clone {
    fn two_point_crossover(&self, mate: &Self) -> [Self; 2];
}

pub trait Mutate: Generate + Clone {
    fn mutate(&self, parameters: Self::GeneratorParameters) -> Self;
}

pub trait Generate
where
    Self::GeneratorParameters: Send + Clone + Serialize + DeserializeOwned,
{
    type GeneratorParameters;

    fn generate(parameters: Self::GeneratorParameters) -> Self;
}

pub trait DuplicateNew {
    fn duplicate_new(&self) -> Self;
}
