use crate::utils::benchmark_tools::create_path;
use std::error::Error;
use std::path::Path;
use std::path::PathBuf;

use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fs::{read_to_string, OpenOptions};
use std::io::prelude::*;

pub trait Load
where
    Self: Sized + DeserializeOwned,
{
    fn load(path: impl Into<PathBuf>) -> Self {
        // Read the file contents into a string for debugging purposes
        let contents = read_to_string(&path.into()).unwrap();

        // Deserialize the data from the file
        let deserialized: Self = serde_json::from_str(&contents).unwrap();

        deserialized
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

impl<T> Load for T where T: Sized + DeserializeOwned {}
impl<T> Save for T where T: Serialize {}
impl<T> Reproduce for T where T: Load + Save {}

pub trait ResetNew: Reset + Clone {
    fn reset_new(&self) -> Self {
        let mut new = self.clone();
        new.reset();
        return new;
    }
}

pub trait Reset {
    fn reset(&mut self);
}

impl<T> ResetNew for T where T: Reset + Clone {}
