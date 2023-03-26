use std::{
    error::Error,
    fs::{read_to_string, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use serde::{de::DeserializeOwned, Serialize};

use crate::utils::benchmark_tools::create_path;

pub trait Load
where
    Self: Sized + DeserializeOwned,
{
    fn load(path: impl Into<PathBuf>) -> Self {
        let contents = read_to_string(&path.into()).unwrap();
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

        let serialized = serde_json::to_string_pretty(&self)?;

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(Path::new(path))?;

        file.write_all(serialized.as_bytes())?;

        Ok(serialized)
    }
}

pub trait Reproduce: Load + Save {}

impl<T> Load for T where T: Sized + DeserializeOwned {}
impl<T> Save for T where T: Serialize {}
impl<T> Reproduce for T where T: Load + Save {}
