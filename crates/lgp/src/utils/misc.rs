use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use crate::core::engines::reset_engine::{Reset, ResetEngine};

pub type VoidResultAnyError = Result<(), Box<dyn Error>>;

/// Create a path, optionally creating a file at the path.
/// If `file` is true, creates the file; otherwise creates a directory.
pub fn create_path(path: &str, file: bool) -> Result<PathBuf, Box<dyn Error>> {
    let path = Path::new(path);

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    if file {
        fs::File::create(path)?;
    } else {
        fs::create_dir_all(path)?;
    }

    Ok(path.to_owned())
}

impl Reset<uuid::Uuid> for ResetEngine {
    fn reset(item: &mut uuid::Uuid) {
        *item = uuid::Uuid::new_v4();
    }
}
