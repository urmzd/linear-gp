//! Config discovery module
//!
//! Discovers experiment configurations from the configs/ directory.

use std::env;
use std::path::PathBuf;

/// A discovered experiment configuration.
#[derive(Debug, Clone)]
pub struct DiscoveredConfig {
    /// Name of the experiment (directory name)
    pub name: String,
    /// Path to the config file
    pub config_path: PathBuf,
}

/// Get the configs directory from environment or default.
pub fn get_configs_dir() -> PathBuf {
    env::var("LGP_CONFIGS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("configs"))
}

/// Discover all experiment configurations.
///
/// Finds all directories in configs/ that contain a default.toml file.
pub fn discover_configs() -> Result<Vec<DiscoveredConfig>, Box<dyn std::error::Error>> {
    let configs_dir = get_configs_dir();

    if !configs_dir.exists() {
        return Ok(Vec::new());
    }

    let mut configs = Vec::new();

    for entry in std::fs::read_dir(&configs_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            let config_path = path.join("default.toml");
            if config_path.exists() {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .ok_or("Invalid directory name")?
                    .to_string();

                configs.push(DiscoveredConfig { name, config_path });
            }
        }
    }

    // Sort by name for consistent ordering
    configs.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(configs)
}

/// Find a specific experiment configuration by name and optional variant.
///
/// # Arguments
/// - `name`: Directory name under configs/ (e.g., "iris_baseline")
/// - `variant`: Config file name without .toml extension (default: "default")
///
/// # Examples
/// ```ignore
/// // Uses configs/iris_baseline/default.toml
/// find_config("iris_baseline", "default")?;
///
/// // Uses configs/iris_baseline/optimal.toml
/// find_config("iris_baseline", "optimal")?;
/// ```
pub fn find_config(
    name: &str,
    variant: &str,
) -> Result<DiscoveredConfig, Box<dyn std::error::Error>> {
    let configs_dir = get_configs_dir();
    let config_dir = configs_dir.join(name);
    let config_filename = format!("{}.toml", variant);
    let config_path = config_dir.join(&config_filename);

    if !config_path.exists() {
        return Err(format!(
            "Config '{}/{}' not found. Expected: {}",
            name,
            config_filename,
            config_path.display()
        )
        .into());
    }

    Ok(DiscoveredConfig {
        name: name.to_string(),
        config_path,
    })
}
