//! Example command: run Rust examples

use clap::Args;
use std::fs;
use std::path::Path;
use std::process::Command;

use crate::ui;

#[derive(Args)]
pub struct ExampleArgs {
    /// Example name to run (without .rs extension)
    #[arg(required_unless_present = "list")]
    pub name: Option<String>,

    /// List available examples
    #[arg(short, long)]
    pub list: bool,
}

pub fn execute(args: &ExampleArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.list {
        list_examples()
    } else if let Some(name) = &args.name {
        run_example(name)
    } else {
        Err("Either provide an example name or use --list".into())
    }
}

fn list_examples() -> Result<(), Box<dyn std::error::Error>> {
    let examples_dir = Path::new("examples");

    if !examples_dir.exists() {
        ui::warn("No examples directory found");
        return Ok(());
    }

    let mut examples: Vec<String> = fs::read_dir(examples_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()?.to_str()? == "rs" {
                path.file_stem()?.to_str().map(String::from)
            } else {
                None
            }
        })
        .collect();

    if examples.is_empty() {
        ui::warn("No examples found in examples/");
        return Ok(());
    }

    examples.sort();

    ui::header("Available examples");
    for example in examples {
        ui::line(&example);
    }
    ui::info("Run with: lgp example <name>");

    Ok(())
}

fn run_example(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let example_path = Path::new("examples").join(format!("{}.rs", name));

    if !example_path.exists() {
        return Err(format!(
            "Example '{}' not found. Use 'lgp example --list' to see available examples.",
            name
        )
        .into());
    }

    ui::header(&format!("Running example: {}", name));

    let sp = ui::spinner("Compiling and running...");
    let status = Command::new("cargo")
        .args(["run", "--example", name, "--release"])
        .status()?;
    sp.finish_and_clear();

    if !status.success() {
        return Err(format!(
            "Example '{}' failed with exit code: {:?}",
            name,
            status.code()
        )
        .into());
    }

    ui::phase_ok(&format!("Example '{}' completed", name));

    Ok(())
}
