//! Example command: run Rust examples

use clap::Args;
use std::fs;
use std::path::Path;
use std::process::Command;

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
        println!("No examples directory found");
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
        println!("No examples found in examples/");
        return Ok(());
    }

    examples.sort();

    println!("Available examples:");
    for example in examples {
        println!("  {}", example);
    }
    println!();
    println!("Run with: lgp example <name>");

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

    println!("Running example: {}", name);
    println!();

    let status = Command::new("cargo")
        .args(["run", "--example", name, "--release"])
        .status()?;

    if !status.success() {
        return Err(format!(
            "Example '{}' failed with exit code: {:?}",
            name,
            status.code()
        )
        .into());
    }

    Ok(())
}
