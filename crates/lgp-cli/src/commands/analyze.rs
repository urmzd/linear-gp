//! Analyze command: generate statistics tables and optional plots from experiment results.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use clap::Args;
use tracing::{debug, info, warn};

use crate::ui;

#[cfg(feature = "plot")]
use lgp::core::experiment_config::ExperimentConfig;

#[cfg(feature = "plot")]
use crate::config_discovery::get_configs_dir;

#[derive(Args)]
pub struct AnalyzeArgs {
    /// Directory containing experiment output subdirectories
    #[arg(short, long, default_value = "outputs")]
    pub input: PathBuf,

    /// Directory where tables/ and figures/ will be written
    #[arg(short, long, default_value = "outputs")]
    pub output: PathBuf,
}

/// Per-generation fitness statistics.
struct GenerationStats {
    max: f64,
    mean: f64,
    median: f64,
    min: f64,
}

/// Extract fitness from a serde_json::Value, handling both plain and Q-learning wrapper formats.
fn extract_fitness(individual: &serde_json::Value) -> f64 {
    // Q-learning wrapper: { "fitness": 0.0, "program": { "fitness": <real> } }
    if let Some(inner) = individual.get("program") {
        if let Some(f) = inner.get("fitness").and_then(|v| v.as_f64()) {
            return f;
        }
    }
    individual
        .get("fitness")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

/// Compute stats for one generation of individuals.
fn compute_generation_stats(individuals: &[serde_json::Value]) -> GenerationStats {
    let mut fitnesses: Vec<f64> = individuals.iter().map(extract_fitness).collect();
    fitnesses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = fitnesses.len() as f64;
    let sum: f64 = fitnesses.iter().sum();
    let median = if fitnesses.is_empty() {
        0.0
    } else {
        let mid = fitnesses.len() / 2;
        if fitnesses.len().is_multiple_of(2) {
            (fitnesses[mid - 1] + fitnesses[mid]) / 2.0
        } else {
            fitnesses[mid]
        }
    };

    GenerationStats {
        max: fitnesses.last().copied().unwrap_or(0.0),
        mean: if n > 0.0 { sum / n } else { 0.0 },
        median,
        min: fitnesses.first().copied().unwrap_or(0.0),
    }
}

/// Generate a CSV table from a population.json file inside an experiment run directory.
fn generate_table(
    population_path: &Path,
    experiment_name: &str,
    tables_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let data = std::fs::read_to_string(population_path)?;
    let generations: Vec<Vec<serde_json::Value>> = serde_json::from_str(&data)?;

    std::fs::create_dir_all(tables_dir)?;
    let csv_path = tables_dir.join(format!("{}.csv", experiment_name));

    let mut writer = csv::Writer::from_path(&csv_path)?;
    writer.write_record([
        "Generation",
        "Max Fitness",
        "Mean Fitness",
        "Median Fitness",
        "Min Fitness",
    ])?;

    for (gen_idx, generation) in generations.iter().enumerate() {
        let stats = compute_generation_stats(generation);
        writer.write_record(&[
            gen_idx.to_string(),
            stats.max.to_string(),
            stats.mean.to_string(),
            stats.median.to_string(),
            stats.min.to_string(),
        ])?;
    }

    writer.flush()?;
    info!(path = %csv_path.display(), "Generated table");
    Ok(())
}

/// Generate a PNG plot from a CSV table (only available with `plot` feature).
#[cfg(feature = "plot")]
fn generate_figure(
    csv_path: &Path,
    experiment_name: &str,
    figures_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;

    // Read CSV
    let mut reader = csv::Reader::from_path(csv_path)?;
    let mut generations = Vec::new();
    let mut max_vals = Vec::new();
    let mut mean_vals = Vec::new();
    let mut median_vals = Vec::new();
    let mut min_vals = Vec::new();

    for result in reader.records() {
        let record = result?;
        generations.push(record[0].parse::<f64>()?);
        max_vals.push(record[1].parse::<f64>()?);
        mean_vals.push(record[2].parse::<f64>()?);
        median_vals.push(record[3].parse::<f64>()?);
        min_vals.push(record[4].parse::<f64>()?);
    }

    if generations.is_empty() {
        return Ok(());
    }

    // Load metadata for labels
    let configs_dir = get_configs_dir();
    let config_path = configs_dir.join(experiment_name).join("default.toml");
    let (title, x_label, y_label) = if config_path.exists() {
        match ExperimentConfig::load(&config_path) {
            Ok(config) => (
                config
                    .metadata
                    .title
                    .unwrap_or_else(|| experiment_name.replace('_', " ")),
                config
                    .metadata
                    .x_label
                    .unwrap_or_else(|| "Generation".to_string()),
                config
                    .metadata
                    .y_label
                    .unwrap_or_else(|| "Fitness".to_string()),
            ),
            Err(_) => (
                experiment_name.replace('_', " "),
                "Generation".to_string(),
                "Fitness".to_string(),
            ),
        }
    } else {
        (
            experiment_name.replace('_', " "),
            "Generation".to_string(),
            "Fitness".to_string(),
        )
    };

    // Compute y-axis range
    let all_vals: Vec<f64> = max_vals.iter().chain(min_vals.iter()).copied().collect();
    let y_min = all_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = all_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_margin = (y_max - y_min).abs() * 0.05;
    let x_max = *generations.last().unwrap_or(&1.0);

    std::fs::create_dir_all(figures_dir)?;
    let png_path = figures_dir.join(format!("{}.png", experiment_name));

    let root = BitMapBackend::new(&png_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .right_y_label_area_size(80)
        .build_cartesian_2d(0.0..x_max, (y_min - y_margin)..(y_max + y_margin))?;

    chart
        .configure_mesh()
        .x_desc(&x_label)
        .y_desc(&y_label)
        .draw()?;

    let series_data: Vec<(&str, &[f64], RGBColor)> = vec![
        ("max", &max_vals, RED),
        ("mean", &mean_vals, BLUE),
        ("median", &median_vals, GREEN),
        ("min", &min_vals, MAGENTA),
    ];

    for (label, values, color) in series_data {
        let points: Vec<(f64, f64)> = generations
            .iter()
            .zip(values.iter())
            .map(|(&x, &y)| (x, y))
            .collect();
        chart
            .draw_series(LineSeries::new(points, color.stroke_width(2)))?
            .label(label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
            });
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    info!(path = %png_path.display(), "Generated figure");
    Ok(())
}

pub fn execute(args: &AnalyzeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = &args.input;
    let output_dir = &args.output;
    let tables_dir = output_dir.join("tables");

    if !input_dir.exists() {
        return Err(format!("Input directory not found: {}", input_dir.display()).into());
    }

    ui::header("Analysis");

    // Collect all population.json files across experiment directories.
    // Structure: outputs/<experiment>/<timestamp>/outputs/population.json
    let mut table_count = 0;
    let mut experiment_names: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();

    for entry in std::fs::read_dir(input_dir)? {
        let entry = entry?;
        let exp_path = entry.path();
        if !exp_path.is_dir() {
            continue;
        }
        let exp_name = exp_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        // Skip non-experiment directories
        if exp_name == "tables" || exp_name == "figures" || exp_name == "parameters" {
            continue;
        }

        // Walk timestamp subdirectories
        for ts_entry in std::fs::read_dir(&exp_path)? {
            let ts_entry = ts_entry?;
            let ts_path = ts_entry.path();
            if !ts_path.is_dir() {
                continue;
            }
            let pop_path = ts_path.join("outputs").join("population.json");
            if pop_path.exists() {
                experiment_names
                    .entry(exp_name.clone())
                    .or_default()
                    .push(pop_path);
            }
        }
    }

    if experiment_names.is_empty() {
        return Err("No experiment output directories with population.json found".into());
    }

    let sp = ui::spinner("Generating tables...");

    // For each experiment, use the most recently modified population.json
    for (exp_name, mut pop_files) in experiment_names {
        pop_files.sort();
        if let Some(pop_path) = pop_files.last() {
            debug!(experiment = %exp_name, path = %pop_path.display(), "Processing population");
            match generate_table(pop_path, &exp_name, &tables_dir) {
                Ok(()) => table_count += 1,
                Err(e) => {
                    warn!(experiment = %exp_name, error = %e, "Failed to generate table");
                    sp.suspend(|| {
                        ui::warn(&format!("Failed to generate table for {}: {}", exp_name, e));
                    });
                }
            }
        }
    }

    sp.finish_and_clear();
    ui::phase_ok(&format!(
        "Generated {} tables in {}",
        table_count,
        tables_dir.display()
    ));

    // Generate figures if plot feature is enabled
    #[cfg(feature = "plot")]
    {
        let figures_dir = output_dir.join("figures");
        let csv_files: Vec<_> = std::fs::read_dir(&tables_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "csv")
                    .unwrap_or(false)
            })
            .collect();

        let sp = ui::spinner("Generating figures...");
        let mut fig_count = 0;
        for csv_entry in &csv_files {
            let csv_path = csv_entry.path();
            let exp_name = csv_path.file_stem().and_then(|n| n.to_str()).unwrap_or("");
            match generate_figure(&csv_path, exp_name, &figures_dir) {
                Ok(()) => fig_count += 1,
                Err(e) => {
                    warn!(experiment = %exp_name, error = %e, "Failed to generate figure");
                    sp.suspend(|| {
                        ui::warn(&format!(
                            "Failed to generate figure for {}: {}",
                            exp_name, e
                        ));
                    });
                }
            }
        }
        sp.finish_and_clear();
        ui::phase_ok(&format!(
            "Generated {} figures in {}",
            fig_count,
            figures_dir.display()
        ));
    }

    #[cfg(not(feature = "plot"))]
    {
        ui::info("Plotting disabled (build with --features plot to enable)");
    }

    ui::phase_ok("Analysis complete!");
    Ok(())
}
