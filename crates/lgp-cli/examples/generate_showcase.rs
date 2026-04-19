//! Generate showcase comparison figure from Iris experiment outputs.
//!
//! Run all four Iris experiments first, then:
//!   cargo run -p lgp --example generate_showcase --features plot

use plotters::prelude::*;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

struct FitnessSeries {
    best: Vec<f64>,
    #[allow(dead_code)]
    median: Vec<f64>,
    worst: Vec<f64>,
}

fn extract_fitness(individual: &Value) -> f64 {
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

fn load_fitness_series(population_path: &Path) -> FitnessSeries {
    let data = fs::read_to_string(population_path).expect("failed to read population.json");
    let generations: Vec<Vec<Value>> = serde_json::from_str(&data).expect("invalid JSON");

    let mut best = Vec::with_capacity(generations.len());
    let mut median = Vec::with_capacity(generations.len());
    let mut worst = Vec::with_capacity(generations.len());

    for gen in &generations {
        let mut fitnesses: Vec<f64> = gen.iter().map(|ind| extract_fitness(ind)).collect();
        fitnesses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        worst.push(*fitnesses.first().unwrap_or(&0.0));
        median.push(fitnesses[fitnesses.len() / 2]);
        best.push(*fitnesses.last().unwrap_or(&0.0));
    }

    FitnessSeries {
        best,
        median,
        worst,
    }
}

fn find_latest_run(experiment_dir: &Path) -> Option<PathBuf> {
    let mut runs: Vec<PathBuf> = fs::read_dir(experiment_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    runs.sort();
    runs.last().cloned()
}

fn main() {
    let experiments: Vec<(&str, &str)> = vec![
        ("Baseline (no operators)", "iris_baseline"),
        ("Mutation only", "iris_mutation"),
        ("Crossover only", "iris_crossover"),
        ("Full (mut + xover)", "iris_full"),
    ];

    let colors: Vec<RGBColor> = vec![
        RGBColor(139, 148, 158), // gray
        RGBColor(240, 136, 62),  // orange
        RGBColor(88, 166, 255),  // blue
        RGBColor(63, 185, 80),   // green
    ];

    let outputs_dir = Path::new("outputs");
    let mut series_map: BTreeMap<String, (FitnessSeries, RGBColor)> = BTreeMap::new();

    for (i, (label, name)) in experiments.iter().enumerate() {
        let exp_dir = outputs_dir.join(name);
        if !exp_dir.exists() {
            eprintln!("Skipping {} — no output directory found", name);
            continue;
        }
        let run_dir = match find_latest_run(&exp_dir) {
            Some(d) => d,
            None => {
                eprintln!("Skipping {} — no runs found", name);
                continue;
            }
        };
        let pop_path = run_dir.join("outputs").join("population.json");
        if !pop_path.exists() {
            eprintln!("Skipping {} — no population.json", name);
            continue;
        }
        let series = load_fitness_series(&pop_path);
        series_map.insert(label.to_string(), (series, colors[i]));
    }

    if series_map.is_empty() {
        eprintln!("No experiment data found. Run the Iris experiments first.");
        std::process::exit(1);
    }

    fs::create_dir_all("showcase").expect("failed to create showcase/");

    // --- Figure 1: Fitness over generations comparison ---
    let n_gens = series_map
        .values()
        .map(|(s, _)| s.best.len())
        .max()
        .unwrap_or(1);

    let out_path = "showcase/iris-comparison.png";
    let root = BitMapBackend::new(out_path, (960, 600)).into_drawing_area();
    root.fill(&RGBColor(13, 17, 23)).unwrap();

    let bg = RGBColor(22, 27, 34);
    let grid_color = RGBColor(33, 38, 45);
    let text_color = RGBColor(201, 209, 217);
    let title_color = RGBColor(240, 246, 252);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Iris Classification — Genetic Operator Comparison",
            ("sans-serif", 22).into_font().color(&title_color),
        )
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(65)
        .build_cartesian_2d(0usize..n_gens.saturating_sub(1), -0.05f64..1.1f64)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Fitness (accuracy)")
        .x_label_style(("sans-serif", 14).into_font().color(&text_color))
        .y_label_style(("sans-serif", 14).into_font().color(&text_color))
        .x_label_formatter(&|x| format!("{}", x))
        .y_label_formatter(&|y| format!("{:.0}%", y * 100.0))
        .axis_desc_style(("sans-serif", 15).into_font().color(&text_color))
        .axis_style(grid_color)
        .light_line_style(grid_color)
        .bold_line_style(grid_color)
        .set_all_tick_mark_size(0)
        .draw()
        .unwrap();

    // Fill background
    chart.plotting_area().fill(&bg).unwrap();

    // Redraw mesh on top of background
    chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Fitness (accuracy)")
        .x_label_style(("sans-serif", 14).into_font().color(&text_color))
        .y_label_style(("sans-serif", 14).into_font().color(&text_color))
        .x_label_formatter(&|x| format!("{}", x))
        .y_label_formatter(&|y| format!("{:.0}%", y * 100.0))
        .axis_desc_style(("sans-serif", 15).into_font().color(&text_color))
        .axis_style(grid_color)
        .light_line_style(grid_color)
        .bold_line_style(grid_color)
        .set_all_tick_mark_size(0)
        .draw()
        .unwrap();

    for (label, (series, color)) in &series_map {
        // Shaded region between worst and best
        let area_data: Vec<(usize, f64, f64)> = (0..series.best.len())
            .map(|i| (i, series.worst[i], series.best[i]))
            .collect();
        chart
            .draw_series(area_data.iter().map(|&(x, low, high)| {
                Rectangle::new([(x, low), (x + 1, high)], color.mix(0.15).filled())
            }))
            .unwrap();

        // Best fitness line
        let best_points: Vec<(usize, f64)> = series
            .best
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        chart
            .draw_series(LineSeries::new(best_points, color.stroke_width(2)))
            .unwrap()
            .label(label.as_str())
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(3))
            });
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .margin(12)
        .background_style(bg.mix(0.9))
        .border_style(grid_color)
        .label_font(("sans-serif", 13).into_font().color(&text_color))
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Saved {}", out_path);

    // --- Figure 2: Final accuracy bar chart ---
    let bar_path = "showcase/final-accuracy.png";
    let root2 = BitMapBackend::new(bar_path, (960, 380)).into_drawing_area();
    root2.fill(&RGBColor(13, 17, 23)).unwrap();

    let labels: Vec<String> = series_map.keys().cloned().collect();
    let values: Vec<f64> = series_map
        .values()
        .map(|(s, _)| *s.best.last().unwrap_or(&0.0))
        .collect();
    let bar_colors: Vec<RGBColor> = series_map.values().map(|(_, c)| *c).collect();
    let n_bars = labels.len();

    // Use f64 x-axis for precise bar positioning with padding
    let mut chart2 = ChartBuilder::on(&root2)
        .caption(
            "Final Best Accuracy by Experiment",
            ("sans-serif", 20).into_font().color(&title_color),
        )
        .margin(20)
        .margin_bottom(10)
        .x_label_area_size(55)
        .y_label_area_size(65)
        .build_cartesian_2d(0.0f64..(n_bars as f64), 0.0f64..1.15)
        .unwrap();

    chart2
        .configure_mesh()
        .disable_x_mesh()
        .disable_x_axis()
        .y_desc("Accuracy")
        .y_label_formatter(&|y| format!("{:.0}%", y * 100.0))
        .y_label_style(("sans-serif", 14).into_font().color(&text_color))
        .axis_desc_style(("sans-serif", 15).into_font().color(&text_color))
        .axis_style(grid_color)
        .light_line_style(grid_color)
        .bold_line_style(grid_color)
        .set_all_tick_mark_size(0)
        .draw()
        .unwrap();

    chart2.plotting_area().fill(&bg).unwrap();

    // Redraw mesh on filled background
    chart2
        .configure_mesh()
        .disable_x_mesh()
        .disable_x_axis()
        .y_desc("Accuracy")
        .y_label_formatter(&|y| format!("{:.0}%", y * 100.0))
        .y_label_style(("sans-serif", 14).into_font().color(&text_color))
        .axis_desc_style(("sans-serif", 15).into_font().color(&text_color))
        .axis_style(grid_color)
        .light_line_style(grid_color)
        .bold_line_style(grid_color)
        .set_all_tick_mark_size(0)
        .draw()
        .unwrap();

    let pad = 0.15; // padding on each side of bar
    for (i, (val, color)) in values.iter().zip(bar_colors.iter()).enumerate() {
        let x0 = i as f64 + pad;
        let x1 = (i + 1) as f64 - pad;

        chart2
            .draw_series(std::iter::once(Rectangle::new(
                [(x0, 0.0), (x1, *val)],
                color.mix(0.85).filled(),
            )))
            .unwrap();

        // Value label on top of bar
        let label_text = format!("{:.1}%", val * 100.0);
        let center_x = (x0 + x1) / 2.0;
        chart2
            .draw_series(std::iter::once(Text::new(
                label_text,
                (center_x - 0.15, val + 0.03),
                ("sans-serif", 15).into_font().color(&title_color),
            )))
            .unwrap();

        // X-axis label below bar
        chart2
            .draw_series(std::iter::once(Text::new(
                labels[i].clone(),
                (center_x - 0.35, -0.06),
                ("sans-serif", 11).into_font().color(&text_color),
            )))
            .unwrap();
    }

    root2.present().unwrap();
    println!("Saved {}", bar_path);
}
