use std::{
    error::Error,
    fs,
    ops::Range,
    path::{Path, PathBuf},
};

use plotters::{
    prelude::{BitMapBackend, ChartBuilder, IntoDrawingArea, LineSeries, Rectangle},
    style::{Color, IntoFont, Palette, Palette99, BLACK, WHITE},
};

use crate::core::{characteristics::Organism, population::Population};

use super::types::VoidResultAnyError;

const BENCHMARK_PREFIX: &'static str = "assets/benchmarks/";

pub fn create_path(path: &str) -> Result<PathBuf, Box<dyn Error>> {
    let path = Path::new(path);

    if path.is_dir() {
        fs::create_dir_all(path)?;
        return Ok(path.to_owned());
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::File::create(path)?;

    Ok(path.to_owned())
}

pub fn log_benchmarks<T>(population: &Vec<Population<T>>, test_name: &str) -> VoidResultAnyError
where
    T: Organism,
{
    let best_path = create_path(
        Path::new(BENCHMARK_PREFIX)
            .join(test_name)
            .join("best.json")
            .to_str()
            .unwrap(),
    )?;

    let median_path = create_path(
        Path::new(BENCHMARK_PREFIX)
            .join(test_name)
            .join("median.json")
            .to_str()
            .unwrap(),
    )?;

    let worst_path = create_path(
        Path::new(BENCHMARK_PREFIX)
            .join(test_name)
            .join("worst.json")
            .to_str()
            .unwrap(),
    )?;

    let (worst, median, best) = population
        .last()
        .map(|p| (p.worst(), p.median(), p.best()))
        .unwrap();

    worst.unwrap().save(worst_path.to_str().unwrap())?;
    median.unwrap().save(median_path.to_str().unwrap())?;
    best.unwrap().save(best_path.to_str().unwrap())?;

    Ok(())
}

pub fn plot_benchmarks<T>(
    populations: &Vec<Population<T>>,
    test_name: &str,
    y_range: Range<f64>,
) -> VoidResultAnyError
where
    T: Organism,
{
    let plot_path = create_path(
        Path::new(BENCHMARK_PREFIX)
            .join(format!("{}.png", test_name))
            .to_str()
            .unwrap(),
    )?;

    let root = BitMapBackend::new(plot_path.as_path(), (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let n_benchmarks = populations.len();

    let mut chart = ChartBuilder::on(&root)
        .caption("Benchmark", ("sans-serif", 50).into_font())
        .margin(15)
        .x_label_area_size(100)
        .y_label_area_size(100)
        .margin(20)
        .build_cartesian_2d(0..n_benchmarks, y_range.clone())?;

    chart
        .configure_mesh()
        .y_desc("Fitness")
        .x_desc("Generation")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    let benchmarks: Vec<[f64; 3]> = populations
        .into_iter()
        .map(|population| {
            let best = population.best();
            let median = population.median();
            let worst = population.worst();

            let benchmark = [best, median, worst].map(|quantile| {
                quantile
                    .map(|v| v.get_fitness().unwrap_or(y_range.start))
                    .expect("Population should not be empty.")
            });

            benchmark
        })
        .collect();

    for (idx, label) in [(0, "Best"), (1, "Median"), (2, "Worst")] {
        let color = Palette99::pick(idx).mix(0.9);

        chart
            .draw_series(LineSeries::new(
                benchmarks.iter().enumerate().map(|(i, b)| (i, b[idx])),
                color.stroke_width(3),
            ))?
            .label(label)
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));
    }

    chart
        .configure_series_labels()
        .background_style(Color::filled(&WHITE.mix(0.9)))
        .legend_area_size(50)
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
