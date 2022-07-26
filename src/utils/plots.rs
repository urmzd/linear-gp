use std::{error, fmt, ops::Range};

use plotters::{
    prelude::{BitMapBackend, ChartBuilder, IntoDrawingArea, LineSeries, Rectangle},
    style::{Color, IntoFont, Palette, Palette99, BLACK, WHITE},
};

use crate::core::{characteristics::Fitness, population::Population};

pub fn plot_population_benchmarks<T>(
    populations: Vec<Population<T>>,
    plot_path: &str,
    y_range: Range<f64>,
) -> Result<(), Box<dyn error::Error>>
where
    T: Fitness + Clone + PartialOrd + fmt::Debug,
{
    let root = BitMapBackend::new(plot_path, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let n_benchmarks = populations.len();

    let mut chart = ChartBuilder::on(&root)
        .caption("Fitness Over Generations", ("sans-serif", 50).into_font())
        .margin(5u32)
        .x_label_area_size(30u32)
        .y_label_area_size(30u32)
        .build_cartesian_2d(0..n_benchmarks, y_range.clone())?;

    chart.configure_mesh().draw()?;

    let benchmarks: Vec<[f64; 3]> = populations
        .into_iter()
        .map(|population| {
            let best = population.first();
            let median = population.middle();
            let worst = population.last();

            let benchmark = [best, median, worst]
                .map(|quantile| quantile.unwrap().get_fitness().unwrap_or(y_range.start));

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
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}
