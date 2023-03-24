use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use crate::core::{
    algorithm::HyperParameters,
    characteristics::{Organism, Save},
    population::Population,
};

use super::types::VoidResultAnyError;

pub const BENCHMARK_PREFIX: &'static str = "assets/benchmarks/";

#[allow(unused_macros)]
macro_rules! with_named_logger {
    ($name:expr, $($body:tt)*) => {{
        const NAME: &'static str = $name;

        let shared_dir = std::path::Path::new($crate::utils::benchmark_tools::BENCHMARK_PREFIX).join(format!("{}/",NAME));

        $crate::utils::benchmark_tools::create_path(shared_dir.to_str().unwrap(), false).unwrap();

        let file_appender = tracing_appender::rolling::hourly(shared_dir, "default.log");
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

        let subscriber = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .json()
        .with_writer(non_blocking)
        .finish();

        tracing::subscriber::with_default(subscriber, || {
            $($body)*
        })
    }};
}

#[allow(unused_imports)]
pub(crate) use with_named_logger;

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

pub fn log_benchmarks<T>(
    population: &Vec<Population<T>>,
    params: &HyperParameters<T>,
    test_name: &str,
) -> VoidResultAnyError
where
    T: Organism,
{
    let best_path = create_path(
        Path::new(BENCHMARK_PREFIX)
            .join(test_name)
            .join("best.json")
            .to_str()
            .unwrap(),
        true,
    )?;

    let median_path = create_path(
        Path::new(BENCHMARK_PREFIX)
            .join(test_name)
            .join("median.json")
            .to_str()
            .unwrap(),
        true,
    )?;

    let worst_path = create_path(
        Path::new(BENCHMARK_PREFIX)
            .join(test_name)
            .join("worst.json")
            .to_str()
            .unwrap(),
        true,
    )?;

    let params_path = create_path(
        Path::new(BENCHMARK_PREFIX)
            .join(test_name)
            .join("params.json")
            .to_str()
            .unwrap(),
        true,
    )?;

    let (worst, median, best) = population
        .last()
        .map(|p| (p.worst(), p.median(), p.best()))
        .unwrap();

    worst.unwrap().save(worst_path.to_str().unwrap())?;
    median.unwrap().save(median_path.to_str().unwrap())?;
    best.unwrap().save(best_path.to_str().unwrap())?;
    params.save(params_path.to_str().unwrap())?;

    Ok(())
}

pub fn output_benchmarks<T>(populations: &Vec<Population<T>>, test_name: &str) -> VoidResultAnyError
where
    T: Organism,
{
    let plot_path = create_path(
        Path::new(BENCHMARK_PREFIX)
            .join(test_name)
            .join("plot.json")
            .to_str()
            .unwrap(),
        true,
    )?;

    populations.save(plot_path.to_str().unwrap())?;

    Ok(())
}
