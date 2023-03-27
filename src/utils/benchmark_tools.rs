use std::{
    error::Error,
    fs,
    iter::repeat_with,
    path::{Path, PathBuf},
};

use crate::core::{
    characteristics::{Load, Save},
    engines::{
        core_engine::{Core, HyperParameters},
        freeze_engine::Freeze,
        status_engine::Status,
    },
    engines::{fitness_engine::FitnessScore, generate_engine::Generate},
};

use super::misc::VoidResultAnyError;

pub const BENCHMARK_PREFIX: &'static str = "assets/benchmarks/";
pub const LOG_PREFIX: &'static str = "assets/logs/";

#[allow(unused_macros)]
macro_rules! with_named_logger {
    ($name:expr, $($body:tt)*) => {{
        const NAME: &'static str = $name;

        let shared_dir = std::path::Path::new($crate::utils::benchmark_tools::LOG_PREFIX).join(format!("{}/",NAME));

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

use itertools::Itertools;
use serde::Serialize;
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

pub fn save_benchmarks<C>(
    populations: &Vec<Vec<C::Individual>>,
    params: &HyperParameters<C>,
    test_name: &str,
) -> VoidResultAnyError
where
    C: Core,
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

    let last_population = populations.last().unwrap();

    let (mut worst, mut median, mut best) = populations
        .last()
        .map(|p| {
            (
                p.last().cloned().unwrap(),
                p.get(last_population.len() / 2).cloned().unwrap(),
                p.first().cloned().unwrap(),
            )
        })
        .unwrap();

    C::Freeze::freeze(&mut worst);
    C::Freeze::freeze(&mut median);
    C::Freeze::freeze(&mut best);

    worst.save(worst_path.to_str().unwrap())?;
    median.save(median_path.to_str().unwrap())?;
    best.save(best_path.to_str().unwrap())?;
    params.save(params_path.to_str().unwrap())?;

    Ok(())
}

pub fn save_results<T>(populations: &Vec<Vec<T>>, test_name: &str) -> VoidResultAnyError
where
    T: Serialize,
{
    let plot_path = create_path(
        Path::new(LOG_PREFIX)
            .join(test_name)
            .join("population.json")
            .to_str()
            .unwrap(),
        true,
    )?;

    populations.save(plot_path.to_str().unwrap())?;

    Ok(())
}

pub fn load_and_run_program<C>(
    program_path: &str,
    n_trials: usize,
) -> Result<(FitnessScore, FitnessScore), Box<dyn Error>>
where
    C: Core,
{
    let program = C::Individual::load(program_path);
    let original_fitness = C::Status::get_fitness(&program);

    let mut trials: Vec<C::State> = repeat_with(|| C::Generate::generate(()))
        .take(n_trials)
        .collect_vec();

    let mut population = vec![program];
    C::eval_fitness(&mut population, &mut trials);

    let new_fitness = C::Status::get_fitness(population.first().unwrap());

    Ok((original_fitness, new_fitness))
}
