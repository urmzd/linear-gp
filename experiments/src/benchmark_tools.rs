use std::{env, error::Error, iter::repeat_with, path::{Path, PathBuf}};

use lgp::core::{
    characteristics::{Load, Save},
    engines::generate_engine::Generate,
    engines::{
        core_engine::{Core, HyperParameters},
        freeze_engine::Freeze,
        status_engine::Status,
    },
};
use lgp::utils::misc::create_path;

use itertools::Itertools;

pub type VoidResultAnyError = Result<(), Box<dyn Error>>;

pub fn benchmark_prefix() -> String {
    env::var("BENCHMARK_PREFIX").unwrap_or_else(|_| "experiments/assets/output".to_string())
}

pub fn log_prefix() -> String {
    env::var("LOG_PREFIX").unwrap_or_else(|_| "experiments/assets/logs".to_string())
}

pub fn save_experiment<C>(
    populations: &Vec<Vec<C::Individual>>,
    params: &HyperParameters<C>,
    test_name: &str,
) -> VoidResultAnyError
where
    C: Core,
{
    let best_path = create_path(
        Path::new(&benchmark_prefix())
            .join(test_name)
            .join("best.json")
            .to_str()
            .unwrap(),
        true,
    )?;

    let median_path = create_path(
        Path::new(&benchmark_prefix())
            .join(test_name)
            .join("median.json")
            .to_str()
            .unwrap(),
        true,
    )?;

    let worst_path = create_path(
        Path::new(&benchmark_prefix())
            .join(test_name)
            .join("worst.json")
            .to_str()
            .unwrap(),
        true,
    )?;

    let params_path = create_path(
        Path::new(&benchmark_prefix())
            .join(test_name)
            .join("params.json")
            .to_str()
            .unwrap(),
        true,
    )?;

    let plot_path = create_path(
        Path::new(&benchmark_prefix())
            .join(test_name)
            .join("population.json")
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
    populations.save(plot_path.to_str().unwrap())?;

    Ok(())
}

pub fn load_and_run_program<C>(
    program_path: impl Into<PathBuf> + Clone,
    n_trials: usize,
    default_fitness: f64,
) -> Result<(f64, f64), Box<dyn Error>>
where
    C: Core,
{
    let program = C::Individual::load(program_path);
    let original_fitness = C::Status::get_fitness(&program);

    let mut trials: Vec<C::State> = repeat_with(|| C::Generate::generate(()))
        .take(n_trials)
        .collect_vec();

    let mut population = vec![program];
    C::eval_fitness(&mut population, &mut trials, default_fitness);

    let new_fitness = C::Status::get_fitness(population.first().unwrap());

    Ok((original_fitness, new_fitness))
}
