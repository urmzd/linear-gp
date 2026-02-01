//! Benchmark tools for loading and running programs.

use std::{error::Error, iter::repeat_with, path::PathBuf};

use itertools::Itertools;

use crate::core::{
    characteristics::Load,
    engines::{core_engine::Core, generate_engine::Generate, status_engine::Status},
};

/// Load a program and run it, returning (original_fitness, new_fitness).
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
