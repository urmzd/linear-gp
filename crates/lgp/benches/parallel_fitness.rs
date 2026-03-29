use std::iter::repeat_with;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use lgp::{
    core::{
        engines::{
            core_engine::Core, fitness_engine::Fitness, generate_engine::Generate,
            reset_engine::Reset, status_engine::Status,
        },
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    },
    problems::iris::{IrisEngine, IrisState},
};
use rayon::prelude::*;

/// Sequential fitness evaluation (pre-parallelization baseline).
fn eval_fitness_sequential<C: Core>(
    population: &mut Vec<C::Individual>,
    trials: &[C::State],
    default_fitness: f64,
) {
    let n_trials = trials.len();
    for individual in population.iter_mut() {
        let total: f64 = trials
            .iter()
            .cloned()
            .map(|mut trial| {
                C::Reset::reset(individual);
                C::Reset::reset(&mut trial);
                let score = C::Fitness::eval_fitness(individual, &mut trial);
                if score.is_finite() {
                    score
                } else {
                    default_fitness
                }
            })
            .sum();
        C::Status::set_fitness(individual, total / n_trials as f64);
    }
}

/// Parallel fitness evaluation (current implementation).
fn eval_fitness_parallel<C: Core>(
    population: &mut Vec<C::Individual>,
    trials: &[C::State],
    default_fitness: f64,
) {
    let n_trials = trials.len();
    population.par_iter_mut().for_each(|individual| {
        let total: f64 = trials
            .iter()
            .cloned()
            .map(|mut trial| {
                C::Reset::reset(individual);
                C::Reset::reset(&mut trial);
                let score = C::Fitness::eval_fitness(individual, &mut trial);
                if score.is_finite() {
                    score
                } else {
                    default_fitness
                }
            })
            .sum();
        C::Status::set_fitness(individual, total / n_trials as f64);
    });
}

fn parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("fitness_evaluation");

    let n_trials = 5;
    let default_fitness = 0.0;

    let trials: Vec<IrisState> = repeat_with(|| <IrisEngine as Core>::Generate::generate(()))
        .take(n_trials)
        .collect_vec();

    for pop_size in [50, 100, 200, 500] {
        let population: Vec<_> = repeat_with(|| {
            <IrisEngine as Core>::Generate::generate(ProgramGeneratorParameters {
                max_instructions: 100,
                instruction_generator_parameters: InstructionGeneratorParameters {
                    n_extras: 1,
                    external_factor: 10.0,
                    n_actions: 3,
                    n_inputs: 4,
                },
            })
        })
        .take(pop_size)
        .collect();

        group.bench_with_input(
            BenchmarkId::new("sequential", pop_size),
            &pop_size,
            |b, _| {
                b.iter_batched(
                    || population.clone(),
                    |mut pop| {
                        eval_fitness_sequential::<IrisEngine>(&mut pop, &trials, default_fitness);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(BenchmarkId::new("parallel", pop_size), &pop_size, |b, _| {
            b.iter_batched(
                || population.clone(),
                |mut pop| {
                    eval_fitness_parallel::<IrisEngine>(&mut pop, &trials, default_fitness);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, parallel_vs_sequential);
criterion_main!(benches);
