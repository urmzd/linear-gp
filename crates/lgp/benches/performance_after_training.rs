use criterion::{criterion_group, criterion_main, Criterion};
use glob::glob;
use gymnasia::envs::classical_control::{cartpole::CartPoleEnv, mountain_car::MountainCarEnv};
use lgp::{
    problems::gym::{GymRsEngine, GymRsQEngine},
    utils::benchmark_tools::load_and_run_program,
};

/// Maps benchmark type to the config directory name used by the experiment runner.
const TYPES: &[(&str, &str)] = &[
    ("mountain_car_q", "mountain_car_with_q"),
    ("mountain_car_lgp", "mountain_car_lgp"),
    ("cart_pole_q", "cart_pole_with_q"),
    ("cart_pole_lgp", "cart_pole_lgp"),
];

fn performance_benchmark(c: &mut Criterion) {
    let n_trials = 5;

    for (program_type, config_dir) in TYPES {
        let mut better_count = 0;
        let mut improvement_values = Vec::new();

        let pattern = format!("outputs/{}/**/outputs/best.json", config_dir);

        for path in glob(&pattern).unwrap().flatten() {
            // Layout: outputs/<name>/<run_id>/outputs/best.json
            let run_id = path
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            let bench_id = format!("perf_{}_{}", program_type, run_id);

            c.bench_function(&bench_id, |b| {
                b.iter(|| {
                    let (new_fitness, original_fitness) = match *program_type {
                        "mountain_car_q" => load_and_run_program::<GymRsQEngine<MountainCarEnv>>(
                            &path, n_trials, -200.,
                        )
                        .unwrap(),

                        "mountain_car_lgp" => load_and_run_program::<GymRsEngine<MountainCarEnv>>(
                            &path, n_trials, -200.,
                        )
                        .unwrap(),

                        "cart_pole_q" => {
                            load_and_run_program::<GymRsQEngine<CartPoleEnv>>(&path, n_trials, 500.)
                                .unwrap()
                        }
                        "cart_pole_lgp" => {
                            load_and_run_program::<GymRsEngine<CartPoleEnv>>(&path, n_trials, 500.)
                                .unwrap()
                        }
                        _ => panic!("Unknown program type"),
                    };

                    let improvement = new_fitness - original_fitness;

                    if improvement.is_finite() {
                        better_count += 1;
                    }

                    improvement_values.push(improvement);
                });
            });
        }

        if improvement_values.is_empty() {
            eprintln!(
                "No benchmark data found for '{}' (glob: {}). Skipping.",
                program_type, pattern
            );
            continue;
        }

        let mean =
            improvement_values.iter().cloned().sum::<f64>() / improvement_values.len() as f64;
        let median = {
            let mut sorted = improvement_values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let variance = improvement_values
            .iter()
            .map(|v| (*v - mean).powi(2))
            .sum::<f64>()
            / (improvement_values.len() - 1) as f64;
        let std_deviation = variance.sqrt();

        println!("Benchmark for '{}'", program_type);
        println!(
            "Number of times new_fitness is better than original_fitness: {}",
            better_count
        );
        println!("Mean improvement: {}", mean);
        println!("Median improvement: {}", median);
        println!("Standard deviation: {}", std_deviation);
        println!("----------------------------------------------");
    }
}

criterion_group!(benches, performance_benchmark);
criterion_main!(benches);
