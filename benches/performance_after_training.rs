use criterion::{criterion_group, criterion_main, Criterion};
use gym_rs::envs::classical_control::mountain_car::MountainCarEnv;
use lgp::{problems::gym::GymRsQEngine, utils::benchmark_tools::load_and_run_program};

const TYPES: &'static [&str] = &[
    "mountain_car_q",
    "mountain_car_lgp",
    "cart_pole_q",
    "cart_pole_lgp",
];

fn performance_benchmark(c: &mut Criterion) {
    let n_trials = 5;
    let n_iterations = 100;

    for program_type in TYPES {
        let mut better_count = 0;
        let mut improvement_values = Vec::with_capacity(n_iterations);

        let path = format!("assets/benchmarks/{}/best.json", program_type);

        c.bench_function(
            &format!("performance_after_trained_{}", program_type),
            |b| {
                b.iter(|| {
                    let (new_fitness, original_fitness) =
                        load_and_run_program::<GymRsQEngine<MountainCarEnv, 2, 3>>(&path, n_trials)
                            .unwrap();

                    let improvement = new_fitness - original_fitness;

                    if improvement > 0. {
                        better_count += 1;
                    }

                    improvement_values.push(improvement);
                });
            },
        );

        let mean = improvement_values.iter().sum::<f64>() / improvement_values.len() as f64;
        let median = {
            let mut sorted = improvement_values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let variance = improvement_values
            .iter()
            .map(|v| (v - mean).powi(2))
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
