use criterion::{criterion_group, criterion_main, Criterion};
use glob::glob;
use gym_rs::envs::classical_control::{cartpole::CartPoleEnv, mountain_car::MountainCarEnv};
use lgp::{
    problems::gym::{GymRsEngine, GymRsQEngine},
    utils::benchmark_tools::load_and_run_program,
};

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

        for entry in glob(&format!(
            "assets/experiments/**/benchmarks/{}/best.json",
            program_type
        ))
        .unwrap()
        {
            if let Ok(path) = entry {
                let parent = path.parent().unwrap().parent().unwrap().parent().unwrap();
                let iteration_count = parent
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .replace("iteration_", "")
                    .parse::<u32>()
                    .unwrap();

                let bench_id = format!("perf_{}_iteration_{}", program_type, iteration_count);

                c.bench_function(&bench_id, |b| {
                    b.iter(|| {
                        let (new_fitness, original_fitness) = match program_type {
                            &"mountain_car_q" => load_and_run_program::<
                                GymRsQEngine<MountainCarEnv, 2, 3>,
                            >(&path, n_trials)
                            .unwrap(),

                            &"mountain_car_lgp" => load_and_run_program::<
                                GymRsEngine<MountainCarEnv, 2, 3>,
                            >(&path, n_trials)
                            .unwrap(),

                            &"cart_pole_q" => {
                                load_and_run_program::<GymRsQEngine<CartPoleEnv, 4, 2>>(
                                    &path, n_trials,
                                )
                                .unwrap()
                            }
                            &"cart_pole_lgp" => load_and_run_program::<
                                GymRsEngine<CartPoleEnv, 4, 2>,
                            >(&path, n_trials)
                            .unwrap(),
                            _ => panic!("Unknown program type"),
                        };

                        let improvement = new_fitness - original_fitness;

                        if improvement > 0. {
                            better_count += 1;
                        }

                        improvement_values.push(improvement);
                    });
                });
            }
        }

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
