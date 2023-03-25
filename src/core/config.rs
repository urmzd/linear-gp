use config::{Config, Environment, File, FileFormat};

// pub fn load_config<T>(config_path: &str) -> HyperParameters<T>
// where
//     T: Fitness + Generate + Clone,
// {
//     let mut builder = Config::builder();
//     builder = builder.set_default("population_size", 100).unwrap();
//     builder = builder.set_default("gap", 0.5).unwrap();
//     builder = builder.set_default("mutation_percent", 0.5).unwrap();
//     builder = builder.set_default("crossover_percent", 0.5).unwrap();
//     builder = builder.set_default("n_generations", 100).unwrap();
//     builder = builder.add_source(File::new(config_path, FileFormat::Json));
//     builder = builder.add_source(Environment::default());

//     // Add generic parameters and attempt to load into HyperParameters;
//     todo!("hyper parameters")
// }

// pub population_size: usize,
// pub gap: f64,
// pub mutation_percent: f64,
// pub crossover_percent: f64,
// pub n_generations: usize,
// pub fitness_parameters: T::FitnessParameters,
// pub program_parameters: T::GeneratorParameters,
