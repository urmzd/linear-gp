// use clap::{Args, Parser, ValueEnum};
// use lgp::{
//     core::{
//         algorithm::{GeneticAlgorithm, HyperParameters},
//         characteristics::Fitness,
//         input_engine::InputSource,
//         instruction::InstructionGeneratorParameters,
//         program::ProgramGeneratorParameters,
//     },
//     extensions::{
//         interactive::{ILgp, InteractiveLearningParameters, InteractiveLearningParametersArgs},
//         q_learning::{QConsts, QProgram, QProgramGeneratorParameters},
//     },
//     problems::{cart_pole::CartPoleInput, mountain_car::MountainCarInput},
// };

// #[derive(Parser, Debug)]
// #[command(author, version, about, long_about=None)]
// #[command(propagate_version = true)]
// struct Cli<T>
// where
//     T: InputSource,
// {
//     #[arg(value_enum)]
//     problem_type: ProblemType,
//     #[arg(value_enum)]
//     learning_type: LearningType,
//     #[command(flatten)]
//     basic_args: BasicArgs,
//     #[command(flatten)]
//     program_parameter: ProgramGeneratorParameters<T>,
// }

// #[derive(Args, Clone, Debug)]
// struct BasicArgs {
//     #[arg(long, default_value = "100")]
//     population_size: usize,
//     #[arg(long, default_value = "0.5")]
//     gap: f64,
//     #[arg(long, default_value = "0.5")]
//     mutation_percent: f64,
//     #[arg(long, default_value = "0.5")]
//     crossover_percent: f64,
//     #[arg(long, default_value = "100")]
//     n_generations: usize,
//     #[command(flatten)]
//     consts: QConsts,
//     #[arg(long, default_value = "5")]
//     n_trials: usize,
// }

// #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
// pub enum ProblemType {
//     MountainCar,
//     CartPole,
// }

// #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
// pub enum LearningType {
//     Q,
//     Norm,
// }

// macro_rules! generate_program_params {
//     ($cli:expr, $input_type:ty, $use_q_params:expr) => {{
//         let max_instructions = $cli.program_parameter.max_instructions;
//         let instruction_generator_parameters = InstructionGeneratorParameters::new(
//             $cli.program_parameter
//                 .instruction_generator_parameters
//                 .n_extras,
//             $cli.program_parameter
//                 .instruction_generator_parameters
//                 .external_factor,
//         );
//         let prog_parameters =
//             ProgramGeneratorParameters::new(max_instructions, instruction_generator_parameters);

//         QProgramGeneratorParameters::new(prog_parameters, $cli.basic_args.consts)
//     }};
//     ($cli: expr, $input_type:ty) => {{
//         let max_instructions = $cli.program_parameter.max_instructions;
//         let instruction_generator_parameters = InstructionGeneratorParameters::new(
//             $cli.program_parameter
//                 .instruction_generator_parameters
//                 .n_extras,
//             $cli.program_parameter
//                 .instruction_generator_parameters
//                 .external_factor,
//         );
//         let prog_parameters =
//             ProgramGeneratorParameters::new(max_instructions, instruction_generator_parameters);

//         prog_parameters
//     }};
// }

// macro_rules! run_lgp {
//     ($cli: expr, $input_type:ty, $prog_params:expr, $gp: ty) => {
//         $cli.basic_args.consts.reset_active_properties();

//         for population in ILgp::<$gp>::build(HyperParameters {
//             population_size: $cli.basic_args.population_size,
//             gap: $cli.basic_args.gap,
//             crossover_percent: $cli.basic_args.crossover_percent,
//             mutation_percent: $cli.basic_args.mutation_percent,
//             n_generations: $cli.basic_args.n_generations,
//             fitness_parameters: InteractiveLearningParameters::<$input_type>::new(
//                 InteractiveLearningParametersArgs::new(
//                     $cli.basic_args.n_generations,
//                     $cli.basic_args.n_trials,
//                 ),
//             ),
//             program_parameters: $prog_params,
//         }) {
//             if let Some(program) = population.best() {
//                 let fitness = program.get_fitness().unwrap_or(f64::NAN);
//                 println!("{}", fitness);
//             }
//         }
//     };
// }

// fn main() {
//     let mut cli: Cli<MountainCarInput> = Cli::parse();

//     if cli.problem_type == ProblemType::MountainCar {
//         if cli.learning_type == LearningType::Q {
//             let program_params = generate_program_params!(cli, MountainCarInput, true);
//             run_lgp!(
//                 cli,
//                 MountainCarInput,
//                 program_params,
//                 QProgram<MountainCarInput>
//             );
//         } else {
//             let program_params = generate_program_params!(cli, MountainCarInput);
//             run_lgp!(cli, MountainCarInput, program_params, MountainCarInput);
//         }
//     } else {
//         if cli.learning_type == LearningType::Q {
//             let program_params = generate_program_params!(cli, CartPoleInput, true);
//             run_lgp!(cli, CartPoleInput, program_params, QProgram<CartPoleInput>);
//         } else {
//             let program_params = generate_program_params!(cli, CartPoleInput);
//             run_lgp!(cli, CartPoleInput, program_params, CartPoleInput);
//         }
//     }
// }

fn main() {
    // Example Usage
    // GenerateEngine -> Programs
    // FitnessEngine -> Eval
    // GenerateEngine -> State
    // MutateEngine + BreedEngine (variation operators)
    // Environment + State from Generate Engine -> Trials
    println!("Hello World!")
}
