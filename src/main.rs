use clap::{Args, Parser, ValueEnum};
use lgp::{
    core::{
        algorithm::{GeneticAlgorithm, HyperParameters},
        characteristics::Fitness,
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    },
    extensions::{
        interactive::{InteractiveLearningInput, InteractiveLearningParameters},
        q_learning::{QConsts, QLgp, QProgramGeneratorParameters},
    },
    problems::{cart_pole::CartPoleInput, mountain_car::MountainCarInput},
};

#[derive(Parser)]
#[command(author, version, about, long_about=None)]
#[command(propagate_version = true)]
struct Cli {
    #[arg(value_enum)]
    problem_type: ProblemType,
    #[command(flatten)]
    basic_args: BasicArgs,
    #[command(flatten)]
    program_parameter: ProgramGeneratorParameters,
}

#[derive(Args, Clone)]
struct BasicArgs {
    #[arg(long, default_value = "100")]
    population_size: usize,
    #[arg(long, default_value = "0.5")]
    gap: f64,
    #[arg(long, default_value = "0.5")]
    mutation_percent: f64,
    #[arg(long, default_value = "0.5")]
    crossover_percent: f64,
    #[arg(long, default_value = "100")]
    n_generations: usize,
    #[command(flatten)]
    consts: QConsts,
    #[arg(long, default_value = "5")]
    n_trials: usize,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum ProblemType {
    MountainCar,
    CartPole,
}

fn main() {
    let cli = Cli::parse();

    if cli.problem_type == ProblemType::MountainCar {
        let input = MountainCarInput::new();
        let n_generations = cli.basic_args.n_generations;
        let initial_states =
            MountainCarInput::get_initial_states(n_generations, cli.basic_args.n_trials);

        let hyper_params = HyperParameters {
            population_size: cli.basic_args.population_size,
            gap: cli.basic_args.gap,
            crossover_percent: cli.basic_args.crossover_percent,
            mutation_percent: cli.basic_args.mutation_percent,
            n_generations,
            fitness_parameters: InteractiveLearningParameters::new(initial_states, input),
            program_parameters: QProgramGeneratorParameters::new(
                ProgramGeneratorParameters::new(
                    cli.program_parameter.max_instructions,
                    InstructionGeneratorParameters::from::<MountainCarInput>(
                        cli.program_parameter
                            .instruction_generator_parameters
                            .n_extras,
                        cli.program_parameter
                            .instruction_generator_parameters
                            .external_factor,
                    ),
                ),
                cli.basic_args.consts,
            ),
        };

        let best_score = QLgp::build(hyper_params)
            .last()
            .as_ref()
            .and_then(|p| p.best())
            .map(|p| p.get_fitness().unwrap_or(f64::NAN))
            .unwrap_or(f64::NAN);

        println!("{}", best_score)
    } else {
        let input = CartPoleInput::new();
        let n_generations = cli.basic_args.n_generations;
        let initial_states =
            CartPoleInput::get_initial_states(n_generations, cli.basic_args.n_trials);

        let hyper_params = HyperParameters {
            population_size: cli.basic_args.population_size,
            gap: cli.basic_args.gap,
            crossover_percent: cli.basic_args.crossover_percent,
            mutation_percent: cli.basic_args.mutation_percent,
            n_generations,
            fitness_parameters: InteractiveLearningParameters::new(initial_states, input),
            program_parameters: QProgramGeneratorParameters::new(
                ProgramGeneratorParameters::new(
                    cli.program_parameter.max_instructions,
                    InstructionGeneratorParameters::from::<CartPoleInput>(
                        cli.program_parameter
                            .instruction_generator_parameters
                            .n_extras,
                        cli.program_parameter
                            .instruction_generator_parameters
                            .external_factor,
                    ),
                ),
                cli.basic_args.consts,
            ),
        };

        let best_score = QLgp::build(hyper_params)
            .last()
            .as_ref()
            .and_then(|p| p.best())
            .map(|p| p.get_fitness().unwrap_or(f64::NAN))
            .unwrap_or(f64::NAN);

        println!("{}", best_score)
    }
}
