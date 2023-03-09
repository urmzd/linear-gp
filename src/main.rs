use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(author, version, about, long_about=None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
    #[arg(value_enum)]
    problem_type: ProblemType,
    #[arg(value_enum)]
    learning_type: LearningType,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum ProblemType {
    MountainCar,
    CartPole,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum LearningType {
    Q,
    Gp,
}

#[derive(Subcommand)]
enum Commands {
    QLearning {
        alpha: f64,
        alpha_decay: f64,
        gamma: f64,
        epsilon: f64,
        epsilon_decay: f64,
    },
}

fn main() {}
