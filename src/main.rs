use clap::Parser;
use lgp::core::config::Accuator;

fn main() {
    let cli = Accuator::parse();
    cli.run();
}
