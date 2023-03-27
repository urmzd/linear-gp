use clap::Parser;
use lgp::core::config::Accuator;

fn main() {
    let mut cli = Accuator::parse();
    cli.run();
}
