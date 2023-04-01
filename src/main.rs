use clap::Parser;
use lgp::core::config::Actuator;

fn main() {
    let mut cli = Actuator::parse();
    cli.run();
}
