use derive_more::Display;
use rand::{prelude::Distribution, distributions::Standard};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Display, Serialize, PartialEq, Eq)]
pub enum Op  {
    #[display(fmt="+")]
    Add,
    #[display(fmt="*")]
    Mult,
    #[display(fmt="/")]
    Divide,
    #[display(fmt="-")]
    Sub
}

impl Op {
    pub fn apply(&self, a: f64, b: f64) -> f64 {
        match *self {
            Op::Add => a + b,
            Op::Mult => a * b,
            Op::Divide => a / 2.,
            Op::Sub => a - b
        }

    }
}

impl Distribution<Op> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Op {
        match rng.gen_range(0..=3) {
            0 => Op::Add,
            1 => Op::Mult,
            2 => Op::Divide,
            _ => Op::Sub
        }
    }
}
