// For testing purposes only (binary classification).

use rand::{distributions::Standard, prelude::Distribution};
use serde::{Deserialize, Serialize};
use strum::EnumCount;

use crate::core::input_engine::State;

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Clone, EnumCount, Deserialize, Serialize, Debug)]
pub enum TestOutput {
    One = 0,
    Two = 1,
}

#[derive(PartialEq, PartialOrd, Clone, Debug, Deserialize, Serialize)]
pub struct TestInput {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: TestOutput,
}

impl State for TestInput {
    const N_INPUTS: usize = 4;
    const N_ACTIONS: usize = 2;

    fn get_value(&self, at_idx: usize) -> f64 {
        match at_idx {
            0 => self.a,
            1 => self.b,
            2 => self.c,
            3 => self.d,
            _ => unreachable!(),
        }
    }

    fn execute_action(&mut self, action: usize) -> f64 {
        (action == self.e as usize) as usize as f64
    }
}

impl Default for TestInput {
    fn default() -> Self {
        TestInput {
            a: 0.,
            b: 0.,
            c: 0.,
            d: 0.,
            e: TestOutput::One,
        }
    }
}

impl Distribution<TestInput> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> TestInput {
        let data: [f64; 4] = [0.0; 4].map(|_| rng.gen_range(0.0..=1.0));
        TestInput {
            a: data[0],
            b: data[1],
            c: data[2],
            d: data[3],
            e: rng.gen(),
        }
    }
}

impl Distribution<TestOutput> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> TestOutput {
        match rng.gen_bool(0.5) {
            true => TestOutput::One,
            false => TestOutput::Two,
        }
    }
}
