// For testing purposes only (binary classification).

use rand::{distributions::Standard, prelude::Distribution};
use serde::{Deserialize, Serialize};
use strum::EnumCount;

use crate::core::{
    engines::reset_engine::{Reset, ResetEngine},
    environment::State,
};

#[derive(
    Eq, PartialEq, Ord, PartialOrd, Hash, Clone, EnumCount, Deserialize, Serialize, Debug, Copy,
)]
pub enum TestOutput {
    One = 0,
    Two = 1,
}

#[derive(PartialEq, PartialOrd, Clone, Debug, Deserialize, Serialize)]
pub struct TestInput {
    data: Vec<SingleInput>,
    idx: usize,
}

#[derive(PartialEq, PartialOrd, Clone, Debug, Deserialize, Serialize)]
struct SingleInput {
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
        let item = &self.data[self.idx];

        match at_idx {
            0 => item.a,
            1 => item.b,
            2 => item.c,
            3 => item.d,
            _ => unreachable!(),
        }
    }

    fn execute_action(&mut self, action: usize) -> f64 {
        let class = self.data[self.idx].e as usize;
        self.idx += 1;
        (action == class) as usize as f64
    }

    fn get(&mut self) -> Option<&mut Self> {
        if self.idx >= self.data.len() {
            return None;
        }
        return Some(self);
    }
}

impl Reset<TestInput> for ResetEngine {
    fn reset(item: &mut TestInput) {
        item.idx = 0;
    }
}

impl Default for TestInput {
    fn default() -> Self {
        TestInput {
            data: vec![SingleInput {
                a: 0.,
                b: 0.,
                c: 0.,
                d: 0.,
                e: TestOutput::One,
            }],
            idx: 0,
        }
    }
}

impl Distribution<SingleInput> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> SingleInput {
        let data: [f64; 4] = [0.0; 4].map(|_| rng.gen_range(0.0..=1.0));
        SingleInput {
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
