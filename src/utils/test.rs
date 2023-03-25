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
    inputs: Vec<SingleInput>,
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

impl Iterator for TestInput {
    type Item = TestInput;

    fn next(&mut self) -> Option<Self::Item> {
        let current_state = if self.idx < self.inputs.len() {
            Some(Self {
                inputs: self.inputs,
                idx: self.idx,
            })
        } else {
            return None;
        };

        self.idx += 1;

        current_state
    }
}

impl State for TestInput {
    const N_INPUTS: usize = 4;
    const N_ACTIONS: usize = 2;

    fn get_value(&self, at_idx: usize) -> f64 {
        let item = self.inputs[self.idx];

        match at_idx {
            0 => item.a,
            1 => item.b,
            2 => item.c,
            3 => item.d,
            _ => unreachable!(),
        }
    }

    fn execute_action(&mut self, action: usize) -> f64 {
        (action == self.inputs[self.idx].e as usize) as usize as f64
    }

    fn reset(&mut self) {
        self.idx = 0;
    }
}

impl Default for TestInput {
    fn default() -> Self {
        TestInput {
            inputs: vec![SingleInput {
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
