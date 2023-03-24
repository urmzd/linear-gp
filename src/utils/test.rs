// For testing purposes only (binary classification).

use rand::{distributions::Standard, prelude::Distribution};
use serde::{Deserialize, Serialize};
use strum::EnumCount;

use crate::{
    core::{algorithm::GeneticAlgorithm, inputs::ValidInput, program::Program},
    extensions::classification::{ClassificationInput, ClassificationParameters},
};

#[derive(PartialEq, PartialOrd, Clone, Debug, Deserialize, Serialize)]
pub struct TestInput(pub [f64; 5]);

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Clone, EnumCount, Deserialize, Serialize)]
pub enum TestRepresent {
    One = 0,
    Two = 1,
}

impl ValidInput for TestInput {
    const N_INPUTS: usize = 4;
    const N_ACTIONS: usize = 2;

    fn get(&self, idx: usize) -> f64 {
        match idx {
            0..=3 => self.0[idx],
            _ => panic!("Idx out of range."),
        }
    }
}

impl ClassificationInput for TestInput {
    fn get_class(&self) -> usize {
        self.0[Self::N_INPUTS] as usize
    }
}

impl Default for TestInput {
    fn default() -> Self {
        TestInput([0.; 5])
    }
}

impl Distribution<TestInput> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> TestInput {
        let data: [f64; 5] = [0.; 5].map(|_| rng.gen_range((0.)..=(1.)));
        TestInput(data)
    }
}

pub struct TestLgp;

impl GeneticAlgorithm for TestLgp {
    type O = Program<ClassificationParameters<TestInput>>;
}
