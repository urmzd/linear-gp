// For testing purposes only (binary classification).

use derive_new::new;
use noisy_float::prelude::r64;
use rand::{distributions::Standard, prelude::Distribution};
use strum::EnumCount;

use crate::{
    core::{
        algorithm::GeneticAlgorithm, inputs::ValidInput, program::Program, registers::RegisterValue,
    },
    extensions::classification::{ClassificationInput, ClassificationParameters},
};

#[derive(PartialEq, PartialOrd, Clone, Debug, new)]
pub struct TestInput(pub [RegisterValue; 5]);

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Clone, EnumCount)]
pub enum TestRepresent {
    One = 0,
    Two = 1,
}

impl ValidInput for TestInput {
    const N_INPUT_REGISTERS: usize = 4;
    const N_ACTION_REGISTERS: usize = 2;

    fn flat(&self) -> Vec<RegisterValue> {
        vec![self.0[0], self.0[1], self.0[2], self.0[3]]
    }
}

impl ClassificationInput for TestInput {
    fn get_class(&self) -> usize {
        self.0[Self::N_INPUT_REGISTERS].const_raw() as usize
    }
}

pub struct TestLgp;
impl GeneticAlgorithm for TestLgp {
    type O = Program<ClassificationParameters<TestInput>>;
}

impl Default for TestInput {
    fn default() -> Self {
        TestInput::new([r64(0.); 5])
    }
}

impl Distribution<TestInput> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> TestInput {
        let data: [RegisterValue; 5] = [0.; 5].map(|_| r64(rng.gen_range((0.)..=(1.))));
        TestInput(data)
    }
}
