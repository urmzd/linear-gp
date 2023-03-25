use clap::Args;
use derivative::Derivative;
use derive_builder::Builder;
use rand::distributions::uniform::{UniformInt, UniformSampler};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::utils::executables::Op;
use crate::utils::random::generator;

use super::engines::generate_engine::{Generate, GenerateEngine};
use super::engines::mutate_engine::{Mutate, MutateEngine};
use super::inputs::ValidInput;
use super::registers::Registers;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Copy, Deserialize)]
pub enum Mode {
    External,
    Internal,
}

impl Distribution<Mode> for Standard {
    fn sample<R: Rng + ?Sized>(rng: &mut R) -> Mode {
        let mode_repr = UniformInt::<usize>::new_inclusive(0, 1).sample(rng);

        if mode_repr == 0 {
            Mode::External
        } else {
            Mode::Internal
        }
    }
}

#[derive(Clone, Derivative, Debug, Serialize, Args, PartialEq, Deserialize, Builder)]
#[derivative(Copy)]
pub struct InstructionGeneratorParameters {
    #[arg(long, default_value = "1")]
    #[builder(default = "1")]
    pub n_extras: usize,
    #[arg(long, default_value = "10.")]
    #[builder(default = "10.")]
    pub external_factor: f64,
    pub n_actions: usize,
    pub n_inputs: usize,
}

impl InstructionGeneratorParameters {
    pub fn n_registers(&self) -> usize {
        // | -1 | 0 | 1 | Extra |
        self.n_actions + self.n_extras
    }
}

#[derive(Serialize, PartialEq, Debug, Deserialize, Derivative)]
#[derivative(Copy, Clone)]
pub struct Instruction {
    src_idx: usize,
    tgt_idx: usize,
    mode: Mode,
    op: Op,
    external_factor: f64,
}

impl Generate<InstructionGeneratorParameters, Instruction> for GenerateEngine {
    fn generate(using: InstructionGeneratorParameters) -> Instruction {
        let current_generator = &mut generator();

        let src_idx = UniformInt::<usize>::new(0, using.n_registers()).sample(current_generator);

        let mode = generator().gen();

        let upper_bound_target_index = if mode == Mode::External {
            using.n_inputs
        } else {
            using.n_registers()
        };

        let target_index =
            UniformInt::<usize>::new(0, upper_bound_target_index).sample(current_generator);

        let executable = generator().gen();

        Instruction {
            src_idx,
            tgt_idx: target_index,
            mode,
            op: executable,
            external_factor: using.external_factor,
        }
    }
}

impl Mutate<InstructionGeneratorParameters, Instruction> for MutateEngine {
    fn mutate(instruction: &mut Instruction, using: InstructionGeneratorParameters) -> Self {
        let mut mutated = GenerateEngine::generate(using);
        let cloned_object = instruction.clone();

        let swap_target = generator().gen();
        let swap_source = generator().gen();
        let swap_exec = generator().gen();

        // Flip a Coin: Target
        if swap_target {
            cloned_object.mode = mutated.clone();
            mutated.tgt_idx = mutated.tgt_idx;
        }

        // Flip a Coin: Source
        if swap_source {
            cloned_object.src_idx = mutated.src_idx;
        }

        // Flip a Coin: Executable
        if swap_exec {
            mutated.op = mutated.op;
        }

        mutated
    }
}

impl Instruction {
    pub fn apply<'b>(&self, registers: &'b mut Registers, input: &impl ValidInput) {
        let target_data = if self.mode == Mode::External {
            Registers::from(input)
        } else {
            registers.clone()
        };

        let target_value = *target_data.get(self.tgt_idx);

        let amplied_target_value = if self.mode == Mode::External {
            self.external_factor * target_value
        } else {
            target_value
        };

        let source_value = *registers.get(self.src_idx);
        let new_source_value = self.op.apply(source_value, amplied_target_value);

        registers.update(self.src_idx, new_source_value);
    }
}
