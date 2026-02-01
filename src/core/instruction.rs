use clap::Args;
use derivative::Derivative;
use derive_builder::Builder;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::utils::random::generator;

use super::engines::generate_engine::{Generate, GenerateEngine};
use super::engines::mutate_engine::{Mutate, MutateEngine};
use super::environment::State;
use super::registers::Registers;
use derive_more::Display;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Copy, Deserialize)]
pub enum Mode {
    External,
    Internal,
}

#[derive(Clone, Copy, Debug, Display, Serialize, PartialEq, Eq, Deserialize)]
pub enum Op {
    #[display(fmt = "+")]
    Add,
    #[display(fmt = "*")]
    Mult,
    #[display(fmt = "/")]
    Divide,
    #[display(fmt = "-")]
    Sub,
}

impl Op {
    pub fn apply(&self, a: f64, b: f64) -> f64 {
        match *self {
            Op::Add => a + b,
            Op::Mult => a * b,
            Op::Divide => a / 2.,
            Op::Sub => a - b,
        }
    }
}

impl Distribution<Op> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Op {
        match rng.gen_range(0..=3) {
            0 => Op::Add,
            1 => Op::Mult,
            2 => Op::Divide,
            _ => Op::Sub,
        }
    }
}

impl Distribution<Mode> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Mode {
        match rng.gen_bool(0.5) {
            false => Mode::External,
            true => Mode::Internal,
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
    #[arg(skip)]
    pub n_actions: usize,
    #[arg(skip)]
    pub n_inputs: usize,
}

impl InstructionGeneratorParameters {
    pub fn n_registers(&self) -> usize {
        // Mountain Car Example: | -1 | 0 | 1 | Extra |
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
        let src_idx = generator().gen_range(0..using.n_registers());

        let mode = generator().gen();

        let upper_bound_target_index = if mode == Mode::External {
            using.n_inputs
        } else {
            using.n_registers()
        };

        let target_index = generator().gen_range(0..upper_bound_target_index);

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
    fn mutate(instruction: &mut Instruction, using: InstructionGeneratorParameters) {
        let mutated = GenerateEngine::generate(using);

        let swap_target = generator().gen();
        let swap_source = generator().gen();
        let swap_exec = generator().gen();

        // Flip a Coin: Target
        if swap_target {
            instruction.mode = mutated.mode;
            instruction.tgt_idx = mutated.tgt_idx;
        }

        // Flip a Coin: Source
        if swap_source {
            instruction.src_idx = mutated.src_idx;
        }

        // Flip a Coin: Executable
        if swap_exec {
            instruction.op = mutated.op;
        }
    }
}

impl Instruction {
    pub fn apply(&self, registers: &mut Registers, input: &impl State) {
        let target_value = match self.mode {
            Mode::External => self.external_factor * input.get_value(self.tgt_idx),
            _ => *registers.get(self.tgt_idx),
        };

        let source_value = *registers.get(self.src_idx);
        let new_source_value = self.op.apply(source_value, target_value);

        registers.update(self.src_idx, new_source_value);
    }
}
