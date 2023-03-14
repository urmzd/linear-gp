use clap::Args;
use derive_new::new;
use rand::distributions::uniform::{UniformInt, UniformSampler};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::utils::executables::Op;
use crate::utils::random::generator;

use super::characteristics::{Generate, Mutate};
use super::inputs::ValidInput;
use super::registers::Registers;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Copy, Deserialize)]
pub enum Mode {
    External,
    Internal,
}

impl Mode {
    fn sample<R: Rng + ?Sized>(rng: &mut R) -> Mode {
        let mode_repr = UniformInt::<usize>::new_inclusive(0, 1).sample(rng);

        if mode_repr == 0 {
            Mode::External
        } else {
            Mode::Internal
        }
    }
}

#[derive(Clone, Debug, Serialize, new, Copy, Args, PartialEq)]
pub struct InstructionGeneratorParameters {
    #[arg(skip)]
    n_input_features: usize,
    #[arg(skip)]
    n_input_classes: usize,
    #[arg(long, default_value = "1")]
    pub n_extras: usize,
    #[arg(long, default_value = "10.")]
    pub external_factor: f64,
}

impl InstructionGeneratorParameters {
    pub fn from<T: ValidInput>(n_extras: usize, external_factor: f64) -> Self {
        InstructionGeneratorParameters::new(
            T::N_INPUT_REGISTERS,
            T::N_ACTION_REGISTERS,
            n_extras,
            external_factor,
        )
    }

    pub fn n_registers(&self) -> usize {
        self.n_extras + self.n_input_classes
    }

    pub fn n_inputs(&self) -> usize {
        self.n_input_features
    }

    pub fn n_actions(&self) -> usize {
        self.n_input_classes
    }
}

#[derive(Serialize, Clone, Copy, PartialEq, Debug, Deserialize)]
pub struct Instruction {
    source_index: usize,
    target_index: usize,
    mode: Mode,
    executable: Op,
    external_factor: f64,
}

impl Generate for Instruction {
    type GeneratorParameters = InstructionGeneratorParameters;

    fn generate(parameters: Self::GeneratorParameters) -> Self {
        let current_generator = &mut generator();

        let source_index =
            UniformInt::<usize>::new(0, parameters.n_registers()).sample(current_generator);

        let mode = Mode::sample(current_generator);

        let upper_bound_target_index = if mode == Mode::External {
            parameters.n_input_features
        } else {
            parameters.n_registers()
        };
        let target_index =
            UniformInt::<usize>::new(0, upper_bound_target_index).sample(current_generator);

        let executable = generator().gen();

        Instruction {
            source_index,
            target_index,
            executable,
            mode,
            external_factor: parameters.external_factor,
        }
    }
}

impl Mutate for Instruction {
    fn mutate(&self, params: Self::GeneratorParameters) -> Self {
        let mut mutated = Self::generate(params);

        let swap_target = generator().gen_bool(0.5);
        let swap_source = generator().gen_bool(0.5);
        let swap_exec = generator().gen_bool(0.5);

        // Flip a Coin: Target
        if swap_target {
            mutated.mode = self.mode.clone();
            mutated.target_index = self.target_index;
        }

        // Flip a Coin: Source
        if swap_source {
            mutated.source_index = self.source_index;
        }

        // Flip a Coin: Executable
        if swap_exec {
            mutated.executable = self.executable.clone();
        }

        mutated
    }
}

impl Instruction {
    pub fn apply<'b, T>(&self, registers: &'b mut Registers, input: &'b T)
    where
        T: ValidInput,
    {
        let target_data = if self.mode == Mode::External {
            Registers::from(input)
        } else {
            registers.clone()
        };

        let target_value = *target_data.get(self.target_index);

        let amplied_target_value = if self.mode == Mode::External {
            self.external_factor * target_value
        } else {
            target_value
        };

        let source_value = *registers.get(self.source_index);
        let new_source_value = self.executable.apply(source_value, amplied_target_value);

        registers.update(self.source_index, new_source_value);
    }
}
