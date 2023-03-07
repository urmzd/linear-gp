use derive_new::new;
use rand::distributions::uniform::{UniformInt, UniformSampler};
use rand::prelude::SliceRandom;
use rand::Rng;
use serde::Serialize;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;

use crate::utils::executables::{Op, DEFAULT_EXECUTABLES};
use crate::utils::random::generator;

use super::characteristics::{Generate, Mutate};
use super::inputs::ValidInput;
use super::registers::Registers;

/// The EXTERNAL_FACTOR is a constant float value that is used to amplify the effect of external inputs on the program's execution.
/// Specifically, in the Instruction struct's apply method, if the Mode is Mode::External,
/// then the target value (i.e., the value in the register that the instruction writes to) is multiplied by the EXTERNAL_FACTOR.
/// This can be useful for giving more weight to external inputs, allowing the program to better respond to changes in the environment.
const EXTERNAL_FACTOR: f64 = 10.;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
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

#[derive(Clone, Debug, Serialize, new)]
pub struct InstructionGeneratorParameters {
    n_input_features: usize,
    n_input_classes: usize,
    n_extras: usize,
}

impl InstructionGeneratorParameters {
    pub fn from<T: ValidInput>(n_extras: usize) -> Self {
        InstructionGeneratorParameters::new(T::N_INPUT_REGISTERS, T::N_ACTION_REGISTERS, n_extras)
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

#[derive(Serialize, Eq, Clone)]
pub struct Instruction {
    source_index: usize,
    target_index: usize,
    mode: Mode,
    #[serde(skip_serializing)]
    executable: Op,
}

impl Generate for Instruction {
    type GeneratorParameters = InstructionGeneratorParameters;

    fn generate(parameters: &Self::GeneratorParameters) -> Self {
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

        let executable = DEFAULT_EXECUTABLES
            .choose(current_generator)
            .unwrap()
            .to_owned();

        Instruction {
            source_index,
            target_index,
            executable,
            mode,
        }
    }
}

impl PartialEq for Instruction {
    fn eq(&self, other: &Self) -> bool {
        self.source_index == other.source_index
            && self.target_index == other.target_index
            && self.mode == other.mode
            && self.executable as usize == other.executable as usize
    }
}

impl Debug for Instruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Instruction")
            .field("mode", &self.mode)
            .field("source_index", &self.source_index)
            .field("target_index", &self.target_index)
            .finish()
    }
}

impl Mutate for Instruction {
    fn mutate(&self, params: &Self::GeneratorParameters) -> Self {
        let mut mutated = Self::generate(&params);

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
            EXTERNAL_FACTOR * target_value
        } else {
            target_value
        };

        let source_value = *registers.get(self.source_index);

        let new_source_value = (self.executable)(source_value, amplied_target_value);
        registers.update(self.source_index, new_source_value);
    }
}
