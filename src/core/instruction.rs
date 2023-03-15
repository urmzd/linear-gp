use clap::Args;
use derive_new::new;
use rand::distributions::uniform::{UniformInt, UniformSampler};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::marker::PhantomData;

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

#[derive(Clone, Debug, Serialize, new, Args, PartialEq, Deserialize)]
pub struct InstructionGeneratorParameters<T>
where
    T: ValidInput,
{
    #[arg(long, default_value = "1")]
    pub n_extras: usize,
    #[arg(long, default_value = "10.")]
    pub external_factor: f64,
    #[arg(skip)]
    marker: PhantomData<T>,
}

impl<T> Copy for InstructionGeneratorParameters<T> where T: ValidInput {}
impl<T> Copy for Instruction<T> where T: ValidInput {}

impl<T> InstructionGeneratorParameters<T>
where
    T: ValidInput,
{
    pub fn n_registers(&self) -> usize {
        // | A1 | A2 | A3 | Extra |
        T::N_ACTIONS + self.n_extras
    }

    pub fn n_inputs(&self) -> usize {
        T::N_INPUTS
    }

    pub fn n_actions(&self) -> usize {
        T::N_ACTIONS
    }
}

#[derive(Serialize, Clone, PartialEq, Debug, Deserialize, new)]
pub struct Instruction<T> {
    source_index: usize,
    target_index: usize,
    mode: Mode,
    executable: Op,
    external_factor: f64,
    marker: PhantomData<T>,
}

impl<T> Generate for Instruction<T>
where
    T: ValidInput,
{
    type GeneratorParameters = InstructionGeneratorParameters<T>;

    fn generate(parameters: Self::GeneratorParameters) -> Self {
        let current_generator = &mut generator();

        let source_index =
            UniformInt::<usize>::new(0, parameters.n_registers()).sample(current_generator);

        let mode = Mode::sample(current_generator);

        let upper_bound_target_index = if mode == Mode::External {
            T::N_INPUTS
        } else {
            parameters.n_registers()
        };
        let target_index =
            UniformInt::<usize>::new(0, upper_bound_target_index).sample(current_generator);

        let executable = generator().gen();

        Instruction::new(
            source_index,
            target_index,
            mode,
            executable,
            parameters.external_factor,
        )
    }
}

impl<T> Mutate for Instruction<T>
where
    T: ValidInput,
{
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

impl<T> Instruction<T> {
    pub fn apply<'b>(&self, registers: &'b mut Registers, input: &'b T)
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
