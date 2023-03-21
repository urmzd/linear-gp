use std::{iter::repeat_with, marker::PhantomData};

use crate::utils::random::generator;
use clap::Args;
use derive_new::new;
use rand::{
    distributions::Uniform,
    prelude::{Distribution, IteratorRandom},
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{
    characteristics::{Breed, DuplicateNew, FitnessScore, Generate, Mutate},
    inputs::ValidInput,
    instruction::{Instruction, InstructionGeneratorParameters},
    instructions::Instructions,
    registers::Registers,
};

#[derive(Clone, Debug, new, Args, Deserialize, Serialize)]
#[serde(bound = "T: ValidInput")]
pub struct ProgramGeneratorParameters<T>
where
    T: ValidInput,
{
    #[arg(long, default_value = "12")]
    pub max_instructions: usize,
    #[command(flatten)]
    pub instruction_generator_parameters: InstructionGeneratorParameters<T>,
    #[arg(skip)]
    #[serde(skip)]
    marker: PhantomData<T>,
}

impl<T> Copy for ProgramGeneratorParameters<T> where T: ValidInput {}

impl<T> Clone for Program<T>
where
    T: ProgramParameters,
{
    fn clone(&self) -> Self {
        Self {
            instructions: self.instructions.clone(),
            registers: self.registers.clone(),
            fitness: self.fitness.clone(),
            marker: self.marker.clone(),
            id: self.id.clone(),
        }
    }
}

impl<T> DuplicateNew for Program<T>
where
    T: ProgramParameters,
{
    fn duplicate_new(&self) -> Self {
        Self {
            instructions: self.instructions.clone(),
            registers: self.registers.duplicate_new(),
            fitness: self.fitness.clone(),
            marker: self.marker.clone(),
            id: Uuid::new_v4(),
        }
    }
}

#[derive(Debug, new, Serialize, Deserialize)]
pub struct Program<T>
where
    T: ProgramParameters,
{
    pub id: Uuid,
    pub instructions: Instructions<T::InputType>,
    pub registers: Registers,
    pub fitness: FitnessScore,
    #[serde(skip)]
    marker: PhantomData<T>,
}

impl<T> PartialEq for Program<T>
where
    T: ProgramParameters,
{
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> PartialOrd for Program<T>
where
    T: ProgramParameters,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        return self.fitness.partial_cmp(&other.fitness);
    }
}

pub trait ProgramParameters
where
    Self::InputType: ValidInput,
{
    type InputType;
}

impl<T> Program<T>
where
    T: ProgramParameters,
{
    pub fn run(&mut self, input: &T::InputType) {
        for instruction in &self.instructions {
            instruction.apply(&mut &mut self.registers, input)
        }
    }
}

impl<T> Generate for Program<T>
where
    T: ProgramParameters,
{
    type GeneratorParameters = ProgramGeneratorParameters<T::InputType>;

    fn generate(parameters: Self::GeneratorParameters) -> Self {
        let ProgramGeneratorParameters {
            max_instructions,
            instruction_generator_parameters,
            ..
        } = parameters;

        let registers = Registers::new(instruction_generator_parameters.n_registers());
        let n_instructions = Uniform::new_inclusive(1, max_instructions).sample(&mut generator());
        let instructions = repeat_with(|| Instruction::generate(instruction_generator_parameters))
            .take(n_instructions)
            .collect();

        Self::new(
            Uuid::new_v4(),
            instructions,
            registers,
            FitnessScore::NotEvaluated,
        )
    }
}

impl<T> Mutate for Program<T>
where
    T: ProgramParameters,
{
    fn mutate(&self, params: Self::GeneratorParameters) -> Self {
        let mut mutated = self.clone();

        // Pick instruction to mutate.
        let instruction = mutated
            .instructions
            .iter_mut()
            .choose(&mut generator())
            .unwrap();

        let mutated_instruction = instruction.mutate(params.instruction_generator_parameters);
        *instruction = mutated_instruction;

        mutated.fitness = FitnessScore::NotEvaluated;
        mutated.id = Uuid::new_v4();
        mutated.registers = mutated.registers.duplicate_new();

        mutated
    }
}

impl<T> Breed for Program<T>
where
    T: ProgramParameters,
{
    fn two_point_crossover(&self, mate: &Self) -> [Self; 2] {
        let child_instructions = self.instructions.two_point_crossover(&mate.instructions);

        child_instructions.map(|instructions| {
            Self::new(
                Uuid::new_v4(),
                instructions,
                self.registers.duplicate_new(),
                FitnessScore::NotEvaluated,
            )
        })
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        core::instruction::InstructionGeneratorParameters,
        extensions::classification::ClassificationParameters, utils::test::TestInput,
    };

    use super::*;

    #[test]
    fn given_instructions_when_breed_then_two_children_are_produced_using_genes_of_parents() {
        let params = InstructionGeneratorParameters::new(1, 10.);
        let instructions_a: Instructions<TestInput> =
            (0..10).map(|_| Instruction::generate(params)).collect();
        let instructions_b: Instructions<TestInput> =
            (0..10).map(|_| Instruction::generate(params)).collect();

        let [child_a, child_b] = instructions_a.two_point_crossover(&instructions_b);

        assert_ne!(child_a, child_b);

        assert_ne!(instructions_a, child_a);
        assert_ne!(instructions_a, child_b);

        assert_ne!(instructions_b, child_a);
        assert_ne!(instructions_b, child_b);
    }

    #[test]
    fn given_programs_when_two_point_crossover_then_two_children_are_produced() {
        let instruction_params = InstructionGeneratorParameters::new(1, 10.);
        let program_params = ProgramGeneratorParameters::new(100, instruction_params);

        let program_a = Program::<ClassificationParameters<TestInput>>::generate(program_params);
        let program_b = Program::<ClassificationParameters<TestInput>>::generate(program_params);

        let [child_a, child_b] = program_a.two_point_crossover(&program_b);

        assert_ne!(child_a, child_b);

        assert_ne!(program_a, child_a);
        assert_ne!(program_a, child_b);

        assert_ne!(program_b, child_a);
        assert_ne!(program_b, child_b);
    }
}
