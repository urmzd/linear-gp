use std::iter::repeat_with;

use crate::utils::random::generator;
use clap::Args;
use derivative::Derivative;
use derive_builder::Builder;
use rand::{
    distributions::Uniform,
    prelude::{Distribution, IteratorRandom},
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{
    characteristics::{Breed, FitnessScore, Generate, Mutate, Reset, ResetNew},
    inputs::ValidInput,
    instruction::{Instruction, InstructionGeneratorParameters},
    instructions::Instructions,
    registers::Registers,
};

#[derive(Clone, Debug, Args, Deserialize, Serialize, Derivative, Builder)]
#[derivative(Copy)]
pub struct ProgramGeneratorParameters {
    #[arg(long, default_value = "12")]
    pub max_instructions: usize,
    #[command(flatten)]
    pub instruction_generator_parameters: InstructionGeneratorParameters,
}

impl Reset for Program {
    fn reset(&mut self) {
        self.registers.reset();
        self.fitness.reset();
        self.id = Uuid::new_v4();
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Derivative)]
#[derivative(PartialEq, PartialOrd)]
pub struct Program {
    #[derivative(PartialOrd = "ignore")]
    pub id: Uuid,
    #[derivative(PartialEq = "ignore", PartialOrd = "ignore")]
    pub instructions: Instructions,
    #[derivative(PartialEq = "ignore", PartialOrd = "ignore")]
    pub registers: Registers,
    #[derivative(PartialEq = "ignore")]
    pub fitness: FitnessScore,
}

pub trait ProgramParameters
where
    Self::InputType: ValidInput,
{
    type InputType;
}

impl Program {
    pub fn run(&mut self, input: &impl ValidInput) {
        for instruction in &self.instructions {
            instruction.apply(&mut self.registers, input)
        }
    }
}

impl Generate for ProgramGeneratorParameters {
    type Output = Program;

    fn generate(&self) -> Self {
        let ProgramGeneratorParameters {
            max_instructions,
            instruction_generator_parameters,
            ..
        } = self;

        let registers = Registers::new(instruction_generator_parameters.n_registers());
        let n_instructions = Uniform::new_inclusive(1, max_instructions).sample(&mut generator());
        let instructions = repeat_with(|| Instruction::generate(instruction_generator_parameters))
            .take(n_instructions)
            .collect();

        Program {
            id: Uuid::new_v4(),
            instructions,
            registers,
            fitness: FitnessScore::NotEvaluated,
        }
    }
}

impl Mutate for ProgramGeneratorParameters {
    type Input = Program;
    fn mutate(&self, params: Self::Input) -> Self {
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
        mutated.registers = mutated.registers.reset_new();

        mutated
    }
}

impl Breed for Program {
    fn two_point_crossover(&self, mate: &Self) -> [Self; 2] {
        let child_instructions = self.instructions.two_point_crossover(&mate.instructions);

        child_instructions.map(|instructions| {
            let mut parent = self.reset_new();
            parent.instructions = instructions;
            parent
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
        let params = InstructionGeneratorParameters {
            n_extras: 1,
            external_factor: 10.,
            n_actions: TestInput::N_ACTIONS,
            n_inputs: TestInput::N_INPUTS,
        };
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
        let instruction_generator_parameters = InstructionGeneratorParameters {
            n_extras: 1,
            external_factor: 10.,
            n_actions: TestInput::N_ACTIONS,
            n_inputs: TestInput::N_INPUTS,
        };
        let program_params = ProgramGeneratorParameters {
            max_instructions: 100,
            instruction_generator_parameters,
        };

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
