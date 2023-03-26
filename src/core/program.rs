use std::iter::repeat_with;

use crate::utils::random::generator;
use clap::Args;
use derivative::Derivative;
use derive_builder::Builder;
use rand::{seq::IteratorRandom, Rng};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{
    engines::{
        breed_engine::{Breed, BreedEngine},
        fitness_engine::FitnessScore,
        generate_engine::{Generate, GenerateEngine},
        mutate_engine::{Mutate, MutateEngine},
        reset_engine::{Reset, ResetEngine},
        status_engine::{Status, StatusEngine},
    },
    environment::State,
    instruction::InstructionGeneratorParameters,
    instructions::Instructions,
    registers::Registers,
};

#[derive(Clone, Debug, Args, Deserialize, Serialize, Derivative, Builder)]
#[derivative(Copy)]
pub struct ProgramGeneratorParameters {
    #[arg(long, default_value = "12")]
    #[builder(default = "12")]
    pub max_instructions: usize,
    #[command(flatten)]
    pub instruction_generator_parameters: InstructionGeneratorParameters,
}

impl Reset<Program> for ResetEngine {
    fn reset(item: &mut Program) {
        ResetEngine::reset(&mut item.registers);
        ResetEngine::reset(&mut item.fitness);
    }
}

impl Status<Program> for StatusEngine {
    fn set_fitness(program: &mut Program, fitness: FitnessScore) {
        program.fitness = fitness;
    }

    fn valid(item: &Program) -> bool {
        item.fitness.is_valid()
    }

    fn evaluated(item: &Program) -> bool {
        item.fitness.is_evaluated()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Derivative, Builder)]
#[derivative(PartialEq, PartialOrd, Ord, Eq)]
pub struct Program {
    #[derivative(PartialOrd = "ignore", Ord = "ignore")]
    pub id: Uuid,
    #[derivative(PartialEq = "ignore", PartialOrd = "ignore", Ord = "ignore")]
    pub instructions: Instructions,
    #[derivative(PartialEq = "ignore", PartialOrd = "ignore", Ord = "ignore")]
    pub registers: Registers,
    #[derivative(PartialEq = "ignore", Ord = "ignore")]
    pub fitness: FitnessScore,
}

impl Program {
    pub fn run(&mut self, input: &impl State) {
        for instruction in &self.instructions {
            instruction.apply(&mut self.registers, input)
        }
    }
}

impl Generate<ProgramGeneratorParameters, Program> for GenerateEngine {
    fn generate(using: ProgramGeneratorParameters) -> Program {
        let ProgramGeneratorParameters {
            max_instructions,
            instruction_generator_parameters,
            ..
        } = using;

        let registers = Registers::new(instruction_generator_parameters.n_registers());
        let n_instructions = generator().gen_range(1..=max_instructions);
        let instructions =
            repeat_with(|| GenerateEngine::generate(instruction_generator_parameters))
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

impl Mutate<ProgramGeneratorParameters, Program> for MutateEngine {
    fn mutate(item: &mut Program, using: ProgramGeneratorParameters) {
        // Pick instruction to mutate.
        let instruction = item
            .instructions
            .iter_mut()
            .choose(&mut generator())
            .unwrap();

        let _mutated_instruction =
            MutateEngine::mutate(instruction, using.instruction_generator_parameters);

        ResetEngine::reset(&mut item.id);
        ResetEngine::reset(item);
    }
}

impl Breed<Program> for BreedEngine {
    fn two_point_crossover(mate_1: &Program, mate_2: &Program) -> (Program, Program) {
        let (child_1_instructions, child_2_instructions) =
            BreedEngine::two_point_crossover(&mate_1.instructions, &mate_2.instructions);

        let mut child_1 = mate_1.clone();
        let mut child_2 = mate_2.clone();

        child_1.instructions = child_1_instructions;
        child_2.instructions = child_2_instructions;

        ResetEngine::reset(&mut child_1.id);
        ResetEngine::reset(&mut child_2.id);

        ResetEngine::reset(&mut child_1);
        ResetEngine::reset(&mut child_2);

        (child_1, child_2)
    }
}

#[cfg(test)]
mod tests {

    use crate::{core::instruction::InstructionGeneratorParameters, utils::test::TestInput};

    use super::*;

    #[test]
    fn given_instructions_when_breed_then_two_children_are_produced_using_genes_of_parents() {
        let params = InstructionGeneratorParameters {
            n_extras: 1,
            external_factor: 10.,
            n_actions: TestInput::N_ACTIONS,
            n_inputs: TestInput::N_INPUTS,
        };
        let instructions_a: Instructions =
            (0..10).map(|_| GenerateEngine::generate(params)).collect();
        let instructions_b: Instructions =
            (0..10).map(|_| GenerateEngine::generate(params)).collect();

        let (child_a, child_b) = BreedEngine::two_point_crossover(&instructions_a, &instructions_b);

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

        let program_a = GenerateEngine::generate(program_params);
        let program_b = GenerateEngine::generate(program_params);

        let (child_a, child_b) = BreedEngine::two_point_crossover(&program_a, &program_b);

        assert_ne!(child_a, child_b);

        assert_ne!(program_a, child_a);
        assert_ne!(program_a, child_b);

        assert_ne!(program_b, child_a);
        assert_ne!(program_b, child_b);
    }
}
