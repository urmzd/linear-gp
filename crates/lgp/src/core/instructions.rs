use crate::utils::random::generator;
use rand::Rng;

use super::{
    engines::breed_engine::{Breed, BreedEngine},
    instruction::Instruction,
};

impl Breed<Instructions> for BreedEngine {
    fn two_point_crossover(
        mate_1: &Instructions,
        mate_2: &Instructions,
    ) -> (Instructions, Instructions) {
        debug_assert!(!mate_1.is_empty());
        debug_assert!(!mate_2.is_empty());

        let a_len = mate_1.len();
        let b_len = mate_2.len();

        // Draw the cut points in a fixed order: both starts, then each end.
        // A start on the last instruction swaps the tail, so its end is the length.
        let a_start = generator().gen_range(0..a_len);
        let b_start = generator().gen_range(0..b_len);

        let a_end = if a_start == a_len - 1 {
            a_len
        } else {
            generator().gen_range(a_start + 1..a_len)
        };

        let b_end = if b_start == b_len - 1 {
            b_len
        } else {
            generator().gen_range(b_start + 1..b_len)
        };

        // Build each child once from slices of the parents: head, the other
        // parent's chunk, tail. This copies each instruction one time.
        let splice = |base: &[Instruction], start: usize, end: usize, chunk: &[Instruction]| {
            let mut child = Vec::with_capacity(base.len() - (end - start) + chunk.len());
            child.extend_from_slice(&base[..start]);
            child.extend_from_slice(chunk);
            child.extend_from_slice(&base[end..]);
            child
        };

        let instructions_a = splice(mate_1, a_start, a_end, &mate_2[b_start..b_end]);
        let instructions_b = splice(mate_2, b_start, b_end, &mate_1[a_start..a_end]);

        debug_assert!(!instructions_a.is_empty(), "instructions A after crossover");
        debug_assert!(!instructions_b.is_empty(), "instructions B after crossover");

        (instructions_a, instructions_b)
    }
}

pub type Instructions = Vec<Instruction>;

#[cfg(test)]
mod tests {

    use crate::core::{
        engines::{
            breed_engine::{Breed, BreedEngine},
            generate_engine::{Generate, GenerateEngine},
        },
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    };

    #[test]
    fn given_two_programs_when_two_point_crossover_multiple_times_then_instruction_set_never_grows()
    {
        let max_instructions = 100;
        let parameters = ProgramGeneratorParameters {
            max_instructions,
            instruction_generator_parameters: InstructionGeneratorParameters {
                n_extras: 1,
                external_factor: 10.,
                n_inputs: 4,
                n_actions: 2,
            },
        };

        let mut program_a = GenerateEngine::generate(parameters);
        let mut program_b = GenerateEngine::generate(parameters);

        for _ in 0..100 {
            let parent_a_instruction_len = program_a.instructions.len();
            let parent_b_instruction_len = program_b.instructions.len();

            let (new_parent_a, new_parent_b) =
                BreedEngine::two_point_crossover(&program_a, &program_b);

            debug_assert!(!new_parent_a.instructions.is_empty());
            debug_assert!(!new_parent_b.instructions.is_empty());

            debug_assert!(
                new_parent_a.instructions.len()
                    <= parent_a_instruction_len + parent_b_instruction_len
            );
            debug_assert!(
                new_parent_b.instructions.len()
                    <= parent_a_instruction_len + parent_b_instruction_len
            );

            program_a = new_parent_a;
            program_b = new_parent_b;
        }
    }
}
