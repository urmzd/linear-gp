use crate::utils::random::generator;
use itertools::Itertools;
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
        let mut instructions_a = mate_1.clone();
        let mut instructions_b = mate_2.clone();

        debug_assert!(!instructions_a.is_empty());
        debug_assert!(!instructions_b.is_empty());

        let a_start = generator().gen_range(0..instructions_a.len());
        let b_start = generator().gen_range(0..instructions_b.len());

        let a_end = if a_start == instructions_a.len() - 1 {
            None
        } else {
            debug_assert!(instructions_a.len() > a_start);
            Some(generator().gen_range(a_start + 1..instructions_a.len()))
        };

        let b_end = if b_start == instructions_b.len() - 1 {
            None
        } else {
            debug_assert!(instructions_b.len() > b_start);
            Some(generator().gen_range(b_start + 1..instructions_b.len()))
        };

        let a_chunk = match a_end {
            None => &instructions_a[a_start..],
            Some(a_end_idx) => &instructions_a[a_start..a_end_idx],
        }
        .iter()
        .cloned()
        .collect_vec();

        let b_chunk = match b_end {
            None => &instructions_b[b_start..],
            Some(b_end_idx) => &instructions_b[b_start..b_end_idx],
        }
        .iter()
        .cloned()
        .collect_vec();

        if let Some(a_end_idx) = a_end {
            instructions_a.splice(a_start..a_end_idx, b_chunk)
        } else {
            instructions_a.splice(a_start.., b_chunk)
        }
        .collect_vec();

        if let Some(b_end_idx) = b_end {
            instructions_b.splice(b_start..b_end_idx, a_chunk)
        } else {
            instructions_b.splice(b_start.., a_chunk)
        }
        .collect_vec();

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
