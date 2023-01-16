use more_asserts::assert_le;
use rand::{distributions::Uniform, prelude::Distribution};

use crate::utils::{linked_list::LinkedList, random::generator};

use super::{characteristics::Breed, instruction::Instruction};

impl Breed for Instructions {
    fn two_point_crossover(&self, mate: &Self) -> [Self; 2] {
        let mut instructions_a = self.clone();
        let mut instructions_b = mate.clone();
        let current_generator = &mut generator();

        let a_start = Uniform::new(0, instructions_a.len()).sample(current_generator);
        let a_end = if a_start == instructions_a.len() - 1 {
            None
        } else {
            let tmp_end = Uniform::new(a_start + 1, instructions_a.len()).sample(current_generator);
            Some(tmp_end)
        };

        let b_start = Uniform::new(0, instructions_b.len()).sample(current_generator);
        let b_end = if b_start == instructions_b.len() - 1 {
            None
        } else {
            let tmp_end = Uniform::new(b_start + 1, instructions_b.len()).sample(current_generator);
            Some(tmp_end)
        };

        let mut cursor_a = instructions_a.cursor_mut();
        let mut cursor_b = instructions_b.cursor_mut();

        cursor_a.swap(&mut cursor_b, a_start, b_start, a_end, b_end);

        assert_le!(
            instructions_a.len(),
            instructions_a.len() + instructions_b.len()
        );
        assert_le!(
            instructions_b.len(),
            instructions_a.len() + instructions_b.len()
        );

        [instructions_a, instructions_b]
    }
}

pub type Instructions = LinkedList<Instruction>;

#[cfg(test)]
mod tests {
    use more_asserts::assert_le;

    use crate::{
        core::{
            characteristics::{Breed, Generate},
            instruction::InstructionGeneratorParameters,
            program::{Program, ProgramGeneratorParameters},
        },
        utils::test::TestInput,
    };

    #[test]
    fn given_two_programs_when_two_point_crossover_multiple_times_then_instruction_set_never_grows()
    {
        let max_instructions_length = 100;
        let parameters = ProgramGeneratorParameters::new(
            max_instructions_length,
            InstructionGeneratorParameters::new(4, 2, 1),
        );

        let program_a = Program::<TestInput>::generate(&parameters);
        let program_b = Program::<TestInput>::generate(&parameters);

        let mut parents = [program_a, program_b];

        for _ in 0..100 {
            let parent_a_instruction_len = parents[0].instructions.len();
            let parent_b_instruction_len = parents[1].instructions.len();

            let new_parents = Breed::two_point_crossover(&parents[0], &parents[1]);

            assert_le!(
                new_parents[0].instructions.len(),
                parent_a_instruction_len + parent_b_instruction_len
            );
            assert_le!(
                new_parents[1].instructions.len(),
                parent_a_instruction_len + parent_b_instruction_len
            );

            parents = new_parents;
        }
    }
}
