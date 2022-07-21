use derive_new::new;
use serde::Serialize;

use crate::core::inputs::{Inputs, ValidInput};

#[derive(Clone, Debug, Serialize, new)]
pub struct ClassificationParameters<'a, InputType>
where
    InputType: ClassificationInput,
{
    inputs: &'a Inputs<InputType>,
}

pub trait ClassificationInput: ValidInput {
    fn get_class(&self) -> usize;
}

// impl<'a, T> Organism<'a> for Program<'a, T> where T: ClassificationInput {}

// impl<'a, T> Fitness for Program<'a, T>
// where
//     T: ClassificationInput,
// {
//     fn eval_fitness(&mut self) -> FitnessScore {
//         let inputs = self.problem_parameters.inputs;

//         let mut n_correct = 0;

//         for input in inputs {
//             self.exec(input);

//             let predicted_class = T::argmax(&self.registers);
//             let correct_class = input.get_class().to_i32().unwrap();

//             if predicted_class == correct_class {
//                 n_correct += 1;
//             }

//             self.registers.reset();
//         }

//         let fitness = OrderedFloat(n_correct as f32 / inputs.len() as f32);

//         self.fitness = Some(fitness);

//         fitness
//     }

//     fn get_fitness(&self) -> Option<FitnessScore> {
//         self.fitness
//     }
// }
