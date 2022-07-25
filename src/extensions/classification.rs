use derive_new::new;
use serde::Serialize;

use crate::core::{
    characteristics::Fitness,
    inputs::{Inputs, ValidInput},
    program::Program,
};

#[derive(Clone, Debug, Serialize, new)]
pub struct ClassificationParameters<InputType>
where
    InputType: ClassificationInput,
{
    inputs: Inputs<InputType>,
}

pub trait ClassificationInput: ValidInput {
    fn get_class(&self) -> usize;
}

impl<T> Fitness for Program<ClassificationParameters<T>>
where
    T: ClassificationInput,
{
    type FitnessParameters = ClassificationParameters<T>;

    fn eval_fitness(&mut self, parameters: &mut Self::FitnessParameters) -> f64 {
        let inputs = &parameters.inputs;

        let mut n_correct = 0.0;

        for input in inputs {
            self.exec(input);

            let mut winning_registers = self.registers.all_argmax(Some(0..T::N_ACTION_REGISTERS));
            let predicted_class = if winning_registers.len() == 1 {
                winning_registers.pop().expect("Register to have exist") as i32
            } else {
                -1
            };
            let correct_class = input.get_class() as i32;

            if predicted_class == correct_class {
                n_correct += 1.;
            }

            self.registers.reset();
        }

        let fitness = n_correct / inputs.len() as f64;

        self.fitness = Some(fitness);

        fitness
    }

    fn get_fitness(&self) -> Option<f64> {
        self.fitness
    }
}
