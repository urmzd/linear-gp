use derive_new::new;

use crate::core::{
    characteristics::{Fitness, FitnessScore},
    inputs::{Inputs, ValidInput},
    program::{Program, ProgramParameters},
    registers::{ArgmaxInput, AR},
};

#[derive(Clone, Debug, new)]
pub struct ClassificationParameters<InputType>
where
    InputType: ClassificationInput,
{
    inputs: Inputs<InputType>,
}

pub trait ClassificationInput: ValidInput {
    fn get_class(&self) -> usize;
}

impl<T> ProgramParameters for ClassificationParameters<T>
where
    T: ClassificationInput,
{
    type InputType = T;
}

impl<T> Fitness for Program<ClassificationParameters<T>>
where
    T: ClassificationInput,
{
    type FitnessParameters = ClassificationParameters<T>;

    fn eval_fitness(&mut self, parameters: Self::FitnessParameters) {
        let inputs = &parameters.inputs;

        self.registers.reset();

        let mut n_correct: usize = 0;

        for input in inputs {
            self.run(input);

            match self.registers.argmax(ArgmaxInput::To(T::N_ACTIONS)).one() {
                AR::Overflow => {
                    return {
                        self.fitness = FitnessScore::OutOfBounds;
                    }
                }
                AR::Value(predicted_class) => {
                    n_correct += (predicted_class == input.get_class()) as usize;
                }
            };

            self.registers.reset();
        }

        let fitness = n_correct as f64 / inputs.len() as f64;

        self.fitness = FitnessScore::Valid(fitness);
    }

    fn get_fitness(&self) -> FitnessScore {
        self.fitness
    }
}
