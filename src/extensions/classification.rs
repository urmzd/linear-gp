use crate::core::{
    characteristics::{Fitness, FitnessScore, Reset},
    inputs::{Inputs, ValidInput},
    program::Program,
    registers::{ArgmaxInput, AR},
};

pub trait ClassificationInput: ValidInput {
    fn get_class(&self) -> usize;
}

pub struct Classifier<T>(Inputs<T>);

impl<T> Fitness for Classifier<T>
where
    T: ClassificationInput,
{
    type Parameters = ();

    fn eval_fitness(
        &mut self,
        program: Program,
        parameters: Self::Parameters,
    ) -> (FitnessScore, Self::Parameters) {
        let inputs = &self.0;

        program.registers.reset();

        let mut n_correct: usize = 0;

        for input in inputs {
            program.run(input);

            match program
                .registers
                .argmax(ArgmaxInput::To(T::N_ACTIONS))
                .one()
            {
                AR::Overflow => {
                    return {
                        FitnessScore::OutOfBounds;
                    }
                }
                AR::Value(predicted_class) => {
                    n_correct += (predicted_class == input.get_class()) as usize;
                }
            };

            program.registers.reset();
        }

        let fitness = n_correct as f64 / inputs.len() as f64;

        (FitnessScore::Valid(fitness), ())
    }
}
