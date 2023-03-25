use crate::core::{
    characteristics::Reset,
    engines::fitness_engine::{Fitness, FitnessEngine, FitnessScore},
    inputs::{Inputs, ValidInput},
    program::Program,
    registers::{ArgmaxInput, AR},
};

pub trait ClassificationInput: ValidInput {
    fn get_class(&self) -> usize;
}

impl<T: ClassificationInput> Fitness<Program, Inputs<T>> for FitnessEngine {
    fn eval_fitness(program: &mut Program, parameters: &mut Inputs<T>) {
        program.registers.reset();

        let mut n_correct: usize = 0;

        for input in parameters {
            program.run(input);

            match program
                .registers
                .argmax(ArgmaxInput::To(T::N_ACTIONS))
                .one()
            {
                AR::Overflow => {
                    return {
                        program.fitness = FitnessScore::OutOfBounds;
                    }
                }
                AR::Value(predicted_class) => {
                    n_correct += (predicted_class == input.get_class()) as usize;
                }
            };

            program.registers.reset();
        }

        let fitness = n_correct as f64 / parameters.len() as f64;
        program.fitness = FitnessScore::Valid(fitness);
    }
}
