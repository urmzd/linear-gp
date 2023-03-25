use gym_rs::core::Env;
use strum::additional_attributes;

use crate::core::{
    characteristics::Reset,
    engines::fitness_engine::{Fitness, FitnessEngine, FitnessScore},
    input_engine::{ClassificationEnvironment, EnvironmentalFactor},
    program::Program,
    registers::{ArgmaxInput, AR},
};

impl<S: EnvironmentalFactor, E: ClassificationEnvironment<S>> Fitness<E, ()> for FitnessEngine {
    fn eval_fitness(program: &mut Program, environment: &mut E, parameters: &mut ()) {
        /// TODO: Reset all registers and states before running.
        program.registers.reset();

        let mut n_correct: usize = 0;

        for input in parameters {
            program.run(&input);

            match program
                .registers
                .argmax(ArgmaxInput::To(S::N_ACTIONS))
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

        let accuracy = n_correct as f64 / parameters.len() as f64;
        program.fitness = FitnessScore::Valid(accuracy);
    }
}
