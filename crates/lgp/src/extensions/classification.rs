use crate::core::{
    engines::fitness_engine::{Fitness, FitnessEngine},
    environment::State,
    program::Program,
    registers::{ActionRegister, ArgmaxInput},
};

impl<T> Fitness<Program, T, ()> for FitnessEngine
where
    T: State,
{
    fn eval_fitness(program: &mut Program, states: &mut T) -> f64 {
        let mut n_correct = 0.;
        let mut n_total = 0.;

        while let Some(state) = states.get() {
            program.run(state);

            match program.registers.argmax(ArgmaxInput::ActionRegisters).one() {
                ActionRegister::Overflow => {
                    return f64::NEG_INFINITY;
                }
                ActionRegister::Value(predicted_class) => {
                    n_correct += state.execute_action(predicted_class);
                }
            };

            n_total += 1.;
        }

        n_correct / n_total
    }
}
