use crate::core::{
    engines::fitness_engine::{Fitness, FitnessEngine, FitnessScore},
    environment::State,
    program::Program,
    registers::{ActionRegister, ArgmaxInput},
};


impl<T> Fitness<Program, T, ()> for FitnessEngine
where
    T: State,
{
    fn eval_fitness(program: &mut Program, states: &mut T) -> FitnessScore {
        let mut n_correct = 0.;
        let mut n_total = 0.;

        while let Some(state) = states.get() {
            program.run(state);

            match program
                .registers
                .argmax(ArgmaxInput::To(T::N_ACTIONS))
                .one()
            {
                ActionRegister::Overflow => return FitnessScore::OutOfBounds,
                ActionRegister::Value(predicted_class) => {
                    n_correct += state.execute_action(predicted_class);
                }
            };

            n_total += 1.;
        }

        FitnessScore::Valid(n_correct / n_total)
    }
}
