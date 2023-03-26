use super::fitness_engine::FitnessScore;

pub struct StatusEngine;

pub trait Status<T> {
    fn valid(item: &T) -> bool;
    fn evaluated(item: &T) -> bool;
    fn set_fitness(program: &mut T, fitness: FitnessScore);
    fn get_fitness(program: &T) -> FitnessScore;
}
