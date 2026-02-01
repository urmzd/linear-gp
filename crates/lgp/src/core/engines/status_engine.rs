pub struct StatusEngine;

pub trait Status<T> {
    fn valid(item: &T) -> bool;
    fn evaluated(item: &T) -> bool;
    fn set_fitness(program: &mut T, fitness: f64);
    fn get_fitness(program: &T) -> f64;
}
