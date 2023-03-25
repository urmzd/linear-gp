pub trait Breed<T>
where
    T: Clone,
{
    fn two_point_crossover(mate_1: &T, mate_2: &T) -> [T; 2];
}

pub struct BreedEngine;
