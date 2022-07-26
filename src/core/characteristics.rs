pub trait Fitness {
    type FitnessParameters;

    fn eval_fitness(&mut self, parameters: &mut Self::FitnessParameters);
    fn get_fitness(&self) -> Option<f64>;
}

pub trait Breed: Clone {
    fn two_point_crossover(&self, mate: &Self) -> [Self; 2];
}

pub trait Mutate: Generate + Clone {
    fn mutate(&self, parameters: &Self::GeneratorParameters) -> Self;
}

pub trait Generate {
    type GeneratorParameters;

    fn generate(parameters: &Self::GeneratorParameters) -> Self;
}

pub trait DuplicateNew {
    fn duplicate_new(&self) -> Self;
}
