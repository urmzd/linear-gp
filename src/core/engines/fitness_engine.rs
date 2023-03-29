use super::reset_engine::{Reset, ResetEngine};

pub trait Fitness<I, S, P> {
    fn eval_fitness(program: &mut I, states: &mut S) -> f64;
}

impl Reset<f64> for ResetEngine {
    fn reset(item: &mut f64) {
        *item = f64::NAN;
    }
}

pub struct FitnessEngine;
