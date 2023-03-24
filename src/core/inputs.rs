use core::fmt::Debug;

use super::registers::Registers;

pub type Inputs<InputType> = Vec<InputType>;

pub trait ValidInput: Clone + Send + Debug + Sized
where
    Registers: for<'a> From<&'a Self>,
{
    const N_INPUTS: usize;
    const N_ACTIONS: usize;

    fn get(&self, idx: usize) -> f64;
}
