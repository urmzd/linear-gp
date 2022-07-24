use core::slice::Iter;
use std::{ops::Index, slice::SliceIndex};

use serde::Serialize;

use super::characteristics::DuplicateNew;

pub type RegisterValue = f64;

#[derive(Debug, Clone, Serialize)]
pub struct Registers {
    data: Vec<RegisterValue>,
}

impl From<Vec<RegisterValue>> for Registers {
    fn from(data: Vec<RegisterValue>) -> Self {
        Registers { data }
    }
}

impl DuplicateNew for Registers {
    fn duplicate_new(&self) -> Self {
        Self::new(self.len())
    }
}

impl Registers {
    pub fn new(n_registers: usize) -> Self {
        let data = vec![0.; n_registers];

        Registers { data }
    }

    pub fn reset(&mut self) {
        let Registers { data } = self;
        for value in data.as_mut_slice() {
            *value = 0.
        }
    }

    pub fn len(&self) -> usize {
        let Registers { data } = self;
        data.len()
    }

    pub fn update(&mut self, index: usize, value: RegisterValue) {
        let Registers { data } = self;
        data[index] = value;
    }

    pub fn get(&self, index: usize) -> &RegisterValue {
        let Registers { data } = self;
        data.get(index).unwrap()
    }

    pub fn iter<'a>(&'a self) -> Iter<'a, RegisterValue> {
        self.data.iter()
    }
}

impl<Idx> Index<Idx> for Registers
where
    Idx: SliceIndex<[RegisterValue]>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use crate::core::registers::Registers;

    #[test]
    fn given_registers_when_indexed_with_range_then_slice_is_returned() {
        let mut registers = Registers::new(10);
        registers.update(0, 1.);

        let slice = &registers[0..2];

        assert_eq!(slice, &[1., 0.]);
    }
}
