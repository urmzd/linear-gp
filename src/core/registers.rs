use core::slice::Iter;
use std::{ops::Index, slice::SliceIndex};

use itertools::Itertools;
use noisy_float::prelude::{r64, Float, R64};

use super::characteristics::DuplicateNew;

pub type RegisterValue = R64;

#[derive(Debug, Clone)]
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
        let values = vec![0.; n_registers];
        let data = values.into_iter().map(|v| r64(v)).collect_vec();

        Registers { data }
    }

    pub fn reset(&mut self) {
        let Registers { data } = self;
        for value in data.as_mut_slice() {
            *value = r64(0.)
        }
    }

    pub fn len(&self) -> usize {
        let Registers { data } = self;
        data.len()
    }

    pub fn update(&mut self, index: usize, value: RegisterValue) {
        let Registers { data } = self;
        data[index] = value;
        self.softmax();
    }

    fn softmax(&mut self) {
        let max = self
            .data
            .iter()
            .max()
            .expect("Max value to have been found.");
        let shifted_data = self.data.iter().map(|v| *v - *max).collect_vec();
        let sum: R64 = shifted_data.iter().map(|v| v.exp()).sum();
        if sum != 0. {
            self.data = shifted_data.iter().map(|v| *v - sum.log2()).collect_vec();
        }
    }

    pub fn get(&self, index: usize) -> &RegisterValue {
        let Registers { data } = self;
        data.get(index).unwrap()
    }

    pub fn iter(&self) -> Iter<RegisterValue> {
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
    use noisy_float::prelude::r64;

    use crate::core::registers::Registers;

    #[test]
    fn given_registers_when_indexed_with_range_then_slice_is_returned() {
        let mut registers = Registers::new(10);
        registers.update(0, r64(1.));

        let slice = &registers[0..2];

        assert_eq!(slice, &[1., 0.]);
    }
}
