use core::slice::Iter;
use std::{ops::Index, slice::SliceIndex};

use itertools::Itertools;

use super::characteristics::DuplicateNew;

#[derive(Debug, Clone)]
pub struct Registers {
    data: Vec<f64>,
}

impl From<Vec<f64>> for Registers {
    fn from(data: Vec<f64>) -> Self {
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

    pub fn update(&mut self, index: usize, value: f64) {
        let Registers { data } = self;
        data[index] = value;
    }

    pub fn update_then_softmax(&mut self, index: usize, value: f64) {
        self.update(index, value);
        self.softmax();
    }

    fn softmax(&mut self) {
        let max = self
            .data
            .iter()
            .reduce(|current_max, value| {
                if value > current_max {
                    value
                } else {
                    current_max
                }
            })
            .expect("Max value to have been found.");
        let shifted_data = self.data.iter().map(|v| *v - *max).collect_vec();
        let sum: f64 = shifted_data.iter().map(|v| v.exp()).sum();
        if sum != 0. {
            self.data = shifted_data.iter().map(|v| *v / sum).collect_vec();
        }
    }

    pub fn get(&self, index: usize) -> &f64 {
        let Registers { data } = self;
        data.get(index).unwrap()
    }

    pub fn iter(&self) -> Iter<f64> {
        self.data.iter()
    }
}

impl<Idx> Index<Idx> for Registers
where
    Idx: SliceIndex<[f64]>,
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
