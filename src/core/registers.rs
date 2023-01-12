use core::slice::Iter;
use std::{
    ops::{Index, Range},
    slice::SliceIndex,
};

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

    pub fn all_argmax(&self, range: Option<Range<usize>>) -> Option<Vec<usize>> {
        let Registers { data } = self;
        let range_to_use = range.unwrap_or(0..data.len());
        let sliced_data = &data[range_to_use];
        let max_value = sliced_data
            .iter()
            .copied()
            .reduce(f64::max)
            .expect("Sliced values to not be of cardinality 0.");

        if max_value.is_infinite() || max_value.is_nan() {
            return None;
        }

        let max_indices = sliced_data
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v == &max_value)
            .map(|(i, _)| i)
            .collect_vec();

        Some(max_indices)
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
    use test_log::test;

    #[test]
    fn given_registers_when_indexed_with_range_then_slice_is_returned() {
        let mut registers = Registers::new(10);
        registers.update(0, 1.);

        let slice = &registers[0..2];

        assert_eq!(slice, &[1., 0.]);
    }
}
