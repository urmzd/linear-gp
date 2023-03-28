use core::slice::Iter;
use std::{ops::Index, slice::SliceIndex};

use itertools::Itertools;
use rand::seq::SliceRandom;
use serde::{Deserialize, Deserializer, Serialize};

use crate::utils::random::generator;

use super::engines::reset_engine::{Reset, ResetEngine};

fn deserialize_vec_with_null<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let vec_opt: Option<Vec<Option<f64>>> = Deserialize::deserialize(deserializer)?;
    Ok(vec_opt
        .unwrap_or_default()
        .into_iter()
        .map(|x| x.unwrap_or(f64::NAN))
        .collect())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Registers {
    #[serde(deserialize_with = "deserialize_vec_with_null")]
    data: Vec<f64>,
}

impl From<Vec<f64>> for Registers {
    fn from(data: Vec<f64>) -> Self {
        Registers { data }
    }
}

pub enum ArgmaxResult {
    MaxValues(Vec<usize>),
    Overflow,
}

pub enum ActionRegister {
    Value(usize),
    Overflow,
}

impl ArgmaxResult {
    pub fn one(&self) -> ActionRegister {
        match self {
            ArgmaxResult::MaxValues(indices) if indices.len() == 1 => {
                ActionRegister::Value(indices[0])
            }
            _ => ActionRegister::Overflow,
        }
    }

    pub fn any(&self) -> ActionRegister {
        match self {
            ArgmaxResult::MaxValues(indices) if indices.len() >= 1 => {
                ActionRegister::Value(indices.choose(&mut generator()).copied().unwrap())
            }
            _ => ActionRegister::Overflow,
        }
    }
}

pub enum ArgmaxInput {
    All,
    To(usize),
}

impl Reset<Registers> for ResetEngine {
    fn reset(item: &mut Registers) {
        for value in item.data.as_mut_slice() {
            *value = 0.
        }
    }
}

impl Registers {
    pub fn new(n_registers: usize) -> Self {
        let data = vec![0.; n_registers];

        Registers { data }
    }

    pub fn argmax(&self, range: ArgmaxInput) -> ArgmaxResult {
        let range_to_use = match range {
            ArgmaxInput::All => 0..(self.data.len()),
            ArgmaxInput::To(to) => 0..(to),
        };

        let sliced_data = &self.data[range_to_use];
        let max_value = sliced_data
            .iter()
            .copied()
            .reduce(f64::max)
            .expect("Sliced values to not be of cardinality 0.");

        if max_value.is_infinite() || max_value.is_nan() {
            return ArgmaxResult::Overflow;
        }

        let max_indices = sliced_data
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v == &max_value)
            .map(|(i, _)| i)
            .collect_vec();

        ArgmaxResult::MaxValues(max_indices)
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

    #[test]
    fn given_registers_when_indexed_with_range_then_slice_is_returned() {
        let mut registers = Registers::new(10);
        registers.update(0, 1.);

        let slice = &registers[0..2];

        assert_eq!(slice, &[1., 0.]);
    }
}
