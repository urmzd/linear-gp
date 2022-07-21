use ordered_float::OrderedFloat;
use serde::Serialize;

pub type RegisterValue = OrderedFloat<f32>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Registers(Vec<RegisterValue>);

impl From<Vec<RegisterValue>> for Registers {
    fn from(data: Vec<RegisterValue>) -> Self {
        Registers(data)
    }
}

impl Registers {
    pub fn new(n_registers: usize) -> Self {
        let data = vec![OrderedFloat(0.); n_registers];

        Registers(data)
    }

    pub fn reset(&mut self) {
        let Registers(data) = self;
        for value in data.as_mut_slice() {
            *value = OrderedFloat(0.)
        }
    }

    pub fn reset_clone(&self) -> Self {
        let mut registers = self.clone();
        registers.reset();
        registers
    }

    pub fn len(&self) -> usize {
        let Registers(data) = self;
        data.len()
    }

    pub fn update(&mut self, index: usize, value: RegisterValue) {
        let Registers(data) = self;
        data[index] = value;
    }

    pub fn get(&self, index: usize) -> &RegisterValue {
        let Registers(data) = self;
        data.get(index).unwrap()
    }
}
