pub type Op = fn(a: f64, b: f64) -> f64;

pub type Executables = &'static [Op];

pub const DEFAULT_EXECUTABLES: Executables = &[add, subtract, multiply, divide];

pub fn add(a: f64, b: f64) -> f64 {
    a + b
}

pub fn subtract(a: f64, b: f64) -> f64 {
    a - b
}

pub fn multiply(a: f64, b: f64) -> f64 {
    a * b
}

pub fn divide(a: f64, _b: f64) -> f64 {
    a / 2.
}
