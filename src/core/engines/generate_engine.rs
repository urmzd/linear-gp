pub struct GenerateEngine;

pub trait Generate<U, T> {
    fn generate(using: U) -> T;
}
