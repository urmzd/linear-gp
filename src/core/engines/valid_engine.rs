pub struct ValidEngine;

pub trait Valid<T> {
    fn valid(item: &T) -> bool;
}
