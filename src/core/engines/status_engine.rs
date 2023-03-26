pub struct StatusEngine;

pub trait Status<T> {
    fn valid(item: &T) -> bool;
    fn evaluated(item: &T) -> bool;
}
