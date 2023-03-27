pub struct FreezeEngine;

pub trait Freeze<T> {
    fn freeze(_item: &mut T) {}
}
