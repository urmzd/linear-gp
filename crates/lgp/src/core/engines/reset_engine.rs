pub struct ResetEngine;

pub trait Reset<T> {
    fn reset(item: &mut T);
}
