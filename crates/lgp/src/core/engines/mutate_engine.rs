pub struct MutateEngine;

pub trait Mutate<F, I> {
    fn mutate(item: &mut I, using: F);
}
