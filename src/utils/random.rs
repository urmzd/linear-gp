use std::{cell::UnsafeCell, rc::Rc};

use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

#[derive(Clone, Debug)]
pub struct Random {
    rng: InternalGenerator,
}

type InternalGenerator = Rc<UnsafeCell<Xoshiro256PlusPlus>>;

thread_local! {
    static GENERATOR: InternalGenerator = {
        let prng = Xoshiro256PlusPlus::from_entropy();

        Rc::new(UnsafeCell::new(prng))
    }
}

pub fn generator() -> Random {
    let rng = GENERATOR.with(|t| t.clone());
    Random { rng }
}

impl Default for Random {
    fn default() -> Self {
        generator()
    }
}

impl RngCore for Random {
    fn next_u32(&mut self) -> u32 {
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let rng = unsafe { &mut *self.rng.get() };
        rng.fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        let rng = unsafe { &mut *self.rng.get() };
        rng.try_fill_bytes(dest)
    }
}
