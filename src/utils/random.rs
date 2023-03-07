use std::{cell::UnsafeCell, rc::Rc};

use rand::{
    rngs::{adapter::ReseedingRng, OsRng},
    RngCore, SeedableRng,
};
use rand_chacha::ChaCha8Core;

#[derive(Clone, Debug)]
pub struct Random {
    rng: InternalGenerator,
}

type InternalGenerator = Rc<UnsafeCell<ReseedingRng<ChaCha8Core, OsRng>>>;

thread_local! {
    static GENERATOR: InternalGenerator = {
        let prng = ChaCha8Core::from_entropy();
        let reseeding_rng = ReseedingRng::new(prng, 10, OsRng);

        Rc::new(UnsafeCell::new(reseeding_rng))
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
