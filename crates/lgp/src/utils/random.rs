use std::{cell::UnsafeCell, rc::Rc};

use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

type InternalGenerator = Rc<UnsafeCell<Xoshiro256PlusPlus>>;

#[derive(Clone, Debug)]
pub struct Random {
    rng: InternalGenerator,
}

thread_local! {
    static GENERATOR: InternalGenerator = {
        let prng = Xoshiro256PlusPlus::from_entropy();

        Rc::new(UnsafeCell::new(prng))
    }
}

/// This function should only be called once and at the top level of a program.
pub fn update_seed(seed: Option<u64>) {
    let prng = match seed {
        Some(internal_seed) => Xoshiro256PlusPlus::seed_from_u64(internal_seed),
        None => Xoshiro256PlusPlus::from_entropy(),
    };

    GENERATOR.with(|t| {
        let generator = unsafe { &mut *t.get() };
        *generator = prng;
    });
}

/// Runs `f` with this thread's generator seeded from `seed`, then restores the
/// previous generator state.
///
/// Rayon worker threads own their own generators, which `update_seed` never
/// reaches. Parallel work draws a seed per task from the calling thread and
/// runs the task through this function, so seeded runs repeat exactly no matter
/// which worker picks up the task. Restoring the old state keeps a worker that
/// steals a task from another engine from corrupting that engine's stream.
pub fn with_seed<R>(seed: u64, f: impl FnOnce() -> R) -> R {
    let previous = GENERATOR.with(|t| {
        let generator = unsafe { &mut *t.get() };
        std::mem::replace(generator, Xoshiro256PlusPlus::seed_from_u64(seed))
    });

    let result = f();

    GENERATOR.with(|t| {
        let generator = unsafe { &mut *t.get() };
        *generator = previous;
    });

    result
}

/// Runs `f` with this thread's generator, without cloning a handle to it.
/// Draws the same values as `generator()`; use it in hot loops.
pub fn with_generator<R>(f: impl FnOnce(&mut Xoshiro256PlusPlus) -> R) -> R {
    GENERATOR.with(|t| {
        let generator = unsafe { &mut *t.get() };
        f(generator)
    })
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

/// Returns a version 4 UUID drawn from this thread's generator, so seeded runs
/// produce the same program ids.
pub fn random_uuid() -> uuid::Uuid {
    let mut bytes = [0u8; 16];
    generator().fill_bytes(&mut bytes);
    uuid::Builder::from_random_bytes(bytes).into_uuid()
}
