use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub const SEED_NO: u64 = 42;

pub struct Random {
    rng: ChaCha8Rng,
}

pub fn generator() -> Random {
    Random {
        rng: ChaCha8Rng::seed_from_u64(SEED_NO),
    }
}

impl Default for Random {
    fn default() -> Self {
        generator()
    }
}

impl RngCore for Random {
    fn next_u32(&mut self) -> u32 {
        let rng = { &mut self.rng };
        rng.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        let rng = { &mut self.rng };
        rng.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let rng = { &mut self.rng };
        rng.fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        let rng = { &mut self.rng };
        rng.try_fill_bytes(dest)
    }
}
