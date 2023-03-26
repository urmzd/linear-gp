use std::error::Error;

use crate::core::engines::reset_engine::{Reset, ResetEngine};

pub type VoidResultAnyError = Result<(), Box<dyn Error>>;

impl Reset<uuid::Uuid> for ResetEngine {
    fn reset(item: &mut uuid::Uuid) {
        *item = uuid::Uuid::new_v4();
    }
}
