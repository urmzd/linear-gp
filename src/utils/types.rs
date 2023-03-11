use std::error::Error;

pub type VoidResultAnyError = Result<(), Box<dyn Error>>;
