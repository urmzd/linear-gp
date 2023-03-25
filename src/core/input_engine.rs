/// Defines a single "factor" / data point.
pub trait EnvironmentalFactor {
    const N_INPUTS: usize;
    const N_ACTIONS: usize;

    /// Returns a value from the feature set.
    fn get_value(&self, at_idx: usize) -> f64;
    fn execute_action(&mut self, action: usize) -> f64;
}

/// Holds all the environment required to assess the fitness of an indiviidual.
pub trait Environment<S: EnvironmentalFactor> {
    /// Returns the input for the current trial.
    fn get_trial(&mut self) -> Vec<S>;
}

pub trait ClassificationEnvironment<S: EnvironmentalFactor>: Environment<S> {}

pub trait RlEnvironment<S: EnvironmentalFactor>: Environment<S> {
    fn set_state(&mut self, state: S);
    fn is_terminal_state(&self) -> bool;
}
