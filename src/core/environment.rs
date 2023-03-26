/// Defines a single state which can use the current context to get the next data.
pub trait State: Sized {
    const N_INPUTS: usize;
    const N_ACTIONS: usize;

    fn get_value(&self, at_idx: usize) -> f64;
    /// Updates the impact of the factor.
    /// For example, if data[0] has been accessed, we increase the index so data[1] is accessed next (in classification).
    /// In RL, we act on the environment and internally update the termination state, and hold the new state.
    fn execute_action(&mut self, action: usize) -> f64;

    /// We take a mutable reference and return self.
    fn get(&mut self) -> Option<&mut Self>;
}

pub trait RlState: State {
    /// Returns true if episode count > MAX or terminal_signal sent from environment.
    fn is_terminal(&mut self) -> bool;

    // Returns the initial state.
    fn get_initial_state(&self) -> Vec<f64>;
}
