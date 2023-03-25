/// Defines a single state which can use the current context to get the next data.
pub trait State
where
    Self: Sized,
{
    const N_INPUTS: usize;
    const N_ACTIONS: usize;

    fn next_state(&mut self) -> Option<Self>;
    /// Returns a value from the feature set.
    fn get_value(&self, at_idx: usize) -> f64;
    /// Updates the impact of the factor.
    /// For example, if data[0] has been accessed, we increase the index so data[1] is accessed next (in classification).
    /// In RL, we act on the environment and internally update the termination state, and hold the new state.
    fn execute_action(&mut self, action: usize) -> f64;
}

pub struct Environment<T>
where
    T: State,
{
    state: T,
    trial_idx: usize,
    n_trials: usize,
    trial: Vec<T>,
}

impl<T> Environment<T>
where
    T: State,
{
    fn init(n_trials: usize, state: T) -> Self {
        Environment {
            n_trials,
            trial_idx: 0,
            trial: Vec::with_capacity(n_trials),
            state,
        }
    }
    /// A trial consists of several starting states.
    /// returns true if next trial is available, false otherwise
    fn next_trial(&mut self) -> bool {
        self.trial_idx += 1;
        self.n_trials >= self.trial_idx
    }
    // /// Reseting a trial reverts to the 0th index of the current trial collection.
    fn reset_trial(&mut self) {
        self.trial_idx = 0;
    }

    // A new trial generates a new set of starting states.
    fn new_trials(&mut self) {
        // self.trials = GenerateEngine::generate()
    }
}

pub trait RlState: State {
    /// Returns true if episode count > MAX or terminal_signal sent from environment.
    fn is_terminal(&mut self) -> bool;
}
