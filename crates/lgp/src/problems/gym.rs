use std::marker::PhantomData;

use gymnasia::core::Env;
use gymnasia::core::Flatten;
use gymnasia::core::StepResult;
use gymnasia::envs::classical_control::cartpole::CartPoleEnv;
use gymnasia::envs::classical_control::mountain_car::MountainCarEnv;

use crate::core::engines::breed_engine::BreedEngine;
use crate::core::engines::core_engine::Core;
use crate::core::engines::fitness_engine::FitnessEngine;
use crate::core::engines::freeze_engine::FreezeEngine;
use crate::core::engines::generate_engine::Generate;
use crate::core::engines::generate_engine::GenerateEngine;
use crate::core::engines::mutate_engine::MutateEngine;
use crate::core::engines::reset_engine::Reset;
use crate::core::engines::reset_engine::ResetEngine;
use crate::core::engines::status_engine::StatusEngine;
use crate::core::environment::RlState;
use crate::core::environment::State;
use crate::core::program::Program;
use crate::core::program::ProgramGeneratorParameters;
use crate::extensions::interactive::UseRlFitness;
use crate::extensions::q_learning::QProgram;
use crate::extensions::q_learning::QProgramGeneratorParameters;

pub trait GymRsEnvExt: Env<Action = i64> + Clone + Send + Sync
where
    Self::Observation: Copy + Flatten + Send + Sync,
{
    fn create() -> Self;
    fn max_steps() -> usize;
    fn set_state(&mut self, obs: Self::Observation);
}

impl GymRsEnvExt for CartPoleEnv {
    fn create() -> Self {
        CartPoleEnv::new()
    }
    fn max_steps() -> usize {
        500
    }
    fn set_state(&mut self, obs: Self::Observation) {
        self.state = obs;
    }
}

impl GymRsEnvExt for MountainCarEnv {
    fn create() -> Self {
        MountainCarEnv::new()
    }
    fn max_steps() -> usize {
        200
    }
    fn set_state(&mut self, obs: Self::Observation) {
        self.state = obs;
    }
}

#[derive(Clone, Debug)]
pub struct GymRsInput<E: GymRsEnvExt>
where
    E::Observation: Copy + Flatten + Send + Sync,
{
    environment: E,
    terminated: bool,
    episode_idx: usize,
    initial_state: E::Observation,
    observation: Vec<f64>,
}

impl<E> State for GymRsInput<E>
where
    E: GymRsEnvExt,
    E::Observation: Copy + Flatten + Send + Sync,
{
    fn get_value(&self, idx: usize) -> f64 {
        self.observation[idx]
    }

    fn execute_action(&mut self, action: usize) -> f64 {
        let step_result: StepResult<E::Observation> = self.environment.step(action as i64);
        self.episode_idx += 1;
        self.observation = step_result.observation.flatten();
        self.terminated =
            self.episode_idx >= E::max_steps() || step_result.terminated || step_result.truncated;
        step_result.reward
    }

    fn get(&mut self) -> Option<&mut Self> {
        if self.terminated {
            return None;
        }

        Some(self)
    }
}

impl<T> RlState for GymRsInput<T>
where
    T: GymRsEnvExt,
    T::Observation: Copy + Flatten + Send + Sync,
{
    fn is_terminal(&mut self) -> bool {
        self.terminated
    }

    fn get_initial_state(&self) -> Vec<f64> {
        self.initial_state.flatten()
    }
}

impl<T> Reset<GymRsInput<T>> for ResetEngine
where
    T: GymRsEnvExt,
    T::Observation: Copy + Flatten + Send + Sync,
{
    fn reset(item: &mut GymRsInput<T>) {
        item.environment.reset(None, Default::default());
        item.environment.set_state(item.initial_state);
        item.observation = item.initial_state.flatten();
        item.terminated = false;
        item.episode_idx = 0;
    }
}

impl<T> Generate<(), GymRsInput<T>> for GenerateEngine
where
    T: GymRsEnvExt,
    T::Observation: Copy + Flatten + Send + Sync,
{
    fn generate(_from: ()) -> GymRsInput<T> {
        let mut environment: T = T::create();
        let initial_state = environment.reset(None, Default::default());
        let observation = initial_state.flatten();

        GymRsInput {
            environment,
            terminated: false,
            episode_idx: 0,
            initial_state,
            observation,
        }
    }
}

#[derive(Clone)]
pub struct GymRsQEngine<T>(PhantomData<T>);
#[derive(Clone)]
pub struct GymRsEngine<T>(PhantomData<T>);

impl<T> Core for GymRsQEngine<T>
where
    T: GymRsEnvExt,
    T::Observation: Copy + Flatten + Send + Sync,
{
    type Individual = QProgram;
    type ProgramParameters = QProgramGeneratorParameters;
    type State = GymRsInput<T>;
    type FitnessMarker = ();
    type Generate = GenerateEngine;
    type Fitness = FitnessEngine;
    type Reset = ResetEngine;
    type Breed = BreedEngine;
    type Mutate = MutateEngine;
    type Status = StatusEngine;
    type Freeze = FreezeEngine;
}

impl<T> Core for GymRsEngine<T>
where
    T: GymRsEnvExt,
    T::Observation: Copy + Flatten + Send + Sync,
{
    type Individual = Program;
    type ProgramParameters = ProgramGeneratorParameters;
    type State = GymRsInput<T>;
    type FitnessMarker = UseRlFitness;
    type Generate = GenerateEngine;
    type Fitness = FitnessEngine;
    type Reset = ResetEngine;
    type Breed = BreedEngine;
    type Mutate = MutateEngine;
    type Status = StatusEngine;
    type Freeze = FreezeEngine;
}
