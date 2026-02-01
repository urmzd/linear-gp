use std::marker::PhantomData;

use gym_rs::core::Env;

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

#[derive(Clone, Debug)]
pub struct GymRsInput<E: Env> {
    environment: E,
    terminated: bool,
    episode_idx: usize,
    initial_state: E::Observation,
}

impl<E> State for GymRsInput<E>
where
    E: Env,
{
    fn get_value(&self, idx: usize) -> f64 {
        self.environment.get_observation_property(idx)
    }

    fn execute_action(&mut self, action: usize) -> f64 {
        let action_reward = self.environment.step(action);
        self.episode_idx += 1;
        self.terminated = self.episode_idx >= E::episode_length() || action_reward.done;
        action_reward.reward
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
    T: Env,
{
    fn is_terminal(&mut self) -> bool {
        self.terminated
    }

    fn get_initial_state(&self) -> Vec<f64> {
        self.initial_state.into()
    }
}

impl<T> Reset<GymRsInput<T>> for ResetEngine
where
    T: Env,
{
    fn reset(item: &mut GymRsInput<T>) {
        item.environment.reset(None, false, None);
        item.environment.set_observation(item.initial_state);
        item.terminated = false;
        item.episode_idx = 0;
    }
}

impl<T> Generate<(), GymRsInput<T>> for GenerateEngine
where
    T: Env,
{
    fn generate(_from: ()) -> GymRsInput<T> {
        let mut environment: T = Env::new();
        let (initial_state, _) = environment.reset(None, false, None);

        GymRsInput {
            environment,
            terminated: false,
            episode_idx: 0,
            initial_state,
        }
    }
}

#[derive(Clone)]
pub struct GymRsQEngine<T>(PhantomData<T>);
#[derive(Clone)]
pub struct GymRsEngine<T>(PhantomData<T>);

impl<T> Core for GymRsQEngine<T>
where
    T: Env,
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
    T: Env,
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

