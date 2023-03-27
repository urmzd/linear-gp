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
pub struct GymRsInput<E: Env, const N_INPUTS: usize, const N_ACTIONS: usize> {
    environment: E,
    terminated: bool,
    episode_idx: usize,
    initial_state: E::Observation,
}

impl<T, const N_PUTS: usize, const N_ACTS: usize> State for GymRsInput<T, N_PUTS, N_ACTS>
where
    T: Env,
{
    const N_INPUTS: usize = N_PUTS;
    const N_ACTIONS: usize = N_ACTS;

    fn get_value(&self, idx: usize) -> f64 {
        self.environment.get_observation_property(idx)
    }

    fn execute_action(&mut self, action: usize) -> f64 {
        let action_reward = self.environment.step(action);
        self.episode_idx += 1;
        self.terminated = self.episode_idx >= T::episode_length() || action_reward.done;
        action_reward.reward
    }

    fn get(&mut self) -> Option<&mut Self> {
        if self.terminated {
            return None;
        }

        Some(self)
    }
}

impl<T, const N_PUTS: usize, const N_ACTS: usize> RlState for GymRsInput<T, N_PUTS, N_ACTS>
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

impl<T, const N_PUTS: usize, const N_ACTS: usize> Reset<GymRsInput<T, N_PUTS, N_ACTS>>
    for ResetEngine
where
    T: Env,
{
    fn reset(item: &mut GymRsInput<T, N_PUTS, N_ACTS>) {
        item.environment.reset(None, false, None);
        item.environment.set_observation(item.initial_state);
        item.terminated = false;
        item.episode_idx = 0;
    }
}

impl<T, const N_PUTS: usize, const N_ACTS: usize> Generate<(), GymRsInput<T, N_PUTS, N_ACTS>>
    for GenerateEngine
where
    T: Env,
{
    fn generate(_from: ()) -> GymRsInput<T, N_PUTS, N_ACTS> {
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
pub struct GymRsQEngine<T, const N_PUTS: usize, const N_ACTS: usize>(PhantomData<T>);
#[derive(Clone)]
pub struct GymRsEngine<T, const N_PUTS: usize, const N_ACTS: usize>(PhantomData<T>);

impl<T, const N_PUTS: usize, const N_ACTS: usize> Core for GymRsQEngine<T, N_PUTS, N_ACTS>
where
    T: Env,
{
    type Individual = QProgram;
    type ProgramParameters = QProgramGeneratorParameters;
    type State = GymRsInput<T, N_PUTS, N_ACTS>;
    type FitnessMarker = ();
    type Generate = GenerateEngine;
    type Fitness = FitnessEngine;
    type Reset = ResetEngine;
    type Breed = BreedEngine;
    type Mutate = MutateEngine;
    type Status = StatusEngine;
    type Freeze = FreezeEngine;
}

impl<T, const N_PUTS: usize, const N_ACTS: usize> Core for GymRsEngine<T, N_PUTS, N_ACTS>
where
    T: Env,
{
    type Individual = Program;
    type ProgramParameters = ProgramGeneratorParameters;
    type State = GymRsInput<T, N_PUTS, N_ACTS>;
    type FitnessMarker = UseRlFitness;
    type Generate = GenerateEngine;
    type Fitness = FitnessEngine;
    type Reset = ResetEngine;
    type Breed = BreedEngine;
    type Mutate = MutateEngine;
    type Status = StatusEngine;
    type Freeze = FreezeEngine;
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;
    use crate::core::config::load_hyper_parameters;
    use crate::core::engines::core_engine::HyperParameters;
    
    use crate::utils::benchmark_tools::{save_benchmarks, save_results, with_named_logger};
    use crate::utils::misc::VoidResultAnyError;

    use gym_rs::envs::classical_control::cartpole::CartPoleEnv;
    use gym_rs::envs::classical_control::mountain_car::MountainCarEnv;

    #[test]
    fn cart_pole_q() -> VoidResultAnyError {
        with_named_logger!("cart_pole_q", {
            let parameters: HyperParameters<GymRsQEngine<CartPoleEnv, 4, 2>> =
                load_hyper_parameters("assets/parameters/cart-pole-q.json")?;
            let populations = parameters.build_engine().take(parameters.n_generations).collect_vec();

            save_benchmarks(&populations, &parameters, NAME)?;
            save_results(&populations, NAME)?;

            Ok(())
        })
    }

    #[test]
    fn cart_pole_lgp() -> VoidResultAnyError {
        with_named_logger!("cart_pole_lgp", {
            let parameters: HyperParameters<GymRsEngine<CartPoleEnv, 4, 2>> =
                load_hyper_parameters("assets/parameters/cart-pole-lgp.json")?;

            let populations = parameters.build_engine().take(parameters.n_generations).collect_vec();

            save_benchmarks(&populations, &parameters, NAME)?;
            save_results(&populations, NAME)?;

            Ok(())
        })
    }

    #[test]
    fn mountain_cart_lgp() -> VoidResultAnyError {
        with_named_logger!("mountain_car_lgp", {
            let parameters: HyperParameters<GymRsEngine<MountainCarEnv, 2, 3>> =
                load_hyper_parameters("assets/parameters/mountain-car-lgp.json")?;
            let populations = parameters.build_engine().take(parameters.n_generations).collect_vec();

            save_benchmarks(&populations, &parameters, NAME)?;
            save_results(&populations, NAME)?;

            Ok(())
        })
    }

    #[test]
    fn mountain_car_q() -> VoidResultAnyError {
        with_named_logger!("mountain_car_q", {
            let parameters: HyperParameters<GymRsQEngine<MountainCarEnv, 2, 3>> =
                load_hyper_parameters("assets/parameters/mountain-car-q.json")?;
            let populations = parameters.build_engine().take(parameters.n_generations).collect_vec();

            save_benchmarks(&populations, &parameters, NAME)?;
            save_results(&populations, NAME)?;

            Ok(())
        })
    }
}
