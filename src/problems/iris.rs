use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use strum::EnumCount;
use tokio::runtime::Runtime;

use crate::{
    core::{
        engines::{
            breed_engine::BreedEngine,
            core_engine::Core,
            fitness_engine::FitnessEngine,
            freeze_engine::FreezeEngine,
            generate_engine::{Generate, GenerateEngine},
            mutate_engine::MutateEngine,
            reset_engine::{Reset, ResetEngine},
            status_engine::StatusEngine,
        },
        environment::State,
        program::{Program, ProgramGeneratorParameters},
    },
    utils::{loader::download_and_load_csv, random::generator},
};

pub const IRIS_DATASET_LINK: &'static str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data";

#[derive(
    Debug,
    Clone,
    Copy,
    Eq,
    PartialEq,
    EnumCount,
    PartialOrd,
    Ord,
    strum::Display,
    Serialize,
    Deserialize,
    Hash,
)]
pub enum IrisClass {
    #[serde(rename = "Iris-setosa")]
    Setosa = 0,
    #[serde(rename = "Iris-versicolor")]
    Versicolour = 1,
    #[serde(rename = "Iris-virginica")]
    Virginica = 2,
}

pub struct IrisLgp;

#[derive(Deserialize, Debug, Clone, PartialEq, PartialOrd, Serialize)]
pub struct IrisInput {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    class: IrisClass,
}

pub struct IrisState {
    data: Vec<IrisInput>,
    idx: usize,
}

impl State for IrisState {
    fn get_value(&self, idx: usize) -> f64 {
        let item = &self.data[self.idx];

        match idx {
            0 => item.sepal_length,
            1 => item.sepal_width,
            2 => item.petal_length,
            3 => item.petal_width,
            _ => unreachable!(),
        }
    }

    fn execute_action(&mut self, action: usize) -> f64 {
        let item = &self.data[self.idx];
        self.idx += 1;
        let correct_class = item.class as usize;
        let is_correct = correct_class == action;
        is_correct as usize as f64
    }

    fn get(&mut self) -> Option<&mut Self> {
        if self.idx >= self.data.len() {
            return None;
        }

        Some(self)
    }
}

impl Reset<IrisState> for ResetEngine {
    fn reset(item: &mut IrisState) {
        item.idx = 0;
    }
}

impl Generate<(), IrisState> for GenerateEngine {
    fn generate(_using: ()) -> IrisState {
        let runtime = Runtime::new().unwrap();
        let mut data = runtime
            .block_on(download_and_load_csv(IRIS_DATASET_LINK))
            .expect("Failed to download and load the dataset");

        data.shuffle(&mut generator());

        IrisState { data, idx: 0 }
    }
}

#[derive(Clone)]
pub struct IrisEngine;

impl Core for IrisEngine {
    type State = IrisState;
    type Individual = Program;
    type ProgramParameters = ProgramGeneratorParameters;
    type FitnessMarker = ();
    type Generate = GenerateEngine;
    type Fitness = FitnessEngine;
    type Reset = ResetEngine;
    type Breed = BreedEngine;
    type Mutate = MutateEngine;
    type Status = StatusEngine;
    type Freeze = FreezeEngine;
}

#[cfg(test)]
mod test {

    use itertools::Itertools;

    use crate::core::engines::core_engine::HyperParametersBuilder;
    use crate::core::engines::status_engine::Status;
    use crate::core::instruction::InstructionGeneratorParametersBuilder;
    use crate::core::program::ProgramGeneratorParametersBuilder;
    use crate::utils::benchmark_tools::save_experiment;
    use crate::utils::misc::VoidResultAnyError;

    use super::*;

    #[test]
    fn baseline() -> VoidResultAnyError {
        let name = "iris_baseline";
        let instruction_parameters = InstructionGeneratorParametersBuilder::default()
            .n_actions(3)
            .n_inputs(4)
            .build()?;
        let program_parameters = ProgramGeneratorParametersBuilder::default()
            .max_instructions(100)
            .instruction_generator_parameters(instruction_parameters)
            .build()?;
        let parameters = HyperParametersBuilder::<IrisEngine>::default()
            .program_parameters(program_parameters)
            .n_trials(1)
            .n_generations(200)
            .mutation_percent(0.)
            .crossover_percent(0.)
            .build()?;

        let populations = parameters
            .build_engine()
            .take(parameters.n_generations)
            .collect_vec();

        save_experiment(&populations, &parameters, name)?;

        let last_population = populations.last().unwrap();
        assert!(last_population
            .iter()
            .all(|individual| Some(StatusEngine::get_fitness(individual))
                == last_population.first().map(StatusEngine::get_fitness)));

        Ok(())
    }

    #[test]
    fn mutation() -> VoidResultAnyError {
        let name = "iris_mutation";
        let instruction_parameters = InstructionGeneratorParametersBuilder::default()
            .n_actions(3)
            .n_inputs(4)
            .build()?;
        let program_parameters = ProgramGeneratorParametersBuilder::default()
            .max_instructions(100)
            .instruction_generator_parameters(instruction_parameters)
            .build()?;
        let parameters = HyperParametersBuilder::<IrisEngine>::default()
            .program_parameters(program_parameters)
            .mutation_percent(1.0)
            .crossover_percent(0.)
            .n_trials(1)
            .build()?;

        let populations = parameters
            .build_engine()
            .take(parameters.n_generations)
            .collect_vec();

        save_experiment(&populations, &parameters, name)?;

        Ok(())
    }

    #[test]
    fn crossover() -> VoidResultAnyError {
        let name = "iris_crossover";
        let instruction_parameters = InstructionGeneratorParametersBuilder::default()
            .n_actions(3)
            .n_inputs(4)
            .build()?;
        let program_parameters = ProgramGeneratorParametersBuilder::default()
            .max_instructions(100)
            .instruction_generator_parameters(instruction_parameters)
            .build()?;
        let parameters = HyperParametersBuilder::<IrisEngine>::default()
            .program_parameters(program_parameters)
            .mutation_percent(0.)
            .crossover_percent(1.0)
            .n_trials(1)
            .build()?;

        let populations = parameters
            .build_engine()
            .take(parameters.n_generations)
            .collect_vec();

        save_experiment(&populations, &parameters, name)?;

        Ok(())
    }

    #[test]
    fn full() -> VoidResultAnyError {
        let name = "iris_full";

        let instruction_parameters = InstructionGeneratorParametersBuilder::default()
            .n_actions(3)
            .n_inputs(4)
            .build()?;
        let program_parameters = ProgramGeneratorParametersBuilder::default()
            .max_instructions(100)
            .instruction_generator_parameters(instruction_parameters)
            .build()?;
        let parameters = HyperParametersBuilder::<IrisEngine>::default()
            .program_parameters(program_parameters)
            .mutation_percent(0.5)
            .crossover_percent(0.5)
            .n_trials(1)
            .build()?;

        let populations = parameters
            .build_engine()
            .take(parameters.n_generations)
            .collect_vec();

        save_experiment(&populations, &parameters, name)?;

        Ok(())
    }
}
