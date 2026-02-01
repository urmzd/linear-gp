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

pub const IRIS_DATASET_LINK: &str =
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

