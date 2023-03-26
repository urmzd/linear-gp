use serde::{Deserialize, Serialize};
use strum::EnumCount;
use tokio::runtime::Runtime;

use crate::{
    core::{
        engines::{
            generate_engine::{Generate, GenerateEngine},
            reset_engine::{Reset, ResetEngine},
        },
        environment::State,
    },
    utils::loader::download_and_load_csv,
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
    const N_INPUTS: usize = 4;
    const N_ACTIONS: usize = 3;

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
        (item.class as usize == action) as usize as f64
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
        item.idx = 0
    }
}

impl Generate<(), IrisState> for GenerateEngine {
    fn generate(_using: ()) -> IrisState {
        let runtime = Runtime::new().unwrap();
        let data = runtime
            .block_on(download_and_load_csv(IRIS_DATASET_LINK))
            .expect("Failed to download and load the dataset");

        IrisState { data, idx: 0 }
    }
}
