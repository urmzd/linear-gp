use core::fmt;
use std::fmt::Display;

use serde::{Deserialize, Serialize};
use strum::EnumCount;

use crate::{
    core::{
        algorithm::{GeneticAlgorithm, Loader},
        inputs::ValidInput,
        program::Program,
    },
    extensions::classification::{ClassificationInput, ClassificationParameters},
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

impl GeneticAlgorithm for IrisLgp {
    type O = Program<ClassificationParameters<IrisInput>>;
}

impl<'a> Loader for IrisLgp {
    type InputType = IrisInput;
}

#[derive(Deserialize, Debug, Clone, PartialEq, PartialOrd, Serialize)]
pub struct IrisInput {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    class: IrisClass,
}

impl ClassificationInput for IrisInput {
    fn get_class(&self) -> usize {
        self.class as usize
    }
}

impl Display for IrisInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let serialized = serde_json::to_string(&self).unwrap();
        f.write_str(&serialized)
    }
}

impl ValidInput for IrisInput {
    const N_INPUTS: usize = 4;
    const N_ACTIONS: usize = 3;

    fn flat(&self) -> Vec<f64> {
        [
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width,
        ]
        .to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        core::{
            algorithm::{GeneticAlgorithm, HyperParameters, Loader},
            instruction::InstructionGeneratorParameters,
            program::ProgramGeneratorParameters,
        },
        extensions::classification::ClassificationParameters,
        utils::benchmark_tools::{log_benchmarks, output_benchmarks},
    };
    use itertools::Itertools;
    use std::error;

    use super::{IrisInput, IrisLgp, IRIS_DATASET_LINK};

    use tempfile::NamedTempFile;

    use std::io::Write;

    struct ContentFilePair(pub String, pub NamedTempFile);

    async fn get_iris_content() -> Result<ContentFilePair, Box<dyn error::Error>> {
        let tmp_file = NamedTempFile::new()?;
        let response = reqwest::get(IRIS_DATASET_LINK).await?;
        let content = response.text().await?;
        writeln!(&tmp_file, "{}", &content)?;

        Ok(ContentFilePair(content, tmp_file))
    }
    // TODO: Update tests to include assertions about benchmark trends.
    #[tokio::test]
    async fn sanity_test_mutation_crossover() -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;
        let inputs = IrisLgp::load_from_csv(tmp_file.path());

        let hyper_params = HyperParameters {
            population_size: 100,
            n_generations: 100,
            gap: 0.5,
            mutation_percent: 0.5,
            crossover_percent: 0.5,
            fitness_parameters: ClassificationParameters::new(inputs),
            program_parameters: ProgramGeneratorParameters::new(
                100,
                InstructionGeneratorParameters::new(1, 10.),
            ),
        };

        debug_assert_eq!(hyper_params.crossover_percent, 0.5);
        debug_assert_eq!(hyper_params.mutation_percent, 0.5);

        let populations = IrisLgp::build(hyper_params.clone()).collect_vec();

        const TEST_NAME: &'static str = "iris-mutate-crossover";

        output_benchmarks(&populations, TEST_NAME)?;
        log_benchmarks(&populations, &hyper_params, TEST_NAME)?;

        Ok(())
    }
    #[tokio::test]
    async fn sanity_test_mutation() -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;
        let inputs = IrisLgp::load_from_csv(tmp_file.path());

        let hyper_params = HyperParameters {
            population_size: 100,
            n_generations: 100,
            gap: 0.5,
            mutation_percent: 1.,
            crossover_percent: 0.,
            fitness_parameters: ClassificationParameters::new(inputs),
            program_parameters: ProgramGeneratorParameters::new(
                100,
                InstructionGeneratorParameters::new(1, 10.),
            ),
        };

        assert_eq!(hyper_params.crossover_percent, 0.);
        assert_eq!(hyper_params.mutation_percent, 1.);

        let populations = IrisLgp::build(hyper_params.clone()).collect_vec();

        const TEST_NAME: &'static str = "iris-mutate";
        output_benchmarks(&populations, TEST_NAME)?;
        log_benchmarks(&populations, &hyper_params, TEST_NAME)?;

        Ok(())
    }

    #[tokio::test]
    async fn sanity_test_crossover() -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;
        let inputs = IrisLgp::load_from_csv(tmp_file.path());

        let hyper_params = HyperParameters {
            population_size: 100,
            n_generations: 100,
            gap: 0.5,
            mutation_percent: 0.,
            crossover_percent: 1.,
            fitness_parameters: ClassificationParameters::new(inputs),
            program_parameters: ProgramGeneratorParameters::new(
                100,
                InstructionGeneratorParameters::new(1, 10.),
            ),
        };

        assert_eq!(hyper_params.crossover_percent, 1.);
        assert_eq!(hyper_params.mutation_percent, 0.);

        let populations = IrisLgp::build(hyper_params.clone()).collect_vec();

        const TEST_NAME: &'static str = "iris-crossover";
        output_benchmarks(&populations, TEST_NAME)?;
        log_benchmarks(&populations, &hyper_params, TEST_NAME)?;

        Ok(())
    }

    #[tokio::test]
    async fn sanity_test_base() -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;
        let inputs = IrisLgp::load_from_csv(tmp_file.path());

        let hyper_params = HyperParameters {
            population_size: 100,
            n_generations: 250,
            gap: 0.5,
            mutation_percent: 0.,
            crossover_percent: 0.,
            fitness_parameters: ClassificationParameters::new(inputs),
            program_parameters: ProgramGeneratorParameters::new(
                100,
                InstructionGeneratorParameters::new(1, 10.),
            ),
        };

        assert_eq!(hyper_params.crossover_percent, 0.);
        assert_eq!(hyper_params.mutation_percent, 0.);

        let populations = IrisLgp::build(hyper_params.clone()).collect_vec();

        let worst = populations.last().unwrap().worst().unwrap().clone();
        let median = populations.last().unwrap().median().unwrap().clone();
        let best = populations.last().unwrap().best().unwrap().clone();

        const TEST_NAME: &'static str = "iris-default";

        output_benchmarks(&populations, TEST_NAME)?;
        log_benchmarks(&populations, &hyper_params, TEST_NAME)?;

        if worst.fitness != median.fitness || median.fitness != best.fitness {
            // TODO: Create concrete error type; SNAFU or Failure?
            panic!("GP was unable to converge in given time... \n Best: {:?}, \n Median: {:?} \n Worst: {:?} \n", best.fitness, median.fitness, worst.fitness);
        }

        Ok(())
    }

    #[tokio::test]
    async fn given_population_when_breeding_occurs_then_population_capacity_is_met(
    ) -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;

        let inputs = IrisLgp::load_from_csv(tmp_file.path());
        let hyper_params = HyperParameters {
            population_size: 100,
            n_generations: 100,
            gap: 0.5,
            mutation_percent: 0.,
            crossover_percent: 0.,
            fitness_parameters: ClassificationParameters::new(inputs),
            program_parameters: ProgramGeneratorParameters::new(
                100,
                InstructionGeneratorParameters::new(1, 10.),
            ),
        };

        let (mut population, mut hyper_params) = IrisLgp::init_pop(hyper_params);

        (population, hyper_params) = IrisLgp::eval_fitness(population, hyper_params);
        (population, hyper_params) = IrisLgp::rank(population, hyper_params);
        (population, hyper_params) = IrisLgp::survive(population, hyper_params);

        let dropped_pop_len = population.len();

        assert!(dropped_pop_len < hyper_params.population_size);

        (population, hyper_params) = IrisLgp::variation(population, hyper_params);

        assert!(population.len() == hyper_params.population_size);

        Ok(())
    }

    #[tokio::test]
    async fn given_population_and_retention_rate_when_selection_occurs_then_population_is_cut_by_dropout(
    ) -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;

        let inputs = IrisLgp::load_from_csv(tmp_file.path());

        let hyper_params = HyperParameters {
            population_size: 100,
            n_generations: 100,
            gap: 0.5,
            mutation_percent: 0.,
            crossover_percent: 0.,
            fitness_parameters: ClassificationParameters::new(inputs),
            program_parameters: ProgramGeneratorParameters::new(
                100,
                InstructionGeneratorParameters::new(1, 10.),
            ),
        };

        let (mut population, mut hyper_params) = IrisLgp::init_pop(hyper_params);

        (population, hyper_params) = IrisLgp::eval_fitness(population, hyper_params);
        (population, hyper_params) = IrisLgp::rank(population, hyper_params);
        (population, hyper_params) = IrisLgp::survive(population, hyper_params);

        assert_eq!(
            population.len(),
            ((hyper_params.population_size as f64 * (1.0 - hyper_params.gap)).floor() as usize)
        );

        Ok(())
    }

    #[tokio::test]
    async fn given_inputs_and_hyperparams_when_population_is_initialized_then_population_generated_with_hyperparams_and_inputs(
    ) -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;

        let inputs = IrisLgp::load_from_csv(tmp_file.path());

        let hyper_params = HyperParameters {
            population_size: 100,
            n_generations: 100,
            gap: 0.5,
            mutation_percent: 0.,
            crossover_percent: 0.,
            fitness_parameters: ClassificationParameters::new(inputs),
            program_parameters: ProgramGeneratorParameters::new(
                100,
                InstructionGeneratorParameters::new(1, 10.),
            ),
        };

        let (population, hyper_params) = IrisLgp::init_pop(hyper_params);

        assert_eq!(population.len(), hyper_params.population_size);

        for individual in population {
            assert!(individual.instructions.len() <= 100)
        }

        Ok(())
    }

    #[tokio::test]
    async fn given_iris_dataset_when_csv_is_read_then_rows_are_deserialized_as_structs(
    ) -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(content, _) = get_iris_content().await?;
        assert_ne!(content.len(), 0);

        let content_bytes = content.as_bytes();

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(content_bytes);

        let data = reader.deserialize();
        let mut count = 0;

        for result in data {
            let _record: IrisInput = result?;
            count += 1;
        }

        assert_ne!(count, 0);

        Ok(())
    }

    #[tokio::test]
    async fn given_iris_dataset_when_csv_path_is_provided_then_collection_of_iris_structs_are_returned(
    ) -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmpfile) = get_iris_content().await?;
        let inputs = IrisLgp::load_from_csv(tmpfile.path());
        assert_ne!(inputs.len(), 0);
        Ok(())
    }
}
