mod config;

use std::error;

use config::{get_iris_content, ContentFilePair, IrisInput, IrisLgp};
use lgp::{
    core::{
        algorithm::{GeneticAlgorithm, HyperParameters, Loader},
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    },
    extensions::classification::ClassificationParameters,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error>> {
    let ContentFilePair(_, file) = get_iris_content().await?;
    let inputs = IrisLgp::load_from_csv(file.path());

    let hyper_params = HyperParameters {
        population_size: 1,
        n_generations: 1,
        gap: 0.5,
        mutation_percent: 0.5,
        lazy_evaluate: false,
        crossover_percent: 0.5,
        fitness_parameters: ClassificationParameters::new(inputs),
        program_parameters: ProgramGeneratorParameters::new(
            100,
            InstructionGeneratorParameters::from::<IrisInput>(1),
        ),
    };

    IrisLgp::build(hyper_params).last();
    Ok(())
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use lgp::{
        core::{
            algorithm::{GeneticAlgorithm, HyperParameters, Loader},
            instruction::InstructionGeneratorParameters,
            program::{Program, ProgramGeneratorParameters},
        },
        extensions::classification::ClassificationParameters,
        utils::plots::plot_benchmarks,
    };
    use more_asserts::{assert_le, assert_lt};
    use pretty_assertions::{assert_eq, assert_ne};
    use std::error;

    use crate::config::{get_iris_content, ContentFilePair, IrisInput, IrisLgp};

    // TODO: Update tests to include assertions about benchmark trends.
    #[tokio::test]
    async fn sanity_test_mutation_crossover() -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;
        let inputs = IrisLgp::load_from_csv(tmp_file.path());

        let hyper_params: HyperParameters<Program<ClassificationParameters<IrisInput>>> =
            HyperParameters {
                population_size: 100,
                n_generations: 100,
                gap: 0.5,
                mutation_percent: 0.5,
                crossover_percent: 0.5,
                lazy_evaluate: false,
                fitness_parameters: ClassificationParameters::new(inputs),
                program_parameters: ProgramGeneratorParameters::new(
                    100,
                    InstructionGeneratorParameters::from::<IrisInput>(1),
                ),
            };

        assert_eq!(hyper_params.crossover_percent, 0.5);
        assert_eq!(hyper_params.mutation_percent, 0.5);

        let populations = IrisLgp::build(hyper_params).collect_vec();

        const PLOT_FILE_NAME: &'static str = "assets/plots/tests/iris/smoke/mutate_crossover.png";
        plot_benchmarks(populations, PLOT_FILE_NAME, 0.0..1.0)?;
        Ok(())
    }
    #[tokio::test]
    async fn sanity_test_mutation() -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;
        let inputs = IrisLgp::load_from_csv(tmp_file.path());

        let hyper_params: HyperParameters<Program<ClassificationParameters<IrisInput>>> =
            HyperParameters {
                population_size: 100,
                n_generations: 100,
                gap: 0.5,
                mutation_percent: 1.,
                crossover_percent: 0.,
                lazy_evaluate: false,
                fitness_parameters: ClassificationParameters::new(inputs),
                program_parameters: ProgramGeneratorParameters::new(
                    100,
                    InstructionGeneratorParameters::from::<IrisInput>(1),
                ),
            };

        assert_eq!(hyper_params.crossover_percent, 0.);
        assert_eq!(hyper_params.mutation_percent, 1.);

        let populations = IrisLgp::build(hyper_params).collect_vec();

        const PLOT_FILE_NAME: &'static str = "assets/plots/tests/iris/smoke/mutate.png";
        plot_benchmarks(populations, PLOT_FILE_NAME, 0.0..1.0)?;
        Ok(())
    }

    #[tokio::test]
    async fn sanity_test_crossover() -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;
        let inputs = IrisLgp::load_from_csv(tmp_file.path());

        let hyper_params: HyperParameters<Program<ClassificationParameters<IrisInput>>> =
            HyperParameters {
                population_size: 100,
                n_generations: 100,
                gap: 0.5,
                mutation_percent: 0.,
                crossover_percent: 1.,
                lazy_evaluate: false,
                fitness_parameters: ClassificationParameters::new(inputs),
                program_parameters: ProgramGeneratorParameters::new(
                    100,
                    InstructionGeneratorParameters::from::<IrisInput>(1),
                ),
            };

        assert_eq!(hyper_params.crossover_percent, 1.);
        assert_eq!(hyper_params.mutation_percent, 0.);

        let populations = IrisLgp::build(hyper_params).collect_vec();

        const PLOT_FILE_NAME: &'static str = "assets/plots/tests/iris/smoke/crossover.png";
        plot_benchmarks(populations, PLOT_FILE_NAME, 0.0..1.0)?;

        Ok(())
    }

    #[tokio::test]
    async fn sanity_test_base() -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;
        let inputs = IrisLgp::load_from_csv(tmp_file.path());

        let hyper_params: HyperParameters<Program<ClassificationParameters<IrisInput>>> =
            HyperParameters {
                population_size: 100,
                n_generations: 250,
                gap: 0.5,
                mutation_percent: 0.,
                crossover_percent: 0.,
                lazy_evaluate: false,
                fitness_parameters: ClassificationParameters::new(inputs),
                program_parameters: ProgramGeneratorParameters::new(
                    100,
                    InstructionGeneratorParameters::from::<IrisInput>(1),
                ),
            };

        assert_eq!(hyper_params.crossover_percent, 0.);
        assert_eq!(hyper_params.mutation_percent, 0.);

        let populations = IrisLgp::build(hyper_params).collect_vec();

        let worst = populations.last().unwrap().worst().unwrap().clone();
        let median = populations.last().unwrap().median().unwrap().clone();
        let best = populations.last().unwrap().best().unwrap().clone();

        // TODO: Pull the graph section out into a seperate function.
        const PLOT_FILE_NAME: &'static str = "assets/plots/tests/iris/smoke/default.png";
        plot_benchmarks(populations, PLOT_FILE_NAME, 0.0..1.0)?;

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
        let mut hyper_params: HyperParameters<Program<ClassificationParameters<IrisInput>>> =
            HyperParameters {
                population_size: 100,
                n_generations: 100,
                gap: 0.5,
                mutation_percent: 0.,
                crossover_percent: 0.,
                lazy_evaluate: false,
                fitness_parameters: ClassificationParameters::new(inputs),
                program_parameters: ProgramGeneratorParameters::new(
                    100,
                    InstructionGeneratorParameters::from::<IrisInput>(1),
                ),
            };

        let mut population = IrisLgp::init_pop(&hyper_params);

        IrisLgp::eval_fitness(&mut population, &mut hyper_params);
        IrisLgp::rank(&mut population);
        IrisLgp::apply_selection(&mut population, hyper_params.gap);

        let dropped_pop_len = population.len();

        assert_lt!(dropped_pop_len, hyper_params.population_size);

        IrisLgp::breed(&mut population, 0., 0., &hyper_params.program_parameters);

        assert_eq!(population.len(), hyper_params.population_size);

        Ok(())
    }

    #[tokio::test]
    async fn given_population_and_retention_rate_when_selection_occurs_then_population_is_cut_by_dropout(
    ) -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;

        let inputs = IrisLgp::load_from_csv(tmp_file.path());

        let mut hyper_params: HyperParameters<Program<ClassificationParameters<IrisInput>>> =
            HyperParameters {
                population_size: 100,
                n_generations: 100,
                gap: 0.5,
                mutation_percent: 0.,
                crossover_percent: 0.,
                lazy_evaluate: false,
                fitness_parameters: ClassificationParameters::new(inputs),
                program_parameters: ProgramGeneratorParameters::new(
                    100,
                    InstructionGeneratorParameters::from::<IrisInput>(1),
                ),
            };

        let mut population = IrisLgp::init_pop(&hyper_params);

        IrisLgp::eval_fitness(&mut population, &mut hyper_params);
        IrisLgp::rank(&mut population);
        IrisLgp::apply_selection(&mut population, hyper_params.gap);

        self::assert_eq!(
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

        let hyper_params: HyperParameters<Program<ClassificationParameters<IrisInput>>> =
            HyperParameters {
                population_size: 100,
                n_generations: 100,
                gap: 0.5,
                mutation_percent: 0.,
                crossover_percent: 0.,
                lazy_evaluate: false,
                fitness_parameters: ClassificationParameters::new(inputs),
                program_parameters: ProgramGeneratorParameters::new(
                    100,
                    InstructionGeneratorParameters::from::<IrisInput>(1),
                ),
            };

        let population = IrisLgp::init_pop(&hyper_params);

        self::assert_eq!(population.len(), hyper_params.population_size);

        for individual in population {
            assert_le!(individual.instructions.len(), 100)
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
