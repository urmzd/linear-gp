mod set_up;

use std::error;

use lgp::{
    core::{
        algorithm::{EventHooks, GeneticAlgorithm, HyperParameters, Loader},
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    },
    extensions::classification::ClassificationParameters,
};
use set_up::{get_iris_content, ContentFilePair, IrisInput, IrisLgp};

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error>> {
    let ContentFilePair(_, file) = get_iris_content().await?;
    let inputs = IrisLgp::load_inputs(file.path());

    let mut hyper_params = HyperParameters {
        population_size: 1,
        n_generations: 1,
        gap: 0.5,
        mutation_percent: 0.5,
        lazy_evaluate: true,
        crossover_percent: 0.5,
        fitness_parameters: ClassificationParameters::new(inputs),
        program_parameters: ProgramGeneratorParameters::new(
            100,
            InstructionGeneratorParameters::from::<IrisInput>(1),
        ),
    };

    IrisLgp::execute(&mut hyper_params, EventHooks::default())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use lgp::{
        core::{
            algorithm::{EventHooks, GeneticAlgorithm, HyperParameters, Loader},
            characteristics::{Fitness, FitnessScore},
            instruction::InstructionGeneratorParameters,
            program::{Program, ProgramGeneratorParameters},
        },
        extensions::classification::ClassificationParameters,
        utils::plots::plot_benchmarks,
    };
    use log::debug;
    use more_asserts::{assert_le, assert_lt};
    use pretty_assertions::{assert_eq, assert_ne};
    use std::error;

    use crate::set_up::{get_iris_content, ContentFilePair, IrisInput, IrisLgp};

    // TODO: Update tests to include assertions about benchmark trends.
    #[tokio::test]
    async fn given_lgp_instance_with_mutation_and_crossover_operations_when_sufficient_iterations_have_been_met_then_population_shows_increase_in_median_and_best_fitness(
    ) -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;
        let inputs = IrisLgp::load_inputs(tmp_file.path());

        let mut hyper_params: HyperParameters<Program<ClassificationParameters<IrisInput>>> =
            HyperParameters {
                population_size: 100,
                n_generations: 100,
                gap: 0.5,
                mutation_percent: 0.5,
                crossover_percent: 0.5,
                lazy_evaluate: true,
                fitness_parameters: ClassificationParameters::new(inputs),
                program_parameters: ProgramGeneratorParameters::new(
                    100,
                    InstructionGeneratorParameters::from::<IrisInput>(1),
                ),
            };

        assert_eq!(hyper_params.crossover_percent, 0.5);
        assert_eq!(hyper_params.mutation_percent, 0.5);

        let mut populations = vec![];

        IrisLgp::execute(
            &mut hyper_params,
            EventHooks::default().with_on_post_rank(&mut |population, _| {
                populations.push(population.clone());
            }),
        )?;

        const PLOT_FILE_NAME: &'static str = "plots/tests/lgp_with_mutate_crossover_test.png";
        plot_benchmarks(populations, PLOT_FILE_NAME, 0.0..1.0)?;
        Ok(())
    }
    #[tokio::test]
    async fn given_lgp_instance_with_mutation_operations_when_sufficient_iterations_have_been_met_then_population_shows_increase_in_median_and_best_fitness(
    ) -> Result<(), Box<dyn error::Error>> {
        pretty_env_logger::init();
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;
        let inputs = IrisLgp::load_inputs(tmp_file.path());

        let mut hyper_params: HyperParameters<Program<ClassificationParameters<IrisInput>>> =
            HyperParameters {
                population_size: 100,
                n_generations: 100,
                gap: 0.5,
                mutation_percent: 1.,
                crossover_percent: 0.,
                lazy_evaluate: true,
                fitness_parameters: ClassificationParameters::new(inputs),
                program_parameters: ProgramGeneratorParameters::new(
                    100,
                    InstructionGeneratorParameters::from::<IrisInput>(1),
                ),
            };

        assert_eq!(hyper_params.crossover_percent, 0.);
        assert_eq!(hyper_params.mutation_percent, 1.);

        let mut populations = vec![];

        IrisLgp::execute(
            &mut hyper_params,
            EventHooks::default().with_on_post_rank(&mut |population, _| {
                populations.push(population.clone());
                debug!("{:?}", population.iter().map(|v| v.fitness).collect_vec())
            }),
        )?;

        const PLOT_FILE_NAME: &'static str = "plots/tests/lgp_with_mutate_test.png";
        plot_benchmarks(populations, PLOT_FILE_NAME, 0.0..1.0)?;
        Ok(())
    }

    #[tokio::test]
    async fn given_lgp_instance_with_crossover_operations_when_sufficient_iterations_have_been_met_then_population_shows_increase_in_worst_and_median_fitness(
    ) -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;
        let inputs = IrisLgp::load_inputs(tmp_file.path());

        let mut hyper_params: HyperParameters<Program<ClassificationParameters<IrisInput>>> =
            HyperParameters {
                population_size: 100,
                n_generations: 100,
                gap: 0.5,
                mutation_percent: 0.,
                crossover_percent: 1.,
                lazy_evaluate: true,
                fitness_parameters: ClassificationParameters::new(inputs),
                program_parameters: ProgramGeneratorParameters::new(
                    100,
                    InstructionGeneratorParameters::from::<IrisInput>(1),
                ),
            };

        assert_eq!(hyper_params.crossover_percent, 1.);
        assert_eq!(hyper_params.mutation_percent, 0.);

        let mut populations = vec![];

        IrisLgp::execute(
            &mut hyper_params,
            EventHooks::default().with_on_post_rank(&mut |population, _| {
                populations.push(population.clone());
            }),
        )?;

        const PLOT_FILE_NAME: &'static str = "./plots/tests/lgp_with_crossover_test.png";
        plot_benchmarks(populations, PLOT_FILE_NAME, 0.0..1.0)?;

        Ok(())
    }

    #[tokio::test]
    async fn given_lgp_instance_when_sufficient_iterations_have_been_used_then_population_contains_the_same_benchmark_fitness(
    ) -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;
        let inputs = IrisLgp::load_inputs(tmp_file.path());

        let mut hyper_params: HyperParameters<Program<ClassificationParameters<IrisInput>>> =
            HyperParameters {
                population_size: 100,
                n_generations: 250,
                gap: 0.5,
                mutation_percent: 0.,
                crossover_percent: 0.,
                lazy_evaluate: true,
                fitness_parameters: ClassificationParameters::new(inputs),
                program_parameters: ProgramGeneratorParameters::new(
                    100,
                    InstructionGeneratorParameters::from::<IrisInput>(1),
                ),
            };

        assert_eq!(hyper_params.crossover_percent, 0.);
        assert_eq!(hyper_params.mutation_percent, 0.);

        let mut populations = vec![];

        let mut best = 0;
        let mut median = 0;
        let mut worst = 0;

        let mut best_f = FitnessScore::NotEvaluated;
        let mut median_f = FitnessScore::NotEvaluated;
        let mut worst_f = FitnessScore::NotEvaluated;

        IrisLgp::execute(
            &mut hyper_params,
            EventHooks::default().with_on_post_rank(&mut |population, _| {
                populations.push(population.clone());

                worst_f = population.last().unwrap().get_fitness();
                median_f = population.middle().unwrap().get_fitness();
                best_f = population.first().unwrap().get_fitness();

                worst = population
                    .last()
                    .unwrap()
                    .get_fitness()
                    .unwrap_or(-200.)
                    .to_bits();
                median = population
                    .middle()
                    .unwrap()
                    .get_fitness()
                    .unwrap_or(-200.)
                    .to_bits();
                best = population
                    .first()
                    .unwrap()
                    .get_fitness()
                    .unwrap_or(-200.)
                    .to_bits();
            }),
        )?;

        // TODO: Pull the graph section out into a seperate function.
        const PLOT_FILE_NAME: &'static str = "plots/tests/lgp_smoke_test.png";
        plot_benchmarks(populations, PLOT_FILE_NAME, 0.0..1.0)?;

        debug!(
            "Total: Compare Worst to Median {:?}",
            worst_f.unwrap().total_cmp(&median_f.unwrap())
        );
        debug!(
            "Total: Compare Median to Best {:?}",
            median_f.unwrap().total_cmp(&best_f.unwrap())
        );

        debug!(
            "Partial: Compare Worst to Median {:?}",
            worst_f.unwrap().partial_cmp(&median_f.unwrap())
        );
        debug!(
            "Partial: Compare Median to Best {:?}",
            median_f.unwrap().partial_cmp(&best_f.unwrap())
        );

        debug!("Normal: Compare Worst to Median {:?}", worst_f == median_f);
        debug!("Normal: Compare Median to Best {:?}", median_f == best_f);

        if worst != median || median != best {
            // TODO: Create concrete error type; SNAFU or Failure?
            panic!("GP was unable to converge in given time... \n Best: {:b}, \n Median: {:b} \n Worst: {:b} \n", best, median, worst);
        }

        Ok(())
    }

    #[tokio::test]
    async fn given_population_when_breeding_occurs_then_population_capacity_is_met(
    ) -> Result<(), Box<dyn error::Error>> {
        let ContentFilePair(_, tmp_file) = get_iris_content().await?;

        let inputs = IrisLgp::load_inputs(tmp_file.path());
        let mut hyper_params: HyperParameters<Program<ClassificationParameters<IrisInput>>> =
            HyperParameters {
                population_size: 100,
                n_generations: 100,
                gap: 0.5,
                mutation_percent: 0.,
                crossover_percent: 0.,
                lazy_evaluate: true,
                fitness_parameters: ClassificationParameters::new(inputs),
                program_parameters: ProgramGeneratorParameters::new(
                    100,
                    InstructionGeneratorParameters::from::<IrisInput>(1),
                ),
            };

        let mut population = IrisLgp::init_population(&hyper_params);

        IrisLgp::rank(&mut population, &mut hyper_params.fitness_parameters, true);
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

        let inputs = IrisLgp::load_inputs(tmp_file.path());

        let mut hyper_params: HyperParameters<Program<ClassificationParameters<IrisInput>>> =
            HyperParameters {
                population_size: 100,
                n_generations: 100,
                gap: 0.5,
                mutation_percent: 0.,
                crossover_percent: 0.,
                lazy_evaluate: true,
                fitness_parameters: ClassificationParameters::new(inputs),
                program_parameters: ProgramGeneratorParameters::new(
                    100,
                    InstructionGeneratorParameters::from::<IrisInput>(1),
                ),
            };

        let mut population = IrisLgp::init_population(&hyper_params);

        IrisLgp::rank(&mut population, &mut hyper_params.fitness_parameters, true);
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

        let inputs = IrisLgp::load_inputs(tmp_file.path());

        let hyper_params: HyperParameters<Program<ClassificationParameters<IrisInput>>> =
            HyperParameters {
                population_size: 100,
                n_generations: 100,
                gap: 0.5,
                mutation_percent: 0.,
                crossover_percent: 0.,
                lazy_evaluate: true,
                fitness_parameters: ClassificationParameters::new(inputs),
                program_parameters: ProgramGeneratorParameters::new(
                    100,
                    InstructionGeneratorParameters::from::<IrisInput>(1),
                ),
            };

        let population = IrisLgp::init_population(&hyper_params);

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
        let inputs = IrisLgp::load_inputs(tmpfile.path());
        assert_ne!(inputs.len(), 0);
        Ok(())
    }
}
