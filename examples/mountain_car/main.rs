use gym_rs::{
    envs::classical_control::mountain_car::{MountainCarEnv, MountainCarObservation},
    utils::{custom::Sample, renderer::RenderMode},
};
use itertools::Itertools;
use lgp::{
    core::{
        algorithm::{EventHooks, GeneticAlgorithm, HyperParameters},
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    },
    extensions::reinforcement_learning::ReinforcementLearningParameters,
    utils::random::generator,
};
use set_up::{MountainCarInput, MountainCarLgp};

mod set_up;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let environment = MountainCarEnv::new(RenderMode::Human, None);
    let input = MountainCarInput::new(environment);
    let initial_states = (vec![0; 5])
        .into_iter()
        .map(|_| MountainCarObservation::sample_between(&mut generator(), None))
        .collect_vec();

    let mut hyper_params = HyperParameters {
        population_size: 1,
        gap: 0.5,
        crossover_percent: 0.,
        mutation_percent: 0.,
        n_generations: 1,
        lazy_evaluate: true,
        fitness_parameters: ReinforcementLearningParameters::new(initial_states, 200, input),
        program_parameters: ProgramGeneratorParameters::new(
            100,
            InstructionGeneratorParameters::from::<MountainCarInput>(1),
        ),
    };

    MountainCarLgp::execute(&mut hyper_params, EventHooks::default())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use gym_rs::{
        envs::classical_control::mountain_car::{MountainCarEnv, MountainCarObservation},
        utils::{custom::Sample, renderer::RenderMode},
    };
    use itertools::Itertools;
    use lgp::{
        core::{
            algorithm::{EventHooks, GeneticAlgorithm, HyperParameters},
            instruction::InstructionGeneratorParameters,
            program::ProgramGeneratorParameters,
        },
        extensions::reinforcement_learning::ReinforcementLearningParameters,
        utils::{plots::plot_population_benchmarks, random::generator},
    };

    use crate::set_up::{MountainCarInput, MountainCarLgp};

    #[test]
    fn given_mountain_car_example_when_lgp_executed_then_task_is_solved(
    ) -> Result<(), Box<dyn std::error::Error>> {
        MountainCarLgp::init_env();

        let environment = MountainCarEnv::new(RenderMode::None, None);
        let input = MountainCarInput::new(environment);
        let initial_states = (vec![0; 5])
            .into_iter()
            .map(|_| MountainCarObservation::sample_between(&mut generator(), None))
            .collect_vec();

        let mut hyper_params = HyperParameters {
            population_size: 100,
            gap: 0.5,
            crossover_percent: 0.5,
            mutation_percent: 0.5,
            n_generations: 100,
            lazy_evaluate: false,
            fitness_parameters: ReinforcementLearningParameters::new(initial_states, 200, input),
            program_parameters: ProgramGeneratorParameters::new(
                32,
                InstructionGeneratorParameters::from::<MountainCarInput>(1),
            ),
        };

        let mut populations = vec![];

        MountainCarLgp::execute(
            &mut hyper_params,
            EventHooks::default()
                .with_on_after_rank(&mut |population| Ok(populations.push(population.clone())))
                .with_on_post_fitness_params(
                    &mut &mut |params: &mut ReinforcementLearningParameters<MountainCarInput>| {
                        params.update(
                            (vec![0; 5])
                                .into_iter()
                                .map(|_| {
                                    MountainCarObservation::sample_between(&mut generator(), None)
                                })
                                .collect_vec(),
                        );

                        Ok(())
                    },
                ),
        )?;

        const PLOT_FILE_NAME: &'static str = "./assets/tests/plots/mountain_car.png";
        plot_population_benchmarks(populations, PLOT_FILE_NAME, -200.0..0.0)?;
        Ok(())
    }
}
