use gym_rs::{
    envs::classical_control::mountain_car::{MountainCarEnv},
    utils::{renderer::RenderMode},
};

use lgp::{
    core::{
        algorithm::{EventHooks, GeneticAlgorithm, HyperParameters},
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    },
    extensions::{
        gym_rs::ExtendedGymRsEnvironment, reinforcement_learning::ReinforcementLearningParameters,
    },
};
use set_up::{MountainCarInput, MountainCarLgp};

mod set_up;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let environment = MountainCarEnv::new(RenderMode::Human, None);
    let input = MountainCarInput::new(environment);
    let n_generations = 1;
    let n_trials = 5;
    let initial_states = MountainCarInput::get_initial_states(n_generations, n_trials);

    let mut hyper_params = HyperParameters {
        population_size: 1,
        gap: 0.5,
        crossover_percent: 0.,
        mutation_percent: 0.,
        n_generations,
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
        envs::classical_control::mountain_car::{MountainCarEnv},
        utils::{renderer::RenderMode},
    };
    
    use lgp::{
        core::{
            algorithm::{EventHooks, GeneticAlgorithm, HyperParameters},
            instruction::InstructionGeneratorParameters,
            program::ProgramGeneratorParameters,
        },
        extensions::{
            gym_rs::ExtendedGymRsEnvironment,
            reinforcement_learning::ReinforcementLearningParameters,
        },
        utils::{plots::plot_population_benchmarks},
    };

    use crate::set_up::{MountainCarInput, MountainCarLgp};

    #[test]
    fn given_mountain_car_example_when_lgp_executed_then_task_is_solved(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let environment = MountainCarEnv::new(RenderMode::None, None);
        let input = MountainCarInput::new(environment);
        let n_generations = 100;
        let n_trials = 5;
        let initial_states = MountainCarInput::get_initial_states(n_generations, n_trials);

        let mut hyper_params = HyperParameters {
            population_size: 100,
            gap: 0.5,
            crossover_percent: 0.5,
            mutation_percent: 0.5,
            n_generations,
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
            EventHooks::default().with_on_post_rank(&mut |population, params| {
                populations.push(population.clone());
                params.fitness_parameters.next_generation()
            }),
        )?;

        const PLOT_FILE_NAME: &'static str = "./assets/tests/plots/mountain_car.png";
        plot_population_benchmarks(populations, PLOT_FILE_NAME, -200.0..0.0)?;
        Ok(())
    }
}
