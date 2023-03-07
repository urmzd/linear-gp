use gym_rs::{envs::classical_control::mountain_car::MountainCarEnv, utils::renderer::RenderMode};
use itertools::Itertools;
use lgp::{
    core::{
        algorithm::{GeneticAlgorithm, HyperParameters},
        instruction::InstructionGeneratorParameters,
        program::ProgramGeneratorParameters,
    },
    extensions::{
        gym_rs::ExtendedGymRsEnvironment,
        interactive::{ILgp, InteractiveLearningInput, InteractiveLearningParameters},
        q_learning::{QConsts, QLgp, QProgram, QProgramGeneratorParameters},
    },
    utils::{plots::plot_benchmarks, types::VoidResultAnyError},
};
mod config;
use config::MountainCarInput;

fn main() -> VoidResultAnyError {
    let environment = MountainCarEnv::new(RenderMode::None);
    let input = MountainCarInput::new(environment);
    let n_generations = 100;
    let n_trials = 5;
    let initial_states = MountainCarInput::get_initial_states(n_generations, n_trials);
    let fitness_parameters = InteractiveLearningParameters::new(initial_states, input);
    let instruction_parameters = InstructionGeneratorParameters::from::<MountainCarInput>(1);
    let program_parameters = ProgramGeneratorParameters::new(12, instruction_parameters);

    let lgp_hp = HyperParameters {
        population_size: 100,
        gap: 0.5,
        crossover_percent: 0.5,
        mutation_percent: 0.5,
        n_generations,
        fitness_parameters: fitness_parameters.clone(),
        program_parameters,
    };

    let lgpq_hp: HyperParameters<QProgram<MountainCarInput>> = HyperParameters {
        population_size: lgp_hp.population_size,
        gap: lgp_hp.gap,
        mutation_percent: lgp_hp.mutation_percent,
        crossover_percent: lgp_hp.crossover_percent,
        n_generations: lgp_hp.n_generations,
        fitness_parameters: fitness_parameters.clone(),
        program_parameters: QProgramGeneratorParameters::new(
            lgp_hp.program_parameters,
            QConsts::default(),
        ),
    };

    let lgp_pops = ILgp::build(lgp_hp).collect_vec();
    let q_pops = QLgp::build(lgpq_hp).collect_vec();

    const PLOT_FILE_NAME: &'static str = "assets/plots/examples/mountain_car/default.png";
    plot_benchmarks(
        lgp_pops,
        PLOT_FILE_NAME,
        (-(MountainCarInput::MAX_EPISODE_LENGTH as isize) as f64)..0.0,
    )?;

    const Q_PLOT_FILE_NAME: &'static str = "assets/plots/examples/mountain_car/q.png";
    plot_benchmarks(
        q_pops,
        Q_PLOT_FILE_NAME,
        (-(MountainCarInput::MAX_EPISODE_LENGTH as isize) as f64)..0.0,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {

    use gym_rs::{
        envs::classical_control::mountain_car::MountainCarEnv, utils::renderer::RenderMode,
    };

    use itertools::Itertools;
    use lgp::{
        core::{
            algorithm::{GeneticAlgorithm, HyperParameters},
            instruction::InstructionGeneratorParameters,
            program::ProgramGeneratorParameters,
        },
        extensions::{
            gym_rs::ExtendedGymRsEnvironment,
            interactive::{ILgp, InteractiveLearningParameters},
            q_learning::{QConsts, QLgp, QProgram, QProgramGeneratorParameters},
        },
        utils::{plots::plot_benchmarks, types::VoidResultAnyError},
    };

    use crate::config::MountainCarInput;

    #[test]
    fn given_mountain_car_example_when_lgp_executed_then_task_is_solved(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let environment = MountainCarEnv::new(RenderMode::None);
        let input = MountainCarInput::new(environment);
        let n_generations = 100;
        let n_trials = 5;
        let initial_states = MountainCarInput::get_initial_states(n_generations, n_trials);

        let hyper_params = HyperParameters {
            population_size: 100,
            gap: 0.5,
            crossover_percent: 0.5,
            mutation_percent: 0.5,
            n_generations,
            fitness_parameters: InteractiveLearningParameters::new(initial_states, input),
            program_parameters: ProgramGeneratorParameters::new(
                12,
                InstructionGeneratorParameters::from::<MountainCarInput>(1),
            ),
        };

        let populations = ILgp::build(hyper_params).collect_vec();

        const PLOT_FILE_NAME: &'static str = "assets/plots/tests/mountain_car/smoke/default.png";
        plot_benchmarks(populations, PLOT_FILE_NAME, -200.0..0.0)?;
        Ok(())
    }

    #[test]
    fn given_mountain_car_task_when_q_learning_lgp_is_used_then_task_is_solved(
    ) -> VoidResultAnyError {
        let game = MountainCarEnv::new(RenderMode::None);
        let environment = MountainCarInput::new(game);
        let n_generations = 100;
        let n_trials = 5;
        let initial_states = MountainCarInput::get_initial_states(n_generations, n_trials);
        let parameters = InteractiveLearningParameters::new(initial_states, environment);

        let hyper_params: HyperParameters<QProgram<MountainCarInput>> = HyperParameters {
            population_size: 100,
            gap: 0.5,
            mutation_percent: 0.5,
            crossover_percent: 0.5,
            n_generations,
            fitness_parameters: parameters,
            program_parameters: QProgramGeneratorParameters::new(
                ProgramGeneratorParameters::new(
                    12,
                    InstructionGeneratorParameters::from::<MountainCarInput>(1),
                ),
                QConsts::default(),
            ),
        };

        let pops = QLgp::build(hyper_params).collect_vec();

        const PLOT_FILE_NAME: &'static str = "assets/plots/tests/mountain_car/smoke/q.png";
        plot_benchmarks(pops, PLOT_FILE_NAME, -200.0..0.0)?;
        Ok(())
    }
}
