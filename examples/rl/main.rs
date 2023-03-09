use std::marker::PhantomData;

use cart_pole_config::CartPoleInput;
use derive_more::Display;
use gym_rs::{
    core::Env,
    envs::classical_control::{cartpole::CartPoleEnv, mountain_car::MountainCarEnv},
};
use itertools::Itertools;
use kurobako_core::{
    domain::{self, Constraint},
    problem::{Evaluator, Problem, ProblemFactory, ProblemSpecBuilder},
    trial,
};
use lgp::{
    core::{
        algorithm::{GeneticAlgorithm, HyperParameters, Organism},
        inputs::ValidInput,
        instruction::InstructionGeneratorParameters,
        program::{Program, ProgramGeneratorParameters},
    },
    extensions::{
        gym_rs::ExtendedGymRsEnvironment,
        interactive::{InteractiveLearningInput, InteractiveLearningParameters},
        q_learning::{QConsts, QProgramGeneratorParameters},
    },
};
use mountain_car_config::MountainCarInput;

mod cart_pole_config;
mod mountain_car_config;

#[derive(Display, PartialEq, Eq)]
pub enum Variant {
    #[display(fmt = "linear")]
    Linear,
    #[display(fmt = "q")]
    Q,
}

#[derive(Display, PartialEq, Eq)]
pub enum ProblemTypes {
    #[display(fmt = "cart_pole")]
    CartPole,
    #[display(fmt = "mountain_car")]
    MountainCar,
}

pub struct LgpProblemFactory<G, O, P> {
    problem_type: ProblemTypes,
    variant: Variant,
    g_marker: PhantomData<G>,
    o_marker: PhantomData<O>,
    p_marker: PhantomData<P>,
}

pub struct LgpProblem<G, O, P>
where
    P: ExtendedGymRsEnvironment,
    O: Organism<FitnessParameters = InteractiveLearningParameters<P>>,
    G: GeneticAlgorithm<O = O>,
{
    algorithm: PhantomData<G>,
    problem_type: ProblemTypes,
    variant: Variant,
}

pub struct LgpProblemEvaluator<G, O, P>
where
    P: ExtendedGymRsEnvironment + InteractiveLearningInput,
    O: Organism<FitnessParameters = InteractiveLearningParameters<P>>,
    G: GeneticAlgorithm<O = O>,
{
    g_marker: PhantomData<G>,
    hyper_parameters: HyperParameters<O>,
}

impl<G, O, P> Evaluator for LgpProblemEvaluator<G, O, P>
where
    P: ExtendedGymRsEnvironment + InteractiveLearningInput,
    O: Organism<FitnessParameters = InteractiveLearningParameters<P>>,
    G: GeneticAlgorithm<O = O>,
{
    fn evaluate(
        &mut self,
        next_step: u64,
    ) -> kurobako_core::Result<(u64, kurobako_core::trial::Values)> {
        let evaluated_programs = G::build(self.hyper_parameters.clone()).collect_vec();
        let best_program = evaluated_programs
            .last()
            .take()
            .unwrap()
            .best()
            .take()
            .unwrap()
            .get_fitness()
            .unwrap();
        let values = trial::Values::new(vec![best_program]);

        return Ok((next_step, values));
    }
}

impl<G, O, P> Problem for LgpProblem<G, O, P>
where
    P: ExtendedGymRsEnvironment + InteractiveLearningInput,
    O: Organism<FitnessParameters = InteractiveLearningParameters<P>>,
    G: GeneticAlgorithm<O = O>,
{
    type Evaluator = LgpProblemEvaluator<G, O, P>;

    fn create_evaluator(
        &self,
        params: kurobako_core::trial::Params,
    ) -> kurobako_core::Result<Self::Evaluator> {
        let n_generations = params[ParamMapping::Gens as usize] as usize;
        let n_trials = params[ParamMapping::Trials as usize] as usize;

        let input = P::new();
        let initial_states = P::get_initial_states(n_generations, n_trials);

        let fitness_parameters = InteractiveLearningParameters::new(initial_states, input);

        let basic_params = ProgramGeneratorParameters::new(
            params[ParamMapping::Instructions as usize] as usize,
            InstructionGeneratorParameters::from::<P>(
                params[ParamMapping::Registers as usize] as usize,
            ),
        );

        if self.variant == Variant::Q {
            return Ok(LgpProblemEvaluator {
                g_marker: PhantomData,
                hyper_parameters: HyperParameters {
                    population_size: params[ParamMapping::PopSize as usize] as usize,
                    gap: params[ParamMapping::Gap as usize],
                    crossover_percent: params[ParamMapping::Crossover as usize],
                    mutation_percent: params[ParamMapping::Mutation as usize],
                    n_generations,
                    fitness_parameters,
                    program_parameters: QProgramGeneratorParameters::new(
                        basic_params,
                        QConsts::new(
                            params[ParamMapping::Alpha as usize],
                            params[ParamMapping::Gamma as usize],
                            params[ParamMapping::Epsilon as usize],
                            params[ParamMapping::AlphaDecay as usize],
                            params[ParamMapping::EpsilonDecay as usize],
                        ),
                    ),
                },
            });
        } else {
            return Ok(LgpProblemEvaluator {
                g_marker: PhantomData,
                hyper_parameters: HyperParameters {
                    population_size: params[ParamMapping::PopSize as usize] as usize,
                    gap: params[ParamMapping::Gap as usize],
                    crossover_percent: params[ParamMapping::Crossover as usize],
                    mutation_percent: params[ParamMapping::Mutation as usize],
                    n_generations,
                    fitness_parameters,
                    program_parameters: basic_params,
                },
            });
        }
    }
}

pub enum ParamMapping {
    Gens,
    Trials,
    PopSize,
    Gap,
    Mutation,
    Crossover,
    Registers,
    Instructions,
    Alpha,
    Epsilon,
    Gamma,
    AlphaDecay,
    EpsilonDecay,
}

impl<G, O, P> ProblemFactory for LgpProblemFactory<G, O, P>
where
    P: ExtendedGymRsEnvironment + InteractiveLearningInput,
    O: Organism<FitnessParameters = InteractiveLearningParameters<P>>,
    G: GeneticAlgorithm<O = O>,
{
    type Problem = LgpProblem<G, O, P>;

    fn specification(&self) -> kurobako_core::Result<kurobako_core::problem::ProblemSpec> {
        let name = format!("{}-{}", self.problem_type, self.variant);

        let mut spec = ProblemSpecBuilder::new(&name)
            .param(domain::var("n_generations").discrete(1, i64::MAX))
            .param(domain::var("n_trials").discrete(1, i64::MAX))
            .param(domain::var("population_size").discrete(0, i64::MAX))
            .param(
                domain::var("gap")
                    .continuous(0., 1.)
                    .constraint(Constraint::new("gap > 0. && gap < 1.")),
            )
            .param(
                domain::var("mutation_percent")
                    .continuous(0., 1.)
                    .constraint(Constraint::new(
                        "mutation_percent + crossover_perecent <= 1.0",
                    )),
            )
            .param(
                domain::var("crossover_percent")
                    .continuous(0., 1.)
                    .constraint(Constraint::new(
                        "mutation_percent + crossover_percent <= 1.0",
                    )),
            )
            .param(domain::var("ip_n_extras").discrete(1, i64::MAX))
            .param(domain::var("pp_n_instructions").discrete(1, i64::MAX));

        if self.variant == Variant::Q {
            spec = spec
                .param(domain::var("q_alpha").continuous(0., 1.))
                .param(domain::var("q_epsilon").continuous(0., 1.0))
                .param(domain::var("q_gamma").continuous(0.9, 0.99))
                .param(
                    domain::var("q_alpha_decay")
                        .continuous(0., 1.0)
                        .constraint(Constraint::new("q_alpha_decay < q_alpha")),
                )
                .param(
                    domain::var("q_epsilon_decay")
                        .continuous(0., 1.0)
                        .constraint(Constraint::new("q_epsilon_decay < q_alpha")),
                );
        }

        return spec.finish();
    }

    fn create_problem(
        &self,
        rng: kurobako_core::rng::ArcRng,
    ) -> kurobako_core::Result<Self::Problem> {
        let problem = match self.problem_type {
            ProblemTypes::MountainCar => {
                todo!()
            }
            _ => {
                todo!()
            }
        };

        return Ok(problem);
    }
}

fn main() {}
