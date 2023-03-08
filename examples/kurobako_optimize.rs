use std::marker::PhantomData;

use derive_more::Display;
use kurobako_core::{
    domain::{self, Constraint},
    problem::{Evaluator, Problem, ProblemFactory, ProblemSpecBuilder},
};
use lgp::core::algorithm::{GeneticAlgorithm, HyperParameters, Organism};

#[derive(Display, PartialEq, Eq)]
pub enum Variant {
    #[display(fmt = "linear")]
    Linear,
    #[display(fmt = "q")]
    Q,
}

#[derive(Display)]
pub enum ProblemTypes {
    #[display(fmt = "cart_pole")]
    CartPole,
    #[display(fmt = "mountain_car")]
    MountainCar,
}

pub struct LgpProblemFactory<G, O> {
    problem_type: ProblemTypes,
    variant: Variant,
    g_marker: PhantomData<G>,
    o_marker: PhantomData<O>,
}

pub struct LgpProblem<G, O>
where
    O: Organism,
    G: GeneticAlgorithm<O = O> + Send,
{
    algorithm: G,
}

pub struct LgpProblemEvaluator<G, O>
where
    O: Organism + Send,
    G: GeneticAlgorithm<O = O> + Send,
{
    g_marker: PhantomData<G>,
    o_marker: PhantomData<O>,
    hyper_parameters: HyperParameters<O>,
}

impl<G, O> Evaluator for LgpProblemEvaluator<G, O>
where
    O: Organism,
    G: GeneticAlgorithm<O = O>,
{
    fn evaluate(
        &mut self,
        next_step: u64,
    ) -> kurobako_core::Result<(u64, kurobako_core::trial::Values)> {
        todo!()
    }
}

impl<G, O> Problem for LgpProblem<G, O>
where
    O: Organism,
    G: GeneticAlgorithm<O = O>,
{
    type Evaluator = LgpProblemEvaluator<G, O>;

    fn create_evaluator(
        &self,
        params: kurobako_core::trial::Params,
    ) -> kurobako_core::Result<Self::Evaluator> {
        todo!()
    }
}

impl<G, O> ProblemFactory for LgpProblemFactory<G, O>
where
    O: Organism,
    G: GeneticAlgorithm<O = O>,
{
    type Problem = LgpProblem<G, O>;

    fn specification(&self) -> kurobako_core::Result<kurobako_core::problem::ProblemSpec> {
        let name = format!("{}-{}", self.problem_type, self.variant);

        let mut spec = ProblemSpecBuilder::new(&name)
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
            .param(domain::var("n_generations").discrete(0, i64::MAX))
            .param(domain::var("ip_n_extras").discrete(1, i64::MAX))
            .param(domain::var("pp_n_instructions").discrete(1, i64::MAX))
            .param(domain::var("n_generations").discrete(1, i64::MAX))
            .param(domain::var("n_trials").discrete(1, i64::MAX));

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
