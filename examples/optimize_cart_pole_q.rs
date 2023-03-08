use std::{fmt, marker::PhantomData};

use derive_more::Display;
use kurobako_core::{
    domain::{self, Constraint},
    problem::{Problem, ProblemFactory, ProblemSpecBuilder},
};
use lgp::core::{
    algorithm::{GeneticAlgorithm, Organism},
    characteristics::{Breed, DuplicateNew, Fitness, Generate, Mutate},
};

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

//let environment = MountainCarEnv::new(RenderMode::None);
//let input = MountainCarInput::new(environment);
//let n_generations = 100;
//let n_trials = 5;
//let initial_states = MountainCarInput::get_initial_states(n_generations, n_trials);
//let fitness_parameters = InteractiveLearningParameters::new(initial_states, input);
//let instruction_parameters = InstructionGeneratorParameters::from::<MountainCarInput>(1);
//let program_parameters = ProgramGeneratorParameters::new(12, instruction_parameters);

//let lgp_hp = HyperParameters {
//population_size: 100,
//gap: 0.5,
//crossover_percent: 0.5,
//mutation_percent: 0.5,
//n_generations,
//fitness_parameters: fitness_parameters.clone(),
//program_parameters,
//};

//let lgpq_hp: HyperParameters<QProgram<MountainCarInput>> = HyperParameters {
//population_size: lgp_hp.population_size,
//gap: lgp_hp.gap,
//mutation_percent: lgp_hp.mutation_percent,
//crossover_percent: lgp_hp.crossover_percent,
//n_generations: lgp_hp.n_generations,
//fitness_parameters: fitness_parameters.clone(),
//program_parameters: QProgramGeneratorParameters::new(
//lgp_hp.program_parameters,
//QConsts::default(),
//),
//};
//

struct LgpProblem<G, O>
where
    O: Organism,
    G: GeneticAlgorithm<O = O>,
{
    algorithm: G,
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
            .param(domain::var("population_size").discrete(0, usize::MAX))
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
        todo!()
    }
}

fn main() {}
