use std::fmt::{self, Debug};

use clap::Args;
use derivative::Derivative;
use rand::distributions::uniform::{UniformFloat, UniformInt, UniformSampler};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{
    core::{
        algorithm::{GeneticAlgorithm, HyperParameters},
        characteristics::{Breed, Fitness, FitnessScore, Generate, Mutate, Reset, ResetNew},
        inputs::ValidInput,
        program::{Program, ProgramGeneratorParameters},
        registers::{ArgmaxInput, Registers, AR},
    },
    utils::{float_ops, random::generator},
};

use super::interactive::{ILgp, InteractiveLearningInput, InteractiveLearningParameters};

#[derive(Clone, Serialize, Deserialize)]
pub struct QTable {
    table: Vec<Vec<f64>>,
    n_actions: usize,
    n_registers: usize,
    q_consts: QConsts,
}

impl Debug for QTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.table.iter()).finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ActionRegisterPair {
    action: usize,
    register: usize,
}

impl Reset for QTable {
    fn reset(&mut self) {
        self.q_consts.reset();
    }
}

impl QTable {
    pub fn new(n_actions: usize, n_registers: usize, q_consts: QConsts) -> Self {
        let table = vec![vec![0.; n_actions]; n_registers];
        QTable {
            table,
            n_actions,
            n_registers,
            q_consts,
        }
    }

    pub fn action_random(&self) -> usize {
        UniformInt::<usize>::new(0, self.n_actions).sample(&mut generator())
    }

    pub fn action_argmax(&self, register_number: usize) -> usize {
        let available_actions = self
            .table
            .get(register_number)
            .expect("Register number to be less than length of QTable.");

        let iter = available_actions.iter().copied();
        let max = float_ops::argmax(iter);

        max.expect("Available action to yield an index.")
    }

    pub fn get_action_register(&self, registers: &Registers) -> Option<ActionRegisterPair> {
        let winning_register = match registers.argmax(ArgmaxInput::All).any() {
            AR::Value(register) => register,
            _ => return None,
        };

        let prob = UniformFloat::<f64>::new_inclusive(0., 1.).sample(&mut generator());

        let winning_action = if prob <= self.q_consts.epsilon_active {
            self.action_random()
        } else {
            self.action_argmax(winning_register)
        };

        Some(ActionRegisterPair {
            action: winning_action,
            register: winning_register,
        })
    }

    pub fn update(
        &mut self,
        current_action_state: ActionRegisterPair,
        current_reward: f64,
        next_action_state: ActionRegisterPair,
    ) {
        let current_q_value =
            self.table[current_action_state.register][current_action_state.action];
        let next_q_value = self.action_argmax(next_action_state.register) as f64;

        let new_q_value = self.q_consts.alpha_active
            * (current_reward + (self.q_consts.gamma * next_q_value) - current_q_value);

        self.table[current_action_state.register][current_action_state.action] += new_q_value;
        self.q_consts.decay();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Derivative)]
pub struct QProgram {
    #[derivative(PartialEq = "ignore", PartialOrd = "ignore")]
    pub q_table: QTable,
    pub program: Program,
}

impl Reset for QProgram {
    fn reset(&mut self) {
        self.q_table.reset();
        self.program.reset();
    }
}

fn get_action_state<T>(
    environment: &mut T,
    q_table: &mut QTable,
    program: &mut Program,
) -> Option<ActionRegisterPair>
where
    T: InteractiveLearningInput,
{
    // Run the program on the current state.
    program.run(environment);

    // Get the winning action-register pair.
    let action_state = q_table.get_action_register(&program.registers);

    action_state
}

impl<T> Fitness<InteractiveLearningParameters<T>> for QProgram {
    fn eval_fitness(&mut self, mut parameters: InteractiveLearningParameters<T>) {
        let mut score = 0.;

        // We run the program and determine what action to take at the step = 0.
        let mut current_action_state = match get_action_state(
            &mut parameters.environment,
            &mut self.q_table,
            &mut self.program,
        ) {
            Some(action_state) => action_state,
            None => {
                self.program.fitness = FitnessScore::OutOfBounds;
                return;
            }
        };

        // We execute the selected action and continue to repeat the cycle until termination.
        for _step in 0..T::MAX_EPISODE_LENGTH {
            // Act.
            let state_reward_pair = parameters
                .environment
                .execute_action(current_action_state.action);

            let reward = state_reward_pair.get_value();
            score += reward;

            if state_reward_pair.is_terminal() {
                break;
            }

            let next_action_state = match get_action_state(
                &mut parameters.environment,
                &mut self.q_table,
                &mut self.program,
            ) {
                None => {
                    // We've encountered numerical instability. The program is not considered valid, and thus
                    // has the lowest score.
                    return {
                        self.program.fitness = FitnessScore::OutOfBounds;
                    };
                }
                Some(action_state) => action_state,
            };

            // We only update when there is a transition.
            if current_action_state.register != next_action_state.register {
                self.q_table
                    .update(current_action_state, reward, next_action_state)
            }

            current_action_state = next_action_state;
        }

        // Reset for next evaluation.
        self.program.registers.reset_new();

        info!(
            id = serde_json::to_string(&self.program.id.to_string()).unwrap(),
            q_table = serde_json::to_string(&self.q_table).unwrap(),
            initial_state = serde_json::to_string(&initial_state.into()).unwrap(),
            score = serde_json::to_string(&score).unwrap()
        );

        let fitness_score = FitnessScore::Valid(*score);

        self.program.fitness = fitness_score;
    }
}

impl Breed for QProgram {
    fn two_point_crossover(&self, mate: &Self) -> [Self; 2] {
        let children = self.program.two_point_crossover(&mate.program);
        children.map(|program| QProgram {
            program,
            q_table: self.q_table.reset_new(),
        })
    }
}

impl Mutate for QProgram {
    fn mutate(&self, parameters: Self::GeneratorParameters) -> Self {
        let mutated = self.program.mutate(parameters.program_parameters);
        QProgram {
            program: mutated,
            q_table: self.q_table.reset_new(),
        }
    }
}

impl<T> Generate for QProgram<T>
where
    T: InteractiveLearningInput,
{
    type GeneratorParameters = QProgramGeneratorParameters<T>;

    fn generate(parameters: Self::GeneratorParameters) -> Self {
        let program = Program::generate(parameters.program_parameters);

        let instruction_params = &parameters
            .program_parameters
            .instruction_generator_parameters;

        let q_table = QTable::new(
            instruction_params.n_actions(),
            instruction_params.n_registers(),
            parameters.consts,
        );

        QProgram { q_table, program }
    }
}

#[derive(Debug, Clone, Args, Deserialize, Serialize)]
pub struct QProgramGeneratorParameters {
    #[command(flatten)]
    program_parameters: ProgramGeneratorParameters,
    #[command(flatten)]
    consts: QConsts,
}

#[derive(Debug, Clone, Copy, Args, Serialize, Deserialize)]
pub struct QConsts {
    /// Learning Factor
    #[arg(long, default_value = "0.1")]
    alpha: f64,
    /// Discount Factor
    #[arg(long, default_value = "0.9")]
    gamma: f64,
    /// Exploration Factor
    #[arg(long, default_value = "0.05")]
    epsilon: f64,
    /// Learning Rate Decay
    #[arg(long, default_value = "0.01")]
    alpha_decay: f64,
    /// Exploration Decay
    #[arg(long, default_value = "0.001")]
    epsilon_decay: f64,

    /// To allow new programs to start from the new state, we have active
    /// properties to mutuate.
    #[arg(skip)]
    #[serde(skip)]
    alpha_active: f64,
    #[serde(skip)]
    #[arg(skip)]
    epsilon_active: f64,
}

impl Reset for QConsts {
    fn reset(&mut self) {
        self.alpha_active = self.alpha;
        self.epsilon_active = self.epsilon;
    }
}

impl QConsts {
    pub fn new(alpha: f64, gamma: f64, epsilon: f64, alpha_decay: f64, epsilon_decay: f64) -> Self {
        Self {
            alpha_active: alpha,
            epsilon_active: epsilon,
            alpha,
            gamma,
            epsilon,
            alpha_decay,
            epsilon_decay,
        }
    }

    pub fn decay(&mut self) {
        self.alpha_active *= 1. - self.alpha_decay;
        self.epsilon_active *= 1. - self.epsilon_decay
    }
}

impl Default for QConsts {
    fn default() -> Self {
        Self {
            alpha: 0.25,
            gamma: 0.90,
            epsilon: 0.05,
            alpha_decay: 0.0,
            epsilon_decay: 0.0,
            alpha_active: 0.25,
            epsilon_active: 0.05,
        }
    }
}

impl<T> GeneticAlgorithm for ILgp<QProgram<T>>
where
    T: InteractiveLearningInput,
{
    type O = QProgram<T>;

    fn on_pre_init(mut parameters: HyperParameters<Self::O>) -> HyperParameters<Self::O> {
        parameters.generator.consts.reset();
        parameters
    }

    fn on_post_rank(
        population: Population<Self::O>,
        mut parameters: HyperParameters<Self::O>,
    ) -> (Population<Self::O>, HyperParameters<Self::O>) {
        parameters.evaluator.next_generation();

        return (population, parameters);
    }
}
