use std::{
    fmt::{self, Debug},
    marker::PhantomData,
};

use derive_new::new;
use more_asserts::{assert_ge, assert_le};
use rand::{
    distributions::uniform::{UniformFloat, UniformInt, UniformSampler},
    prelude::SliceRandom,
};
use tracing::debug;
use tracing::field::valuable;
use valuable::Valuable;

use crate::{
    core::{
        algorithm::GeneticAlgorithm,
        characteristics::{Breed, DuplicateNew, Fitness, FitnessScore, Generate, Mutate},
        program::{Program, ProgramGeneratorParameters},
        registers::Registers,
    },
    utils::{float_ops, random::generator},
};

use super::interactive::{InteractiveLearningInput, InteractiveLearningParameters};

#[derive(Clone, Valuable)]
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

#[derive(Debug, Clone, Copy, Valuable)]
pub struct ActionRegisterPair {
    action: usize,
    register: usize,
}

impl DuplicateNew for QTable {
    fn duplicate_new(&self) -> Self {
        QTable::new(self.n_actions, self.n_registers, self.q_consts)
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

    pub fn eval<T>(&self, registers: &Registers) -> Option<ActionRegisterPair> {
        let winning_registers = registers.all_argmax(None);

        let winning_register = match winning_registers {
            // NOTE: Should we panic instead?
            None => return None,
            // Select any register.
            Some(registers) => registers
                .choose(&mut generator())
                .cloned()
                .expect("Register to have been chosen."),
        };

        assert_le!(self.q_consts.epsilon, 1.0);
        assert_ge!(self.q_consts.epsilon, 0.);

        // TODO: Move generator to structs.
        let prob = UniformFloat::<f64>::new_inclusive(0., 1.).sample(&mut generator());

        let winning_action = if prob < self.q_consts.epsilon {
            self.action_random()
        } else {
            self.action_argmax(winning_register)
        };

        Some(ActionRegisterPair {
            action: winning_action,
            register: winning_register as usize,
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

        let new_q_value = self.q_consts.alpha
            * (current_reward + (self.q_consts.gamma * next_q_value) - current_q_value);

        self.table[current_action_state.register][current_action_state.action] += new_q_value;
    }
}

#[derive(Debug, Clone)]
pub struct QProgram<T>
where
    T: InteractiveLearningInput,
    T::State: Clone + fmt::Debug,
{
    pub q_table: QTable,
    pub program: Program<InteractiveLearningParameters<T>>,
}

impl<T> DuplicateNew for QProgram<T>
where
    T: InteractiveLearningInput,
    T::State: Clone + fmt::Debug,
{
    fn duplicate_new(&self) -> Self {
        return Self {
            q_table: self.q_table.duplicate_new(),
            program: self.program.duplicate_new(),
        };
    }
}

impl<T> PartialEq for QProgram<T>
where
    T: InteractiveLearningInput,
    T::State: Clone + fmt::Debug,
{
    fn eq(&self, other: &Self) -> bool {
        self.program == other.program
    }
}

impl<T> PartialOrd for QProgram<T>
where
    T: InteractiveLearningInput,
    T::State: Clone + fmt::Debug,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.program.fitness.partial_cmp(&other.program.fitness)
    }
}

fn get_action_state<T>(
    environment: &mut T,
    q_table: &mut QTable,
    program: &mut Program<InteractiveLearningParameters<T>>,
) -> Option<ActionRegisterPair>
where
    T: InteractiveLearningInput,
{
    // Run the program on the current state.
    program.exec(environment);

    // Get the winning action-register pair.
    let action_state = q_table.eval::<InteractiveLearningParameters<T>>(&program.registers);

    action_state
}

impl<T> Fitness for QProgram<T>
where
    T: InteractiveLearningInput,
    T::State: Clone + Debug,
{
    type FitnessParameters = InteractiveLearningParameters<T>;

    fn eval_fitness(&mut self, parameters: &mut Self::FitnessParameters) {
        let mut scores = vec![];

        for initial_state in parameters.get_states() {
            let mut score = 0.;

            parameters.environment.set_state(initial_state.clone());

            // We run the program and determine what action to take at the current step.
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
                let state_reward_pair = parameters.environment.sim(current_action_state.action);

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
                        self.program.fitness = FitnessScore::OutOfBounds;
                        return;
                    }
                    Some(action_state) => action_state,
                };

                if current_action_state.register != next_action_state.register {
                    self.q_table
                        .update(current_action_state, reward, next_action_state)
                }

                current_action_state = next_action_state;
            }

            // Reset for next evaluation.
            self.program.registers.reset();
            parameters.environment.reset();

            let initial_state_vec: &Vec<f64> = &initial_state.into();

            debug!(
                id = valuable(&self.program.id.to_string()),
                q_table = valuable(&self.q_table),
                initial_state = valuable(&initial_state_vec),
                score = valuable(&score)
            );

            scores.push(score);
        }

        scores.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        let median = scores.swap_remove(scores.len() / 2);

        let fitness_score = FitnessScore::Valid(median);

        self.program.fitness = fitness_score;
        self.q_table.q_consts.decay();
    }

    fn get_fitness(&self) -> FitnessScore {
        self.program.fitness
    }
}

impl<T> Breed for QProgram<T>
where
    T: InteractiveLearningInput,
    T::State: Clone + Debug,
{
    fn two_point_crossover(&self, mate: &Self) -> [Self; 2] {
        let children = self.program.two_point_crossover(&mate.program);
        children.map(|program| QProgram {
            program,
            q_table: self.q_table.duplicate_new(),
        })
    }
}

impl<T> Mutate for QProgram<T>
where
    T: InteractiveLearningInput,
    T::State: Clone + Debug,
{
    fn mutate(&self, parameters: &Self::GeneratorParameters) -> Self {
        let mutated = self.program.mutate(&parameters.program_parameters);
        QProgram {
            program: mutated,
            q_table: self.q_table.duplicate_new(),
        }
    }
}

impl<T> Generate for QProgram<T>
where
    T: InteractiveLearningInput,
    T::State: Clone + Debug,
{
    type GeneratorParameters = QProgramGeneratorParameters;

    fn generate(parameters: &Self::GeneratorParameters) -> Self {
        let program =
            Program::<InteractiveLearningParameters<T>>::generate(&parameters.program_parameters);

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

#[derive(Debug, new)]
pub struct QProgramGeneratorParameters {
    program_parameters: ProgramGeneratorParameters,
    consts: QConsts,
}

#[derive(Debug, Clone, Copy, new, Valuable)]
pub struct QConsts {
    /// Learning Factor
    alpha: f64,
    /// Discount Factor
    gamma: f64,
    /// Exploration Factor
    epsilon: f64,
    /// Learning Rate Decay
    alpha_decay: f64,
    /// Exploration Decay
    epsilon_decay: f64,
}

impl QConsts {
    pub fn decay(&mut self) {
        self.alpha *= 1. - self.alpha_decay;
        self.epsilon *= 1. - self.epsilon_decay
    }
}

impl Default for QConsts {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            gamma: 0.99,
            epsilon: 0.05,
            alpha_decay: 0.01,
            epsilon_decay: 0.001,
        }
    }
}

pub struct QLgp<T>(PhantomData<T>);

impl<T> GeneticAlgorithm for QLgp<T>
where
    T: InteractiveLearningInput + fmt::Debug,
    T::State: Clone + fmt::Debug,
{
    type O = QProgram<T>;

    fn on_post_rank(
        _population: &mut crate::core::population::Population<Self::O>,
        parameters: &mut crate::core::algorithm::HyperParameters<Self::O>,
    ) {
        parameters.fitness_parameters.environment.finish();
    }

    fn on_pre_eval_fitness(
        _population: &mut crate::core::population::Population<Self::O>,
        parameters: &mut crate::core::algorithm::HyperParameters<Self::O>,
    ) {
        parameters.fitness_parameters.environment.reset();
    }
}
