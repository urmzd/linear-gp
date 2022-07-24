use std::fmt::{self, Debug};

use itertools::Itertools;
use more_asserts::{assert_ge, assert_le};
use ordered_float::OrderedFloat;
use rand::distributions::uniform::{UniformFloat, UniformInt, UniformSampler};

use crate::{
    core::{
        characteristics::{Breed, DuplicateNew, Fitness, Generate, Mutate},
        instruction::InstructionGeneratorParameters,
        program::{Program, ProgramGeneratorParameters},
        registers::{Registers, R32},
    },
    utils::random::generator,
};

use super::{
    core::ExtensionParameters,
    reinforcement_learning::{ReinforcementLearningInput, ReinforcementLearningParameters},
};

#[derive(Clone, Debug)]
pub struct QTable {
    table: Vec<Vec<R32>>,
    n_actions: usize,
    n_registers: usize,
    /// Step size parameter.
    alpha: R32,
    /// Discount.
    gamma: R32,
    // Greedy selection.
    epsilon: R32,
}
pub trait QLearningInput: ReinforcementLearningInput {
    type State;

    fn set_state(&mut self, state: Self::State);
}

pub struct ActionRegisterPair {
    action: usize,
    register: usize,
}

impl<T> QLearningParameters<T>
where
    T: QLearningInput,
{
    pub fn update_states(&mut self, states: Vec<T::State>) {
        self.initial_states = states;
    }
}

impl<T> ExtensionParameters for QLearningParameters<T>
where
    T: QLearningInput,
{
    fn argmax(registers: &Registers) -> i32 {
        let selected_register = registers.iter().map(|v| OrderedFloat(*v)).position_max();

        selected_register.expect("Register to be of cardinality non-zero.") as i32
    }
}

#[derive(Debug, Clone)]
pub struct QLearningParameters<InputType>
where
    InputType: QLearningInput,
{
    rl_parameters: ReinforcementLearningParameters<InputType>,
    initial_states: Vec<InputType::State>,
}

impl DuplicateNew for QTable {
    fn duplicate_new(&self) -> Self {
        QTable::new(
            self.n_actions,
            self.n_registers,
            self.alpha,
            self.gamma,
            self.epsilon,
        )
    }
}

impl QTable {
    pub fn new(n_actions: usize, n_registers: usize, alpha: R32, gamma: R32, epsilon: R32) -> Self {
        let table = vec![vec![0.; n_actions]; n_registers];
        QTable {
            table,
            n_actions,
            n_registers,
            alpha,
            gamma,
            epsilon,
        }
    }

    pub fn action_random(&self, register_number: usize) -> usize {
        let QTable { table, .. } = &self;

        let action_size = table[register_number].len();

        UniformInt::<usize>::new(0, action_size).sample(&mut generator())
    }

    pub fn action_argmax(&self, register_number: usize) -> usize {
        let QTable { table, .. } = &self;
        let available_actions = table
            .get(register_number)
            .expect("Register number to be less than length of QTable.");

        available_actions
            .into_iter()
            .map(|v| OrderedFloat(*v))
            .position_max()
            .expect("Actions length to greater than 0.")
    }

    pub fn eval<T>(&self, registers: &Registers) -> ActionRegisterPair
    where
        T: ExtensionParameters,
    {
        let winning_register = T::argmax(registers);
        assert_le!(self.epsilon, 1.0);
        assert_ge!(self.epsilon, 0.);

        // TODO: Move generator to structs.
        let prob = UniformFloat::<f32>::new_inclusive(0., 1.).sample(&mut generator());

        let winning_action = if prob < self.epsilon {
            self.action_random(winning_register as usize)
        } else {
            self.action_argmax(winning_register as usize)
        };

        ActionRegisterPair {
            action: winning_action,
            register: winning_register as usize,
        }
    }

    pub fn update(
        &mut self,
        current_register: usize,
        current_action: usize,
        current_reward: R32,
        next_register: usize,
    ) {
        let current_q_value = self.table[current_register][current_action];
        let next_q_value = self.action_argmax(next_register) as f32;

        let new_q_value = current_q_value
            + self.alpha * (current_reward + self.gamma * next_q_value - current_q_value);

        self.table[current_register][current_action] = new_q_value;
    }
}

#[derive(Debug, Clone)]
pub struct QProgram<T>
where
    T: QLearningInput,
    T::State: Clone + fmt::Debug,
{
    q_table: QTable,
    program: Program<QLearningParameters<T>>,
}

impl<T> PartialEq for QProgram<T>
where
    T: QLearningInput,
    T::State: Clone + fmt::Debug,
{
    fn eq(&self, other: &Self) -> bool {
        self.program == other.program
    }
}

impl<T> PartialOrd for QProgram<T>
where
    T: QLearningInput,
    T::State: Clone + fmt::Debug,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.program.partial_cmp(&other.program)
    }
}

impl<T> Fitness for QProgram<T>
where
    T: QLearningInput,
    T::State: Clone + Debug,
{
    type FitnessParameters = QLearningParameters<T>;

    fn eval_fitness(
        &mut self,
        parameters: &mut Self::FitnessParameters,
    ) -> crate::core::characteristics::FitnessScore {
        let mut scores = vec![];
        // TODO: Call init and finish after `rank`
        for state in &parameters.initial_states {
            // INIT STEPS.
            let mut score = 0f32;
            parameters
                .rl_parameters
                .environment
                .set_state(state.clone());
            self.program.exec(&parameters.rl_parameters.environment);
            let prev_action_state = self
                .q_table
                .eval::<QLearningParameters<T>>(&self.program.registers);

            for _step in 1..parameters.rl_parameters.max_episode_length {
                let state_reward_pair = parameters
                    .rl_parameters
                    .environment
                    .sim(prev_action_state.action);

                let reward = state_reward_pair.get_value();
                score += reward;

                self.program.exec(&parameters.rl_parameters.environment);
                let current_action_state = self
                    .q_table
                    .eval::<QLearningParameters<T>>(&self.program.registers);

                if prev_action_state.register != current_action_state.register {
                    self.q_table.update(
                        prev_action_state.register,
                        current_action_state.action,
                        reward,
                        current_action_state.register,
                    )
                }

                if state_reward_pair.is_terminal() {
                    break;
                }
            }

            scores.push(score);
        }

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = scores.swap_remove(scores.len() / 2);

        self.program.fitness = Some(median);

        median
    }

    fn get_fitness(&self) -> Option<crate::core::characteristics::FitnessScore> {
        self.program.fitness
    }
}

impl<T> Breed for QProgram<T>
where
    T: QLearningInput,
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
    T: QLearningInput,
    T::State: Clone + Debug,
{
    fn mutate<'a>(&self, parameters: &'a Self::GeneratorParameters) -> Self {
        let mutated = self.program.mutate(&parameters.program_parameters);
        QProgram {
            program: mutated,
            q_table: self.q_table.duplicate_new(),
        }
    }
}

impl<T> Generate for QProgram<T>
where
    T: QLearningInput,
    T::State: Clone + Debug,
{
    type GeneratorParameters = QProgramGeneratorParameters;

    fn generate<'a>(parameters: &'a Self::GeneratorParameters) -> Self {
        let program = Program::<QLearningParameters<T>>::generate(&parameters.program_parameters);

        let InstructionGeneratorParameters {
            n_features,
            n_registers,
        } = parameters
            .program_parameters
            .instruction_generator_parameters;

        let q_table = QTable::new(
            n_features,
            n_registers,
            parameters.alpha,
            parameters.gamma,
            parameters.epsilon,
        );

        QProgram { q_table, program }
    }
}

#[derive(Debug)]
pub struct QProgramGeneratorParameters {
    program_parameters: ProgramGeneratorParameters,
    alpha: R32,
    gamma: R32,
    epsilon: R32,
}
