use std::fmt::{self, Debug};

use clap::Args;
use derivative::Derivative;
use derive_builder::Builder;
use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::{debug, trace};

use crate::{
    core::{
        engines::{
            breed_engine::{Breed, BreedEngine},
            fitness_engine::{Fitness, FitnessEngine},
            freeze_engine::{Freeze, FreezeEngine},
            generate_engine::{Generate, GenerateEngine},
            mutate_engine::{Mutate, MutateEngine},
            reset_engine::{Reset, ResetEngine},
            status_engine::{Status, StatusEngine},
        },
        environment::{RlState, State},
        instruction::InstructionGeneratorParameters,
        program::{Program, ProgramGeneratorParameters},
        registers::{ActionRegister, ArgmaxInput, Registers},
    },
    utils::{float_ops, random::generator},
};

#[derive(Clone, Serialize, Deserialize)]
pub struct QTable {
    table: Vec<Vec<f64>>,
    q_consts: QConsts,
    freeze: bool,
}

impl Freeze<QTable> for FreezeEngine {
    fn freeze(item: &mut QTable) {
        item.freeze = true;
    }
}

impl Generate<(InstructionGeneratorParameters, QConsts), QTable> for GenerateEngine {
    fn generate(using: (InstructionGeneratorParameters, QConsts)) -> QTable {
        let mut table = QTable {
            table: vec![vec![0.; using.0.n_actions]; using.0.n_registers()],
            q_consts: using.1,
            freeze: false,
        };

        ResetEngine::reset(&mut table);
        table
    }
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

impl Reset<QTable> for ResetEngine {
    fn reset(item: &mut QTable) {
        ResetEngine::reset(&mut item.q_consts);
    }
}

impl QTable {
    pub fn action_random(&self) -> usize {
        let n_actions = self.table[0].len();
        generator().gen_range(0..n_actions)
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
            ActionRegister::Value(register) => register,
            _ => {
                return None;
            }
        };

        let prob = generator().gen_range(0.0..1.0);

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

        trace!(
            register = current_action_state.register,
            action = current_action_state.action,
            reward = current_reward,
            old_q = current_q_value,
            delta_q = new_q_value,
            alpha = self.q_consts.alpha_active,
            gamma = self.q_consts.gamma,
            "Q-table update"
        );

        if !self.freeze {
            self.q_consts.decay();
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Derivative)]
#[derivative(PartialEq, PartialOrd, Ord, Eq)]
pub struct QProgram {
    #[derivative(PartialEq = "ignore", PartialOrd = "ignore", Ord = "ignore")]
    pub q_table: QTable,
    pub program: Program,
}

impl Freeze<QProgram> for FreezeEngine {
    fn freeze(item: &mut QProgram) {
        FreezeEngine::freeze(&mut item.q_table);
    }
}

impl Reset<QProgram> for ResetEngine {
    fn reset(item: &mut QProgram) {
        ResetEngine::reset(&mut item.program);
    }
}

fn get_action_state<T>(environment: &mut T, q_program: &mut QProgram) -> Option<ActionRegisterPair>
where
    T: State,
{
    // Run the program on the current state.
    q_program.program.run(environment);

    // Get the winning action-register pair.
    q_program
        .q_table
        .get_action_register(&q_program.program.registers)
}

impl<T: RlState> Fitness<QProgram, T, ()> for FitnessEngine {
    fn eval_fitness(program: &mut QProgram, states: &mut T) -> f64 {
        let mut score = 0.;

        // We run the program and determine what action to take at the step = 0.
        let mut current_action_state = match get_action_state(states, program) {
            Some(action_state) => action_state,
            None => {
                return f64::NEG_INFINITY;
            }
        };

        // We execute the selected action and continue to repeat the cycle until termination.
        while let Some(state) = states.get() {
            // Act.
            let reward = state.execute_action(current_action_state.action);
            score += reward;

            if state.is_terminal() {
                break;
            }

            let next_action_state = match get_action_state(state, program) {
                Some(action_state) => action_state,
                None => {
                    return f64::NEG_INFINITY;
                }
            };

            // We only update when there is a transition.
            // NOTE: Why?
            if current_action_state.register != next_action_state.register {
                program
                    .q_table
                    .update(current_action_state, reward, next_action_state)
            }

            current_action_state = next_action_state;
        }

        debug!(
            program_id = %program.program.id,
            score = score,
            "Q-Learning fitness evaluation complete"
        );

        trace!(
            program_id = %program.program.id,
            q_table = serde_json::to_string(&program.q_table).ok(),
            initial_state = serde_json::to_string(&states.get_initial_state()).ok(),
            "Full Q-Learning evaluation details"
        );

        score
    }
}

impl Breed<QProgram> for BreedEngine {
    fn two_point_crossover(mate_1: &QProgram, mate_2: &QProgram) -> (QProgram, QProgram) {
        let (child_1_program, child_2_program) =
            BreedEngine::two_point_crossover(&mate_1.program, &mate_2.program);

        let mut child_1 = mate_1.clone();
        let mut child_2 = mate_2.clone();

        child_1.program = child_1_program;
        child_2.program = child_2_program;

        ResetEngine::reset(&mut child_1.program.id);
        ResetEngine::reset(&mut child_2.program.id);

        ResetEngine::reset(&mut child_1.program);
        ResetEngine::reset(&mut child_2.program);

        ResetEngine::reset(&mut child_1.q_table);
        ResetEngine::reset(&mut child_2.q_table);

        (child_1, child_2)
    }
}

impl Status<QProgram> for StatusEngine {
    fn valid(item: &QProgram) -> bool {
        StatusEngine::valid(&item.program)
    }

    fn set_fitness(program: &mut QProgram, fitness: f64) {
        program.program.fitness = fitness;
    }

    fn get_fitness(program: &QProgram) -> f64 {
        program.program.fitness
    }

    fn evaluated(item: &QProgram) -> bool {
        StatusEngine::evaluated(&item.program)
    }
}

impl Mutate<QProgramGeneratorParameters, QProgram> for MutateEngine {
    fn mutate(item: &mut QProgram, using: QProgramGeneratorParameters) {
        MutateEngine::mutate(&mut item.program, using.program_parameters);
        ResetEngine::reset(&mut item.program);
        ResetEngine::reset(&mut item.program.id);
        ResetEngine::reset(&mut item.q_table);
    }
}

impl Generate<QProgramGeneratorParameters, QProgram> for GenerateEngine {
    fn generate(using: QProgramGeneratorParameters) -> QProgram {
        let program = GenerateEngine::generate(using.program_parameters);
        let q_table = GenerateEngine::generate((
            using.program_parameters.instruction_generator_parameters,
            using.consts,
        ));

        QProgram { q_table, program }
    }
}

#[derive(Debug, Clone, Args, Deserialize, Serialize, Copy, Builder)]
pub struct QProgramGeneratorParameters {
    #[command(flatten)]
    pub program_parameters: ProgramGeneratorParameters,
    #[builder(default)]
    #[command(flatten)]
    pub consts: QConsts,
}

#[derive(Debug, Clone, Copy, Args, Serialize, Deserialize, Builder)]
pub struct QConsts {
    /// Learning Factor
    #[arg(long, default_value = "0.1")]
    #[builder(default = "0.1")]
    alpha: f64,
    /// Discount Factor
    #[arg(long, default_value = "0.9")]
    #[builder(default = "0.9")]
    gamma: f64,
    /// Exploration Factor
    #[arg(long, default_value = "0.05")]
    #[builder(default = "0.05")]
    epsilon: f64,
    /// Learning Rate Decay
    #[arg(long, default_value = "0.01")]
    #[builder(default = "0.01")]
    alpha_decay: f64,
    /// Exploration Decay
    #[arg(long, default_value = "0.001")]
    #[builder(default = "0.001")]
    epsilon_decay: f64,

    /// To allow new programs to start from the new state, we have active
    /// properties to mutuate.
    #[arg(skip)]
    #[serde(skip)]
    #[builder(setter(skip), default)]
    alpha_active: f64,

    #[serde(skip)]
    #[arg(skip)]
    #[builder(setter(skip), default)]
    epsilon_active: f64,
}

impl Reset<QConsts> for ResetEngine {
    fn reset(item: &mut QConsts) {
        item.alpha_active = item.alpha;
        item.epsilon_active = item.epsilon;
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
        let alpha = generator().gen_range(0.0..1.);
        let gamma = generator().gen_range(0.0..1.);
        let epsilon = generator().gen_range(0.0..1.);
        let alpha_decay = generator().gen_range(0.0..1.);
        let epsilon_decay = generator().gen_range(0.0..1.);
        Self {
            alpha,
            gamma,
            epsilon,
            alpha_decay,
            epsilon_decay,
            alpha_active: alpha,
            epsilon_active: epsilon_decay,
        }
    }
}
