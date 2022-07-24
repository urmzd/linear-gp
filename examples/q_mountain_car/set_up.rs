#[derive(Debug, Serialize, new, Clone)]
pub struct MountainCarInput {
    environment: MountainCarEnv,
}
impl GeneticAlgorithm for QMountainCarLgp {
    type O = QProgram<MountainCarInput>;
}
pub struct QMountainCarLgp;

impl ValidInput for MountainCarInput {
    const N_INPUT_REGISTERS: usize = 2;
    const N_ACTION_REGISTERS: usize = 3;

    fn flat(&self) -> Vec<R32> {
        let state = self.get_state();
        state
    }
}
impl ReinforcementLearningInput for MountainCarInput {
    fn init(&mut self) {
        self.environment.reset(Some(0), false, None);
    }

    fn sim(&mut self, action: usize) -> StateRewardPair {
        let ActionReward { reward, done, .. } = self.environment.step(action);
        let reward_ RegisterValue = reward.into_inner() as  RegisterValue;

        StateRewardPair {
            state: self.get_state(),
            reward: match done {
                true => Reward::Terminal(reward_ RegisterValue),
                false => Reward::Continue(reward_ RegisterValue),
            },
        }
    }

    fn get_state(&self) -> Vec<R32> {
        let state = &self.environment.state;
        [state.position, state.velocity]
            .map(|v| v.into_inner() as  RegisterValue)
            .to_vec()
    }

    fn finish(&mut self) {
        self.environment.close();
    }

    fn reset(&mut self) {
        self.environment.reset(None, false, None);
    }
}

impl QLearningInput for MountainCarInput {
    type State = <MountainCarEnv as Env>::Observation;

    fn set_state(&mut self, state: Self::State) {
        self.environment.state = state;
    }
}
