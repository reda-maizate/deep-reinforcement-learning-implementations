pub mod replay_buffer {
    use std::collections::VecDeque;
    use rand::Rng;

    pub struct ReplayBuffer {
        buffer: VecDeque<Transition>,
        capacity: usize,
    }

    #[derive(Clone)]
    pub struct Transition {
        state: Vec<f32>,
        action: usize,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
    }

    impl ReplayBuffer {
        pub fn new(capacity: usize) -> Self {
            Self {
                buffer: VecDeque::with_capacity(capacity),
                capacity,
            }
        }

        pub fn add_transition(&mut self, state: Vec<f32>, action: usize, reward: f32, next_state: Vec<f32>, done: bool) {
            if self.buffer.len() == self.capacity {
                self.buffer.pop_front();
            }
            self.buffer.push_back(Transition {
                state,
                action,
                reward,
                next_state,
                done,
            });
        }

        pub fn sample_batch(&self, batch_size: usize) -> (Vec<Vec<f32>>, Vec<usize>, Vec<f32>, Vec<Vec<f32>>, Vec<bool>) {
            let mut states = Vec::new();
            let mut actions = Vec::new();
            let mut rewards = Vec::new();
            let mut next_states = Vec::new();
            let mut dones = Vec::new();
            let mut rng = rand::thread_rng();
            let buffer_size = self.buffer.len();
            for _ in 0..batch_size {
                let idx = rng.gen_range(0..buffer_size);
                let transition = self.buffer[idx].clone();
                states.push(transition.state);
                actions.push(transition.action);
                rewards.push(transition.reward);
                next_states.push(transition.next_state);
                dones.push(transition.done);
            }
            (states, actions, rewards, next_states, dones)
        }

        pub fn len(&self) -> usize {
            self.buffer.len()
        }
    }
}
