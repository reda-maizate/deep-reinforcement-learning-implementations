use std::f32;
use std::fmt::Debug;
use pbr::ProgressBar;
use tch::nn::{self, OptimizerConfig, RNN, VarStore};
use rand::Rng;
use tch::{Device, Kind, no_grad, Tensor};
use environnements::contracts::DeepSingleAgentEnv;
use crate::to_do::functions::{argmax, get_data_from_index_list};
use crate::to_do::replay_buffer::replay_buffer::ReplayBuffer;


pub struct DoubleDeepQLearning<T> {
    env: T,
    max_iter_count: u32,
    gamma: f32,
    alpha: f64,
    epsilon: f32,
    q: Model,
    target_q: Model, // Add a field for the "target" Q-function
}

type Model = Box<dyn Fn(&Tensor) -> Tensor>;


impl<T: DeepSingleAgentEnv> DoubleDeepQLearning<T> {
    pub fn new(env: T) -> Self {
        let device = Device::cuda_if_available();
        let model_vs = VarStore::new(device);
        let q = Self::model(&model_vs.root(), env.max_action_count() as i64);
        let target_q = Self::model(&model_vs.root(), env.max_action_count() as i64); // Create the "target" Q-function
        Self {
            env,
            max_iter_count: 10_000,
            gamma: 0.99,
            alpha: 0.1,
            epsilon: 0.1,
            q,
            target_q,
        }
    }

    pub fn model(vs: &nn::Path, nact: i64) -> Model {
        let conf = nn::LinearConfig{
            bias: true,
            ..Default::default()
        };
        let linear = nn::linear(vs, 1, nact, conf);
        let seq = nn::seq()
            .add(linear);
        let device = vs.device();
        Box::new(move |xs: &Tensor| {
            xs.to_device(device).apply(&seq)
        })
    }

    pub fn train(&mut self) -> (VarStore, Vec<f64>, Vec<f64>) {
        let device = Device::cuda_if_available();
        let model_vs = VarStore::new(device);
        let mut optimizer = nn::Sgd::default().build(&model_vs, self.alpha).unwrap();
        let mut ema_score = 0.0;
        let mut ema_nb_steps = 0.0;
        let mut first_episode = true;
        let mut step = 0.0;
        let mut ema_score_progress = Vec::new();
        let mut ema_nb_steps_progress = Vec::new();
        let mut pb = ProgressBar::new(self.max_iter_count as u64);
        pb.format("╢▌▌░╟");
        let tau = 0.05; // Interpolation parameter for soft-update
        let update_target_q = 100; // Update the "target" Q-function every `update_target_q` steps
        let mut replay_buffer = ReplayBuffer::new(10000); // Create a replay buffer

        for i in 0..self.max_iter_count {
            if self.env.is_game_over() {
                if first_episode {
                    ema_score = self.env.score() as f64;
                    ema_nb_steps = step;
                    first_episode = false;
                } else {
                    ema_score = (1.0 - 0.9) * self.env.score() as f64 + 0.9 * ema_score;
                    ema_nb_steps = (1.0 - 0.9) * step + 0.9 * ema_nb_steps;
                    ema_score_progress.push(ema_score);
                    ema_nb_steps_progress.push(ema_nb_steps);
                }

                self.env.reset();
                step = 0.0;
            }

            let s = self.env.state_description();
            let aa = self.env.available_actions_ids();

            let tensor_s = Tensor::of_slice(&s).to_kind(Kind::Float);
            let q_prep = no_grad(|| (self.q)(&tensor_s));
            let q_prep_target = no_grad(|| (self.target_q)(&tensor_s)); // Get Q-values from the "target" Q-function

            let action_id;
            if (rand::thread_rng().gen_range(0..2) as f32).partial_cmp(&self.epsilon).unwrap().is_lt() {
                action_id = aa[rand::thread_rng().gen_range(0..aa.len())];
            } else {
                action_id = aa[argmax(&get_data_from_index_list(&Vec::<f32>::from(&q_prep), aa.as_slice())).0];
            }

            let (new_s, r, done) = self.env.step(action_id);
            let new_aa = self.env.available_actions_ids();
            let tensor_new_s = Tensor::of_slice(&new_s).to_kind(Kind::Float);
            let q_prep_new = no_grad(|| self.q(&tensor_new_s));
            let q_prep_new_target = no_grad(|| self.target_q(&tensor_new_s)); // Get Q-values from the "target" Q-function

            let max_q_new = new_aa.iter()
                .map(|a| q_prep_new_target[*a])
                .max_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap();
            let q_value = q_prep[action_id];
            let target = r as f32 + self.gamma * max_q_new;
            let delta = target - q_value;
            replay_buffer.add_transition(s, action_id, r, new_s, done); // Add transition to replay buffer
            if replay_buffer.len() > 1000 {
                let (s_batch, a_batch, r_batch, next_s_batch, done_batch) = replay_buffer.sample_batch(32);
                let tensor_s_batch = Tensor::of_slice(&s_batch).to_kind(Kind::Float);
                let tensor_next_s_batch = Tensor::of_slice(&next_s_batch).to_kind(Kind::Float);
                let q_prep_batch = no_grad(|| self.q(&tensor_s_batch));
                let q_prep_new_batch = no_grad(|| self.target_q(&tensor_next_s_batch));
                let max_q_new_batch = q_prep_new_batch.max_dim1();
                let target_batch = r_batch + (1. - done_batch) * self.gamma * &max_q_new_batch;
                let q_batch = q_prep_batch.index_select(1, &a_batch);
                let delta_batch = target_batch - &q_batch;
                let loss = delta_batch.powi(2).mean();
                optimizer.backward_step(&loss);
            }
            if i % update_target_q == 0 { // Soft-update the "target" Q-function every `update_target_q` steps
                for (param, target_param) in self.q.parameters().iter().zip(self.target_q.parameters_mut().iter()) {
                    target_param.copy_(tau * param + (1. - tau) * target_param);
                }
            }
            step += 1.0;
            pb.inc();
        }
        pb.finish_print("Training complete!");
        (model_vs, ema_score_progress, ema_nb_steps_progress)
    }
}
