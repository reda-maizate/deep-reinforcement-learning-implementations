use std::borrow::{Borrow, BorrowMut};
use std::f32;
use environnements::contracts::DeepSingleAgentEnv;
use pbr::ProgressBar;
use tch::nn::{self, OptimizerConfig, VarStore};
use rand::Rng;
use tch::{Device, Kind, no_grad, Tensor};
use crate::to_do::functions::{argmax, get_data_from_index_list, get_random_mini_batch};
use crate::utils::score::EMA;

type Model = Box<dyn Fn(&Tensor) -> Tensor>;

#[derive(Debug)]
pub struct DDQN {
    q_model: VarStore,
    q_target_model: VarStore,
}

impl DDQN {
    pub fn new() -> Self {
        let device = Device::cuda_if_available();
        let model_vs = VarStore::new(device);
        let target_model_vs = VarStore::new(device);
        Self {
            q_model: model_vs,
            q_target_model: target_model_vs,
        }
    }

    fn q_model(vs: &nn::Path, nact: i64) -> Model {
        let conf = nn::LinearConfig {
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

    fn update_score<T: DeepSingleAgentEnv>(env: &mut T, ema: &mut EMA) {
        if env.is_game_over() {
            if ema.first_episode {
                ema.score = env.score() as f64;
                ema.nb_steps = ema.step;
                ema.first_episode = false;
            } else {
                ema.score = (1.0 - 0.9) * env.score() as f64 + 0.9 * ema.score;
                ema.nb_steps = (1.0 - 0.9) * ema.step + 0.9 * ema.nb_steps;
                ema.score_progress.push(ema.score);
                ema.nb_steps_progress.push(ema.nb_steps);
            }

            env.reset();
            ema.step = 0.0;
        }
    }

    fn step<T: DeepSingleAgentEnv>(env: &mut T, q: &Model, tensor_s: &Tensor, aa: &Vec<usize>, epsilon: f32) -> (usize, f32, Vec<f64>, Vec<usize>) {
        let action_id;
        if (rand::thread_rng().gen_range(0..2) as f32).partial_cmp(&epsilon).unwrap().is_lt() {
            action_id = aa[rand::thread_rng().gen_range(0..aa.len())];
        } else {
            let q_prep = no_grad(|| q(&tensor_s));
            action_id = aa[argmax(&get_data_from_index_list(&Vec::<f32>::from(&q_prep), aa.as_slice())).0];
        }

        let old_score = env.score();
        env.act_with_action_id(action_id);
        let new_score = env.score();
        let r = new_score - old_score;

        let s_p = env.state_description();
        let aa_p = env.available_actions_ids();

        (action_id, r, s_p, aa_p)
    }

    pub fn train<T: DeepSingleAgentEnv>(&mut self, env: &mut T, max_iter_count: u32, gamma: f32, alpha: f64, epsilon: f32, target_update_frequency: u32) -> (Vec<f64>, Vec<f64>) {
        let q = Self::q_model(&self.q_model.root(), env.max_action_count() as i64);
        let mut optimizer = nn::Sgd::default().build(&self.q_model, alpha).unwrap();
        let q_target = Self::q_model(&self.q_target_model.root(), env.max_action_count() as i64);

        let mut ema = EMA::new();

        // Progress bar
        let mut pb = ProgressBar::new(max_iter_count as u64);
        pb.format("╢▌▌░╟");

        for t in 0..max_iter_count {
            //self.env.view();
            Self::update_score(env.borrow_mut(), ema.borrow_mut());

            let s = env.state_description();
            let tensor_s = Tensor::of_slice(&s).to_kind(Kind::Float);
            let aa = env.available_actions_ids();

            let (a, r, s_p, aa_p) = Self::step(env.borrow_mut(), &q, &tensor_s, &aa, epsilon);

            let y;
            if env.is_game_over() {
                y = r;
            } else {
                let tensor_s_p = Tensor::of_slice(&s_p).to_kind(Kind::Float);
                let q_pred_p = no_grad(|| q_target(&tensor_s_p));
                let max_q_pred_p = argmax(&get_data_from_index_list(&Vec::<f32>::from(&q_pred_p), aa_p.as_slice())).1;
                y = r + gamma * max_q_pred_p;
            }

            optimizer.zero_grad();
            let q_s_a = q(&tensor_s).unsqueeze(0).get(0).get(a as i64);
            // Improvement possible : Parallelize the computation with multiple environments by changing the next line.
            let loss = (y - &q_s_a).pow(&Tensor::of_slice(&[2]));

            optimizer.backward_step(&loss);

            if t % target_update_frequency == 0 {
                self.q_target_model.copy(&self.q_model).unwrap();
            }

            ema.step += 1.0;
            pb.inc();
        }
        (ema.score_progress, ema.nb_steps_progress)
    }

    pub fn train_with_er<T: DeepSingleAgentEnv>(&mut self, env: &mut T, max_iter_count: u32, gamma: f32, alpha: f64, epsilon: f32, target_update_frequency: u32, batch_size: usize) -> (Vec<f64>, Vec<f64>) {
        let q = Self::q_model(&self.q_model.root(), env.max_action_count() as i64);
        let mut optimizer = nn::Sgd::default().build(&self.q_model, alpha).unwrap();
        let q_target = Self::q_model(&self.q_target_model.root(), env.max_action_count() as i64);
        let mut replay_buffer = Vec::new();

        let mut ema = EMA::new();

        // Progress bar
        let mut pb = ProgressBar::new(max_iter_count as u64);
        pb.format("╢▌▌░╟");

        for t in 0..max_iter_count {
            //self.env.view();
            Self::update_score(env.borrow_mut(), ema.borrow_mut());

            let s = env.state_description();
            let tensor_s = Tensor::of_slice(&s).to_kind(Kind::Float);
            let aa = env.available_actions_ids();

            let (a, r, s_p, aa_p) = Self::step(env.borrow_mut(), &q, &tensor_s, &aa, epsilon);

            replay_buffer.push((s, a, r, s_p, aa_p, env.is_game_over()));

            let mini_batch = get_random_mini_batch(&replay_buffer, batch_size);

            for transition in mini_batch {
                let (s, a, r, s_p, aa_p, done) = transition;
                let mut y = r;
                if !done {
                    let tensor_s_p = Tensor::of_slice(&s_p).to_kind(Kind::Float);
                    let q_pred_p = no_grad(|| q_target(&tensor_s_p));
                    let max_q_pred_p = argmax(&get_data_from_index_list(&Vec::<f32>::from(&q_pred_p), aa_p.as_slice())).1;
                    y = r + gamma * max_q_pred_p;
                }

                optimizer.zero_grad();
                let tensor_s = Tensor::of_slice(&s).to_kind(Kind::Float);
                let q_s_a = q(&tensor_s).unsqueeze(0).get(0).get(a as i64);
                // Improvement possible : Parallelize the computation with multiple environments by changing the next line.
                let loss = (y - &q_s_a).pow(&Tensor::of_slice(&[2]));

                optimizer.backward_step(&loss);
            }

            if t % target_update_frequency == 0 {
                self.q_target_model.copy(&self.q_model).unwrap();
            }

            ema.step += 1.0;
            pb.inc();
        }
        (ema.score_progress, ema.nb_steps_progress)
    }

    pub fn get_model(&mut self) -> &VarStore {
        &self.q_target_model
    }
}