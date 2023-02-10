use std::f32;
use std::fmt::Debug;
use pbr::ProgressBar;
use tch::nn::{self, OptimizerConfig, VarStore};
use rand::Rng;
use tch::{Device, Kind, no_grad, Tensor};
use environnements::contracts::DeepSingleAgentEnv;
use crate::to_do::functions::{argmax, get_data_from_index_list};


#[derive(Debug)]
pub struct REinforce<T> {
    env: T,
    max_iter_count: u32,
    gamma: f32,
    alpha_pi: f64,
    alpha_v: f64
}

type Bicephale_Model = Box<dyn Fn(&Tensor) -> (Tensor, Tensor)>;
type Model = Box<dyn Fn(&Tensor) -> Tensor>;


impl<T: DeepSingleAgentEnv> REinforce<T> {
    pub fn new(env: T) -> Self {
        Self {
            env,
            max_iter_count: 10_000,
            gamma: 0.99,
            alpha_pi: 0.0001,
            alpha_v: 0.0001
        }
    }

    fn pi_model(vs: &nn::Path, state_dim: i64, max_action_count: i64) -> Bicephale_Model {
        let conf = nn::LinearConfig{
            bias: true,
            ..Default::default()
        };
        let pi_input_mask = nn::linear(vs, max_action_count,  1, conf);
        let seq = nn::seq()
            .add(nn::linear(vs, state_dim,  128, conf))
            .add_fn(|xs: &Tensor| xs.tanh())
            .add(nn::linear(vs, 128,  128, conf))
            .add_fn(|xs: &Tensor| xs.tanh())
            .add(nn::linear(vs, 128,  max_action_count, conf))
            .add_fn(|xs: &Tensor| xs.tanh());

        let pi_state_desc = nn::Linear::new(vs, 128,  max_action_count, Default::default());
        let pi_mask = nn::Linear::new(vs, 1,  max_action_count, Default::default());
        let device = vs.device();
        Box::new(move |xs: &Tensor| {
            let xs = xs.to_device(device).apply(&seq);
            (xs.apply(&pi_state_desc), xs.apply(&pi_mask))
        })
    }

    fn v_model(vs: &nn::Path, nact: i64) -> Model {
        let conf = nn::LinearConfig{
            bias: true,
            ..Default::default()
        };
        let seq = nn::seq()
            .add(nn::linear(vs, state_dim,  128, conf))
            .add_fn(|xs: &Tensor| xs.tanh())
            .add(nn::linear(vs, 128,  128, conf))
            .add_fn(|xs: &Tensor| xs.tanh())
            .add(nn::linear(vs, 128,  max_action_count, conf))
            .add_fn(|xs: &Tensor| xs.tanh());

        let v = nn::Linear::new(vs, 128,  1, Default::default());
        let device = vs.device();
        Box::new(move |xs: &Tensor| {
            let xs = xs.to_device(device).apply(&seq);
            xs.apply(&v)
        })
    }

    pub fn train(&mut self) -> (VarStore, Vec<f64>, Vec<f64>) {
        let device = Device::cuda_if_available();
        let model_vs = VarStore::new(device);
        let q = Self::model(&model_vs.root(), self.env.max_action_count() as i64);

        let mut ema_score = 0.0;
        let mut ema_nb_steps = 0.0;
        let mut first_episode = true;

        let mut step = 0.0;
        let mut ema_score_progress = Vec::new();
        let mut ema_nb_steps_progress = Vec::new();

        let mut optimizer = nn::Sgd::default().build(&model_vs, self.alpha).unwrap();

        // Progress bar
        let mut pb = ProgressBar::new(self.max_iter_count as u64);
        pb.format("╢▌▌░╟");

        for _ in 0..self.max_iter_count {
            //self.env.view();
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
            let q_prep = no_grad(|| q(&tensor_s));

            let action_id;
            if (rand::thread_rng().gen_range(0..2) as f32).partial_cmp(&self.epsilon).unwrap().is_lt() {
                action_id = aa[rand::thread_rng().gen_range(0..aa.len())];
            } else {
                action_id = aa[argmax(&get_data_from_index_list(&Vec::<f32>::from(&q_prep), aa.as_slice())).0];
            }

            let old_score = self.env.score();
            self.env.act_with_action_id(action_id);
            let new_score = self.env.score();
            let r = new_score - old_score;

            let s_p = self.env.state_description();
            let aa_p = self.env.available_actions_ids();

            let y;
            if self.env.is_game_over() {
                y = r;
            } else {
                let tensor_s_p = Tensor::of_slice(&s_p).to_kind(Kind::Float);
                let q_pred_p = no_grad(|| q(&tensor_s_p));
                let max_q_pred_p = argmax(&get_data_from_index_list(&Vec::<f32>::from(&q_pred_p), aa_p.as_slice())).1;
                y = r + self.gamma * max_q_pred_p;
            }

            optimizer.zero_grad();
            let q_s_a = q(&tensor_s).unsqueeze(0).get(0).get(action_id as i64);
            // Improvement possible : Parallelize the computation with multiple environments by changing the next line.
            let loss = (y - &q_s_a).pow(&Tensor::of_slice(&[2]));

            optimizer.backward_step(&loss);

            step += 1.0;
            pb.inc();
        }
        (model_vs, ema_score_progress, ema_nb_steps_progress)
    }
}