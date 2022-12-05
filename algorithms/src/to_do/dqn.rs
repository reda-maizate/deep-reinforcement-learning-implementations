use std::f32;
use std::fmt::Debug;
use pbr::ProgressBar;
use tch::nn::{self, Module, OptimizerConfig, VarStore};
use rand::Rng;
use tch::{Device, kind, Kind, no_grad, Tensor};
use environnements::contracts::DeepSingleAgentEnv;
use crate::to_do::functions::{argmax, get_data_from_index_list};


#[derive(Debug)]
pub struct DeepQLearning<T> {
    env: T,
    max_iter_count: u32,
    gamma: f32,
    alpha: f64,
    epsilon: f32
}

type Model = Box<dyn Fn(&Tensor) -> Tensor>;


impl<T: DeepSingleAgentEnv> DeepQLearning<T> {
    pub fn new(env: T) -> Self {
        Self {
            env,
            max_iter_count: 10_000,
            gamma: 0.99,
            alpha: 0.1,
            epsilon: 0.2
        }
    }

    fn model(vs: &nn::Path, nact: i64) -> Model {
        let seq = nn::seq()
            .add(nn::linear(vs / "l1", 1,  nact, Default::default()));

        let device = vs.device();
        Box::new(move |xs: &Tensor| {
            xs.to_device(device).apply(&seq)
        })
    }

    pub fn train(&mut self) -> (VarStore, Vec<f64>, Vec<f64>) {
        let device = Device::cuda_if_available();
        let model_vs = VarStore::new(device);
        //let seq = nn::seq()
        //    .add(nn::linear(&model_vs.root() / "linear1", 1, self.env.max_action_count() as i64, Default::default()));
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

        for _ in 0..self.max_iter_count { //self.max_iter_count).progress() {
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

            // let test = tch::no_grad(|| q(&s));

            let tensor_s = Tensor::of_slice(&s).to_kind(Kind::Float);
            // let q_prep = seq.forward(&tensor_s);
            let q_prep = no_grad(|| q(&tensor_s));
            //println!("q_prep: {:?}", Vec::<f32>::
            // from(&q_prep));

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
                // let q_pred_p = seq.forward(&tensor_s_p);
                let q_pred_p = no_grad(|| q(&tensor_s_p));
                //println!("q_pred_p: {:?}", Vec::<f32>::from( &q_pred_p));
                let max_q_pred_p = argmax(&get_data_from_index_list(&Vec::<f32>::from(&q_pred_p), aa_p.as_slice())).1;
                y = r + self.gamma * max_q_pred_p;
            }

            // let q_s_a = seq.forward(&tensor_s).unsqueeze(0).get(0).get(action_id as i64);
            let q_s_a = q(&tensor_s).unsqueeze(0).get(0).get(action_id as i64);
            // Improvement possible : Parallelize the computation with multiple environments by changing the next line.
            let loss = ((y - &q_s_a) * (y - &q_s_a)).requires_grad_(true);
            //q_s_a.print();

            //optimizer.zero_grad();
            //println!("Before: {:?}", model_vs.trainable_variables());
            //println!("Gradients: {:?}", loss.grad());
            optimizer.backward_step(&loss);
            //println!("After: {:?}", model_vs.trainable_variables());

            step += 1.0;
            pb.inc();
        }
        (model_vs, ema_score_progress, ema_nb_steps_progress)
    }
}