use std::f32;
use std::fmt::Debug;
use std::iter::zip;
use std::ops::{Add, AddAssign};
use pbr::ProgressBar;
use tch::nn::{self, OptimizerConfig, VarStore};
use rand::Rng;
use tch::{Device, Kind, no_grad, Tensor};
use environnements::contracts::DeepSingleAgentEnv;
use rand::distributions::Slice;
use crate::to_do::functions::{argmax, get_data_from_index_list};
extern crate random_choice;
use self::random_choice::random_choice;


#[derive(Debug)]
pub struct REinforce<T> {
    env: T,
    max_iter_count: u32,
    gamma: f32,
    alpha: f64,
}

type Model = Box<dyn Fn(&Tensor) -> Tensor>;


impl<T: DeepSingleAgentEnv> REinforce<T> {
    pub fn new(env: T) -> Self {
        Self {
            env,
            max_iter_count: 10_000,
            gamma: 0.99,
            alpha: 0.1,
        }
    }

    fn pi_model(vs: &nn::Path, max_action_count: i64) -> Model {
        let conf = nn::LinearConfig{
            bias: true,
            ..Default::default()
        };
        let pi_input_mask = nn::linear(vs, 1,  1, conf);
        let seq = nn::seq()
            .add(pi_input_mask)
            .add(nn::linear(vs, 1,  max_action_count, conf));

        let device = vs.device();
        Box::new(move |xs: &Tensor| {
            xs.to_device(device).apply(&seq)
        })
    }

    pub fn train(&mut self) -> (VarStore, Vec<f64>, Vec<f64>) {
        println!("Training REINFORCE");
        let device = Device::cuda_if_available();
        let model_vs = VarStore::new(device);
        let pi = Self::pi_model(&model_vs.root(), self.env.max_action_count() as i64);

        let mut ema_score = 0.0;
        let mut ema_nb_steps = 0.0;
        let mut first_episode = true;

        let mut step = 0.0;
        let mut ema_score_progress = Vec::new();
        let mut ema_nb_steps_progress = Vec::new();

        let mut episode_states_buffer: Vec<Vec<f64>> = Vec::new();
        let mut episode_actions_buffer: Vec<usize> = Vec::new();
        let mut episode_rewards_buffer = Vec::new();

        let mut optimizer = nn::Adam::default().build(&model_vs, self.alpha).unwrap();

        // Progress bar
        let mut pb = ProgressBar::new(self.max_iter_count as u64);
        pb.format("╢▌▌░╟");

        for _ in 0..self.max_iter_count {
            // self.env.view();
            if self.env.is_game_over() {
                let mut G = 0.0;

                for t in (0..episode_states_buffer.len()).rev() {
                    G = episode_rewards_buffer[t] + self.gamma * G;

                    let episode_states_buffer_t = Tensor::of_slice(&episode_states_buffer[t])
                        .to_kind(Kind::Float);
                    // println!("episode_states_buffer_t: {:?}", episode_states_buffer_t);
                    let mut pi_s_a_t = pi(&episode_states_buffer_t).softmax(0, Kind::Float);
                    println!("prediction training: {:?}", pi_s_a_t);
                    pi_s_a_t = pi_s_a_t.get(episode_actions_buffer[t] as i64);
                    // println!("pi_s_a_t: {:?}", pi_s_a_t);
                    let log_pi_s_a_t = pi_s_a_t.log(); // Log2 ou log10 ?
                    // println!("log_pi_s_a_t: {:?}", log_pi_s_a_t);

                    let loss = - (self.alpha * (self.gamma.powf(t as f32) as f64) * (G as f64) * log_pi_s_a_t);
                    println!("loss: {:?}", loss);

                    // println!("-- Weights --");
                    // model_vs.variables().get("weight").unwrap().print();
                    // println!("-- Biases --");
                    // model_vs.variables().get("bias").unwrap().print();
                    // Add
                    optimizer.backward_step(&loss);
                    // println!("---------------------");
                    // println!("-- Weights --");
                    // model_vs.variables().get("weight").unwrap().print();
                    // println!("-- Biases --");
                    // model_vs.variables().get("bias").unwrap().print();
                }

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
                episode_states_buffer.clear();
                episode_actions_buffer.clear();
                episode_rewards_buffer.clear();
                step = 0.0;
            }

            let s = self.env.state_description();
            episode_states_buffer.push(s.clone());

            let aa = self.env.available_actions_ids();

            let tensor_s = Tensor::of_slice(&s.clone()).to_kind(Kind::Float);
            let pi_prep = no_grad(|| pi(&tensor_s)).softmax(0, Kind::Float); // TODO: Add softmax to the output of the model
            println!("prediction: {:?}", Vec::<f32>::from(&pi_prep));
            let allowed_pi_s = get_data_from_index_list(&Vec::<f32>::from(&pi_prep), aa.as_slice());
            let sum_allowed_pi_s: f32 = allowed_pi_s.iter().sum();

            let probs;
            if sum_allowed_pi_s == 0.0 {
                probs = vec![1.0 / aa.len() as f32; aa.len()];
            } else {
                probs = allowed_pi_s.iter().map(|x| x / sum_allowed_pi_s).collect();
            }

            let a = random_choice().random_choice_f32(&aa, &probs, 1)[0];
            episode_actions_buffer.push(*a);

            let old_score = self.env.score();
            self.env.act_with_action_id(*a);
            let new_score = self.env.score();
            let r = new_score - old_score;

            episode_rewards_buffer.push(r);
            step += 1.0;
            pb.inc();
        }
        (model_vs, ema_score_progress, ema_nb_steps_progress)
    }
}