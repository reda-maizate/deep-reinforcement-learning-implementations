use std::f32;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::os::unix::raw::time_t;
use std::time::Instant;
use pbr::ProgressBar;
use tch::nn::{self, OptimizerConfig, VarStore};
use rand::Rng;
use tch::{Device, Kind, no_grad, Tensor};
use environnements::contracts::DeepSingleAgentEnv;
use random_choice::random_choice;
use crate::to_do::functions::{argmax, get_data_from_index_list, load_model, save_model};


#[derive(Debug)]
pub struct ReinforceWithLearnedBaseline<T> {
    env: T,
    max_iter_count: u32,
    gamma: f32,
    alpha_pi: f64,
    alpha_v: f64
}

type Bicephale_Model = Box<dyn Fn(&Tensor) -> (Tensor, Tensor)>;
type Model = Box<dyn Fn(&Tensor) -> Tensor>;

pub fn pi_model(vs: &nn::Path, max_action_count: i64) -> Model {
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

fn v_model(vs: &nn::Path) -> Model {
    let conf = nn::LinearConfig{
        bias: true,
        ..Default::default()
    };
    let v_input_mask = nn::linear(vs, 1,  1, conf);
    let seq = nn::seq()
        .add(v_input_mask)
        .add(nn::linear(vs, 1,  1, conf));

    let device = vs.device();
    Box::new(move |xs: &Tensor| {
        xs.to_device(device).apply(&seq)
    })
}

impl<T: DeepSingleAgentEnv> ReinforceWithLearnedBaseline<T> {
    pub fn new(env: T) -> Self {
        Self {
            env,
            max_iter_count: 10_000,
            gamma: 0.99,
            alpha_pi: 0.01,
            alpha_v: 0.01
        }
    }

    pub fn train(&mut self, save: bool) -> (VarStore, Vec<f64>, Vec<f64>) {
        println!("Training REINFORCE with learned baseline");
        let device = Device::cuda_if_available();
        let pi_vs = VarStore::new(device);
        let v_vs = VarStore::new(device);
        let pi = pi_model(&pi_vs.root(), self.env.max_action_count() as i64);
        let v = v_model(&v_vs.root());

        let mut ema_score = 0.0;
        let mut ema_nb_steps = 0.0;
        let mut first_episode = true;

        let mut step = 0.0;
        let mut ema_score_progress = Vec::new();
        let mut ema_nb_steps_progress = Vec::new();

        let mut episode_states_buffer: Vec<Vec<f64>> = Vec::new();
        let mut episode_actions_buffer: Vec<usize> = Vec::new();
        let mut episode_rewards_buffer = Vec::new();

        let mut optimizer_pi = nn::Adam::default().build(&pi_vs, self.alpha_pi).unwrap();
        let mut optimizer_v = nn::Adam::default().build(&v_vs, self.alpha_v).unwrap();

        // Progress bar
        let mut pb = ProgressBar::new(self.max_iter_count as u64);
        pb.format("╢▌▌░╟");

        for _ in 0..self.max_iter_count {
            // self.env.view();
            if self.env.is_game_over() {
                let mut G = 0.0;

                for t in (0..episode_states_buffer.len()).rev() {
                    G = episode_rewards_buffer[t] + self.gamma * G;

                    let v_s_pred = &v(&Tensor::of_slice(&episode_states_buffer[t]).to_kind(Kind::Float)).get(0);
                    // println!("v_s_pred: {:?}", v_s_pred);

                    let delta = G - v_s_pred;
                    // println!("delta: {:?}", delta);
                    let loss_v = self.alpha_v * delta * v_s_pred;
                    // println!("loss_v: {:?}", loss_v);

                    optimizer_v.backward_step(&loss_v);

                    let episode_states_buffer_t = Tensor::of_slice(&episode_states_buffer[t])
                        .to_kind(Kind::Float);
                    // println!("episode_states_buffer_t: {:?}", episode_states_buffer_t);
                    let mut pi_s_a_t = pi(&episode_states_buffer_t).softmax(0, Kind::Float);
                    // println!("prediction training: {:?}", pi_s_a_t);
                    pi_s_a_t = pi_s_a_t.get(episode_actions_buffer[t] as i64);
                    // println!("pi_s_a_t: {:?}", pi_s_a_t);
                    let log_pi_s_a_t = pi_s_a_t.log(); // Log2 ou log10 ?
                    // println!("log_pi_s_a_t: {:?}", log_pi_s_a_t);

                    let loss_pi = - (self.alpha_pi * (self.gamma.powf(t as f32) as f64) * (G as f64) * log_pi_s_a_t);
                    // println!("loss: {:?}", loss);

                    // println!("-- Weights --");
                    // model_vs.variables().get("weight").unwrap().print();
                    // println!("-- Biases --");
                    // model_vs.variables().get("bias").unwrap().print();
                    // Add
                    optimizer_pi.backward_step(&loss_pi);
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
                    ema_score = (1.0 - 0.95) * self.env.score() as f64 + 0.95 * ema_score;
                    ema_nb_steps = (1.0 - 0.95) * step + 0.95 * ema_nb_steps;
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
            // println!("prediction: {:?}", Vec::<f32>::from(&pi_prep));
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
        if save {
            let model_path = format!("src/models/{}/reinforce_lb_max_iter_{}_g_{}_alpha_pi_{}_alpha_v_{}.pt",
                    self.env.name(), self.max_iter_count, self.gamma, self.alpha_pi, self.alpha_v);
            save_model(&pi_vs, &model_path);
        }
        (pi_vs, ema_score_progress, ema_nb_steps_progress)
    }
}

pub fn evaluate<T: DeepSingleAgentEnv>(env: T, path: &str, nb_games: usize) {
    let mut env = env;
    let mut score = 0.0;
    let mut model_vs = nn::VarStore::new(Device::Cpu);
    let model = pi_model(&model_vs.root(), env.max_action_count() as i64);
    model_vs.load(path).unwrap();
    let start_time = Instant::now();
    let mut current_game = 0;
    while current_game < nb_games {
        if env.is_game_over() {
            score += env.score() as f64;
            current_game += 1;
            env.reset();
        }
        let s = env.state_description();
        let tensor_s = Tensor::of_slice(&s.clone()).to_kind(Kind::Float);
        let pi = no_grad(|| model(&tensor_s).softmax(0, Kind::Float));
        let a = argmax(&Vec::<f32>::from(&pi)).0;
        env.act_with_action_id(a);
    }
    println!("Evaluation time : {:.2?}", start_time.elapsed());
    println!("Mean score: {}", score / nb_games as f64);
}