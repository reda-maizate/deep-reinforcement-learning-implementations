use std::borrow::BorrowMut;
use std::f32;
use std::fmt::Debug;
use pbr::ProgressBar;
use tch::nn::{self, OptimizerConfig, VarStore};
use tch::{Device, Kind, no_grad, Tensor};
use environnements::contracts::DeepSingleAgentEnv;
use crate::to_do::functions::{argmax, get_data_from_index_list, update_score, step};
use crate::utils::score::EMA;


#[derive(Debug)]
pub struct DeepQLearning {
    q_model: VarStore,
}

type Model = Box<dyn Fn(&Tensor) -> Tensor>;

impl DeepQLearning {
    pub fn new() -> Self {
        let device = Device::cuda_if_available();
        let model_vs = VarStore::new(device);
        Self {
            q_model: model_vs,
        }
    }

    pub fn create_model(vs: &nn::Path, n_states: i64, n_act: i64) -> Model {
        let conf = nn::LinearConfig{
            bias: true,
            ..Default::default()
        };
        let linear = nn::linear(vs, n_states,  n_act, conf);
        let seq = nn::seq()
            .add(linear);

        let device = vs.device();
        Box::new(move |xs: &Tensor| {
            xs.to_device(device).apply(&seq)
        })
    }

    pub fn train<T: DeepSingleAgentEnv>(&mut self, env: &mut T, max_iter_count: u32,
                                        gamma: f32, alpha: f64, epsilon: f32) -> EMA {
        let q = Self::create_model(
            &self.q_model.root(),
            env.state_dim() as i64,
            env.max_action_count() as i64
        );
        let mut optimizer = nn::Sgd::default().build(
            &self.q_model,
            alpha
        ).unwrap();

        let mut ema = EMA::new();

        // Progress bar
        let mut pb = ProgressBar::new(max_iter_count as u64);
        pb.format("╢▌▌░╟");

        for _ in 0..max_iter_count {
            // env.view();
            update_score(env.borrow_mut(), ema.borrow_mut());

            let s = env.state_description();
            let tensor_s = Tensor::of_slice(&s).to_kind(Kind::Float);
            let aa = env.available_actions_ids();

            let (a, r, s_p, aa_p) = step(env.borrow_mut(), &q, &tensor_s, &aa, epsilon);

            let y;
            if env.is_game_over() {
                y = r;
            } else {
                let tensor_s_p = Tensor::of_slice(&s_p).to_kind(Kind::Float);
                let q_pred_p = no_grad(|| q(&tensor_s_p));
                let max_q_pred_p = argmax(
                    &get_data_from_index_list(
                        &Vec::<f32>::from(&q_pred_p),
                        aa_p.as_slice()
                    )
                ).1;
                y = r + gamma * max_q_pred_p;
            }

            optimizer.zero_grad();
            let q_s_a = q(&tensor_s).unsqueeze(0).get(0).get(a as i64);
            // Improvement possible : Parallelize the computation with multiple environments by changing the next line.
            let loss = (y - &q_s_a).pow(&Tensor::of_slice(&[2]));

            optimizer.backward_step(&loss);

            ema.step += 1.0;
            pb.inc();
        }
        ema
    }

    pub fn get_model(&mut self) -> &VarStore {
        &self.q_model
    }

    pub fn load_model<T: DeepSingleAgentEnv>(env: &mut T, path: &str) -> Model {
        let mut model_vs = VarStore::new(Device::Cpu);
        let model = Self::create_model(
            &model_vs.root(),
            env.state_dim() as i64,
            env.max_action_count() as i64
        );
        model_vs.load(path).unwrap();
        model
    }

    pub fn evaluate_model<T: DeepSingleAgentEnv>(env: &mut T, model: &Model) {
        let mut score = 0.0;
        let mut current_game = 0;
        while current_game < 1000 {
            if env.is_game_over() {
                score += env.score() as f64;
                current_game += 1;
                env.reset();
            }
            let s = env.state_description();
            let tensor_s = Tensor::of_slice(&s).to_kind(Kind::Float);
            let pred = no_grad(|| model(&tensor_s));
            let aa = env.available_actions_ids();
            let a = aa[argmax(&get_data_from_index_list(&Vec::<f32>::from(&pred), aa.as_slice())).0];
            env.act_with_action_id(a);
        }
        println!("Score: {}", score);
        println!("Mean score: {}", score / 1000 as f64);
    }
}