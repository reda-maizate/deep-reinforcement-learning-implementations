use std::borrow::Borrow;
use std::fmt::Debug;
use tch::nn::{self, Module, ModuleT, OptimizerConfig, SequentialT};
use indicatif::ProgressIterator;
use rand::Rng;
use tch::{Device, Kind, kind};
use tch::IValue::Tensor;
use environnements::contracts::DeepSingleAgentEnv;


#[derive(Debug)]
pub struct DeepQLearning<T> {
    env: T,
    max_iter_count: u32,
    gamma: f32,
    alpha: f64,
    epsilon: f32
}

type Model = Box<dyn Fn(&tch::Tensor) -> tch::Tensor>;


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
            .add(nn::linear(vs / "l1", nact, 1, Default::default()));

        let device = vs.device();
        Box::new(move |xs: &tch::Tensor| {
            xs.to_device(device).apply(&seq)
        })
    }

    pub fn train(&mut self) -> (Model, Vec<f64>, Vec<f64>) {
        let device = Device::cuda_if_available();
        let mut model_vs = nn::VarStore::new(device);
        let seq = nn::seq()
            .add(nn::linear(&model_vs.root() / "linear1", 1, self.env.max_action_count() as i64, Default::default()));
        let q = DeepQLearning::<T>::model(&model_vs.root(), self.env.max_action_count() as i64);

        let mut ema_score = 0.0;
        let mut ema_nb_steps = 0.0;
        let mut first_episode = true;

        let mut step = 0.0;
        let mut ema_score_progress = Vec::new();
        let mut ema_nb_steps_progress = Vec::new();

        let mut optimizer = nn::Sgd::default().build(&model_vs, self.alpha).unwrap();

        for _ in (0..1).progress() { //self.max_iter_count).progress() {
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

            //let s = self.env.state_description();
            let s: Vec<f32> = vec![2.34];
            let aa = self.env.available_actions_ids();

            //let test = tch::no_grad(|| q(&s));
            let tensor_s = tch::Tensor::of_slice(&s);
            tensor_s.print();
            let test = seq.forward(&tensor_s);//.to_kind(kind::Kind::Float).unsqueeze(0));
            // let test = &tensor_s.to_device(device).apply(&seq);
            test.print();

            //<dyn Fn<(&Tensor,), Output=Tensor> as FnOnce<(&Tensor,)>>::Output
            // let q_pred = q(&tensor_s);
            // let mut action_id;
            // if (rand::thread_rng().gen_range(0..2) as f32).partial_cmp(&self.epsilon).unwrap().is_lt() {
            //     action_id = aa[rand::thread_rng().gen_range(0..aa.len())];
            // } else {
            //     //TODO
            //     // action_id = aa[q_pred.argmax(0, true).unwrap().item::<usize>()];
            //     action_id = aa[test.gather()];
            // }
            //
            // let old_score = self.env.score();
            // self.env.act_with_action_id(action_id);
            // let new_score = self.env.score();
            // let r = new_score - old_score;
            //
            // let s_p = self.env.state_description();
            // let aa_p = self.env.available_actions_ids();
            //
            // let mut y;
            // if self.env.is_game_over() {
            //     y = r;
            // } else {
            //     let tensor_s_p = tch::Tensor::of_slice(&s_p);
            //     let q_pred_p = q(&tensor_s_p);
            //     println!("q_pred_p: {:?}", q_pred_p);
            //     //TODO
            //     let max_q_pred_p = q_pred_p.max().unwrap().item::<f32>();
            //     y = r + self.gamma * max_q_pred_p;
            // }
            //
            //
            // let q_s_a = q(&tensor_s).get(0).get(action_id);
            // let loss = (f64::powi(y - q_s_a, 2)).mean(Kind::Float);
            // optimizer.backward_step(&loss);
            //
            // step += 1.0;
        }
        (q, ema_score_progress, ema_nb_steps_progress)
    }
}