use tch::nn::{self, Module, ModuleT, OptimizerConfig, SequentialT};
use indicatif::ProgressIterator;
use tch::{Device, Kind, Tensor};

struct DeepQLearning<T> {
    env: T,
    max_iter_count: u32,
    gamma: f32,
    alpha: f64,
    epsilon: f32
}

type Model = Box<dyn Fn(&Tensor) -> Tensor>;


impl<T> DeepQLearning<T> {
    fn new(env: T) -> Self {
        Self {
            env,
            max_iter_count: 10_000,
            gamma: 0.99,
            alpha: 0.1,
            epsilon: 0.2
        }
    }

    fn model(p: &nn::Path, nact: i64) -> Model {
        let seq = nn::seq()
            .add(nn::Linear(model_vs / "l1", nact, 1, Default::default()));

        let device = p.device();
        Box::new(move |xs: &Tensor| {
            xs.to_device(device).apply(&seq)
        })
    }

    fn train(&mut self) -> (SequentialT, Vec<f64>, Vec<f64>) {
        let device = Device::cuda_if_available();
        let mut model_vs = nn::VarStore::new(device);
        let q = model(&model_vs, self.max_action_count());


        let mut ema_score = 0.0;
        let mut ema_nb_steps = 0.0;
        let mut first_episode = true;

        let mut step = 0;
        let mut ema_score_progress = Vec::new();
        let mut ema_nb_steps_progress = Vec::new();

        let mut optimizer = nn::Sgd::default().build(&model_vs, self.alpha).unwrap();

        for _ in (0..max_iter_count).progress() {
            if env.is_game_over() {
                if first_episode {
                    ema_score = env.score();
                    ema_nb_steps = step as f64;
                    first_episode = false;
                } else {
                    ema_score = (1 - 0.9) * env.score() + 0.9 * ema_score;
                    ema_nb_steps = (1 - 0.9) * step + 0.9 * ema_nb_steps;
                    ema_score_progress.push(ema_score);
                    ema_nb_steps_progress.push(ema_nb_steps);
                }

                env.reset();
                step = 0;
            }

            let s = env.state_description();
            let aa = env.available_actions_ids();

            //let test = tch::no_grad(|| q(&s));
            let q_pred = q(&s);
            let mut action_id;
            if rand::thread_rng().gen_range(0..2) < epsilon {
                action_id = aa[rand::thread_rng().gen_range(0..aa.len())];
            } else {
                action_id = aa[q_pred.argmax(0, true).unwrap().item::<usize>()];
            }

            let old_score = env.score();
            env.act_with_action_id(action_id);
            let new_score = env.score();
            let r = new_score - old_score;

            let s_p = env.state_description();
            let aa_p = env.available_actions_ids();

            let mut y;
            if env.is_game_over() {
                y = r;
            } else {
                let q_pred_p = q(&s_p);
                println!("q_pred_p: {:?}", q_pred_p);
                let max_q_pred_p = q_pred_p.max().unwrap().item::<f32>();
                y = r + gamma * max_q_pred_p;
            }


            let q_s_a = q(s).get(0).get(action_id);
            let loss = (f64::powi(y - q_s_a, 2)).mean(Kind::Float);
            optimizer.backward_step(&loss);
            
            step += 1;
        }
        (q, ema_score_progress, ema_nb_steps_progress)
    }
}