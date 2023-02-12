use std::borrow::BorrowMut;
use pbr::ProgressBar;
use tch::nn::{self, Module, OptimizerConfig, VarStore};
use tch::{Device, Kind, no_grad, Tensor};
use environnements::contracts::DeepSingleAgentEnv;
use crate::to_do::functions::{argmin, update_score, vec_zeros};
use crate::utils::score::EMA;

#[derive(Debug)]
pub struct PPO_A2C {
    pi_and_v_model: VarStore,
}

type Model = Box<dyn Fn(&Tensor) -> (Tensor, Tensor)>;

impl PPO_A2C {
    pub fn new() -> Self {
        let device = Device::cuda_if_available();
        let model_vs = VarStore::new(device);
        Self {
            pi_and_v_model: model_vs,
        }
    }

    fn create_model(vs: &nn::Path, state_dim: i64, max_action_count: i64) -> Model {
        let conf_linear = nn::LinearConfig{
            bias: true,
            ..Default::default()
        };
        let seq = nn::seq()
            .add(nn::linear(vs, state_dim, 128, conf_linear))
            .add_fn(|xs| xs.tanh())
            .add(nn::linear(vs, 128, 128, conf_linear))
            .add_fn(|xs| xs.tanh())
            .add(nn::linear(vs, 128, 128, conf_linear))
            .add_fn(|xs| xs.tanh())
            .add(nn::linear(vs, 128, 128, conf_linear))
            .add_fn(|xs| xs.tanh())
            .add(nn::linear(vs, 128, 128, conf_linear))
            .add_fn(|xs| xs.tanh())
            .add(nn::linear(vs, 128, 128, conf_linear))
            .add_fn(|xs| xs.tanh());
        let actor = nn::linear(vs, 128, max_action_count, conf_linear);
        let critic = nn::linear(vs, 128, 1, conf_linear);

        let device = vs.device();
        Box::new(move |xs: &Tensor| {
            let xs = xs.to_device(device).apply(&seq);
            (xs.apply(&actor), xs.apply(&critic))
        })
    }

    pub fn train<T: DeepSingleAgentEnv>(&mut self, env: &mut T, max_iter_count: u32,
                                        gamma: f32, alpha: f64, actors_count: usize,
                                        epochs: usize, c1: f32, c2: f32) -> EMA {
        let pi_and_v = Self::create_model(
            &self.pi_and_v_model.root(),
            env.state_dim() as i64,
            env.max_action_count() as i64
        );
        let mut optimizer = nn::Adam::default().build(
            &self.pi_and_v_model,
            alpha
        ).unwrap();

        let mut ema = EMA::new();

        let mut pb = ProgressBar::new(max_iter_count as u64);
        pb.format("╢▌▌░╟");


        let mut envs:Vec<T> = Vec::new();
        let mut env_ref = env.clone_env();

        for _ in 0..max_iter_count {
            for i in 0..actors_count {
                if envs.len() < actors_count {
                    envs.push(env_ref.clone_env());
                } else {
                    envs[i] = env_ref.clone_env();
                }
            }

            let mut states = vec_zeros(0.0 as f64, actors_count, env.state_dim());
            let mut masks = vec_zeros(0.0 as f64, actors_count, env.max_action_count());

            for i in 0..envs.len() {
                update_score(&mut envs[i], ema.borrow_mut());

                let s = envs[i].state_description();
                let aa = envs[i].available_actions_ids();

                let mut mask = vec![0.0 as f64; envs[i].max_action_count()];
                for j in aa {
                    mask[j] = 1.0;
                }

                states[i] = s.clone();
                masks[i] = mask.clone();
            }

            let states_flatten:Vec<f64> = states.clone().into_iter().flatten().collect();
            let tensor_states = Tensor::of_slice(&states_flatten)
                .reshape(&[states.len() as i64, states[0].len() as i64])
                .to_kind(Kind::Float);
            let masks_flatten:Vec<f64> = masks.clone().into_iter().flatten().collect();
            let tensor_masks = Tensor::of_slice(&masks_flatten)
                .reshape(&[masks.len() as i64, masks[0].len() as i64])
                .to_kind(Kind::Float)*(-1e9);
            let (actor, critic) = no_grad(|| pi_and_v(&tensor_states));
            let critic = Vec::<f32>::from(critic.squeeze_dim(-1));
            let probs = no_grad(|| (actor + &tensor_masks).softmax(-1, Kind::Float));
            let chosen_actions_tensor = probs.multinomial(1, true).squeeze_dim(-1);
            let chosen_actions = Vec::<i64>::from(&chosen_actions_tensor);

            let mut rewards = vec![0.0 as f32; actors_count];
            let mut states_p = vec_zeros(0.0 as f64, actors_count, env.state_dim());
            let mut masks_p = vec_zeros(0.0 as f64, actors_count, env.max_action_count());
            let mut game_overs = vec![1.0; actors_count];

            for i in 0..envs.len() {
                let old_score = envs[i].score();
                envs[i].act_with_action_id(chosen_actions[i] as usize);
                let new_score = envs[i].score();
                let r = new_score - old_score;

                rewards[i] = r;

                let s_p = envs[i].state_description();
                let aa_p = envs[i].available_actions_ids();

                let mut mask_p = vec![0.0 as f64; envs[i].max_action_count()];

                for j in aa_p {
                    mask_p[j] = 1.0;
                }

                states_p[i] = s_p.clone();
                masks_p[i] = mask_p.clone();

                if envs[i].is_game_over() {
                    game_overs[i] = 0.0;
                }
            }
            env_ref = envs[actors_count - 1].clone_env();

            let states_p_flatten:Vec<f64> = states_p.clone().into_iter().flatten().collect();
            let tensor_states_p = Tensor::of_slice(&states_p_flatten)
                .reshape(&[states_p.len() as i64, states_p[0].len() as i64])
                .to_kind(Kind::Float);
            let (actor_p, critic_p) = no_grad(|| pi_and_v(&tensor_states_p));
            let critic_p = Vec::<f32>::from(critic_p.squeeze_dim(-1));

            let mut targets = vec![0.0; actors_count];
            let mut deltas = vec![0.0; actors_count];
            for i in 0..envs.len() {
                targets[i] = rewards[i] + gamma * critic_p[i] + game_overs[i];
                deltas[i] = targets[i] - critic[i];
            }
            let actor_old = probs;

            // Training Step Start
            for _ in 0..epochs {
                optimizer.zero_grad();

                let (actor, critic) = pi_and_v(&tensor_states);
                let probs = (actor + &tensor_masks).softmax(-1, Kind::Float);

                let loss_vf = ((Tensor::of_slice(&targets) - critic.squeeze_dim(-1))
                    .pow(&Tensor::of_slice(&[2])))
                    .mean(Kind::Float);

                let loss_entropy = (&probs * (&probs + 0.000000001).log())
                    .sum(Kind::Float);

                let actor_s_a_pred = probs
                    .gather(-1, &chosen_actions_tensor.unsqueeze(-1), false)
                    .squeeze_dim(-1);

                let actor_old_s_a_pred = actor_old
                    .gather(-1, &chosen_actions_tensor.unsqueeze(-1), false)
                    .squeeze_dim(-1);

                let r = actor_s_a_pred / (actor_old_s_a_pred + 0.0000000001);

                let loss_policy_clipped = ((&r * Tensor::of_slice(&deltas))
                    .minimum(&(&r.clamp(1.0 - 0.2, 1.0 + 0.2) * Tensor::of_slice(&deltas))))
                    .sum(Kind::Float);

                let total_loss = c1 * loss_vf - c2 * loss_entropy - loss_policy_clipped;

                optimizer.backward_step(&total_loss);
            }
            // Training Step End

            ema.step += 1.0;
            pb.inc();
        }
        ema
    }

    pub fn get_model(&mut self) -> &VarStore {
        &self.pi_and_v_model
    }
}