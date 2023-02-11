use pbr::ProgressBar;
use tch::nn::{self, Module, OptimizerConfig, VarStore};
use tch::{Device, Tensor};
use environnements::contracts::DeepSingleAgentEnv;
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
                                        gamma: f32, alpha: f64, actors_count: u32,
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

        for t in 0..max_iter_count {



            ema.step += 1.0;
            pb.inc();
        }
        ema
    }

    pub fn get_model(&mut self) -> &VarStore {
        &self.pi_and_v_model
    }
}