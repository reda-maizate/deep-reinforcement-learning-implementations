use std::f32;
use tch::nn::{self, OptimizerConfig, VarStore};
use rand::Rng;
use tch::{Device, Kind, no_grad, Tensor};

#[derive(Debug)]
struct DDQN {
    max_iter_count: u32,
    gamma: f32,
    alpha: f64,
    epsilon: f32,
    q_model: Box<dyn Fn(&Tensor) -> Tensor>,
    q_model_target: Box<dyn Fn(&Tensor) -> Tensor>,
    optimizer: nn::Optimizer,
}

impl DDQN {
    fn new(nact: i64, device: Device) -> Self {
        let q_model = DDQN::q_model(nact, device);
        let q_model_target = DDQN::q_model(nact, device);
        let optimizer = nn::Adam::default().build(&q_model.vs, 1e-3).unwrap();

        Self {
            max_iter_count: 10_000,
            gamma: 0.99,
            alpha: 0.1,
            epsilon: 0.1,
            q_model,
            q_model_target,
            optimizer,
        }
    }

    fn q_model(nact: i64, device: Device) -> nn::Path {
        let vs = nn::VarStore::new(device);
        let conf = nn::LinearConfig{
            bias: true,
            ..Default::default()
        };
        let linear = nn::linear(&vs.root(), 1, nact, conf);
        let seq = nn::seq()
            .add(linear);
        let device = vs.device();
        Box::new(move |xs: &Tensor| {
            xs.to_device(device).apply(&seq)
        })
    }

    fn train(&mut self, state: &Tensor, next_state: &Tensor, action: i64, reward: f32) {
        let q = no_grad(|| self.q_model.forward(state));
        let q_next = no_grad(|| self.q_model_target.forward(next_state));
        let a = argmax(q_next);
        let q_next_best = q_next.get(0, a).item().to_owned();

        let target = reward + self.gamma * q_next_best;
        let q_target = q.get(0, action).item().to_owned();
        let loss = (target - q_target).powi(2);

        self.optimizer.backward_step(&loss);
    }

    fn act(&mut self, state: &Tensor) -> i64 {
        let q = no_grad(|| self.q_model.forward(state));
        let a = if rand::thread_rng().gen_range(0, 2) as f32 < self.epsilon {
            rand::thread_rng().gen_range(0, q.size()[1])
        } else {
            argmax(q)
        };
        a
    }

    fn update_target(&mut self) {
        self.q_model_target.vs.copy(self.q_model.vs);
    }
}