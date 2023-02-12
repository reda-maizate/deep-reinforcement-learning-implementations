use tch::Tensor;
use crate::to_do::functions::{argmin, calculate_priority_weights};

type Model = Box<dyn Fn(&Tensor) -> Tensor>;

pub struct PrioritizedExperienceReplay {
    pub replay_buffer: Vec<(Vec<f64>, usize, f32, Vec<f64>, Vec<usize>, bool)>,
    pub weights_buffer: Vec<f32>,
    capacity_buffer: usize,
}

impl PrioritizedExperienceReplay {
    pub fn new(capacity_buffer: usize) -> Self {
        let buffer = Vec::new();
        let weights = Vec::new();
        Self {
            replay_buffer: buffer,
            weights_buffer: weights,
            capacity_buffer,
        }
    }

    pub fn add(&mut self, s: Vec<f64>, a: usize, r: f32, s_p: Vec<f64>, aa_p: Vec<usize>, done: bool, priority: f32) {
        self.replay_buffer.push((s, a, r, s_p, aa_p, done));
        self.weights_buffer.push(priority);
    }

    pub fn update_priority(&mut self, q: &Model, q_target: &Model, gamma: f32) {
        for i in 0..self.replay_buffer.len() {
            let (s, a, r, s_p, aa_p, done) = self.replay_buffer[i].clone();
            self.weights_buffer[i] = calculate_priority_weights(q, q_target, &s, a, r, &s_p, &aa_p, done, gamma)
        }
        if self.replay_buffer.len() >= self.capacity_buffer {
            let index_min_priority = argmin(&self.weights_buffer).0;
            self.replay_buffer.remove(index_min_priority);
            self.weights_buffer.remove(index_min_priority);
        }
    }
}