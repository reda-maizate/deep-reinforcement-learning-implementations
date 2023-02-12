use rand::{thread_rng, seq::SliceRandom, Rng, prelude::*};
use tch::{Kind, no_grad, Tensor};
use environnements::contracts::DeepSingleAgentEnv;
use crate::utils::score::EMA;

type Model = Box<dyn Fn(&Tensor) -> Tensor>;

pub fn get_data_from_index_list<T: Copy>(vector: &Vec<T>, index: &[usize]) -> Vec<T> {
    let mut new_vector = vec![];
    for i in index {
        new_vector.push(vector[*i]);
    }
    new_vector
}

pub fn argmax<T: Copy + PartialOrd>(vector: &Vec<T>) -> (usize, T) {
    let mut max = vector[0];
    let mut argmax:usize = 0;
    for (i, &v) in vector.iter().enumerate() {
        if v > max {
            max = v;
            argmax = i;
        }
    }
    (argmax, max)
}

pub fn update_score<T: DeepSingleAgentEnv>(env: &mut T, ema: &mut EMA) {
    if env.is_game_over() {
        if ema.first_episode {
            ema.score = env.score() as f64;
            ema.nb_steps = ema.step;
            ema.first_episode = false;
        } else {
            ema.score = (1.0 - 0.9) * env.score() as f64 + 0.9 * ema.score;
            ema.nb_steps = (1.0 - 0.9) * ema.step + 0.9 * ema.nb_steps;
            ema.score_progress.push(ema.score);
            ema.nb_steps_progress.push(ema.nb_steps);
        }

        env.reset();
        ema.step = 0.0;
    }
}

pub fn step<T: DeepSingleAgentEnv>(env: &mut T, q: &Model, tensor_s: &Tensor, aa: &Vec<usize>, epsilon: f32) -> (usize, f32, Vec<f64>, Vec<usize>) {
    let action_id;
    if (thread_rng().gen_range(0.0..1.0) as f32).partial_cmp(&epsilon).unwrap().is_lt() {
        action_id = aa[thread_rng().gen_range(0..aa.len())];
    } else {
        let q_prep = no_grad(|| q(&tensor_s));
        action_id = aa[argmax(&get_data_from_index_list(&Vec::<f32>::from(&q_prep), aa.as_slice())).0];
    }

    let old_score = env.score();
    env.act_with_action_id(action_id);
    let new_score = env.score();
    let r = new_score - old_score;

    let s_p = env.state_description();
    let aa_p = env.available_actions_ids();

    (action_id, r, s_p, aa_p)
}

pub fn get_random_mini_batch<T: Clone>(vector: &Vec<T>, batch_size: usize) -> Vec<T> {
    let mut rng = thread_rng();
    vector.choose_multiple(&mut rng, batch_size).cloned().collect()
}

pub fn argmin<T: Copy + PartialOrd>(vector: &Vec<T>) -> (usize, T) {
    let mut min = vector[0];
    let mut argmin:usize = 0;
    for (i, &v) in vector.iter().enumerate() {
        if v < min {
            min = v;
            argmin = i;
        }
    }
    (argmin, min)
}

pub fn get_random_prioritized_mini_batch(vector: &Vec<usize>, weights: &Vec<f32>, batch_size: usize) -> Vec<usize> {
    let mut mini_batch = Vec::new();
    for _ in 0..batch_size {
        let rand:f32 = thread_rng().gen_range(0.0..argmax(&weights).1);
        let index = weights.iter().position(|&w| w.partial_cmp(&rand).unwrap().is_ge()).unwrap();
        mini_batch.push(vector[index]);
    }
    mini_batch
}

pub fn calculate_priority_weights(q: &Model, q_target: &Model, s: &Vec<f64>, a: usize, r: f32, s_p: &Vec<f64>, aa_p: &Vec<usize>, done: bool, gamma: f32) -> f32{
    let mut y = r;
    if !done {
        let tensor_s_p = Tensor::of_slice(&s_p).to_kind(Kind::Float);
        let q_pred_p = no_grad(|| q_target(&tensor_s_p));
        let max_q_pred_p = argmax(
            &get_data_from_index_list(
                &Vec::<f32>::from(&q_pred_p),
                aa_p.as_slice()
            )
        ).1;
        y = r + gamma * max_q_pred_p;
    }

    let tensor_s = Tensor::of_slice(&s).to_kind(Kind::Float);
    let q_s_a = no_grad(|| q(&tensor_s).unsqueeze(0).get(0).get(a as i64));

    (y - &Vec::<f32>::from(&q_s_a)[0]).abs()
}

pub fn vec_zeros<T: Clone>(zero_type: T, rows: usize, cols: usize) -> Vec<Vec<T>> {
    vec![vec![zero_type; cols]; rows]
}
