use rand::{thread_rng, seq::SliceRandom, Rng};
use tch::{no_grad, Tensor};
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

pub fn argmax<T: Copy + std::cmp::PartialOrd>(vector: &Vec<T>) -> (usize, T) {
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
    if (rand::thread_rng().gen_range(0..2) as f32).partial_cmp(&epsilon).unwrap().is_lt() {
        action_id = aa[rand::thread_rng().gen_range(0..aa.len())];
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