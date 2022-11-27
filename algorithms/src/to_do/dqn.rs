use tch::nn;
use indicatif::ProgressIterator;

struct DeepQLearning {
    env: DeepSingleAgentEnv,
    max_iter_count: u32,
    gamma: f32,
    alpha: f32,
    epsilon: f32
}


impl DeepQLearning {
    fn new(env: DeepSingleAgentEnv) -> Self {
        Self {
            env,
            max_iter_count: 10_000,
            gamma: 0.99,
            alpha: 0.1,
            epsilon: 0.2
        }
    }

    fn train(&mut self) {
        let mut q = nn::seq()
            .add(nn::linear(self.env.max_action_count(), 1, Default::default()));

        let mut ema_score = 0.0;
        let mut ema_nb_steps = 0.0;
        first_episode = true;

        let mut step = 0;
        ema_score_progess = Vec::new();
        ema_nb_steps_progress = Vec::new();

        let mut optimizer = nn::Adam::default().build(&q, 1e-3).unwrap();

        for _ in (0..max_iter_count).progress() {
            if env.is_game_over() {
                if first_episode {
                    ema_score = env.score();
                    ema_nb_steps = step as f64;
                    first_episode = false;
                }
                else {
                    ema_score = (1 - 0.9) * env.score() + 0.9 * ema_score;
                    ema_nb_steps = (1 - 0.9) * step + 0.9 * ema_nb_steps;
                    ema_score_progess.push(ema_score);
                    ema_nb_steps_progress.push(ema_nb_steps);
                }

                env.reset();
                step = 0;
            }

            let mut s = env.state_description();
            let mut aa = env.available_actions_ids();

            let mut q_pred = q.forward(&s);
            if rand::thread_rng().gen_range(0..1) < epsilon {
                let mut action_id = aa[rand::thread_rng().gen_range(0..aa.len())];
            }
            else {
                let mut action_id = aa[q_pred.argmax(0, true).unwrap().item::<usize>()];
            }

            let mut old_score = env.score();
            env.act_with_action_id(action_id);
            let mut new_score = env.score();
            let mut r = new_score - old_score;

            let s_p = env.state_description();
            let aa_p = env.available_actions_ids();

            if env.is_game_over() {
                let mut y = r;
            } else {
                let mut q_pred_p = q.forward(&s_p);
                let mut max_q_pred_p = q_pred_p.max(0, true).unwrap().item::<f32>();
                let mut y = r + gamma * max_q_pred_p;
            }


            //TODO: Implement this in Rust
            /*
            with tf.GradientTape() as tape:
                q_s_a = q(np.array([s]))[0][a]
                loss = tf.reduce_mean((y - q_s_a) ** 2)

            grads = tape.gradient(loss, q.trainable_variables)
            opt.apply_gradients(zip(grads, q.trainable_variables))

            step += 1
            return q, ema_score_progress, ema_nb_steps_progress
             */
        }
    }
}