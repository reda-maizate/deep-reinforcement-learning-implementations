use environnements::contracts::MCRRSingleAgentEnv;
use environnements::to_do::mcrr_single_agent::LineWorld;
use pbr::ProgressBar;
use rand::seq::SliceRandom;

#[derive(Debug)]
pub struct MonteCarloRandomRollout<T> {
    env: T,
    simulation_count_per_action: usize,
}

impl<T: MCRRSingleAgentEnv> MonteCarloRandomRollout<T> {
    pub fn new(env: T, simulation_count_per_action: Option<usize>) -> Self {
        let sim_count;
        if let Some(x) = simulation_count_per_action {
            sim_count = x;
        } else {
            sim_count = 50;
        }
        Self {
            env,
            simulation_count_per_action: sim_count,
        }
    }

    pub fn monte_carlo_random_rollout_and_choose_action(&mut self) -> Option<usize> {
        let mut best_action = None;
        let mut best_action_average_score = None;

        for a in self.env.available_actions_ids() {
            let mut action_score = 0.0;
            for _ in 0..self.simulation_count_per_action {
                let mut cloned_env = self.env.clone();
                cloned_env.act_with_action_id(a);

                while !cloned_env.is_game_over() {
                    let mut actions = cloned_env.available_actions_ids();
                    let action_id = actions.choose(&mut rand::thread_rng()).unwrap();
                    //println!("action_id: {}", action_id);
                    cloned_env.act_with_action_id(*action_id);
                }

                //println!("score: {}", cloned_env.score() as f64);
                action_score += cloned_env.score() as f64;
            }
            let mut average_action_score = action_score / self.simulation_count_per_action as f64;

            if best_action_average_score == None || best_action_average_score < Some(average_action_score) {
                best_action = Option::from(a);
                best_action_average_score = Some(average_action_score);
                //println!("best_action: {}", best_action.unwrap());
                //println!("best_action_average_score: {}", best_action_average_score.unwrap());
            }
        }
        best_action
    }

    pub fn run_line_world_n_games_and_return_mean_score(&mut self, games_count: u32) -> f64 {
        let mut total: f64 = 0.0;

        // Progress bar
        let mut pb = ProgressBar::new(games_count as u64);
        pb.format("╢▌▌░╟");

        for _ in 0..games_count {
            self.env.reset();

            while !self.env.is_game_over() {
                let chosen_a = self.monte_carlo_random_rollout_and_choose_action();
                self.env.act_with_action_id(chosen_a.unwrap());
            }
            total += self.env.score() as f64;
            pb.inc();
        }
        println!("total: {} / games_count: {}", total, games_count);
        total / games_count as f64
    }
}