use std::collections::HashMap;
use environnements::contracts::MCTSSingleAgentEnv;
use pbr::ProgressBar;
use rand::seq::IteratorRandom;
use rand::prelude::SliceRandom;

pub struct Node {
    pub consideration_count: usize,
    pub selection_count: u32,
    pub mean_score: f32,
}

impl Node {
    pub fn new() -> Self {
        Self {
            consideration_count: 0,
            selection_count: 0,
            mean_score: 0.0,
        }
    }

    pub fn from(consideration_count: usize, selection_count: u32, mean_score: f32) -> Self {
        Self {
            consideration_count,
            selection_count,
            mean_score,
        }
    }
}

pub struct MonteCarloTreeSearch<T> {
    env: T,
    iteration_count: usize,
}

impl<T: MCTSSingleAgentEnv> MonteCarloTreeSearch<T> {
    pub fn new(env: T, iteration_count: Option<usize>) -> Self {
        let iter_count;
        if let Some(x) = iteration_count {
            iter_count = x;
        } else {
            iter_count = 200;
        }
        Self {
            env,
            iteration_count: iter_count,
        }
    }

    pub fn monte_carlo_tree_search_and_choose_action(&mut self) -> Option<usize> {
        let mut tree = HashMap::new();

        let root = self.env.state_id();
        tree.insert(root, HashMap::new());

        for a in self.env.available_actions_ids() {
            tree.get_mut(&root).unwrap().insert(a, Node::new());
        }

        for _ in 0..self.iteration_count {
            let mut cloned_env = self.env.clone();
            let mut current_node = cloned_env.state_id();

            let mut nodes_and_chosen_actions = Vec::new();

            // SELECTION
            // let tree_current_node = tree[&current_node].values().any(|stats| stats.selection_count == 0);
            // println!("tree_current_node: {:?}", tree_current_node);
            while !cloned_env.is_game_over() &&
                tree[&current_node].values().any(|stats| stats.selection_count == 0) {

                let mut best_action = None;
                let mut best_action_score = None;

                for (a, a_stats) in &tree[&current_node] {
                    // Convert this python code to rust:
                    // ucb1_score = a_stats['mean_score'] + math.sqrt(2) * math.sqrt(
                    // math.log(a_stats['consideration_count']) / a_stats['selection_count'])
                    let ucb1_score = a_stats.mean_score + (2.0 * (a_stats.consideration_count as f32).ln() / a_stats.selection_count as f32).sqrt();

                    if best_action_score == None || best_action_score < Some(ucb1_score) {
                        best_action = Option::from(*a);
                        best_action_score = Some(ucb1_score);
                    }
                }

                nodes_and_chosen_actions.push((current_node, best_action.unwrap()));
                cloned_env.act_with_action_id(best_action.unwrap());
                current_node = cloned_env.state_id();

                if !tree.contains_key(&current_node) {
                    let mut new_node = HashMap::new();
                    for a in cloned_env.available_actions_ids() {
                        new_node.insert(a, Node::new());
                    }
                    tree.insert(current_node, new_node);
                }
            }

            // EXPAND
            if !cloned_env.is_game_over() {
                let random_action = tree[&current_node]
                    .iter()
                    .filter(|(_, stats)| stats.selection_count == 0)
                    .map(|(a, _)| a)
                    .choose(&mut rand::thread_rng());

                nodes_and_chosen_actions.push((current_node, *random_action.unwrap()));
                cloned_env.act_with_action_id(*random_action.unwrap());
                current_node = cloned_env.state_id();

                if !tree.contains_key(&current_node) {
                    let mut new_node = HashMap::new();
                    for a in cloned_env.available_actions_ids() {
                        new_node.insert(a, Node::new());
                    }
                    tree.insert(current_node, new_node);
                }
            }

            // EVALUATE / ROLLOUT
            while !cloned_env.is_game_over() {
                let c_env_available_actions_ids = cloned_env.available_actions_ids();
                let random_action = c_env_available_actions_ids.choose(&mut rand::thread_rng());
                cloned_env.act_with_action_id(*random_action.unwrap());
            }

            let score = cloned_env.score();

            // BACKUP / BACKPROPAGATE / UPDATE STATS
            for (node, chose_action) in nodes_and_chosen_actions {
                let mut tree_node = tree.get_mut(&node).unwrap();

                for a in tree_node.keys() {
                    tree_node.get_mut(a).unwrap().consideration_count += 1;
                }

                tree_node.get_mut(&chose_action).unwrap().mean_score =
                    (
                    tree_node.get_mut(&chose_action).unwrap().mean_score *
                        tree_node.get_mut(&chose_action).unwrap().selection_count as f32 + score
                )
                        / (tree_node.get_mut(&chose_action).unwrap().selection_count + 1) as f32;
                tree_node.get_mut(&chose_action).unwrap().selection_count += 1;
            }
        }

        let mut most_selected_action = None;
        let mut most_selected_action_count = None;

        for (a, a_stats) in &tree[&root] {
            if most_selected_action_count == None || most_selected_action_count < Some(a_stats.selection_count) {
                most_selected_action = Option::from(*a);
                most_selected_action_count = Some(a_stats.selection_count);
            }
        }
        most_selected_action
    }

    pub fn run_line_world_n_games_and_return_mean_score(&mut self, games_count: u32) -> f64 {
        let mut total: f64 = 0.0;
        let mut wins: u32 = 0;
        let mut losses: u32 = 0;
        let mut draws: u32 = 0;

        // Progress bar
        let mut pb = ProgressBar::new(games_count as u64);
        pb.format("╢▌▌░╟");

        for _ in 0..games_count {
            self.env.reset();
            while !self.env.is_game_over() {
                let action = self.monte_carlo_tree_search_and_choose_action();
                self.env.act_with_action_id(action.unwrap());
            }

            if self.env.score() > 0.0 {
                wins += 1;
            } else if self.env.score() < 0.0 {
                losses += 1;
            } else {
                draws += 1;
            }
            total += self.env.score() as f64;
            pb.inc();
        }
        println!("MCTS - wins: {}, losses: {}, draws: {}", wins, losses, draws);
        println!("MCTS - mean score: {}", total / games_count as f64);
        total / games_count as f64
    }
}