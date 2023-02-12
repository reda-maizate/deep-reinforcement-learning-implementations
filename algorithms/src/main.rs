use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{LineStyle};
use plotlib::view::ContinuousView;
use crate::to_do::{dqn, mcrr, reinforce_learned_baseline};
use crate::to_do::reinforce;
use environnements::to_do::deep_single_agent::{GridWorld, LineWorld};
use environnements::to_do::mcrr_single_agent::{GridWorld as GridWorldMCRR, LineWorld as LineWorldMCRR};
use environnements::to_do::mcts_single_agent::{GridWorld as GridWorldMCTS, LineWorld as LineWorldMCTS};
use environnements;
use crate::to_do::functions::load_model;
use crate::to_do::reinforce_learned_baseline::evaluate;

pub mod to_do;

fn main() {
    // Environnements classic
    let line_world_env = LineWorld::new(Option::Some(10));
    // let grid_world_env = GridWorld::new(Some(5), Some(5));

    // Environnements MCRR
    // let line_world_env_mcrr = LineWorldMCRR::new(Option::Some(10));
    // let grid_world_env_mcrr = GridWorldMCRR::new(Some(5), Some(5));

    // Environnements MCTS
    // let line_world_env_mcts = LineWorldMCTS::new(Option::Some(10));
    // let grid_world_env_mcts = GridWorldMCTS::new(Some(5), Some(5));

    // Algorithms
    // DQN
    // let (dqn, ema_scores, ema_nb_step) = dqn::DeepQLearning::new(line_world_env).train();
    // let (dqn, ema_scores, ema_nb_step) = dqn::DeepQLearning::new(grid_world_env).train();

    // REINFORCE
    // let (pi, ema_scores, ema_nb_step) = reinforce::REinforce::new(line_world_env).train();
    // let (pi, ema_scores, ema_nb_step) = reinforce_learned_baseline::ReinforceWithLearnedBaseline::new(line_world_env).train(true);
    evaluate(LineWorld::new(Option::Some(10)), "src/models/LineWorld/reinforce_lb_max_iter_10000_g_0.99_alpha_pi_0.01_alpha_v_0.01.pt", 1000);
    // MCRR
    // let mean_score = mcrr::MonteCarloRandomRollout::new(grid_world_env_mcrr, Some(20)).run_line_world_n_games_and_return_mean_score(1000);
    // println!("Mean score: {:.4}", mean_score);

    // MCTS
    // let mean_score = mcts::MonteCarloTreeSearch::new(line_world_env_mcts, Some(20)).run_line_world_n_games_and_return_mean_score(1000);
    // println!("Mean score: {:.4}", mean_score);
    /*
    let mut scores = vec![];
    let mut nb_steps = vec![];
    for i in 0..ema_scores.len() {
        scores.push((i as f64, ema_scores[i]));
    }

    for i in 0..ema_nb_step.len() {
        nb_steps.push((i as f64, ema_nb_step[i]));
    }

    let s1: Plot = Plot::new(scores).line_style(
        LineStyle::new()
    ); // and a custom colour

    let v = ContinuousView::new()
        .add(s1);

    Page::single(&v).save("src/results/scores-reinforce-with-learned-baseline.svg").unwrap();

    let s2: Plot = Plot::new(nb_steps).line_style(
        LineStyle::new()
    ); // and a custom colour

    let v = ContinuousView::new()
        .add(s2);

    Page::single(&v).save("src/results/nb-steps-reinforce-with-learned-baseline.svg").unwrap();
    */
}
