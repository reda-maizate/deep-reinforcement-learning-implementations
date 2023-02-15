extern crate core;

use environnements::contracts::DeepSingleAgentEnv;
use crate::to_do::{dqn::DeepQLearning, ddqn::DDQN};
use environnements::to_do::deep_single_agent::{GridWorld, LineWorld};
use crate::to_do::functions::save_model;

pub mod to_do;
pub mod utils;

fn main() {
    // let mut line_world_env = LineWorld::new(Option::Some(10));
    // line_world_env.play(true);
    // let mut grid_world_env = GridWorld::new(Some(5), Some(5));
    // grid_world_env.play(true);

    // DQN Basic
    // let mut dqn = DeepQLearning::new();
    // let mut ema = dqn.train(&mut line_world_env, 10_000, 0.99, 0.1, 0.1);
    // let model_dqn = dqn.get_model();
    // save_model(model_dqn, "src/models/LineWorld/dqn_max_iter_10000_g_0.99_alpha_0.1.ot");

    // let model = DeepQLearning::load_model(&mut line_world_env, "src/models/LineWorld/dqn_max_iter_10000_g_0.99_alpha_0.1.ot");
    // DeepQLearning::evaluate_model(&mut line_world_env, &model);

    // DDQN Basic
    // let mut ddqn = DDQN::new();
    // let mut ema = ddqn.train(&mut line_world_env, 10_000, 0.99, 0.1, 0.1, 10);
    // let model_ddqn = ddqn.get_model();
    // save_model(model_ddqn, "src/models/LineWorld/ddqn_max_iter_10000_g_0.99_alpha_0.1.ot");

    // let model = DDQN::load_model(&mut line_world_env, "src/models/LineWorld/ddqn_max_iter_10000_g_0.99_alpha_0.1.ot");
    // DDQN::evaluate_model(&mut line_world_env, &model);

    // DDQN with ER
    // let mut ddqn_er = DDQN::new();
    // let mut ema = ddqn_er.train_with_er(&mut line_world_env, 5_000, 0.99, 0.1, 0.1, 10, 100, 1_000);
    // let model_ddqn_er = ddqn_er.get_model();
    // save_model(model_ddqn_er, "src/models/LineWorld/ddqn_er_max_iter_5000_g_0.99_alpha_0.1.ot");

    // let model = DDQN::load_model(&mut line_world_env, "src/models/LineWorld/ddqn_er_max_iter_5000_g_0.99_alpha_0.1.ot");
    // DDQN::evaluate_model(&mut line_world_env, &model);

    // DDQN with PER
    // let mut ddqn_per = DDQN::new();
    // let mut ema = ddqn_per.train_with_per(&mut line_world_env, 10_000, 0.99, 0.1, 0.1, 10, 100, 1_000);
    // let model_ddqn_per = ddqn_per.get_model();
    // save_model(model_ddqn_per, "src/models/LineWorld/ddqn_per_max_iter_10000_g_0.99_alpha_0.1.ot");

    // let model = DDQN::load_model(&mut line_world_env, "src/models/LineWorld/ddqn_per_max_iter_10000_g_0.99_alpha_0.1.ot");
    // DDQN::evaluate_model(&mut line_world_env, &model);

    // println!("\nGradients: {:?}", model_dqn.trainable_variables());
    // model_dqn.variables().get("weight").unwrap().print();

    // ema.display_results("dqn");
    /* The Pacman game is available to play in the file drl-project/src/main.rs */
}
