extern crate core;

use std::borrow::BorrowMut;
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{LineStyle};
use plotlib::view::ContinuousView;
use crate::to_do::{dqn::DeepQLearning, ddqn::DDQN};
use environnements::to_do::deep_single_agent::{GridWorld, LineWorld};
use environnements;
use crate::to_do::ppo_a2c::PPO_A2C;

pub mod to_do;
pub mod utils;

fn main() {
    let mut line_world_env = LineWorld::new(Option::Some(10));
    let mut grid_world_env = GridWorld::new(Some(5), Some(5));

    // DQN Basic
    // let mut dqn = DeepQLearning::new();
    // let mut ema = dqn.train(line_world_env.borrow_mut(), 10_000, 0.99, 0.1, 0.1);
    // let model_dqn = dqn.get_model();
    // DDQN Basic
    // let mut ddqn = DDQN::new();
    // let mut ema = ddqn.train(line_world_env.borrow_mut(), 10_000, 0.99, 0.1, 0.1, 10);
    // let model_ddqn = ddqn.get_model();
    // DDQN with ER
    // let mut ddqn_er = DDQN::new();
    // let mut ema = ddqn_er.train_with_er(line_world_env.borrow_mut(), 5_000, 0.99, 0.1, 0.1, 10, 100, 1_000);
    // let model_ddqn_er = ddqn_er.get_model();
    // DDQN with PER
    // let mut ddqn_per = DDQN::new();
    // let mut ema = ddqn_per.train_with_per(line_world_env.borrow_mut(), 10_000, 0.99, 0.1, 0.1, 10, 100, 1_000);
    // let model_ddqn_per = ddqn_per.get_model();
    // PPO A2C
    let mut ppo_a2c = PPO_A2C::new();
    let mut ema = ppo_a2c.train(line_world_env.borrow_mut(), 100_000, 0.99, 1e-5, 10, 5, 1.0, 0.01);
    let model_ppo_a2c = ppo_a2c.get_model();

    // println!("\nGradients: {:?}", model_ppo_a2c.trainable_variables());
    // model_ppo_a2c.variables().get("weight").unwrap().print();

    ema.display_results("ppo_a2c");
}
