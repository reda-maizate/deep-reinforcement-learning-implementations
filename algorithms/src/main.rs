use std::borrow::BorrowMut;
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{LineStyle};
use plotlib::view::ContinuousView;
use crate::to_do::{dqn::DeepQLearning, ddqn::DDQN};
use environnements::to_do::deep_single_agent::{GridWorld, LineWorld};
use environnements;

pub mod to_do;
pub mod utils;

fn main() {
    let mut line_world_env = LineWorld::new(Option::Some(10));
    let mut grid_world_env = GridWorld::new(Some(5), Some(5));
    // let mut grid_world_env_er = GridWorld::new(Some(5), Some(5));

    // DQN Basic
    let mut dqn = DeepQLearning::new();
    let mut ema = dqn.train(line_world_env.borrow_mut(), 10_000, 0.99, 0.1, 0.1);
    let model_dqn = dqn.get_model();
    // DDQN Basic
    // let mut ddqn = DDQN::new();
    // let mut ema = ddqn.train(line_world_env.borrow_mut(), 10_000, 0.99, 0.1, 0.1, 10);
    // let model_ddqn = ddqn.get_model();
    // DDQN with ER
    // let mut ddqn_er = DDQN::new();
    // let mut ema = ddqn_er.train_with_er(line_world_env.borrow_mut(), 5_000, 0.99, 0.1, 0.1, 10, 100);
    // let model_ddqn_er = ddqn_er.get_model();

    println!("\nGradients: {:?}", model_dqn.trainable_variables());
    model_dqn.variables().get("weight").unwrap().print();

    ema.display_results("dqn");
}
