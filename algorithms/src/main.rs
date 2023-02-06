use std::borrow::BorrowMut;
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{LineStyle};
use plotlib::view::ContinuousView;
use crate::to_do::{dqn, ddqn::DDQN};
use environnements::to_do::deep_single_agent::{GridWorld, LineWorld};
use environnements;

pub mod to_do;
pub mod utils;

fn main() {
    let line_world_env = LineWorld::new(Option::Some(10));
    let mut grid_world_env = GridWorld::new(Some(5), Some(5));
    let mut grid_world_env_er = GridWorld::new(Some(5), Some(5));

    let mut ddqn = DDQN::new();
    let (ema_scores, ema_nb_step) = ddqn.train(grid_world_env.borrow_mut(), 10_000, 0.99, 0.1, 0.1, 10);
    let model_ddqn = ddqn.get_model();
    let mut ddqn_er = DDQN::new();
    let (ema_scores_er, ema_nb_step_er) = ddqn_er.train_with_er(grid_world_env_er.borrow_mut(), 5_000, 0.99, 0.1, 0.1, 10, 100);
    let model_ddqn_er = ddqn_er.get_model();

    println!("\nGradients: {:?}", model_ddqn_er.trainable_variables());
    model_ddqn_er.variables().get("weight").unwrap().print();

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

    Page::single(&v).save("src/results/scores-ddqn.svg").unwrap();

    let s2: Plot = Plot::new(nb_steps).line_style(
        LineStyle::new()
    ); // and a custom colour

    let v = ContinuousView::new()
        .add(s2);

    Page::single(&v).save("src/results/nb-steps-ddqn.svg").unwrap();

    let mut scores_er = vec![];
    let mut nb_steps_er = vec![];
    for i in 0..ema_scores_er.len() {
        scores_er.push((i as f64, ema_scores_er[i]));
    }

    for i in 0..ema_nb_step_er.len() {
        nb_steps_er.push((i as f64, ema_nb_step_er[i]));
    }

    let s1_er: Plot = Plot::new(scores_er).line_style(
        LineStyle::new()
    ); // and a custom colour

    let v_er = ContinuousView::new()
        .add(s1_er);

    Page::single(&v_er).save("src/results/scores-ddqn_with_er.svg").unwrap();

    let s2_er: Plot = Plot::new(nb_steps_er).line_style(
        LineStyle::new()
    ); // and a custom colour

    let v_er = ContinuousView::new()
        .add(s2_er);

    Page::single(&v_er).save("src/results/nb-steps-ddqn_with_er.svg").unwrap();
}
