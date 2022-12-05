use std::iter::zip;
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{LineStyle, PointMarker, PointStyle};
use plotlib::view::ContinuousView;
use crate::to_do::dqn;
use environnements::to_do::deep_single_agent::LineWorld;
use environnements;

pub mod to_do;

fn main() {
    let line_world_env = LineWorld::new(Option::Some(5));
    let (dqn, ema_scores, ema_nb_step) = dqn::DeepQLearning::new(line_world_env).train();
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

    Page::single(&v).save("scores.svg").unwrap();

    let s2: Plot = Plot::new(nb_steps).line_style(
        LineStyle::new()
    ); // and a custom colour

    let v = ContinuousView::new()
        .add(s2);

    Page::single(&v).save("nb_steps.svg").unwrap();
}
