use std::fs::File;
use std::io::{BufReader, BufWriter};
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::LineStyle;
use plotlib::view::ContinuousView;
use tch::{Device, nn};
use tch::nn::VarStore;

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

pub fn save_model(model_vs: &VarStore, path: &str) {
    let mut path: std::path::PathBuf = path.into();
    model_vs.save(&mut path).unwrap();
}

pub fn load_model(path: &str) -> VarStore {
    let mut path: std::path::PathBuf = path.into();
    println!("{:?}", path);
    let mut model_vs = VarStore::new(Device::Cpu);
    println!("{:?}", model_vs.variables());
    model_vs.load(&mut path).unwrap();
    println!("{:?}", model_vs.variables());
    model_vs
}

pub fn plot_scores_and_nb_steps(name: String, ema_scores: Vec<f64>, ema_nb_step: Vec<f64>) {
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
        .add(s1)
        .y_range(0.0, 1.0);

    Page::single(&v).save(&format!("src/results/gridworld/scores-{}.svg", name)).unwrap();

    let s2: Plot = Plot::new(nb_steps).line_style(
        LineStyle::new()
    ); // and a custom colour

    let v = ContinuousView::new()
        .add(s2);

    Page::single(&v).save(&format!("src/results/gridworld/nb-steps-{}.svg", name)).unwrap();
}