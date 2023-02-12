use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::LineStyle;
use plotlib::view::ContinuousView;

pub struct EMA {
    pub score: f64,
    pub nb_steps: f64,
    pub first_episode: bool,
    pub step: f64,
    pub score_progress: Vec<f64>,
    pub nb_steps_progress: Vec<f64>,
}

impl EMA {
    pub fn new() -> Self {
        let score_progress = Vec::new();
        let nb_steps_progress = Vec::new();
        Self {
            score: 0.0,
            nb_steps: 0.0,
            first_episode: true,
            step: 0.0,
            score_progress,
            nb_steps_progress,
        }
    }

    pub fn display_results(&mut self, name_algo: &str) {
        let mut scores = vec![];
        let mut nb_steps = vec![];

        for i in 0..self.score_progress.len() {
            scores.push((i as f64, self.score_progress[i]));
        }

        for i in 0..self.nb_steps_progress.len() {
            nb_steps.push((i as f64, self.nb_steps_progress[i]));
        }

        let s1: Plot = Plot::new(scores).line_style(
            LineStyle::new()
        ); // and a custom colour
        let v = ContinuousView::new().add(s1);
        Page::single(&v).save(format!("src/results/scores-{}.svg", name_algo)).unwrap();

        let s2: Plot = Plot::new(nb_steps).line_style(
            LineStyle::new()
        ); // and a custom colour
        let v = ContinuousView::new().add(s2);
        Page::single(&v).save(format!("src/results/nb-steps-{}.svg", name_algo)).unwrap();
    }
}