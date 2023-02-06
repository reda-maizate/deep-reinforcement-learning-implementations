use environnements::contracts::DeepSingleAgentEnv;

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
}