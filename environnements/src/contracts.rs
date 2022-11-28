pub trait SingleAgentEnv {
    fn state_id(&self) -> usize;
    fn is_game_over(&self) -> bool;
    fn act_with_action_id(&mut self, action_id: usize);
    fn score(&self) -> f32;
    fn available_actions_ids(&self) -> Vec<usize>;
    fn reset(&mut self);
    fn view(&self);
    fn reset_random(&mut self);
}

pub trait DeepSingleAgentEnv {
    fn max_action_count(&self) -> usize;
    fn state_description(&self) -> Vec<f64>;
    fn state_dim(&self) -> usize;
    fn is_game_over(&self) -> bool;
    fn act_with_action_id(&mut self, action_id: usize);
    fn score(&self) -> f32;
    fn available_actions_ids(&self) -> Vec<usize>;
    fn reset(&mut self);
    fn view(&self);
}

pub trait MDPEnv {
    fn states(&self) -> Vec<usize>;
    fn actions(&self) -> Vec<usize>;
    fn rewards(&self) -> Vec<f32>;
    fn is_state_terminal(&self, s: usize) -> bool;
    fn transition_probability(&self, s: usize, a: usize, s_p: usize, r: f32) -> f32;
    fn view_state(&self, s: usize);
}