use rand::Rng;

#[derive(Debug)]
struct LineWorld {
    nb_cells: usize,
    current_cell: usize,
    step_count: u32,
    win_rate: f32,
    game_played: u32,
}

impl LineWorld {
    fn new(nb_cells: Option<usize>) -> Self {
        let cells;
        if let Some(x) = nb_cells {
            cells = x;
        } else {
            cells = 5;
        }
        Self {
            nb_cells: cells,
            current_cell: (cells / 2) as usize,
            step_count: 0,
            win_rate: 0.0,
            game_played: 0,
        }
    }

    fn win_rate(&mut self) {
        self.win_rate += self.score();
        self.game_played += 1;
        println!("Win rate: {}", match self.game_played > 0 {
            true => self.win_rate as f64 / self.game_played as f64,
            false => 0.0,
        });
        println!("Game played: {}", self.nb_cells - 2);
    }
}

impl DeepSingleAgentEnv for LineWorld {
    fn state_dim(&self) -> usize {
        1
    }

    fn state_description(&self) -> Vec<f64> {
        vec![self.current_cell as f64 / (self.nb_cells as f64 - 1.0) * 2.0 - 1.0]
    }

    fn max_action_count(&self) -> usize {
        2
    }

    fn is_game_over(&self) -> bool {
        if self.step_count > (f64::powi(self.nb_cells, 2)) as u32 {
            return true;
        }
        (self.current_cell == 0) || (self.current_cell == self.nb_cells - 1)
    }

    fn act_with_action_id(&mut self, action_id: usize) {
        self.step_count += 1;
        if action_id == 0 {
            self.current_cell -= 1;
        } else {
            self.current_cell += 1;
        }
    }

    fn score(&self) -> f32 {
        if self.current_cell == 0 {
            return -1.0;
        } else if self.current_cell == self.nb_cells - 1 {
            return 1.0;
        }
        0.0
    }

    fn available_actions_ids(&self) -> Vec<usize> {
        vec![0, 1]
    }

    fn reset(&mut self) {
        self.current_cell = self.nb_cells / 2;
        self.step_count = 0;
    }

    fn view(&self) {
        println!("Game Over: {}", self.is_game_over());
        println!("Score: {}", self.score());
        for i in 0..self.nb_cells {
            if i == self.current_cell {
                print!("X");
            } else if i == 0 {
                print!("L");
            } else if i == self.nb_cells - 1 {
                print!("W");
            } else {
                print!("_");
            }
        }
        println!();
    }
}