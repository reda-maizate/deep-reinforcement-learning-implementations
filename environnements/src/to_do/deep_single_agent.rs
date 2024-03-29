use crate::contracts::DeepSingleAgentEnv;
use std::io;
use rand::{thread_rng, Rng};

#[derive(Debug)]
pub struct LineWorld {
    nb_cells: usize,
    current_cell: usize,
    step_count: u32,
}

impl LineWorld {
    pub fn new(nb_cells: Option<usize>) -> Self {
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
        }
    }

    pub fn play(&mut self, random: bool) {
        while !self.is_game_over() {
            self.view();
            if random {
                let aa = self.available_actions_ids();
                let action_id = aa[thread_rng().gen_range(0..aa.len())];
                self.act_with_action_id(action_id);
            } else {
                println!("0:Left 1:Right");
                println!("Saisi action: ");
                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
                let input = input.trim().parse::<usize>().unwrap();
                if input > 1 {
                    continue;
                }
                self.act_with_action_id(input);
            }
        }
        self.view();
        println!("\nSaisir 1 pour relancer une partie: ");
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim().parse::<usize>().unwrap();
        if input == 1 {
            self.reset();
            self.play(random);
        }
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
        if self.step_count > (self.nb_cells * 2) as u32 {
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

#[derive(Debug)]
pub struct GridWorld {
    nb_cols: usize,
    nb_rows: usize,
    nb_cells: usize,
    current_cell: usize,
    step_count: u32,
}

impl GridWorld {
    pub fn new(nb_cols: Option<usize>, nb_rows: Option<usize>) -> Self {
        let cols;
        let rows;
        if let (Some(x), Some(y)) = (nb_cols, nb_rows) {
            cols = x;
            rows = y;
        } else {
            cols = 5;
            rows = 5;
        }
        Self {
            nb_cols: cols,
            nb_rows: rows,
            nb_cells: cols * rows,
            current_cell: 0,
            step_count: 0,
        }
    }

    pub fn play(&mut self, random: bool) {
        while !self.is_game_over() {
            self.view();
            if random {
                let aa = self.available_actions_ids();
                let action_id = aa[thread_rng().gen_range(0..aa.len())];
                self.act_with_action_id(action_id);
            } else {
                println!("0:Left 1:Right 2:Up 3:Down");
                println!("Saisi action: ");
                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
                let input = input.trim().parse::<usize>().unwrap();
                if input > 4 {
                    continue;
                }
                self.act_with_action_id(input);
            }
        }
        self.view();
        println!("\nSaisir 1 pour relancer une partie: ");
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim().parse::<usize>().unwrap();
        if input == 1 {
            self.reset();
            self.play(random);
        }
    }
}

impl DeepSingleAgentEnv for GridWorld {
    fn max_action_count(&self) -> usize {
        4
    }

    fn state_description(&self) -> Vec<f64> {
        vec![self.current_cell as f64 / (self.nb_cells as f64 - 1.0) * 2.0 - 1.0]
    }

    fn state_dim(&self) -> usize {
        1
    }

    fn is_game_over(&self) -> bool {
        if self.step_count > (self.nb_cells * 2) as u32 {
            return true;
        }
        (self.current_cell == self.nb_rows - 1) || (self.current_cell == self.nb_cells - 1)
    }

    fn act_with_action_id(&mut self, action_id: usize) {
        // O: LEFT
        // 1: RIGHT
        // 2: UP
        // 3: DOWN
        self.step_count += 1;
        if (action_id == 0) && (self.current_cell % self.nb_rows != 0) {
            self.current_cell -= 1;
        } else if (action_id == 1) && (self.current_cell % self.nb_rows != self.nb_rows - 1) {
            self.current_cell += 1;
        } else if (action_id == 2) && (self.current_cell as i32 - self.nb_rows as i32 >= 0) {
            self.current_cell -= self.nb_rows;
        } else if (action_id == 3) && (self.current_cell + self.nb_rows <= self.nb_cells - 1) {
            self.current_cell += self.nb_rows;
        }
    }

    fn score(&self) -> f32 {
        if self.current_cell == self.nb_rows - 1 {
            return -3.0;
        } else if self.current_cell == self.nb_cells - 1 {
            return 1.0;
        }
        0.0
    }

    fn available_actions_ids(&self) -> Vec<usize> {
        vec![0, 1, 2, 3]
    }

    fn reset(&mut self) {
        self.current_cell = 0;
        self.step_count = 0;
    }

    fn view(&self) {
        println!("\nGame Over: {}", self.is_game_over());
        println!("Score: {}", self.score());
        for i in 0..self.nb_cols {
            for j in 0..self.nb_rows {
                if (i * self.nb_rows) + j == self.current_cell {
                    print!("X");
                } else if i == 0 && j == 0 {
                    print!("S");
                } else if i == 0 && j == self.nb_rows - 1 {
                    print!("L");
                } else if i == self.nb_cols - 1 && j == self.nb_rows - 1  {
                    print!("W");
                } else {
                    print!("_");
                }
            }
            println!();
        }
    }
}