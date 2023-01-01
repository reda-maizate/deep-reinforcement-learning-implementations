use std::io;

struct Player {
    x: i32,
    y: i32,
    direction: Direction,
}

enum Direction {
    Up,
    Down,
    Left,
    Right,
}

struct Ghost {
    x: i32,
    y: i32,
    direction: Direction,
}

const WALL: char = '#';
const PATH: char = ' ';
const GUM: char = '.';
const SUPER_GUM: char = 'o';
const PLAYER: char = 'P';
const GHOST: char = 'G';
const TOTAL_GUMS: i32 = 100;

struct Game {
    player: Player,
    ghosts: Vec<Ghost>,
    grid: Vec<Vec<char>>,
    score: i32,
    finished: bool,
}

impl Game {
    fn new() -> Game {
        // Initialisez ici la grille de jeu et les personnages
        Game {
            player: Player { x: 0, y: 0, direction: Direction::Right },
            ghosts: vec![],
            grid: vec![],
            score: 0,
            finished: false,
        }
    }

    fn generate_grid(&mut self) {
        // Générez ici la grille de jeu
        self.grid = vec![vec![' ', ' ', '.', ' ', '#', ' ', ' ', ' ', '#'],
                         vec!['#', ' ', '#', ' ', ' ', ' ', '#', ' ', '#'],
                         vec!['#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#'],
                         vec!['#', '#', '#', '#', '#', ' ', ' ', ' ', '#'],
                         vec![' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
                         vec!['#', ' ', '#', '#', ' ', ' ', '#', ' ', '#'],
                         vec!['#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#'],
                         vec!['#', '#', '#', '#', '#', '#', '#', '#', '#']];
    }

    fn generate_ghosts(&mut self) {
        // Générez ici les fantômes
        self.ghosts = vec![Ghost { x: 3, y: 2, direction: Direction::Right },
                           Ghost { x: 4, y: 5, direction: Direction::Right }];
    }

    fn update(&mut self) {
        // Mettez à jour la position et la direction du joueur
        self.player.update(&self.grid);

        // Mettez à jour la direction des fantômes
        for ghost in &mut self.ghosts {
            ghost.update(&self.player, &mut self.grid);
        }

        // Vérifiez si le joueur a mangé une gomme ou une super-gomme et mettez à jour le score en conséquence
        let cell = self.grid[self.player.x as usize][self.player.y as usize];
        if cell == GUM {
            self.score += 10;
            self.grid[self.player.x as usize][self.player.y as usize] = PATH;
            println!("Score: {}", self.score);
        } else if cell == SUPER_GUM {
            self.score += 50;
            self.grid[self.player.x as usize][self.player.y as usize] = PATH;
            println!("Score: {}", self.score);
        } else if cell == GHOST {
            self.finished = true;
            println!("Game over!");
            return;
        } else if self.score == TOTAL_GUMS {
            self.finished = true;
            println!("You win!");
        } else {
            println!("Score: {}", self.score);
        }

        // Effacez la gomme ou la super-gomme de la grille
        self.grid[self.player.x as usize][self.player.y as usize] = 'P';

        // Vérifiez si le joueur a gagné le niveau en mangeant toutes les gommes
        if self.score >= TOTAL_GUMS {
            // Passez au niveau suivant ici
            self.finished = true;
            return;
        }
    }

    fn player_collision(&self) -> bool {
        // Vérifiez si le joueur a collisionné avec un fantôme ici et retournez true en conséquence
        false
    }

    fn handle_input(&mut self, input: Input) {
        // Mettez à jour ici la direction du joueur en fonction de l'entrée du joueur

        match input {
            Input::Up => self.player.direction = Direction::Up,
            Input::Down => self.player.direction = Direction::Down,
            Input::Left => self.player.direction = Direction::Left,
            Input::Right => self.player.direction = Direction::Right,
        }
    }

    fn render(&mut self) {
        // Affichez ici la grille de jeu et les personnages à l'écran
        self.grid[self.player.x as usize][self.player.y as usize] = 'P';
        for row in &self.grid {
            for cell in row {
                print!("{}", cell);
            }
            println!("");
        }
        self.grid[self.player.x as usize][self.player.y as usize] = PATH;
    }
}

enum Input {
    Up,
    Down,
    Left,
    Right,
}

impl Ghost {
    fn update(&mut self, player: &Player, grid: &mut Vec<Vec<char>>) {
        // Mettez à jour ici la position du fantôme en fonction de la position du joueur et des murs
        match self.direction {
            Direction::Up => {
                if grid[self.x as usize - 1][self.y as usize] != WALL {
                    self.x -= 1;
                } else {
                    self.direction = Direction::Down;
                }
            }
            Direction::Down => {
                if grid[self.x as usize + 1][self.y as usize] != WALL {
                    self.x += 1;
                } else {
                    self.direction = Direction::Up;
                }
            }
            Direction::Left => {
                if grid[self.x as usize][self.y as usize - 1] != WALL {
                    self.y -= 1;
                } else {
                    self.direction = Direction::Right;
                }
            }
            Direction::Right => {
                if grid[self.x as usize][self.y as usize + 1] != WALL {
                    self.y += 1;
                } else {
                    self.direction = Direction::Left;
                }
            }
        }



        grid[self.x as usize][self.y as usize] = GHOST;
    }
}


impl Player {
    fn update(&mut self, grid: &Vec<Vec<char>>) {
        // Mettez à jour la position et la direction du joueur ici en fonction de la grille de jeu

        let x = self.x;
        let y = self.y;
        let dir = &self.direction;

        match dir {
            Direction::Up => self.x -= 1,
            Direction::Down => self.x += 1,
            Direction::Left => self.y -= 1,
            Direction::Right => self.y += 1,
        }

        // Gérez les collisions avec les murs et les portes ici
        if self.x < 0 || self.x >= grid.len() as i32 {
            self.x = x;
        } else if self.y < 0 || self.y >= grid[0].len() as i32 {
            self.y = y;
        } else if grid[self.x as usize][self.y as usize] == WALL {
            self.x = x;
            self.y = y;
        }
    }
}

fn get_input() -> Input {
    // Récupérez ici l'entrée du joueur et retournez-la sous forme d'Input

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    match input.trim() {
        "u" => Input::Up,
        "d" => Input::Down,
        "l" => Input::Left,
        "r" => Input::Right,
        _ => get_input(),
    }
}

fn main() {
    let mut game = Game::new();
    game.generate_grid();
    game.generate_ghosts();
    println!("Beginner's Rust - Pacman");
    println!("Use URLD to move");
    game.render();

    while !game.finished {
        game.handle_input(get_input());
        game.update();
        game.render();
    }
}