use std::io;
use rand::seq::SliceRandom;

#[derive(Debug, Copy, Clone)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Debug)]
struct Player {
    x: i32,
    y: i32,
    direction: Direction,
}

#[derive(Debug)]
struct Ghost {
    x: i32,
    y: i32,
}

#[derive(Debug)]
struct Game {
    player: Player,
    ghosts: Vec<Ghost>,
    grid: Vec<Vec<char>>,
    score: i32,
    finished: bool,
}

const WALL: char = '#';
const PATH: char = '.';
const GUM: char = 'o';
const SUPER_GUM: char = 'O';
const PLAYER: char = 'P';
const GHOST: char = 'G';
const TOTAL_GUMS: i32 = 100;

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

impl Ghost {
    fn update(&mut self, grid: &mut Vec<Vec<char>>) {
        // Faire avancer le fantôme de manière aléatoire ici
        let past_x = self.x;
        let past_y = self.y;

        // TODO: A remplacer par la méthode `available_actions` de l'environnement
        let mut actions = vec![];
        // TODO: error -> "expected `i32`, found `usize`"
        if grid[self.x as usize + 1][self.y as usize] != WALL {
            actions.push(Direction::Down);
        }
        if grid[self.x as usize - 1][self.y as usize] != WALL {
            actions.push(Direction::Up);
        }
        if grid[self.x as usize][self.y as usize + 1] != WALL {
            actions.push(Direction::Right);
        }
        if grid[self.x as usize][self.y as usize - 1] != WALL {
            actions.push(Direction::Left);
        }

        let mut rng = rand::thread_rng();
        let action = actions.choose(&mut rng).unwrap();

        match action {
            Direction::Up => self.x -= 1,
            Direction::Down => self.x += 1,
            Direction::Left => self.y -= 1,
            Direction::Right => self.y += 1,
        }

        grid[past_x as usize][past_y as usize] = PATH;
        grid[self.x as usize][self.y as usize] = GHOST;
    }
}

impl Game {
    fn new() -> Game {
        // Initialisez ici la grille de jeu et les personnages
        Game {
            player: Player { x: 1, y: 1, direction: Direction::Right },
            ghosts: vec![],
            grid: vec![],
            score: 0,
            finished: false,
        }
    }

    fn generate_grid(&mut self) {
        // Générez ici la grille de jeu
        self.grid = vec![
            vec!['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
            vec!['#', '.', '.', '.', 'O', '.', '#', '.', '.', '.', '.', '.', '.', '#', '.', '.', '.', '#'],
            vec!['#', '.', '.', '.', '.', '.', '#', '.', '#', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
            vec!['#', '#', '#', '#', '#', '.', '.', '.', '.', '.', '.', '.', '.', '#', '.', '.', '.', '#'],
            vec!['#', '.', '.', '.', 'o', '.', '.', '.', '.', '.', '#', '#', '.', '.', '.', '.', '.', '#'],
            vec!['#', '.', '#', '#', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
            vec!['#', '.', '.', '.', '.', '.', '#', 'O', '#', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
            vec!['#', '.', '.', '#', '#', '#', '#', '#', '#', '.', '.', 'O', '.', '#', '.', '.', '.', '#'],
            vec!['#', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '#', '.', '.', '.', '#'],
            vec!['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#']
        ];
    }

    fn generate_ghosts(&mut self, nb_ghosts: i32) {
        // Générez ici les fantômes
        let mut possible_spawn_positions = vec![];
        for i in 0..self.grid.len() {
            for j in 0..self.grid[i].len() {
                if self.grid[i][j] == PATH {
                    possible_spawn_positions.push((i, j));
                }
            }
        }

        self.ghosts = vec![];
        for _ in 0..nb_ghosts {
            let mut rng = rand::thread_rng();
            let (x, y) = possible_spawn_positions.choose(&mut rng).unwrap();
            self.ghosts.push(Ghost { x: *x as i32, y: *y as i32 });
            self.grid[*x][*y] = GHOST;
        }
    }

    fn update(&mut self) {
        // Mettez à jour la position et la direction du joueur
        println!("--------------------------------");
        self.player.update(&self.grid);

        // Mettez à jour la direction des fantômes
        for ghost in &mut self.ghosts {
            ghost.update( &mut self.grid);
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
        } else {
            println!("Score: {}", self.score);
        }

        // Effacez la gomme ou la super-gomme de la grille
        self.grid[self.player.x as usize][self.player.y as usize] = PLAYER;

        // Vérifiez si le joueur a gagné le niveau en mangeant toutes les gommes
        if self.score >= TOTAL_GUMS {
            // Passez au niveau suivant ici
            self.finished = true;
            println!("You win!");
            return;
        }
    }


    fn handle_input(&mut self, direction: Direction) {
        // Mettez à jour ici la direction du joueur en fonction de l'entrée du joueur
        self.player.direction = direction;
    }

    fn render(&mut self) {
        // Affichez ici la grille de jeu et les personnages à l'écran
        self.grid[self.player.x as usize][self.player.y as usize] = PLAYER;
        for row in &self.grid {
            for cell in row {
                print!("{}", cell);
            }
            println!();
        }
        self.grid[self.player.x as usize][self.player.y as usize] = PATH;
    }
}

fn get_input() -> Direction {
    // Récupérez ici l'entrée du joueur et retournez-la sous forme d'Input
    println!("Enter a direction or quit with 'q' : ");
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    match input.to_lowercase().trim() {
        "u" => Direction::Up,
        "d" => Direction::Down,
        "l" => Direction::Left,
        "r" => Direction::Right,
        // Exit the game here
        "q" => {
            println!("Exiting game...");
            std::process::exit(0);
        }
        _ => get_input(),
    }
}

fn main() {
    let mut game = Game::new();
    let nb_ghosts = 3;

    game.generate_grid();
    game.generate_ghosts(nb_ghosts);
    println!("~~~~~ Pacman ~~~~~");
    println!("Instructions:");
    println!("Use the 'u', 'd', 'l', 'r' keys to move the player");
    println!("Press 'q' to quit the game");
    println!();
    println!();
    println!("Starting game...");
    game.render();

    while !game.finished {
        let input = get_input();
        game.handle_input(input);
        game.update();
        game.render();
    }
}