use crate::to_do::dqn;
use environnements::to_do::deep_single_agent::LineWorld;
use environnements;

pub mod to_do;

fn main() {
    let line_world_env = LineWorld::new(Option::Some(5));
    let dqn = dqn::DeepQLearning::new(line_world_env).train();
    // println!("{:?}", dqn);
}
