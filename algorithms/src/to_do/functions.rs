use std::fs::File;
use std::io::{BufReader, BufWriter};
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