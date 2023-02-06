use rand::{thread_rng, seq::SliceRandom};

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

pub fn get_random_mini_batch<T: Clone>(vector: &Vec<T>, batch_size: usize) -> Vec<T> {
    let mut rng = thread_rng();
    vector.choose_multiple(&mut rng, batch_size).cloned().collect()
}