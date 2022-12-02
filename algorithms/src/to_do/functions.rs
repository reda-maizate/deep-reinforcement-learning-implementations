
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