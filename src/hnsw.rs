#![allow(dead_code)]
pub struct Node {
    id: usize,
    embedding: Vec<f32>,
    neighbours: Vec<usize>,
}

pub struct HnswIndex {
    nodes: Vec<Node>,
    entry_point: Option<usize>,
}

pub fn insert(index: &mut HnswIndex, embedding_vec: Vec<f32>) {
    // empty index case
    if index.nodes.is_empty() {
        let node_to_insert = Node {
            id: 0,
            embedding: embedding_vec,
            neighbours: Vec::new(),
        };

        index.nodes.push(node_to_insert);
        index.entry_point = Some(0);
    } else {
        todo!("Insert case when index is non-empty");
    }
}
