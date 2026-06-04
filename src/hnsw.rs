pub struct Node {
    id: usize,
    embedding: Vec<f32>,
    neighbours: Vec<usize>,
}

pub struct HnswIndex {
    nodes: Vec<Node>,
    entry_point: Option<Node>,
}

pub fn insert(mut index: HnswIndex, embedding_vec: Vec<f32>) {
    // empty index case
    if index.nodes.len() == 0 {
        let node_to_insert = Node {
            id: 0,
            embedding: embedding_vec,
            neighbours: Vec::new(),
        };

        index.nodes.push(node_to_insert);
    }
}
