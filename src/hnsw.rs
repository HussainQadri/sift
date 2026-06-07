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

pub fn calculate_most_similiar_neighbours(
    current_node: &Node,
    query_node: &Node,
    all_nodes: &HnswIndex,
) -> Vec<(usize, f32)> {
    let node_neighbors = &current_node.neighbours;
    let mut result: Vec<(usize, f32)> = node_neighbors
        .iter()
        .map(|&node_id| {
            (
                node_id,
                similarity::cosine_similarity(
                    &all_nodes.nodes[node_id].embedding,
                    &query_node.embedding,
                ),
            )
        })
        .collect();

    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    result
}
