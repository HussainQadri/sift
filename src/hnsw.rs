#![allow(dead_code)]
use crate::similarity;
pub struct Node {
    id: usize,
    embedding: Vec<f32>,
    neighbours: Vec<usize>,
}

pub struct HnswIndex {
    nodes: Vec<Node>,
    entry_point: Option<usize>,
    m: usize,
}

pub fn insert(index: &mut HnswIndex, embedding_vec: Vec<f32>) {
    // empty index case
    let mut node_to_insert = Node {
        id: index.nodes.len(),
        embedding: embedding_vec,
        neighbours: Vec::new(),
    };
    if index.nodes.is_empty() {
        index.nodes.push(node_to_insert);
        index.entry_point = Some(0);
    } else {
        let mut current_id = index.entry_point.unwrap();

        loop {
            let current_similarity = similarity::cosine_similarity(
                &index.nodes[current_id].embedding,
                &node_to_insert.embedding,
            );

            let most_similar_neighbours = calculate_most_similiar_neighbours(
                &index.nodes[current_id],
                &node_to_insert,
                index,
            );

            if most_similar_neighbours.is_empty() {
                break;
            }

            let best_neighbour_id = most_similar_neighbours[0].0;
            let best_neighbour_similarity = most_similar_neighbours[0].1;

            if best_neighbour_similarity > current_similarity {
                current_id = best_neighbour_id;
            } else {
                break;
            }
        }
        node_to_insert.neighbours.push(current_id);
        index.nodes[current_id].neighbours.push(node_to_insert.id);
        index.nodes.push(node_to_insert);
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
