#![allow(dead_code)]
use crate::similarity;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
pub struct Node {
    id: usize,
    embedding: Vec<f32>,
    neighbours: Vec<usize>,
}

#[derive(Clone, Copy)]
pub struct ScoredNode {
    id: usize,
    score: f32,
}

impl Ord for ScoredNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialEq for ScoredNode {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Equal
    }
}

impl PartialOrd for ScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for ScoredNode {}

pub struct HnswIndex {
    nodes: Vec<Node>,
    entry_point: Option<usize>,
    m: usize,
    ef: usize,
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
        let nearby_neighbours: Vec<(usize, f32)> = search_layer(
            index,
            &node_to_insert.embedding,
            index.entry_point.unwrap(),
            index.ef,
        );
        let best_m_neighbours: Vec<usize> = nearby_neighbours
            .into_iter()
            .take(index.m)
            .map(|(id, _score)| id)
            .collect();
        for id in best_m_neighbours {
            node_to_insert.neighbours.push(id);
            index.nodes[id].neighbours.push(node_to_insert.id);
        }
        index.nodes.push(node_to_insert);
    }
}

fn search_greedy(index: &HnswIndex, node_to_insert: &Node) -> usize {
    let mut current_id = index.entry_point.unwrap();

    loop {
        let current_similarity = similarity::cosine_similarity(
            &index.nodes[current_id].embedding,
            &node_to_insert.embedding,
        );

        let most_similar_neighbours =
            calculate_most_similiar_neighbours(&index.nodes[current_id], node_to_insert, index);

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

    current_id
}
fn search_layer(
    index: &HnswIndex,
    query_vector: &[f32],
    entry_point_id: usize,
    ef: usize,
) -> Vec<(usize, f32)> {
    let mut visited: HashSet<usize> = HashSet::new();
    visited.insert(entry_point_id);

    let entry_point_similarity =
        similarity::cosine_similarity(query_vector, &index.nodes[entry_point_id].embedding);
    let entry_scored_node = ScoredNode {
        id: entry_point_id,
        score: entry_point_similarity,
    };

    let mut candidates: BinaryHeap<ScoredNode> = BinaryHeap::new();
    let mut best_found: BinaryHeap<Reverse<ScoredNode>> = BinaryHeap::new();

    candidates.push(entry_scored_node);
    best_found.push(Reverse(entry_scored_node)); // the root is the worst best-found node

    while let Some(best_candidate) = candidates.pop() {
        let worst_found = best_found.peek().unwrap().0;
        if best_candidate.score < worst_found.score {
            break;
        }

        for &neighbour_id in &index.nodes[best_candidate.id].neighbours {
            if !visited.contains(&neighbour_id) {
                visited.insert(neighbour_id);
                let neighbour_vector = &index.nodes[neighbour_id].embedding;
                let neighbour_similarity =
                    similarity::cosine_similarity(query_vector, neighbour_vector);

                if best_found.len() < ef {
                    let new_scored_node = ScoredNode {
                        id: neighbour_id,
                        score: neighbour_similarity,
                    };
                    candidates.push(new_scored_node);
                    best_found.push(Reverse(new_scored_node));
                } else if neighbour_similarity > best_found.peek().unwrap().0.score {
                    let new_scored_node = ScoredNode {
                        id: neighbour_id,
                        score: neighbour_similarity,
                    };
                    candidates.push(new_scored_node);
                    best_found.push(Reverse(new_scored_node));
                    best_found.pop();
                }
            }
        }
    }

    let mut results: Vec<(usize, f32)> = best_found
        .into_iter()
        .map(|Reverse(node)| (node.id, node.score))
        .collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results
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

#[cfg(test)]
mod tests {
    use super::{
        HnswIndex, Node, calculate_most_similiar_neighbours, insert, search_greedy, search_layer,
    };

    fn empty_index() -> HnswIndex {
        HnswIndex {
            nodes: Vec::new(),
            entry_point: None,
            m: 2,
            ef: 2,
        }
    }

    fn node(id: usize, embedding: Vec<f32>, neighbours: Vec<usize>) -> Node {
        Node {
            id,
            embedding,
            neighbours,
        }
    }

    #[test]
    fn first_insert_sets_entry_point_and_stores_node() {
        let mut index = empty_index();

        insert(&mut index, vec![1.0, 0.0]);

        assert_eq!(index.entry_point, Some(0));
        assert_eq!(index.nodes.len(), 1);
        assert_eq!(index.nodes[0].id, 0);
        assert_eq!(index.nodes[0].embedding, vec![1.0, 0.0]);
        assert!(index.nodes[0].neighbours.is_empty());
    }

    #[test]
    fn second_insert_links_new_node_bidirectionally_to_entry_point() {
        let mut index = empty_index();

        insert(&mut index, vec![1.0, 0.0]);
        insert(&mut index, vec![0.9, 0.1]);

        assert_eq!(index.nodes.len(), 2);
        assert_eq!(index.nodes[0].neighbours, vec![1]);
        assert_eq!(index.nodes[1].neighbours, vec![0]);
    }

    #[test]
    fn insert_uses_search_layer_to_link_to_closest_reachable_nodes() {
        let mut index = HnswIndex {
            nodes: vec![
                node(0, vec![1.0, 0.0], vec![1]),
                node(1, vec![0.0, 1.0], vec![0]),
            ],
            entry_point: Some(0),
            m: 2,
            ef: 2,
        };

        insert(&mut index, vec![0.0, 0.9]);

        assert_eq!(index.nodes[2].neighbours, vec![1, 0]);
        assert_eq!(index.nodes[1].neighbours, vec![0, 2]);
        assert_eq!(index.nodes[0].neighbours, vec![1, 2]);
    }

    #[test]
    fn search_layer_returns_up_to_ef_best_reachable_nodes() {
        let index = HnswIndex {
            nodes: vec![
                node(0, vec![1.0, 0.0], vec![1, 2]),
                node(1, vec![0.0, 1.0], vec![0]),
                node(2, vec![0.9, 0.1], vec![0]),
            ],
            entry_point: Some(0),
            m: 2,
            ef: 2,
        };

        let results = search_layer(&index, &[1.0, 0.0], 0, index.ef);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[1].0, 2);
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn search_greedy_stops_when_no_neighbour_is_more_similar() {
        let index = HnswIndex {
            nodes: vec![
                node(0, vec![1.0, 0.0], vec![1]),
                node(1, vec![0.0, 1.0], vec![0]),
            ],
            entry_point: Some(0),
            m: 2,
            ef: 2,
        };
        let query = node(2, vec![0.9, 0.1], Vec::new());

        assert_eq!(search_greedy(&index, &query), 0);
    }

    #[test]
    fn calculate_most_similar_neighbours_sorts_by_descending_similarity() {
        let index = HnswIndex {
            nodes: vec![
                node(0, vec![1.0, 0.0], vec![1, 2]),
                node(1, vec![0.0, 1.0], vec![0]),
                node(2, vec![0.8, 0.2], vec![0]),
            ],
            entry_point: Some(0),
            m: 2,
            ef: 2,
        };
        let query = node(3, vec![1.0, 0.0], Vec::new());

        let neighbours = calculate_most_similiar_neighbours(&index.nodes[0], &query, &index);

        assert_eq!(neighbours.len(), 2);
        assert_eq!(neighbours[0].0, 2);
        assert_eq!(neighbours[1].0, 1);
        assert!(neighbours[0].1 > neighbours[1].1);
    }
}
