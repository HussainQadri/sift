use crate::similarity;
use rand::RngExt;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

impl HnswIndex {
    pub fn new(m: usize, ef: usize) -> Self {
        assert!(m > 1);
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            m,
            ef,
        }
    }

    pub fn search(&self, embedding_vec: &[f32], top_k: usize) -> Vec<usize> {
        search(self, embedding_vec, self.ef, top_k)
    }

    pub fn insert(&mut self, record_id: usize, embedding_vec: Vec<f32>) {
        insert(self, record_id, embedding_vec);
    }
}

#[derive(Clone)]
pub struct Node {
    id: usize, // Internal HNSW id, record_ids are translated into this id.
    embedding: Vec<f32>,
    neighbours: Vec<Vec<usize>>,
    record_id: usize,
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
        self.id == other.id && self.score.total_cmp(&other.score) == Ordering::Equal
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
    max_layer: usize,
    m: usize,
    ef: usize,
}

pub fn create_node(index: &HnswIndex, embedding_vec: Vec<f32>, record_id: usize) -> Node {
    let random_max_layer = get_random_level(index.m);
    Node {
        id: index.nodes.len(),
        embedding: embedding_vec,
        neighbours: vec![Vec::new(); random_max_layer + 1],
        record_id,
    }
}

const MAX_ALLOWED_LAYER: usize = 16;

pub fn get_random_level(m: usize) -> usize {
    let mut rng = rand::rng();
    let m_l = 1.0 / (m as f32).ln();

    let u: f32 = rng.random_range(f32::MIN_POSITIVE..1.0);
    ((-u.ln() * m_l).floor() as usize).min(MAX_ALLOWED_LAYER)
}
pub fn insert(index: &mut HnswIndex, record_id: usize, embedding_vec: Vec<f32>) {
    let node_to_insert = create_node(index, embedding_vec, record_id);
    insert_node(index, node_to_insert);
}

fn insert_node(index: &mut HnswIndex, mut node_to_insert: Node) {
    let mut nodes_to_prune = Vec::new();
    let node_max_layer = node_to_insert.neighbours.len() - 1;

    if index.nodes.is_empty() {
        index.nodes.push(node_to_insert);
        index.entry_point = Some(0);
        index.max_layer = node_max_layer;
    } else {
        let mut current_id = index.entry_point.unwrap();
        let old_max_layer = index.max_layer;

        for layer in ((node_max_layer + 1)..=old_max_layer).rev() {
            current_id = search_greedy(index, &node_to_insert.embedding, layer, current_id);
        }

        let top_connection_layer = node_max_layer.min(old_max_layer);
        for node_layer in (0..=top_connection_layer).rev() {
            let nearby_neighbours: Vec<(usize, f32)> = search_layer(
                index,
                &node_to_insert.embedding,
                current_id,
                index.ef,
                node_layer,
            );
            let best_m_neighbours: Vec<usize> = nearby_neighbours
                .into_iter()
                .take(index.m)
                .map(|(id, _score)| id)
                .collect();
            for id in &best_m_neighbours {
                node_to_insert.neighbours[node_layer].push(*id);
                index.nodes[*id].neighbours[node_layer].push(node_to_insert.id);
            }

            for &neighbour_id in &node_to_insert.neighbours[node_layer] {
                nodes_to_prune.push((neighbour_id, node_layer));
            }

            if let Some(&best_id) = best_m_neighbours.first() {
                current_id = best_id;
            }
        }
        let new_node_id = node_to_insert.id;
        index.nodes.push(node_to_insert);

        if node_max_layer > old_max_layer {
            index.entry_point = Some(new_node_id);
            index.max_layer = node_max_layer;
        }

        for (neighbour_id, node_layer) in nodes_to_prune {
            prune(index, neighbour_id, node_layer);
        }
    }
}

fn prune(index: &mut HnswIndex, node_to_prune_id: usize, layer: usize) {
    if index.nodes[node_to_prune_id].neighbours[layer].len() <= index.m {
        return;
    }
    let node_to_prune = index.nodes[node_to_prune_id].clone();
    let neighbour_ids = index.nodes[node_to_prune_id].neighbours[layer].clone();

    let mut scored: Vec<(usize, f32)> = neighbour_ids
        .into_iter()
        .map(|neighbour_id| {
            let score = similarity::cosine_similarity(
                &node_to_prune.embedding,
                &index.nodes[neighbour_id].embedding,
            );
            (neighbour_id, score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));

    index.nodes[node_to_prune_id].neighbours[layer] = scored
        .iter()
        .take(index.m)
        .map(|(id, _score)| *id)
        .collect();
    let removed: Vec<usize> = scored
        .iter()
        .skip(index.m)
        .map(|(id, _score)| *id)
        .collect();
    for removed_id in removed {
        if let Some(neighbours) = index.nodes[removed_id].neighbours.get_mut(layer) {
            neighbours.retain(|&id| id != node_to_prune_id);
        }
    }
}

fn search_greedy(
    index: &HnswIndex,
    query_vector: &[f32],
    layer: usize,
    entry_point_id: usize,
) -> usize {
    let mut current_id = entry_point_id;

    loop {
        let current_similarity =
            similarity::cosine_similarity(&index.nodes[current_id].embedding, query_vector);

        let most_similar_neighbours = calculate_most_similiar_neighbours(
            &index.nodes[current_id],
            query_vector,
            index,
            layer,
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

    current_id
}
fn search_layer(
    index: &HnswIndex,
    query_vector: &[f32],
    entry_point_id: usize,
    ef: usize,
    layer: usize,
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

        for &neighbour_id in &index.nodes[best_candidate.id].neighbours[layer] {
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

pub fn search(index: &HnswIndex, query_vector: &[f32], ef: usize, top_k: usize) -> Vec<usize> {
    let mut current_id = index.entry_point.unwrap();

    for layer in (1..=index.max_layer).rev() {
        current_id = search_greedy(index, query_vector, layer, current_id);
    }

    let results: Vec<(usize, f32)> = search_layer(index, query_vector, current_id, ef, 0);
    results
        .into_iter()
        .map(|(id, _embedding)| index.nodes[id].record_id)
        .take(top_k)
        .collect()
}

pub fn calculate_most_similiar_neighbours(
    current_node: &Node,
    query_vector: &[f32],
    all_nodes: &HnswIndex,
    layer: usize,
) -> Vec<(usize, f32)> {
    let node_neighbors = &current_node.neighbours[layer];
    let mut result: Vec<(usize, f32)> = node_neighbors
        .iter()
        .map(|&node_id| {
            (
                node_id,
                similarity::cosine_similarity(&all_nodes.nodes[node_id].embedding, query_vector),
            )
        })
        .collect();

    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    result
}

#[cfg(test)]
mod tests {
    use super::{
        HnswIndex, Node, calculate_most_similiar_neighbours, insert, insert_node, search,
        search_greedy, search_layer,
    };

    fn empty_index() -> HnswIndex {
        HnswIndex {
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            m: 2,
            ef: 2,
        }
    }

    fn node(id: usize, embedding: Vec<f32>, neighbours: Vec<Vec<usize>>) -> Node {
        Node {
            id,
            embedding,
            neighbours,
            record_id: id,
        }
    }

    #[test]
    fn first_insert_sets_entry_point_and_stores_node() {
        let mut index = empty_index();

        insert(&mut index, 0, vec![1.0, 0.0]);

        assert_eq!(index.entry_point, Some(0));
        assert_eq!(index.nodes.len(), 1);
        assert_eq!(index.nodes[0].id, 0);
        assert_eq!(index.nodes[0].embedding, vec![1.0, 0.0]);
        assert!(index.nodes[0].neighbours[0].is_empty());
    }

    #[test]
    fn second_insert_links_new_node_bidirectionally_to_entry_point() {
        let mut index = empty_index();

        insert(&mut index, 0, vec![1.0, 0.0]);
        insert(&mut index, 0, vec![0.9, 0.1]);

        assert_eq!(index.nodes.len(), 2);
        assert_eq!(index.nodes[0].neighbours[0], vec![1]);
        assert_eq!(index.nodes[1].neighbours[0], vec![0]);
    }

    #[test]
    fn insert_uses_search_layer_to_link_to_closest_reachable_nodes() {
        let mut index = HnswIndex {
            nodes: vec![
                node(0, vec![1.0, 0.0], vec![vec![1]]),
                node(1, vec![0.0, 1.0], vec![vec![0]]),
            ],
            entry_point: Some(0),
            m: 2,
            ef: 2,
            max_layer: 0,
        };

        insert(&mut index, 0, vec![0.0, 0.9]);

        assert_eq!(index.nodes[2].neighbours[0], vec![1, 0]);
        assert_eq!(index.nodes[1].neighbours[0], vec![0, 2]);
        assert_eq!(index.nodes[0].neighbours[0], vec![1, 2]);
    }

    #[test]
    fn taller_inserted_node_becomes_entry_point_and_updates_max_layer() {
        let mut index = HnswIndex {
            nodes: vec![node(0, vec![1.0, 0.0], vec![Vec::new()])],
            entry_point: Some(0),
            m: 2,
            ef: 2,
            max_layer: 0,
        };
        let taller_node = node(1, vec![0.9, 0.1], vec![Vec::new(), Vec::new(), Vec::new()]);

        insert_node(&mut index, taller_node);

        assert_eq!(index.entry_point, Some(1));
        assert_eq!(index.max_layer, 2);
        assert_eq!(index.nodes[1].neighbours.len(), 3);
    }

    #[test]
    fn search_layer_returns_up_to_ef_best_reachable_nodes() {
        let index = HnswIndex {
            nodes: vec![
                node(0, vec![1.0, 0.0], vec![vec![1, 2]]),
                node(1, vec![0.0, 1.0], vec![vec![0]]),
                node(2, vec![0.9, 0.1], vec![vec![0]]),
            ],
            entry_point: Some(0),
            m: 2,
            ef: 2,
            max_layer: 3,
        };

        let results = search_layer(&index, &[1.0, 0.0], 0, index.ef, 0);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[1].0, 2);
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn search_descends_upper_layers_then_returns_top_k_from_layer_zero() {
        let index = HnswIndex {
            nodes: vec![
                node(0, vec![1.0, 0.0], vec![vec![1], vec![2]]),
                node(1, vec![0.0, 1.0], vec![vec![0, 2]]),
                node(2, vec![0.9, 0.1], vec![vec![0, 1], vec![0]]),
            ],
            entry_point: Some(0),
            m: 2,
            ef: 3,
            max_layer: 1,
        };

        let results = search(&index, &[0.8, 0.2], index.ef, 2);

        assert_eq!(results, vec![2, 0]);
    }

    #[test]
    fn search_greedy_stops_when_no_neighbour_is_more_similar() {
        let index = HnswIndex {
            max_layer: 3,
            nodes: vec![
                node(0, vec![1.0, 0.0], vec![vec![1]]),
                node(1, vec![0.0, 1.0], vec![vec![0]]),
            ],
            entry_point: Some(0),
            m: 2,
            ef: 2,
        };
        let query_node = node(2, vec![0.9, 0.1], Vec::new());

        assert_eq!(search_greedy(&index, &query_node.embedding, 0, 0), 0);
    }

    #[test]
    fn calculate_most_similar_neighbours_sorts_by_descending_similarity() {
        let index = HnswIndex {
            max_layer: 3,
            nodes: vec![
                node(0, vec![1.0, 0.0], vec![vec![1, 2]]),
                node(1, vec![0.0, 1.0], vec![vec![0]]),
                node(2, vec![0.8, 0.2], vec![vec![0]]),
            ],
            entry_point: Some(0),
            m: 2,
            ef: 2,
        };
        let query = node(3, vec![1.0, 0.0], Vec::new());

        let neighbours =
            calculate_most_similiar_neighbours(&index.nodes[0], &query.embedding, &index, 0);

        assert_eq!(neighbours.len(), 2);
        assert_eq!(neighbours[0].0, 2);
        assert_eq!(neighbours[1].0, 1);
        assert!(neighbours[0].1 > neighbours[1].1);
    }

    #[test]
    fn prune_keeps_only_the_m_most_similar_neighbours() {
        let mut index = HnswIndex {
            nodes: vec![
                node(0, vec![1.0, 0.0], vec![vec![1, 2, 3]]),
                node(1, vec![1.0, 0.0], vec![vec![0]]),
                node(2, vec![0.0, 1.0], vec![vec![0]]),
                node(3, vec![0.8, 0.2], vec![vec![0]]),
            ],
            max_layer: 3,
            entry_point: Some(0),
            m: 2,
            ef: 3,
        };

        super::prune(&mut index, 0, 0);

        assert_eq!(index.nodes[0].neighbours[0], vec![1, 3]);
        assert!(!index.nodes[2].neighbours[0].contains(&0));
    }

    #[test]
    fn insertion_keeps_every_node_within_max_degree() {
        let mut index = empty_index();

        for i in 0..20 {
            insert(&mut index, 0, vec![1.0, i as f32 / 100.0]);
        }

        assert!(
            index
                .nodes
                .iter()
                .all(|node| node.neighbours[0].len() <= index.m)
        );
    }
}
