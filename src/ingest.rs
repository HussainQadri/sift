use crate::embeddings_generator;
use crate::hnsw;
use crate::index;
use crate::language_specs;
use crate::treesitter_parse;
use fastembed::TextEmbedding;
use ignore::Walk;
use rayon::prelude::*;
use std::fs;

const EMBEDDING_BATCH_SIZE: usize = 64;

struct PendingFunction {
    path: String,
    header: String,
    source: String,
    line_number: usize,
}

pub struct IngestOutput {
    pub indexed_functions: Vec<index::IndexedFunction>,
    pub hnsw_index: index::PersistedHnswIndex,
}
pub fn ingest_directory(
    model: &mut TextEmbedding,
    path: &std::path::PathBuf,
) -> anyhow::Result<IngestOutput> {
    let mut all_indexed_functions = Vec::new();
    let mut hnsw_index = hnsw::HnswIndex::new(32, 256);

    // Read files in parallel and return a Vec of Vecs containing pending functions
    let pending_by_file: Vec<Vec<PendingFunction>> = Walk::new(path)
        .par_bridge() // Convert iterator from Walk into a parallel iterator
        .map(|result| -> anyhow::Result<Vec<PendingFunction>> {
            let entry = match result {
                Ok(entry) => entry,
                Err(err) => {
                    eprintln!("Invalid directory entry: {err}");
                    return Ok(Vec::new());
                }
            };
            let file_path = entry.path();
            let spec = match language_specs::spec_for_file(file_path) {
                Ok(spec) => spec,
                Err(_) => return Ok(Vec::new()),
            };
            let source_code = fs::read_to_string(file_path)?;
            let tree = treesitter_parse::generate_tree_from_source(&spec, &source_code);
            let functions =
                treesitter_parse::extract_functions(tree.root_node(), &source_code, &spec);

            // Create an empty vector for a file, go through the functions and push them into this
            // vector. We cannot create a vector for storing PendingFunction outside this map
            // because if multiple threads try to push PendingFunction structs to that vector that
            // could cause corruption. After this map work is done, we will flatten this vec
            let mut pending_functions_for_file = Vec::new();
            for function in functions {
                pending_functions_for_file.push(PendingFunction {
                    source: function.source,
                    header: function.header,
                    line_number: function.line_number,
                    path: file_path.display().to_string(),
                });
            }
            Ok(pending_functions_for_file)
        })
        .collect::<anyhow::Result<Vec<Vec<PendingFunction>>>>()?;

    // Take this Vec<Vec<PendingFunction>> and convert into just Vec<PendingFunction>
    let mut pending_functions = pending_by_file.into_iter().flatten().collect();

    embed_pending_functions(
        model,
        &mut pending_functions,
        &mut all_indexed_functions,
        &mut hnsw_index,
    )?;
    let mut persisted_nodes = Vec::new();
    for node in hnsw_index.nodes {
        persisted_nodes.push(index::PersistedHnswNode {
            neighbours: node.neighbours,
            record_id: node.record_id,
            embedding: node.embedding,
        })
    }

    let persisted = index::PersistedHnswIndex {
        nodes: persisted_nodes,
        entry_point: hnsw_index.entry_point,
        ef: hnsw_index.ef,
        m: hnsw_index.m,
        max_layer: hnsw_index.max_layer,
    };
    Ok(IngestOutput {
        indexed_functions: all_indexed_functions,
        hnsw_index: persisted,
    })
}

fn embed_pending_functions(
    model: &mut TextEmbedding,
    pending_function_list: &mut Vec<PendingFunction>,
    all_indexed_functions: &mut Vec<index::IndexedFunction>,
    hnsw_index: &mut hnsw::HnswIndex,
) -> anyhow::Result<()> {
    if pending_function_list.is_empty() {
        return Ok(());
    }

    pending_function_list.sort_by_key(|pending_function| pending_function.source.len());

    let mut batch = Vec::with_capacity(EMBEDDING_BATCH_SIZE);
    for pending_function in pending_function_list.drain(..) {
        batch.push(pending_function);

        if batch.len() >= EMBEDDING_BATCH_SIZE {
            embed_pending_batch(model, &mut batch, all_indexed_functions, hnsw_index)?;
        }
    }

    embed_pending_batch(model, &mut batch, all_indexed_functions, hnsw_index)
}

fn embed_pending_batch(
    model: &mut TextEmbedding,
    batch: &mut Vec<PendingFunction>,
    all_indexed_functions: &mut Vec<index::IndexedFunction>,
    hnsw_index: &mut hnsw::HnswIndex,
) -> anyhow::Result<()> {
    if batch.is_empty() {
        return Ok(());
    }

    let texts = batch
        .iter()
        .map(|pending_function| &pending_function.source)
        .collect();
    let embeddings = embeddings_generator::create_function_embedding(model, texts)?;
    let start_id = all_indexed_functions.len();

    for (offset, (pending_function, embedding)) in batch.drain(..).zip(embeddings).enumerate() {
        let indexed_function = index::IndexedFunction {
            path: pending_function.path,
            header: pending_function.header,
            source: pending_function.source,
            line_number: pending_function.line_number,
            embedding,
            record_id: start_id + offset,
        };

        hnsw_index.insert(
            indexed_function.record_id,
            indexed_function.embedding.clone(),
        );
        all_indexed_functions.push(indexed_function);
    }

    Ok(())
}
