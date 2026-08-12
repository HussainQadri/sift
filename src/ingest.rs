use crate::embeddings_generator;
use crate::hnsw;
use crate::index;
use crate::index::IndexedFunction;
use crate::language_specs;
use crate::treesitter_parse;
use ignore::Walk;
use model2vec_rs::model::StaticModel;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs;
use std::time::Instant;

const EMBEDDING_BATCH_SIZE: usize = 1024;

struct PendingFunction {
    path: String,
    header: String,
    source: String,
    line_number: usize,
}

struct EmbeddedFunction {
    pending_function: PendingFunction,
    embedding: Vec<f32>,
}

pub struct IngestOutput {
    pub indexed_functions: Vec<index::IndexedFunction>,
    pub hnsw_index: index::PersistedHnswIndex,
}
pub fn ingest_directory(path: &std::path::PathBuf) -> anyhow::Result<IngestOutput> {
    let mut all_indexed_functions = Vec::new();
    let mut hnsw_index = hnsw::HnswIndex::new(32, 256);

    // Read files in parallel and return a Vec of Vecs containing pending functions
    let discovery_started = Instant::now();
    let pending_by_file: Vec<Vec<PendingFunction>> = Walk::new(path)
        .par_bridge() // Convert iterator from Walk into a parallel iterator
        .map(|result| -> anyhow::Result<Vec<PendingFunction>> {
            // If we run into any errors we stop ingesting to prevent a malformed index from being
            // created
            let entry = result
                .map_err(|error| anyhow::anyhow!("failed to walk {}: {error}", path.display()))?;
            let file_path = entry.path();
            let spec = match language_specs::spec_for_file(file_path) {
                Ok(spec) => spec,
                Err(_) => return Ok(Vec::new()),
            };
            let source_code = fs::read_to_string(file_path)?;
            let tree = treesitter_parse::generate_tree_from_source(&spec, &source_code)?;
            let functions =
                treesitter_parse::extract_functions(tree.root_node(), &source_code, &spec)?;

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
    let mut pending_functions: Vec<PendingFunction> =
        pending_by_file.into_iter().flatten().collect();

    // TODO: This timing code doesn't belong in this function; after we collect into
    // Vec<Vec<PendingFunction>> the function should have ended - refactor.
    let discovery_time_elapsed = discovery_started.elapsed();
    let function_count = pending_functions.len();

    let unique_body_count: usize = pending_functions
        .iter()
        .map(|function| function.source.as_str())
        .collect::<HashSet<_>>()
        .len();

    let repeated_body_count = function_count - unique_body_count;

    eprintln!(
        "Discovered {function_count} functions: {unique_body_count} unique bodies, \
       {repeated_body_count} repeated bodies in {:.2}s",
        discovery_time_elapsed.as_secs_f64()
    );

    let model = embeddings_generator::create_embedding_model()?;
    let embedded_functions = embed_pending_functions(&model, &mut pending_functions)?;

    index_embedded_functions(
        &mut hnsw_index,
        embedded_functions,
        &mut all_indexed_functions,
    );

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
    model: &StaticModel,
    pending_function_list: &mut Vec<PendingFunction>,
) -> anyhow::Result<Vec<EmbeddedFunction>> {
    if pending_function_list.is_empty() {
        return Ok(Vec::new());
    }

    let mut all_embedded_functions = Vec::with_capacity(pending_function_list.len());

    pending_function_list.sort_by(|a, b| {
        a.source
            .len()
            .cmp(&b.source.len())
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.line_number.cmp(&b.line_number))
    });

    let mut batch = Vec::with_capacity(EMBEDDING_BATCH_SIZE);
    for pending_function in pending_function_list.drain(..) {
        batch.push(pending_function);

        if batch.len() >= EMBEDDING_BATCH_SIZE {
            let embedded_batch = embed_pending_batch(model, &mut batch)?;
            all_embedded_functions.extend(embedded_batch);
        }
    }

    all_embedded_functions.extend(embed_pending_batch(model, &mut batch)?);
    Ok(all_embedded_functions)
}

fn embed_pending_batch(
    model: &StaticModel,
    batch: &mut Vec<PendingFunction>,
) -> anyhow::Result<Vec<EmbeddedFunction>> {
    if batch.is_empty() {
        return Ok(Vec::new());
    }

    let texts = batch
        .iter()
        .map(|pending_function| &pending_function.source)
        .collect();
    let embeddings = embeddings_generator::create_function_embedding(model, texts)?;
    let embedded_functions = batch
        .drain(..)
        .zip(embeddings)
        .map(|(pending_function, embedding)| EmbeddedFunction {
            pending_function,
            embedding,
        })
        .collect();

    Ok(embedded_functions)
}

fn index_embedded_functions(
    index: &mut hnsw::HnswIndex,
    embedded_functions: Vec<EmbeddedFunction>,
    all_indexed_functions: &mut Vec<IndexedFunction>,
) {
    for embedded_function in embedded_functions {
        // The record_id, which is different to the internal HNSW id is just the position of the
        // function in the array.
        let record_id = all_indexed_functions.len();
        let pending = embedded_function.pending_function;
        let indexed_function = IndexedFunction {
            path: pending.path,
            header: pending.header,
            source: pending.source,
            line_number: pending.line_number,
            embedding: embedded_function.embedding,
            record_id,
        };

        index.insert(record_id, indexed_function.embedding.clone());
        all_indexed_functions.push(indexed_function);
    }
}
