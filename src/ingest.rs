use crate::embeddings_generator;
use crate::index;
use crate::language_specs;
use crate::treesitter_parse;
use fastembed::TextEmbedding;
use ignore::Walk;
use std::fs;

const EMBEDDING_BATCH_SIZE: usize = 64;

struct PendingFunction {
    path: String,
    header: String,
    source: String,
    line_number: usize,
}
pub fn ingest_directory(
    model: &mut TextEmbedding,
    path: &std::path::PathBuf,
) -> anyhow::Result<Vec<index::IndexedFunction>> {
    let mut all_indexed_functions = Vec::new();
    let mut pending_functions = Vec::new();

    for result in Walk::new(path) {
        let entry = match result {
            Ok(entry) => entry,
            Err(err) => {
                eprintln!("Invalid directory entry: {err}");
                continue;
            }
        };
        let file_path = entry.path();

        let spec = match language_specs::spec_for_file(file_path) {
            Ok(spec) => spec,
            Err(_) => continue,
        };

        let tree = treesitter_parse::generate_tree(file_path, &spec);
        let source_code = fs::read_to_string(file_path)?;
        let functions = treesitter_parse::extract_functions(tree.root_node(), &source_code, &spec);

        for function in functions {
            pending_functions.push(PendingFunction {
                source: function.source,
                header: function.header,
                line_number: function.line_number,
                path: file_path.display().to_string(),
            });
        }
    }

    embed_pending_functions(model, &mut pending_functions, &mut all_indexed_functions)?;
    Ok(all_indexed_functions)
}

fn embed_pending_functions(
    model: &mut TextEmbedding,
    pending_function_list: &mut Vec<PendingFunction>,
    all_indexed_functions: &mut Vec<index::IndexedFunction>,
) -> anyhow::Result<()> {
    if pending_function_list.is_empty() {
        return Ok(());
    }

    pending_function_list.sort_by_key(|pending_function| pending_function.source.len());

    let mut batch = Vec::with_capacity(EMBEDDING_BATCH_SIZE);
    for pending_function in pending_function_list.drain(..) {
        batch.push(pending_function);

        if batch.len() >= EMBEDDING_BATCH_SIZE {
            embed_pending_batch(model, &mut batch, all_indexed_functions)?;
        }
    }

    embed_pending_batch(model, &mut batch, all_indexed_functions)
}

fn embed_pending_batch(
    model: &mut TextEmbedding,
    batch: &mut Vec<PendingFunction>,
    all_indexed_functions: &mut Vec<index::IndexedFunction>,
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
        all_indexed_functions.push(index::IndexedFunction {
            path: pending_function.path,
            header: pending_function.header,
            source: pending_function.source,
            line_number: pending_function.line_number,
            embedding,
            id: start_id + offset,
        })
    }

    Ok(())
}
