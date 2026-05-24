use crate::embeddings_generator;
use crate::treesitter_parse::ExtractedFunction;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Serialize, Deserialize)]
pub struct IndexedFunction {
    pub(crate) path: String,
    pub(crate) header: String,
    pub(crate) source: String,
    pub(crate) embedding: Vec<f32>,
}

pub fn create_indexed_functions(
    functions: Vec<ExtractedFunction>,
    path: &Path,
) -> anyhow::Result<Vec<IndexedFunction>> {
    let texts: Vec<&String> = functions
        .iter()
        .map(|function_struct| &function_struct.source)
        .collect();

    let embeddings = embeddings_generator::create_function_embedding(texts)?;

    let indexed_functions = functions
        .into_iter()
        .zip(embeddings)
        .map(|(function, embedding)| IndexedFunction {
            path: path.display().to_string(),
            header: function.header,
            source: function.source,
            embedding,
        })
        .collect();
    Ok(indexed_functions)
}

pub fn save_index(indexed_functions: &[IndexedFunction]) -> anyhow::Result<()> {
    fs::create_dir_all(".sift")?;
    let json = serde_json::to_string_pretty(indexed_functions)?;
    fs::write(".sift/index.json", json)?;
    Ok(())
}

pub fn load_index() -> anyhow::Result<Vec<IndexedFunction>> {
    let json = fs::read_to_string(".sift/index.json")?;

    let indexed_functions: Vec<IndexedFunction> = serde_json::from_str(&json)?;

    Ok(indexed_functions)
}
