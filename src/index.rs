use crate::embeddings_generator;
use crate::treesitter_parse::ExtractedFunction;
use fastembed::TextEmbedding;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

const INDEX_PATH: &str = ".sift-index/index.json";

#[derive(Debug, Serialize, Deserialize)]
pub struct IndexedFunction {
    pub(crate) path: String,
    pub(crate) header: String,
    pub(crate) source: String,
    pub(crate) line_number: usize,
    pub(crate) embedding: Vec<f32>,
    pub(crate) id: usize,
}

pub fn create_indexed_functions(
    model: &mut TextEmbedding,
    functions: Vec<ExtractedFunction>,
    path: &Path,
    start_id: usize,
) -> anyhow::Result<Vec<IndexedFunction>> {
    let texts: Vec<&String> = functions
        .iter()
        .map(|function_struct| &function_struct.source)
        .collect();

    let embeddings = embeddings_generator::create_function_embedding(model, texts)?;

    let indexed_functions = functions
        .into_iter()
        .zip(embeddings)
        .enumerate()
        .map(|(offset, (function, embedding))| IndexedFunction {
            path: path.display().to_string(),
            header: function.header,
            source: function.source,
            line_number: function.line_number,
            embedding,
            id: start_id + offset,
        })
        .collect();
    Ok(indexed_functions)
}

pub fn save_index(indexed_functions: &[IndexedFunction]) -> anyhow::Result<()> {
    save_index_at(indexed_functions, Path::new(INDEX_PATH))
}

fn save_index_at(indexed_functions: &[IndexedFunction], index_path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = index_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(indexed_functions)?;
    fs::write(index_path, json)?;
    Ok(())
}

pub fn load_index() -> anyhow::Result<Vec<IndexedFunction>> {
    load_index_at(Path::new(INDEX_PATH))
}

fn load_index_at(index_path: &Path) -> anyhow::Result<Vec<IndexedFunction>> {
    let json = fs::read_to_string(index_path)?;

    let indexed_functions: Vec<IndexedFunction> = serde_json::from_str(&json)?;

    Ok(indexed_functions)
}

#[cfg(test)]
mod tests {
    use super::{IndexedFunction, load_index_at, save_index_at};
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn saved_index_round_trips_function_metadata_and_embedding() {
        let test_dir = std::env::temp_dir().join(format!(
            "sift-index-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let index_path = test_dir.join(".sift/index.json");
        let records = vec![IndexedFunction {
            path: "src/index.rs".to_string(),
            header: "pub fn load_index()".to_string(),
            source: "pub fn load_index() {}".to_string(),
            line_number: 47,
            embedding: vec![0.25, -0.5, 1.0],
            id: 0,
        }];

        save_index_at(&records, &index_path).unwrap();
        let loaded = load_index_at(&index_path).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].path, records[0].path);
        assert_eq!(loaded[0].header, records[0].header);
        assert_eq!(loaded[0].source, records[0].source);
        assert_eq!(loaded[0].line_number, records[0].line_number);
        assert_eq!(loaded[0].embedding, records[0].embedding);

        fs::remove_dir_all(test_dir).unwrap();
    }
}
