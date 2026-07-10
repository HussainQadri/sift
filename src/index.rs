use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
use tempfile::tempfile;

const INDEX_PATH: &str = ".sift-index/index.json";
const HNSW_INDEX_PATH: &str = ".sift-index/hnsw.bin";

#[derive(Debug, Serialize, Deserialize)]
pub struct IndexedFunction {
    pub(crate) path: String,
    pub(crate) header: String,
    pub(crate) source: String,
    pub(crate) line_number: usize,
    pub(crate) embedding: Vec<f32>,
    pub(crate) record_id: usize,
}

#[derive(PartialEq, Debug, Serialize, Deserialize)]
pub struct PersistedHnswIndex {
    pub(crate) nodes: Vec<PersistedHnswNode>,
    pub(crate) entry_point: Option<usize>,
    pub(crate) max_layer: usize,
    pub(crate) m: usize,
    pub(crate) ef: usize,
}

#[derive(PartialEq, Debug, Serialize, Deserialize)]
pub struct PersistedHnswNode {
    pub(crate) record_id: usize,
    pub(crate) neighbours: Vec<Vec<usize>>,
}

pub fn save_index(indexed_functions: &[IndexedFunction]) -> anyhow::Result<()> {
    save_index_at(indexed_functions, Path::new(INDEX_PATH))
}

pub fn save_hnsw_index_at(
    persisted_hnsw_graph: &PersistedHnswIndex,
    hnsw_path: &Path,
) -> anyhow::Result<()> {
    if let Some(parent) = hnsw_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = postcard::to_allocvec(&persisted_hnsw_graph)?;
    fs::write(hnsw_path, bytes)?;
    Ok(())
}

pub fn save_hnsw_index(persisted_hnsw_index: &PersistedHnswIndex) -> anyhow::Result<()> {
    save_hnsw_index_at(persisted_hnsw_index, Path::new(HNSW_INDEX_PATH))?;
    Ok(())
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
            record_id: 0,
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

#[test]

fn hnsw_index_round_trips() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("hnsw.bin");
    let original_hnsw_index = PersistedHnswIndex {
        nodes: vec![
            PersistedHnswNode {
                record_id: 10,
                neighbours: vec![vec![1], vec![]],
            },
            PersistedHnswNode {
                record_id: 20,
                neighbours: vec![vec![0]],
            },
        ],
        ef: 8,
        entry_point: Some(0),
        m: 32,
        max_layer: 1,
    };

    save_hnsw_index_at(&original_hnsw_index, &path).unwrap();
    let bytes = fs::read(path).unwrap();
    let loaded: PersistedHnswIndex = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(loaded, original_hnsw_index);
}
