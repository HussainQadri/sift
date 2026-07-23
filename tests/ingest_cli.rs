use std::fs;

#[test]

fn invalid_target_path_does_not_change_existing_index() -> anyhow::Result<()> {
    // Create a temporary project directory with valid paths
    let temp_dir = tempfile::tempdir()?;

    let index_dir = temp_dir.path().join(".sift-index");
    fs::create_dir_all(&index_dir)?;
    let index_path = index_dir.join("index.json");
    let hnsw_path = index_dir.join("hnsw.bin");

    let original_index = b"existing index data";
    let original_hnsw = b"existing hnsw data";
    fs::write(&index_path, original_index)?;
    fs::write(&hnsw_path, original_hnsw)?;

    // We will simply not create this directory
    let invalid_path = temp_dir.path().join("does-not-exist");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_sift"))
        .current_dir(temp_dir.path())
        .arg("ingest")
        .arg(&invalid_path)
        .output()?;

    assert!(!output.status.success());
    let error_bytes = output.stderr;
    let captured_error_string = String::from_utf8(error_bytes)?;
    // Specifically check we failed this test because the path doesn't exist, not because of some
    // other failure like parsing, etc.
    assert!(captured_error_string.contains("does not exist"));

    let index_after = fs::read(&index_path)?;
    assert_eq!(index_after, original_index);

    let hnsw_after = fs::read(&hnsw_path)?;
    assert_eq!(hnsw_after, original_hnsw);
    Ok(())
}
