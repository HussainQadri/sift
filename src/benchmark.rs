use std::fs;
use std::path::Path;

pub fn run_benchmark(queries: &Path, top: usize, runs: usize) -> anyhow::Result<()> {
    let queries_vec = read_queries_file(queries)?;
    println!(
        "Loaded {} queries; top={top}, runs={runs}",
        queries_vec.len()
    );

    Ok(())
}

pub fn read_queries_file(queries: &Path) -> anyhow::Result<Vec<String>> {
    let file_contents = fs::read_to_string(queries)?;
    // The queries text file contains individual queries on each line, we must parse that here and
    // collect into a vector
    let individual_queries = file_contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(String::from)
        .collect();
    Ok(individual_queries)
}
