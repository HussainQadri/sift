use crate::{
    embeddings_generator,
    index::{self, IndexedFunction},
    search,
};
use std::{collections::HashSet, fs, path::Path, time::Instant};

pub fn run_benchmark(queries: &Path, top: usize, runs: usize) -> anyhow::Result<()> {
    // Load queries
    let queries_vec = read_queries_file(queries)?;
    // For each query, run that query 'run' times with brute force and hnsw whilst timing both
    // Calculate recall once

    let mut brute_force_timings = Vec::new();
    let mut hnsw_timings = Vec::new();
    let runtime_hnsw_index = search::load_runtime_index()?;
    let loaded_indexed_functions = index::load_index()?;
    let mut total_recall_score: f32 = 0.0;
    let query_count = queries_vec.len() as f32;
    for query in queries_vec {
        let query_embedding = embeddings_generator::create_query_embedding(&query)?;
        for run in 0..runs {
            // TODO: Clean this up, too much repeated code
            let brute_force_start = Instant::now();
            let brute_force_result =
                search::search_using_brute_force(&query_embedding, &loaded_indexed_functions, top)?;
            let brute_force_elapsed = brute_force_start.elapsed();

            let brute_force_milliseconds = brute_force_elapsed.as_secs_f64() * 1000.0;
            println!(
                "Brute force time: {}, run: {}",
                brute_force_milliseconds,
                run + 1
            );
            brute_force_timings.push(brute_force_milliseconds);

            let hnsw_start = Instant::now();
            let hnsw_result = search::search_using_hnsw(
                &runtime_hnsw_index,
                &query_embedding,
                &loaded_indexed_functions,
                top,
            )?;
            let hnsw_elapsed = hnsw_start.elapsed();

            let hnsw_milliseconds = hnsw_elapsed.as_secs_f64() * 1000.0;
            println!("HNSW time: {}, run: {}", hnsw_milliseconds, run + 1);
            hnsw_timings.push(hnsw_milliseconds);
            if run == 0 {
                let recall_score = recall(&brute_force_result, &hnsw_result);
                total_recall_score += recall_score;
            }
        }
    }
    let average_recall_score = total_recall_score / query_count;
    println!("Average recall: {}", average_recall_score);
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

pub fn recall(
    brute_force_results: &[(&IndexedFunction, f32)],
    hnsw_results: &[(&IndexedFunction, f32)],
) -> f32 {
    let brute_force_ids: HashSet<usize> = brute_force_results
        .iter()
        .map(|(indexed_function, _score)| indexed_function.record_id)
        .collect();
    let hnsw_ids: HashSet<usize> = hnsw_results
        .iter()
        .map(|(indexed_function, _score)| indexed_function.record_id)
        .collect();
    let common = brute_force_ids.intersection(&hnsw_ids);
    let overlap = common.count();

    // Dividing by brute_force_results.len() and not top_k flag because we may have less than top_k
    overlap as f32 / brute_force_results.len() as f32
}
