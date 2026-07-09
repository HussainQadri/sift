use crate::embeddings_generator;
use crate::hnsw::HnswIndex;
use crate::index;
use crate::similarity;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const DEFAULT_TOP_KS: &[usize] = &[1, 5, 10];

pub struct BenchmarkConfig {
    pub queries: Vec<String>,
    pub queries_file: Option<PathBuf>,
    pub top_k: Vec<usize>,
    pub limit: usize,
    pub m: usize,
    pub ef: usize,
}

pub fn run(config: BenchmarkConfig) -> anyhow::Result<()> {
    let records = index::load_index()?;
    if records.is_empty() {
        anyhow::bail!("Index is empty. Run `sift ingest <path>` first.");
    }

    let queries = load_queries(&config, &records)?;
    let top_ks = normalize_top_ks(config.top_k, records.len())?;
    let max_k = *top_ks
        .iter()
        .max()
        .expect("top_k normalization should produce at least one value");

    let build_started = Instant::now();
    let mut hnsw = HnswIndex::new(config.m, config.ef);
    for record in &records {
        hnsw.insert(record.id, record.embedding.clone());
    }
    let build_time = build_started.elapsed();

    let mut model = embeddings_generator::create_embedding_model()?;
    let embed_started = Instant::now();
    let query_embeddings = embeddings_generator::create_query_embeddings(&mut model, &queries)?;
    let embed_time = embed_started.elapsed();

    let mut recall_totals = vec![0.0_f64; top_ks.len()];
    let mut hnsw_total = Duration::ZERO;
    let mut brute_total = Duration::ZERO;

    for query_embedding in &query_embeddings {
        let hnsw_started = Instant::now();
        let hnsw_results = hnsw.search(query_embedding, max_k);
        hnsw_total += hnsw_started.elapsed();

        let brute_started = Instant::now();
        let brute_results = brute_force_top_k(&records, query_embedding, max_k);
        brute_total += brute_started.elapsed();

        for (index, &k) in top_ks.iter().enumerate() {
            recall_totals[index] += recall_at_k(&hnsw_results, &brute_results, k);
        }
    }

    let query_count = query_embeddings.len();

    println!("records: {}", records.len());
    println!("queries: {query_count}");
    println!("hnsw: m={} ef={}", config.m, config.ef);
    println!("build_hnsw: {:.3} ms", millis(build_time));
    println!("embed_queries: {:.3} ms", millis(embed_time));
    println!(
        "avg_hnsw_query: {:.3} ms",
        millis(avg_duration(hnsw_total, query_count))
    );
    println!(
        "avg_brute_query: {:.3} ms",
        millis(avg_duration(brute_total, query_count))
    );

    for (index, &k) in top_ks.iter().enumerate() {
        let recall = recall_totals[index] / query_count as f64;
        println!("recall@{k}: {:.3}", recall);
    }

    Ok(())
}

fn load_queries(
    config: &BenchmarkConfig,
    records: &[index::IndexedFunction],
) -> anyhow::Result<Vec<String>> {
    let mut queries = config.queries.clone();

    if let Some(path) = &config.queries_file {
        let contents = fs::read_to_string(path)?;
        queries.extend(
            contents
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .map(str::to_string),
        );
    }

    if queries.is_empty() {
        queries.extend(
            records
                .iter()
                .map(|record| record.header.trim())
                .filter(|header| !header.is_empty())
                .map(str::to_string),
        );
    }

    if config.limit > 0 {
        queries.truncate(config.limit);
    }

    if queries.is_empty() {
        anyhow::bail!("No benchmark queries available.");
    }

    Ok(queries)
}

fn normalize_top_ks(mut top_ks: Vec<usize>, record_count: usize) -> anyhow::Result<Vec<usize>> {
    if top_ks.is_empty() {
        top_ks = DEFAULT_TOP_KS.to_vec();
    }

    top_ks.sort_unstable();
    top_ks.dedup();
    top_ks.retain(|&k| k > 0);

    if top_ks.is_empty() {
        anyhow::bail!("At least one positive --k value is required.");
    }

    for top_k in &mut top_ks {
        *top_k = (*top_k).min(record_count);
    }
    top_ks.dedup();

    Ok(top_ks)
}

fn brute_force_top_k(
    records: &[index::IndexedFunction],
    query_embedding: &[f32],
    top_k: usize,
) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = records
        .iter()
        .map(|record| {
            (
                record.id,
                similarity::cosine_similarity(query_embedding, &record.embedding),
            )
        })
        .collect();

    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
        .into_iter()
        .take(top_k)
        .map(|(id, _score)| id)
        .collect()
}

fn recall_at_k(hnsw_results: &[usize], brute_results: &[usize], k: usize) -> f64 {
    let effective_k = k.min(brute_results.len());
    if effective_k == 0 {
        return 0.0;
    }

    let brute_ids: HashSet<usize> = brute_results.iter().take(effective_k).copied().collect();
    let hits = hnsw_results
        .iter()
        .take(effective_k)
        .filter(|id| brute_ids.contains(id))
        .count();

    hits as f64 / effective_k as f64
}

fn avg_duration(total: Duration, count: usize) -> Duration {
    if count == 0 {
        return Duration::ZERO;
    }

    Duration::from_secs_f64(total.as_secs_f64() / count as f64)
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::{normalize_top_ks, recall_at_k};

    #[test]
    fn recall_at_k_counts_overlap_with_exact_top_k() {
        let hnsw = vec![1, 3, 8, 5, 9];
        let brute = vec![1, 2, 3, 4, 5];

        assert_eq!(recall_at_k(&hnsw, &brute, 1), 1.0);
        assert_eq!(recall_at_k(&hnsw, &brute, 5), 0.6);
    }

    #[test]
    fn normalize_top_ks_defaults_sorts_dedups_and_caps_to_record_count() {
        assert_eq!(normalize_top_ks(Vec::new(), 7).unwrap(), vec![1, 5, 7]);
        assert_eq!(
            normalize_top_ks(vec![10, 1, 5, 5], 8).unwrap(),
            vec![1, 5, 8]
        );
    }
}
