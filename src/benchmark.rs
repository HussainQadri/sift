use crate::{
    embeddings_generator,
    index::{self, IndexedFunction},
    search,
};
use serde::Deserialize;
use std::{collections::HashSet, fs, path::Path, time::Instant};

#[derive(Debug, Deserialize)]
struct JudgedQuery {
    query: String,
    relevant: Vec<RelevantFunction>,
}

#[derive(Debug, Deserialize)]
struct RelevantFunction {
    path: String,
    header: String,
    relevance: u8,
}

fn dcg(relevances: &[u8]) -> f64 {
    let mut dcg_sum = 0.0;
    for (i, relevance) in relevances.iter().copied().enumerate() {
        let gain = 2_f64.powi(relevance as i32) - 1.0;
        let rank = i + 1;
        let discount = ((rank + 1) as f64).log2();
        let contribution = gain / discount;
        dcg_sum += contribution;
    }
    dcg_sum
}

fn ndcg(actual_relevances: &[u8], judged_relevances: &[u8]) -> f64 {
    let dcg_val = dcg(actual_relevances);
    let mut judged_relevances_sorted = Vec::new();
    for relevance in judged_relevances.iter().copied() {
        judged_relevances_sorted.push(relevance);
    }

    judged_relevances_sorted.sort_by(|a, b| b.cmp(a));
    judged_relevances_sorted.truncate(actual_relevances.len());
    let idcg_val = dcg(&judged_relevances_sorted);
    if idcg_val == 0.0 {
        return 0.0;
    }
    dcg_val / idcg_val
}

fn parse_judgements(contents: &str) -> anyhow::Result<Vec<JudgedQuery>> {
    let judgements = serde_json::from_str::<Vec<JudgedQuery>>(contents)?;

    if judgements.is_empty() {
        anyhow::bail!("judgements contains no queries")
    }

    for judgement in &judgements {
        if judgement.query.trim().is_empty() {
            anyhow::bail!("judgement query is empty");
        }
        if judgement.relevant.is_empty() {
            anyhow::bail!("relevant functions for judgement is empty");
        }

        let function_relevance_valid = judgement
            .relevant
            .iter()
            .all(|function| (1..=3).contains(&function.relevance));

        if !function_relevance_valid {
            anyhow::bail!(
                "function relevance in this judgement is not between 1 and 3 (inclusive)"
            );
        }
    }
    Ok(judgements)
}

fn read_judgements(path: &Path) -> anyhow::Result<Vec<JudgedQuery>> {
    let contents = fs::read_to_string(path)?;
    parse_judgements(&contents)
}

fn relevance_for_result(result: &IndexedFunction, relevant_functions: &[RelevantFunction]) -> u8 {
    for relevant_function in relevant_functions {
        if relevant_function.header == result.header
            && Path::new(&result.path).ends_with(Path::new(&relevant_function.path))
        {
            return relevant_function.relevance;
        }
    }

    0
}

pub fn run_evaluation(judgements: &Path, top_k: usize) -> anyhow::Result<f64> {
    if top_k == 0 {
        anyhow::bail!("Top k must be greater than 0");
    }
    let judged_query_vec = read_judgements(judgements)?;
    let loaded_indexed_functions = index::load_index()?;

    if loaded_indexed_functions.is_empty() {
        anyhow::bail!("The index is empty, please run `sift ingest` first");
    }
    let model = embeddings_generator::create_embedding_model()?;

    let mut ndcg_total = 0.0;
    let query_count = judged_query_vec.len();
    for judgement in judged_query_vec {
        let judgement_query_embedding =
            embeddings_generator::create_query_embedding(&model, &judgement.query)?;
        let results = search::search_using_brute_force(
            &judgement_query_embedding,
            &loaded_indexed_functions,
            top_k,
        )?;
        let actual_grades = results
            .iter()
            .map(|(function, _)| relevance_for_result(function, &judgement.relevant))
            .collect::<Vec<u8>>();

        let ideal_grades = judgement
            .relevant
            .iter()
            .map(|relevant_function| relevant_function.relevance)
            .collect::<Vec<u8>>();
        ndcg_total += ndcg(&actual_grades, &ideal_grades)
    }

    Ok(ndcg_total / query_count as f64)
}
pub fn run_benchmark(queries: &Path, top: usize, runs: usize) -> anyhow::Result<()> {
    // Load queries
    let queries_vec = read_queries_file(queries)?;
    if queries_vec.is_empty() {
        anyhow::bail!("Benchmark query file contains no queries");
    }

    if runs == 0 {
        anyhow::bail!("Benchmark runs must be greater than zero");
    }

    if top == 0 {
        anyhow::bail!("Benchmark top must be greater than zero");
    }

    let loaded_indexed_functions = index::load_index()?;

    if loaded_indexed_functions.is_empty() {
        anyhow::bail!("The index is empty, please run `sift ingest` first");
    }

    let model = embeddings_generator::create_embedding_model()?;

    // For each query, run that query 'run' times with brute force and hnsw whilst timing both
    // Calculate recall once
    let mut brute_force_timings = Vec::new();
    let mut hnsw_timings = Vec::new();
    let runtime_hnsw_index = search::load_runtime_index()?;
    let mut total_recall_score: f32 = 0.0;
    let query_count = queries_vec.len() as f32;
    for query in queries_vec {
        let query_embedding = embeddings_generator::create_query_embedding(&model, &query)?;
        for run in 0..runs {
            // TODO: Clean this up, too much repeated code
            let brute_force_start = Instant::now();
            let brute_force_result =
                search::search_using_brute_force(&query_embedding, &loaded_indexed_functions, top)?;
            let brute_force_elapsed = brute_force_start.elapsed();

            let brute_force_milliseconds = brute_force_elapsed.as_secs_f64() * 1000.0;
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
            hnsw_timings.push(hnsw_milliseconds);
            if run == 0 {
                let recall_score = recall(&brute_force_result, &hnsw_result);
                total_recall_score += recall_score;
            }
        }
    }
    let average_recall_score = total_recall_score / query_count;

    let brute_median = median(&mut brute_force_timings).unwrap();
    let hnsw_median = median(&mut hnsw_timings).unwrap();
    println!("Average recall: {}", average_recall_score);
    println!("HNSW median time: {}", hnsw_median);
    println!("Brute force median time: {}", brute_median);

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

fn median(timings: &mut [f64]) -> Option<f64> {
    if timings.is_empty() {
        return None;
    }

    timings.sort_by(f64::total_cmp);
    let middle = timings.len() / 2;
    if timings.len().is_multiple_of(2) {
        Some((timings[middle - 1] + timings[middle]) / 2.0)
    } else {
        Some(timings[middle])
    }
}

#[cfg(test)]
mod tests {
    use crate::index::IndexedFunction;

    use super::{RelevantFunction, dcg, ndcg, parse_judgements, relevance_for_result};

    #[test]
    fn parses_labelled_queries() {
        let contents = r#"
        [
          {
            "query": "walk directories in parallel",
            "relevant": [
              {
                "path": "crates/ignore/src/walk.rs",
                "header": "pub fn build_parallel(self)",
                "relevance": 3
              },
              {
                "path": "crates/ignore/src/walk.rs",
                "header": "fn run(self)",
                "relevance": 2
              }
            ]
          }
        ]
        "#;

        let queries = parse_judgements(contents).unwrap();

        assert_eq!(queries.len(), 1);
        assert_eq!(queries[0].query, "walk directories in parallel");
        assert_eq!(queries[0].relevant.len(), 2);

        assert_eq!(queries[0].relevant[0].path, "crates/ignore/src/walk.rs");
        assert_eq!(queries[0].relevant[0].header, "pub fn build_parallel(self)");
        assert_eq!(queries[0].relevant[0].relevance, 3);

        assert_eq!(queries[0].relevant[1].path, "crates/ignore/src/walk.rs");
        assert_eq!(queries[0].relevant[1].header, "fn run(self)");
        assert_eq!(queries[0].relevant[1].relevance, 2);
    }

    #[test]

    fn invalid_relevance_is_rejected() {
        let contents = r#"
        [
          {
            "query": "walk directories in parallel",
            "relevant": [
              {
                "path": "crates/ignore/src/walk.rs",
                "header": "pub fn build_parallel(self)",
                "relevance": 4
              },
              {
                "path": "crates/ignore/src/walk.rs",
                "header": "fn run(self)",
                "relevance": 2
              }
            ]
          }
        ]
        "#;

        let error = parse_judgements(contents).unwrap_err();
        assert!(error.to_string().contains("between 1 and 3"));
    }

    #[test]
    fn dcg_produces_correct_values() {
        assert_eq!(dcg(&[3]), 7.0);
        let dcg_output = dcg(&[3, 2, 0]);
        let expected = 8.892789;
        assert!((dcg_output - expected).abs() < 0.000001);
        assert_eq!(dcg(&[]), 0.0);
    }

    #[test]
    fn ndcg_normalizes_against_the_ideal_ranking() {
        let perfect = ndcg(&[3, 2], &[2, 3]);
        assert!((perfect - 1.0).abs() < 0.000001);

        let missed_best_result = ndcg(&[2, 0], &[3, 2]);
        let expected = dcg(&[2, 0]) / dcg(&[3, 2]);
        assert!((missed_best_result - expected).abs() < 0.000001);

        assert_eq!(ndcg(&[0, 0], &[3, 2]), 0.0);
    }

    #[test]

    fn relevance_for_result_requires_matching_header_and_path() {
        let test_indexed_function = IndexedFunction {
            path: String::from("/home/hussain/Projects/ripgrep/crates/ignore/src/walk.rs"),
            header: String::from("pub fn build_parallel(self)"),
            source: String::from("let test_source = vec![];"),
            line_number: 8,
            embedding: vec![1.2, 3.0, 5.0, 6.0],
            record_id: 0,
        };

        let matching_judgment = RelevantFunction {
            path: String::from("crates/ignore/src/walk.rs"),
            header: String::from("pub fn build_parallel(self)"),
            relevance: 3,
        };
        assert_eq!(
            relevance_for_result(&test_indexed_function, &[matching_judgment]),
            3
        );

        let same_header_wrong_path = RelevantFunction {
            path: String::from("crates/ignore/src/other.rs"),
            header: String::from("pub fn build_parallel(self)"),
            relevance: 3,
        };
        assert_eq!(
            relevance_for_result(&test_indexed_function, &[same_header_wrong_path]),
            0
        );
    }
}
