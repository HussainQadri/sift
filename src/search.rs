use crate::cli;
use crate::cli_output;
use crate::embeddings_generator;
use crate::hnsw::HnswIndex;
use crate::index;
use crate::similarity::cosine_similarity;
use std::collections::HashMap;
use std::path::Path;

pub fn query_search(args: cli::Cli) -> anyhow::Result<()> {
    let keywords = match args.keywords {
        Some(query) => query,
        None => anyhow::bail!("Please enter a query."),
    };
    let top_k_results = args.top;
    let query = embeddings_generator::create_query_embedding(&keywords)?;
    let loaded_indexed_functions = index::load_index()?;
    let mut index = HnswIndex::new(8, 32);
    println!("HNSW OUTPUT");
    for indexed_function in &loaded_indexed_functions {
        index.insert(indexed_function.id, indexed_function.embedding.clone());
    }

    let records_by_id: HashMap<usize, &index::IndexedFunction> = loaded_indexed_functions
        .iter()
        .map(|record| (record.id, record))
        .collect();
    // These are record_ids from the JSON not the internal HNSW index
    let result_ids = index.search(&query);
    for record_id in result_ids {
        let indexed_function = match records_by_id.get(&record_id) {
            Some(value) => value,
            None => {
                eprintln!("HNSW returned unknown record id: {record_id}");
                continue;
            }
        };
        let score = cosine_similarity(&query, &indexed_function.embedding);

        println!("{:.3} {}: ", score, indexed_function.path);

        let extension = Path::new(&indexed_function.path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        print!("\x1b[32m{}:\x1b[0m ", indexed_function.line_number);
        cli_output::print_highlighted(&indexed_function.header, extension);
        println!("\n");
    }
    println!("BRUTE FORCE OUTPUT");
    let mut result: Vec<(index::IndexedFunction, f32)> = loaded_indexed_functions
        .into_iter()
        .map(|indexed_function| {
            let score = cosine_similarity(&query, &indexed_function.embedding);

            (indexed_function, score)
        })
        .collect();
    result.sort_by(|a, b| b.1.total_cmp(&a.1));

    for (indexed_function, score) in result.iter().take(top_k_results) {
        println!("{:.3} {}: ", score, indexed_function.path);
        let extension = Path::new(&indexed_function.path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        print!("\x1b[32m{}:\x1b[0m ", indexed_function.line_number);
        cli_output::print_highlighted(&indexed_function.header, extension);
        println!("\n");
    }
    Ok(())
}
