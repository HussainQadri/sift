use std::path::Path;

use crate::cli;
use crate::cli_output;
use crate::embeddings_generator;
use crate::index;
use crate::similarity::cosine_similarity;

pub fn query_search(args: cli::Cli) -> anyhow::Result<()> {
    let keywords = match args.keywords {
        Some(query) => query,
        None => anyhow::bail!("Please enter a query."),
    };
    let top_k_results = args.top;
    let query = embeddings_generator::create_query_embedding(&keywords)?;
    let loaded_indexed_functions = index::load_index()?;
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
