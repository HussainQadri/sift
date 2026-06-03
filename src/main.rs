use crate::cli::Commands;
use clap::Parser;
mod cli;
mod cli_output;
mod embeddings_generator;
mod index;
mod ingest;
mod language_specs;
mod similarity;
mod treesitter_parse;
use crate::similarity::cosine_similarity;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args = cli::Cli::parse();
    match args.commands {
        Some(Commands::Ingest { path }) => {
            let mut model = embeddings_generator::create_embedding_model()?;

            let target_path = match path {
                Some(p) => p,
                None => std::path::PathBuf::from("."),
            };

            let all_indexed_functions = ingest::ingest_directory(&mut model, &target_path)?;
            index::save_index(&all_indexed_functions)?;
        }

        None => {
            let keywords = args.keywords.expect("Please enter a search query");
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
        }
    }

    Ok(())
}
