use clap::Parser;
mod index;
use clap::Subcommand;
mod embeddings_generator;
mod language_specs;
mod similarity;
mod treesitter_parse;
use std::fs;

use crate::similarity::cosine_similarity;

#[derive(Subcommand, Debug)]
enum Commands {
    Search { keywords: String },

    Ingest { path: std::path::PathBuf },
}

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}
fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    match args.commands {
        // big mess here, allow embeddings to persist -> todo
        Commands::Search { keywords } => {
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

            for (indexed_function, score) in result.iter().take(5) {
                println!("{:.3} {}: ", score, indexed_function.path);
                println!(
                    "{}: {}\n",
                    indexed_function.line_number, indexed_function.header
                );
            }
        }

        Commands::Ingest { path } => {
            let mut all_indexed_functions = Vec::new();

            for resource_entry_result in fs::read_dir(&path)? {
                let entry = resource_entry_result?;
                let file_path = entry.path();

                if !file_path.is_file() {
                    continue;
                }

                let spec = match language_specs::spec_for_file(&file_path) {
                    Ok(spec) => spec,
                    Err(_) => continue,
                };

                let tree = treesitter_parse::generate_tree(&file_path, &spec);
                let source_code = fs::read_to_string(&file_path)?;
                let functions =
                    treesitter_parse::extract_functions(tree.root_node(), &source_code, &spec);
                let indexed_functions = index::create_indexed_functions(functions, &file_path)?;
                all_indexed_functions.extend(indexed_functions);
            }

            index::save_index(&all_indexed_functions)?;
        }
    }

    Ok(())
}
