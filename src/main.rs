use clap::Parser;
mod index;
use clap::Subcommand;
mod embeddings_generator;
mod language_specs;
mod similarity;
mod treesitter_parse;
use fastembed::TextEmbedding;
use std::fs;

use crate::similarity::cosine_similarity;

#[derive(Subcommand, Debug)]
enum Commands {
    Ingest { path: std::path::PathBuf },
}

#[derive(Parser)]
struct Cli {
    keywords: Option<String>,
    #[command(subcommand)]
    commands: Option<Commands>,
}

fn recursive_ingest_dir(
    model: &mut TextEmbedding,
    path: &std::path::PathBuf,
) -> anyhow::Result<Vec<index::IndexedFunction>> {
    let mut all_indexed_functions = Vec::new();

    for resource_entry_result in fs::read_dir(path)? {
        let entry = resource_entry_result?;
        let file_path = entry.path();

        if file_path.is_dir() {
            let indexed_functions = recursive_ingest_dir(model, &file_path)?;
            all_indexed_functions.extend(indexed_functions);
        }

        let spec = match language_specs::spec_for_file(&file_path) {
            Ok(spec) => spec,
            Err(_) => continue,
        };

        let tree = treesitter_parse::generate_tree(&file_path, &spec);
        let source_code = fs::read_to_string(&file_path)?;
        let functions = treesitter_parse::extract_functions(tree.root_node(), &source_code, &spec);
        let indexed_functions = index::create_indexed_functions(model, functions, &file_path)?;
        all_indexed_functions.extend(indexed_functions);
    }

    Ok(all_indexed_functions)
}
fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    match args.commands {
        Some(Commands::Ingest { path }) => {
            let mut model = embeddings_generator::create_embedding_model()?;
            let all_indexed_functions = recursive_ingest_dir(&mut model, &path)?;
            index::save_index(&all_indexed_functions)?;
        }
        None => {
            let keywords = args.keywords.expect("Please enter a search query");
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
    }

    Ok(())
}
