use clap::Parser;
use clap::Subcommand;
mod embeddings_generator;
mod language_specs;
mod similarity;
mod treesitter_parse;
use std::fs;

use crate::similarity::cosine_similarity;

#[derive(Subcommand, Debug)]
enum Commands {
    Search {
        path: std::path::PathBuf,
        keywords: String,
    },
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
        Commands::Search { path, keywords } => {
            let spec = language_specs::spec_for_file(&path)?;
            let tree = treesitter_parse::parser_demo(&path, &spec);
            let source_code = fs::read_to_string(&path).expect("Failed to read source file");
            let headers =
                treesitter_parse::extract_function_headers(tree.root_node(), &source_code, &spec);
            let header_embeddings =
                embeddings_generator::create_function_header_embeddings(headers)?;
            let query_vector = embeddings_generator::create_embedding(&keywords)?;

            let mut header_similarity_vector = Vec::new();
            for embedding_struct in header_embeddings {
                let cosine_result =
                    cosine_similarity(&embedding_struct.header_embedding, &query_vector);
                header_similarity_vector.push((embedding_struct.header, cosine_result));
            }

            header_similarity_vector.sort_by(|a, b| b.1.total_cmp(&a.1));
            for pair in header_similarity_vector {
                println!("{}", pair.0);
            }
        }
    }

    Ok(())
}
