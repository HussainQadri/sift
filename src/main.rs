use crate::{cli::Commands, search::query_search};
mod search;
use clap::Parser;
mod benchmark;
mod cli;
mod cli_output;
mod embeddings_generator;
mod hnsw;
mod index;
mod ingest;
mod language_specs;
mod similarity;
mod treesitter_parse;

fn main() -> anyhow::Result<()> {
    let args: cli::Cli = cli::Cli::parse();
    match args.commands {
        Some(Commands::Ingest { path }) => {
            let mut model = embeddings_generator::create_embedding_model()?;

            let target_path = match path {
                Some(p) => p,
                None => std::path::PathBuf::from("."),
            };

            let ingest_output = ingest::ingest_directory(&mut model, &target_path)?;
            if ingest_output.indexed_functions.is_empty() {
                anyhow::bail!(
                    "no indexable functions were found under {}",
                    target_path.display()
                );
            }

            index::save_index(&ingest_output.indexed_functions)?;
            let hnsw_index = ingest_output.hnsw_index;
            index::save_hnsw_index(&hnsw_index)?;
        }

        Some(Commands::Benchmark { queries, top, runs }) => {
            benchmark::run_benchmark(&queries, top, runs)?;
        }

        None => {
            query_search(args)?;
        }
    }

    Ok(())
}
