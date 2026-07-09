use crate::{benchmark::BenchmarkConfig, cli::Commands, search::query_search};
mod benchmark;
mod search;
use clap::Parser;
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

            let all_indexed_functions = ingest::ingest_directory(&mut model, &target_path)?;
            index::save_index(&all_indexed_functions)?;
        }

        Some(Commands::Benchmark {
            queries,
            queries_file,
            top_k,
            limit,
            m,
            ef,
        }) => benchmark::run(BenchmarkConfig {
            queries,
            queries_file,
            top_k,
            limit,
            m,
            ef,
        })?,

        None => {
            query_search(args)?;
        }
    }

    Ok(())
}
