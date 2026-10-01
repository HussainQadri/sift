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
            // Check if the path provided was empty or not
            let target_path = match path {
                Some(p) => p,
                None => std::path::PathBuf::from("."),
            };

            // Now check if target_path is valid
            let exists = target_path.try_exists()?;
            if !exists {
                anyhow::bail!("path {} does not exist", target_path.display());
            }

            let ingest_output = ingest::ingest_directory(&target_path)?;
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

        Some(Commands::Evaluate { judgements, top }) => {
            let average_ndcg_score = benchmark::run_evaluation(&judgements, top)?;
            println!("Average nDCG@{}: {}", top, average_ndcg_score);
        }

        None => {
            query_search(args)?;
        }
    }

    Ok(())
}
