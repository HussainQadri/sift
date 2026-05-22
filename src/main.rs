use clap::Parser;
use clap::Subcommand;
mod embeddings_generator;
mod language_specs;
mod similarity;
mod treesitter_parse;
use std::fs;

#[derive(Subcommand, Debug)]
enum Commands {
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
        Commands::Ingest { path } => {
            let spec = language_specs::spec_for_file(&path)?;
            let tree = treesitter_parse::parser_demo(&path, &spec);
            let source_code = fs::read_to_string(&path).expect("Failed to read source file");
            let headers =
                treesitter_parse::extract_function_headers(tree.root_node(), &source_code, &spec);

            for header in headers {
                println!("{}", header);
            }
            treesitter_parse::extract_functions(tree.root_node(), &source_code, &spec);
        }
    }

    Ok(())
}
