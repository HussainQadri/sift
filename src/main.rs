use clap::Parser;
use clap::Subcommand;
mod similarity;
mod treesitter_parse;
use std::fs;
use std::path::Path;

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
            let spec = treesitter_parse::spec_for_file(&path)?;
            let tree = treesitter_parse::parser_demo(&path, &spec);
            let source_code = fs::read_to_string(&path).expect("Failed to read source file");
            treesitter_parse::extract_functions(tree.root_node(), &source_code, &spec);
        }
    }

    Ok(())
}
