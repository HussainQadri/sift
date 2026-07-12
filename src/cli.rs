use std::path::PathBuf;

use clap::Parser;
use clap::Subcommand;
#[derive(Subcommand, Debug)]
pub enum Commands {
    Ingest {
        path: Option<std::path::PathBuf>,
    },
    Benchmark {
        #[arg(long)]
        queries: PathBuf,
        #[arg(long, default_value_t = 10)]
        top: usize,

        #[arg(long, default_value_t = 50)]
        runs: usize,
    },
}

#[derive(Parser)]
#[command(arg_required_else_help = true)]
pub struct Cli {
    pub keywords: Option<String>,
    #[arg(long, default_value_t = 3)]
    pub top: usize,

    #[arg(long)]
    pub hnsw: bool,

    #[command(subcommand)]
    pub commands: Option<Commands>,
}
