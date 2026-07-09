use clap::Parser;
use clap::Subcommand;
#[derive(Subcommand, Debug)]
pub enum Commands {
    Ingest {
        path: Option<std::path::PathBuf>,
    },
    #[command(alias = "bench")]
    Benchmark {
        #[arg(long = "query")]
        queries: Vec<String>,
        #[arg(long)]
        queries_file: Option<std::path::PathBuf>,
        #[arg(long = "k")]
        top_k: Vec<usize>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
        #[arg(long, default_value_t = 32)]
        m: usize,
        #[arg(long, default_value_t = 256)]
        ef: usize,
    },
}

#[derive(Parser)]
#[command(arg_required_else_help = true)]
pub struct Cli {
    pub keywords: Option<String>,
    #[arg(long, default_value_t = 3)]
    pub top: usize,

    #[command(subcommand)]
    pub commands: Option<Commands>,
}
