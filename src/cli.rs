use clap::Parser;
use clap::Subcommand;
#[derive(Subcommand, Debug)]
pub enum Commands {
    Ingest { path: Option<std::path::PathBuf> },
}

#[derive(Parser)]
pub struct Cli {
    pub keywords: Option<String>,
    #[arg(long, default_value_t = 3)]
    pub top: usize,

    #[command(subcommand)]
    pub commands: Option<Commands>,
}
