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

    Evaluate {
        #[arg(long)]
        judgements: PathBuf,

        #[arg(long, default_value_t = 10)]
        top: usize,
    },
}

#[derive(Parser)]
#[command(arg_required_else_help = true)]
pub struct Cli {
    pub keywords: Option<String>,
    #[arg(long, default_value_t = 3)]
    pub top: usize,

    #[arg(long)]
    pub exact: bool,

    #[command(subcommand)]
    pub commands: Option<Commands>,
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::Parser;

    #[test]
    fn exact_search_is_disabled_by_default() {
        let cli = Cli::try_parse_from(["sift", "find a function"]).unwrap();

        assert!(!cli.exact);
    }

    #[test]
    fn exact_flag_enables_exact_search() {
        let cli = Cli::try_parse_from(["sift", "--exact", "find a function"]).unwrap();

        assert!(cli.exact);
    }
}
