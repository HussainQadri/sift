use clap::Parser;
use ignore::Walk;
mod index;
use clap::Subcommand;
mod embeddings_generator;
mod language_specs;
mod similarity;
mod treesitter_parse;
use crate::similarity::cosine_similarity;
use fastembed::TextEmbedding;
use std::{fs, path::Path};
use syntect::{
    easy::HighlightLines,
    highlighting::{Style, ThemeSet},
    parsing::SyntaxSet,
    util::{LinesWithEndings, as_24_bit_terminal_escaped},
};

fn print_highlighted(code: &str, extension: &str) {
    let syntax_set = SyntaxSet::load_defaults_newlines();
    let theme_set = ThemeSet::load_defaults();

    let syntax = syntax_set
        .find_syntax_by_extension(extension)
        .unwrap_or_else(|| syntax_set.find_syntax_plain_text());

    let theme = &theme_set.themes["base16-ocean.dark"];

    let mut highlighter = HighlightLines::new(syntax, theme);

    for line in LinesWithEndings::from(code) {
        let ranges: Vec<(Style, &str)> = highlighter.highlight_line(line, &syntax_set).unwrap();

        print!("{}", as_24_bit_terminal_escaped(&ranges[..], false));
    }
}

#[derive(Subcommand, Debug)]
enum Commands {
    Ingest { path: Option<std::path::PathBuf> },
}

#[derive(Parser)]
struct Cli {
    keywords: Option<String>,
    #[command(subcommand)]
    commands: Option<Commands>,
}

fn recursive_ingest_dir(
    model: &mut TextEmbedding,
    path: &std::path::PathBuf,
) -> anyhow::Result<Vec<index::IndexedFunction>> {
    let mut all_indexed_functions = Vec::new();

    for resource_entry_result in fs::read_dir(path)? {
        let entry = resource_entry_result?;
        let file_path = entry.path();

        if file_path.is_dir() {
            let indexed_functions = recursive_ingest_dir(model, &file_path)?;
            all_indexed_functions.extend(indexed_functions);
        }

        let spec = match language_specs::spec_for_file(&file_path) {
            Ok(spec) => spec,
            Err(_) => continue,
        };

        let tree = treesitter_parse::generate_tree(&file_path, &spec);
        let source_code = fs::read_to_string(&file_path)?;
        let functions = treesitter_parse::extract_functions(tree.root_node(), &source_code, &spec);
        let indexed_functions = index::create_indexed_functions(model, functions, &file_path)?;
        all_indexed_functions.extend(indexed_functions);
    }

    Ok(all_indexed_functions)
}
fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    match args.commands {
        Some(Commands::Ingest { path }) => {
            let mut model = embeddings_generator::create_embedding_model()?;

            let target_path = match path {
                Some(p) => p,
                None => std::path::PathBuf::from("."),
            };

            let all_indexed_functions = recursive_ingest_dir(&mut model, &target_path)?;
            index::save_index(&all_indexed_functions)?;
        }
        None => {
            let keywords = args.keywords.expect("Please enter a search query");
            let query = embeddings_generator::create_query_embedding(&keywords)?;
            let loaded_indexed_functions = index::load_index()?;
            let mut result: Vec<(index::IndexedFunction, f32)> = loaded_indexed_functions
                .into_iter()
                .map(|indexed_function| {
                    let score = cosine_similarity(&query, &indexed_function.embedding);

                    (indexed_function, score)
                })
                .collect();
            result.sort_by(|a, b| b.1.total_cmp(&a.1));

            for (indexed_function, score) in result.iter().take(3) {
                println!("{:.3} {}: ", score, indexed_function.path);
                let extension = Path::new(&indexed_function.path)
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("");

                print!("\x1b[32m{}:\x1b[0m ", indexed_function.line_number);
                print_highlighted(&indexed_function.header, extension);
                println!("\n");
            }
        }
    }

    Ok(())
}
