use crate::index;
use crate::language_specs;
use crate::treesitter_parse;
use fastembed::TextEmbedding;
use ignore::Walk;
use std::fs;

pub fn ingest_directory(
    model: &mut TextEmbedding,
    path: &std::path::PathBuf,
) -> anyhow::Result<Vec<index::IndexedFunction>> {
    let mut all_indexed_functions = Vec::new();

    for result in Walk::new(path) {
        let entry = match result {
            Ok(entry) => entry,
            Err(err) => {
                eprintln!("Invalid directory entry: {err}");
                continue;
            }
        };
        let file_path = entry.path();

        let spec = match language_specs::spec_for_file(file_path) {
            Ok(spec) => spec,
            Err(_) => continue,
        };

        let tree = treesitter_parse::generate_tree(file_path, &spec);
        let source_code = fs::read_to_string(file_path)?;
        let functions = treesitter_parse::extract_functions(tree.root_node(), &source_code, &spec);
        let start_id = all_indexed_functions.len();
        let indexed_functions =
            index::create_indexed_functions(model, functions, file_path, start_id)?;
        all_indexed_functions.extend(indexed_functions);
    }

    Ok(all_indexed_functions)
}
