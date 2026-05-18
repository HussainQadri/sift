mod similarity;
mod treesitter_parse;
use std::fs;
fn main() -> anyhow::Result<()> {
    let path = "src/treesitter_parse.rs";
    let tree = treesitter_parse::parser_demo(path);
    let source_code = fs::read_to_string(path).expect("Failed to read source file");
    treesitter_parse::extract_function_headers(tree.root_node(), &source_code);

    Ok(())
}
