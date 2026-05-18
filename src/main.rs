mod similarity;
mod treesitter_parse;
use std::fs;
fn main() -> anyhow::Result<()> {
    let tree = treesitter_parse::parser_demo("src/main.rs");
    println!("{:?}", tree.root_node());
    let path = "src/treesitter_parse.rs";
    let source_code = fs::read_to_string(path).expect("Failed to read source file");
    treesitter_parse::extract_functions(tree.root_node(), &source_code);

    Ok(())
}
