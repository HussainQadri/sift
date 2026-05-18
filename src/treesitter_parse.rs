use std::fs;
use tree_sitter::{Node, Parser};
pub fn parser_demo(path: &str) -> tree_sitter::Tree {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("Error loading Rust grammar");

    let source_code = fs::read_to_string(path).expect("Failed to read source file");
    let tree: tree_sitter::Tree = parser.parse(&source_code, None).unwrap();

    tree
}

pub fn extract_functions(node: Node, source: &str) {
    if node.kind() == "function_item" {
        let function_text = node.utf8_text(source.as_bytes()).unwrap();
        println!("{}", function_text);
    }

    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        extract_functions(child, source);
    }
}
