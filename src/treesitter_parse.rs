use std::path::PathBuf;
use std::{fs, path::Path};
use tree_sitter::{Node, Parser};
pub fn parser_demo(path: &PathBuf, spec: &LanguageSpec) -> tree_sitter::Tree {
    let mut parser = Parser::new();
    parser
        .set_language(&spec.language)
        .expect("Error loading language grammar");

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

pub fn extract_function_headers(node: Node, source: &str) {
    if node.kind() == "function_item" {
        let function_header_start = node.start_byte();
        let body = node
            .child_by_field_name("body")
            .expect("Body does not exist");
        let function_header_end = body.start_byte();
        let function_header = &source[function_header_start..function_header_end];
        println!("{}", function_header.trim());
    }

    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        extract_function_headers(child, source);
    }
}

pub struct LanguageSpec {
    language: tree_sitter::Language,
    function_header_query: &'static str,
}

pub fn rust_spec() -> LanguageSpec {
    LanguageSpec {
        language: tree_sitter_rust::LANGUAGE.into(),
        function_header_query: r#"
              (function_item
                  body: (_) @body) @function
          "#,
    }
}

pub fn spec_for_file(path: &PathBuf) -> anyhow::Result<LanguageSpec> {
    let extension_string = path.extension().and_then(|ext| ext.to_str());

    match extension_string {
        Some("rs") => Ok(rust_spec()),
        Some(ext) => anyhow::bail!("Unsupported file extension"),
        None => anyhow::bail!("File has no extension"),
    }
}
