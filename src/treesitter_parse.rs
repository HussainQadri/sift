use crate::language_specs::LanguageSpec;
use std::path::PathBuf;
use std::{fs, result};
use tree_sitter::{Node, Parser, Query, QueryCursor, StreamingIterator};
pub fn parser_demo(path: &PathBuf, spec: &LanguageSpec) -> tree_sitter::Tree {
    let mut parser = Parser::new();
    parser
        .set_language(&spec.language)
        .expect("Error loading language grammar");

    let source_code = fs::read_to_string(path).expect("Failed to read source file");
    let tree: tree_sitter::Tree = parser.parse(&source_code, None).unwrap();

    tree
}

pub fn extract_functions(node: Node, source: &str, spec: &LanguageSpec) -> Vec<String> {
    let query = match Query::new(&spec.language, spec.function_query) {
        Ok(query) => query,
        Err(err) => {
            eprintln!("Failed to create the tree-sitter query: {}", err);
            return Vec::new();
        }
    };

    let mut cursor = QueryCursor::new();

    let mut matches = cursor.matches(&query, node, source.as_bytes());

    let mut result_vector = Vec::new();
    while let Some(item) = matches.next() {
        for capture in item.captures {
            let result = capture.node.utf8_text(source.as_bytes()).unwrap_or("");
            result_vector.push(result.to_string());
        }
    }

    result_vector
}

pub fn extract_function_headers(node: Node, source: &str, spec: &LanguageSpec) -> Vec<String> {
    let query = match Query::new(&spec.language, spec.function_query) {
        Ok(query) => query,
        Err(err) => {
            eprintln!("Failed to create the tree-sitter query: {}", err);
            return Vec::new();
        }
    };

    let mut headers = Vec::new();
    let mut cursor = QueryCursor::new();

    let mut function_node = None;
    let mut body_node = None;

    let mut matches = cursor.matches(&query, node, source.as_bytes());
    while let Some(item) = matches.next() {
        for capture in item.captures {
            let capture_name = query.capture_names()[capture.index as usize];

            match capture_name {
                "function" => {
                    function_node = Some(capture.node);
                }
                "body" => {
                    body_node = Some(capture.node);
                }

                _ => {}
            }
        }
        if let (Some(function), Some(body)) = (function_node, body_node) {
            let header = &source[function.start_byte()..body.start_byte()];
            headers.push(header.trim().to_string());
        }
    }
    headers
}
