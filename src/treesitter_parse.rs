use std::path::PathBuf;
use std::{fs, path::Path};
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

pub fn extract_functions(node: Node, source: &str, spec: &LanguageSpec) {
    let query = match Query::new(&spec.language, spec.function_header_query) {
        Ok(query) => query,
        Err(err) => {
            eprintln!("Failed to create the tree-sitter query: {}", err);
            return;
        }
    };

    let mut cursor = QueryCursor::new();

    let mut matches = cursor.matches(&query, node, source.as_bytes());

    while let Some(item) = matches.next() {
        for capture in item.captures {
            let result = capture.node.utf8_text(source.as_bytes()).unwrap_or("");
            println!("{}", result);
        }
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
              (function_item) @function
          "#,
    }
}

pub fn python_spec() -> LanguageSpec {
    LanguageSpec {
        language: tree_sitter_python::LANGUAGE.into(),
        function_header_query: r#"
              (function_definition) @function
          "#,
    }
}

pub fn spec_for_file(path: &PathBuf) -> anyhow::Result<LanguageSpec> {
    let extension_string = path.extension().and_then(|ext| ext.to_str());

    match extension_string {
        Some("rs") => Ok(rust_spec()),
        Some("py") => Ok(python_spec()),
        Some(ext) => anyhow::bail!("Unsupported file extension"),
        None => anyhow::bail!("File has no extension"),
    }
}
