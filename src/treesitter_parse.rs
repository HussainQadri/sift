use crate::language_specs::LanguageSpec;
use tree_sitter::{Node, Parser, Query, QueryCursor, StreamingIterator};

pub struct ExtractedFunction {
    pub(crate) header: String,
    pub(crate) source: String,
    pub(crate) line_number: usize,
}

pub fn generate_tree_from_source(spec: &LanguageSpec, source_code: &str) -> tree_sitter::Tree {
    let mut parser = Parser::new();
    parser
        .set_language(&spec.language)
        .expect("Error loading language grammar");

    let tree: tree_sitter::Tree = parser.parse(source_code, None).unwrap();

    tree
}

pub fn extract_functions(node: Node, source: &str, spec: &LanguageSpec) -> Vec<ExtractedFunction> {
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
        let mut function_node = None;
        let mut body_node = None;

        for capture in item.captures {
            let capture_name = query.capture_names()[capture.index as usize];

            match capture_name {
                "function" => function_node = Some(capture.node),
                "body" => body_node = Some(capture.node),
                _ => {}
            }
        }

        if let (Some(function), Some(body)) = (function_node, body_node) {
            let header = &source[function.start_byte()..body.start_byte()];
            let function_source = function.utf8_text(source.as_bytes()).unwrap_or("");

            result_vector.push(ExtractedFunction {
                header: header.trim().to_string(),
                source: function_source.to_string(),
                line_number: function.start_position().row + 1,
            });
        }
    }

    result_vector
}

#[cfg(test)]
mod tests {
    use super::extract_functions;
    use crate::language_specs;
    use tree_sitter::Parser;

    #[test]
    fn rust_extraction_keeps_full_source_header_and_line_number() {
        let source = "\nfn first() {}\n\npub fn wanted(value: i32) -> i32 {\n    value + 1\n}\n";
        let spec = language_specs::rust_spec();
        let mut parser = Parser::new();
        parser.set_language(&spec.language).unwrap();
        let tree = parser.parse(source, None).unwrap();

        let functions = extract_functions(tree.root_node(), source, &spec);

        assert_eq!(functions.len(), 2);
        assert_eq!(functions[1].header, "pub fn wanted(value: i32) -> i32");
        assert!(functions[1].source.contains("value + 1"));
        assert_eq!(functions[1].line_number, 4);
    }
}
