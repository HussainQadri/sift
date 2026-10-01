use crate::language_specs::LanguageSpec;
use tree_sitter::{Node, Parser, Query, QueryCursor, StreamingIterator};

pub struct ExtractedFunction {
    pub(crate) header: String,
    pub(crate) source: String,
    pub(crate) line_number: usize,
}

pub fn generate_tree_from_source(
    spec: &LanguageSpec,
    source_code: &str,
) -> anyhow::Result<tree_sitter::Tree> {
    let mut parser = Parser::new();
    parser
        .set_language(&spec.language)
        .map_err(|error| anyhow::anyhow!("failed to load Tree-sitter language: {error}"))?;

    let tree: tree_sitter::Tree = parser
        .parse(source_code, None)
        .ok_or_else(|| anyhow::anyhow!("parser returned no tree"))?;

    Ok(tree)
}

pub fn extract_functions(
    node: Node,
    source: &str,
    spec: &LanguageSpec,
) -> anyhow::Result<Vec<ExtractedFunction>> {
    let query = Query::new(&spec.language, spec.function_query)
        .map_err(|error| anyhow::anyhow!("could not create Tree-sitter query: {error}"))?;

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
            let source_start = doc_comment_start(function, source, spec.doc_comment_prefixes);
            let function_source = &source[source_start..function.end_byte()];

            result_vector.push(ExtractedFunction {
                header: header.trim().to_string(),
                source: function_source.to_string(),
                line_number: function.start_position().row + 1,
            });
        }
    }

    Ok(result_vector)
}

// Returns where the function's attached doc comments begin, or the function's own start if it has
// none. The comments must sit directly above the function; Rust attributes and C++ template
// declarations between the comments and the function are allowed.
fn doc_comment_start(function: Node, source: &str, doc_comment_prefixes: &[&str]) -> usize {
    let anchor = match function.parent() {
        Some(parent) if parent.kind() == "template_declaration" => parent,
        _ => function,
    };
    let mut start = anchor.start_byte();
    let mut previous = anchor.prev_named_sibling();
    while let Some(node) = previous {
        match node.kind() {
            "attribute_item" => {}
            "comment" | "line_comment" | "block_comment" => {
                let text = &source[node.start_byte()..node.end_byte()];
                if !doc_comment_prefixes
                    .iter()
                    .any(|prefix| text.starts_with(prefix))
                {
                    break;
                }
                start = node.start_byte();
            }
            _ => break,
        }
        previous = node.prev_named_sibling();
    }
    start
}

#[cfg(test)]
mod tests {
    use super::extract_functions;
    use crate::language_specs;
    use tree_sitter::Parser;

    #[test]
    fn rust_extraction_keeps_full_source_header_and_line_number() -> anyhow::Result<()> {
        let source = "\nfn first() {}\n\npub fn wanted(value: i32) -> i32 {\n    value + 1\n}\n";
        let spec = language_specs::rust_spec();
        let mut parser = Parser::new();
        parser.set_language(&spec.language).unwrap();
        let tree = parser.parse(source, None).unwrap();

        let functions = extract_functions(tree.root_node(), source, &spec)?;

        assert_eq!(functions.len(), 2);
        assert_eq!(functions[1].header, "pub fn wanted(value: i32) -> i32");
        assert!(functions[1].source.contains("value + 1"));
        assert_eq!(functions[1].line_number, 4);
        Ok(())
    }

    fn extract(
        spec: crate::language_specs::LanguageSpec,
        source: &str,
    ) -> Vec<super::ExtractedFunction> {
        let mut parser = Parser::new();
        parser.set_language(&spec.language).unwrap();
        let tree = parser.parse(source, None).unwrap();
        extract_functions(tree.root_node(), source, &spec).unwrap()
    }

    #[test]
    fn rust_doc_comments_are_kept_with_the_function() {
        let source = "// not a doc comment\n/// Adds one.\n#[inline]\npub fn wanted(value: i32) -> i32 {\n    value + 1\n}\n";
        let functions = extract(language_specs::rust_spec(), source);

        assert!(
            functions[0]
                .source
                .starts_with("/// Adds one.\n#[inline]\npub fn wanted")
        );
        assert_eq!(functions[0].header, "pub fn wanted(value: i32) -> i32");
        assert_eq!(functions[0].line_number, 4);
    }

    #[test]
    fn functions_without_doc_comments_start_at_the_function() {
        let source = "// a plain comment\nfn wanted() {}\n";
        let functions = extract(language_specs::rust_spec(), source);

        assert_eq!(functions[0].source, "fn wanted() {}");
    }

    #[test]
    fn javadoc_is_kept_with_the_method() {
        let source = "class A {\n    /** Returns one. */\n    int one() { return 1; }\n}\n";
        let functions = extract(language_specs::java_spec(), source);

        assert!(functions[0].source.starts_with("/** Returns one. */"));
    }

    #[test]
    fn cpp_comments_above_templates_are_kept_with_the_function() {
        let source = "// Returns the larger value.\ntemplate <typename T>\nT larger(T a, T b) { return a > b ? a : b; }\n";
        let functions = extract(language_specs::cpp_spec(), source);

        assert!(
            functions[0]
                .source
                .starts_with("// Returns the larger value.\ntemplate <typename T>")
        );
        assert_eq!(functions[0].line_number, 3);
    }
}
