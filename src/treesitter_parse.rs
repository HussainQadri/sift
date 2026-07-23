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
            let function_source = function
                .utf8_text(source.as_bytes())
                .map_err(|error| anyhow::anyhow!("failed to extract function source {error}"))?;

            result_vector.push(ExtractedFunction {
                header: header.trim().to_string(),
                source: function_source.to_string(),
                line_number: function.start_position().row + 1,
            });
        }
    }

    Ok(result_vector)
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
}
