use std::path::Path;

pub fn rust_spec() -> LanguageSpec {
    LanguageSpec {
        language: tree_sitter_rust::LANGUAGE.into(),
        function_query: r#"(function_item body: (block) @body) @function"#,
    }
}

pub fn python_spec() -> LanguageSpec {
    LanguageSpec {
        language: tree_sitter_python::LANGUAGE.into(),
        function_query: r#"
              (function_definition
                body: (block) @body) @function
          "#,
    }
}
pub fn cpp_spec() -> LanguageSpec {
    LanguageSpec {
        language: tree_sitter_cpp::LANGUAGE.into(),
        function_query: r#"
              (function_definition
                body: (compound_statement) @body) @function
          "#,
    }
}
pub fn spec_for_file(path: &Path) -> anyhow::Result<LanguageSpec> {
    let extension_string = path.extension().and_then(|ext| ext.to_str());

    match extension_string {
        Some("rs") => Ok(rust_spec()),
        Some("py") => Ok(python_spec()),
        Some("cpp") => Ok(cpp_spec()),
        Some(_ext) => anyhow::bail!("Unsupported file extension"),
        None => anyhow::bail!("File has no extension"),
    }
}

pub struct LanguageSpec {
    pub(crate) language: tree_sitter::Language,
    pub(crate) function_query: &'static str,
}
