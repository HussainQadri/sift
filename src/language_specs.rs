use std::path::PathBuf;

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

pub struct LanguageSpec {
    pub(crate) language: tree_sitter::Language,
    pub(crate) function_header_query: &'static str,
}
