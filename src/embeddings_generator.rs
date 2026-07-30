use std::path::PathBuf;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

// A global limit as to how much of a function we will read and the query
const MAX_TOKEN_LENGTH: usize = 256;
const SNOWFLAKE_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

pub fn create_function_embedding(
    model: &mut TextEmbedding,
    texts: Vec<&String>,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let embeddings = model.embed(texts, None)?;
    Ok(embeddings)
}

pub fn create_query_embedding(keywords: &String) -> anyhow::Result<Vec<f32>> {
    let mut model = create_embedding_model()?;
    // Prepending the SNOWFLAKE_PREFIX increases retrieval quality, it is literally a sentence
    // telling the model what the input is - like a system prompt.
    let query = format!("{SNOWFLAKE_PREFIX}{keywords}");
    let mut keyword_embedding = model.embed(vec![query], None)?;
    let query_embedding = keyword_embedding.pop().unwrap();
    Ok(query_embedding)
}
fn model_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("sift")
        .join("fastembed")
}
pub fn create_embedding_model() -> anyhow::Result<TextEmbedding> {
    let options = InitOptions::new(EmbeddingModel::SnowflakeArcticEmbedXSQ)
        .with_max_length(MAX_TOKEN_LENGTH)
        .with_cache_dir(model_cache_dir());

    TextEmbedding::try_new(options)
}
