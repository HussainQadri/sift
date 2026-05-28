use std::path::PathBuf;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

pub fn create_function_embedding(
    model: &mut TextEmbedding,
    texts: Vec<&String>,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let embeddings = model.embed(texts, None)?;
    Ok(embeddings)
}

pub fn create_query_embedding(keywords: &String) -> anyhow::Result<Vec<f32>> {
    let mut model = create_embedding_model()?;
    let mut keyword_embedding = model.embed(vec![keywords], None)?;
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
    let options = InitOptions::new(EmbeddingModel::JinaEmbeddingsV2BaseCode)
        .with_cache_dir(model_cache_dir());

    TextEmbedding::try_new(options)
}
