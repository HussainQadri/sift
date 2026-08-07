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
    build_embedding_model(None)
}

fn build_embedding_model(intra_threads: Option<usize>) -> anyhow::Result<TextEmbedding> {
    let options = InitOptions::new(EmbeddingModel::SnowflakeArcticEmbedXSQ)
        .with_max_length(MAX_TOKEN_LENGTH)
        .with_cache_dir(model_cache_dir());

    let options = match intra_threads {
        Some(threads) => options.with_intra_threads(threads),
        None => options,
    };

    TextEmbedding::try_new(options)
}

pub fn create_embedding_model_with_intra_threads(
    intra_threads: usize,
) -> anyhow::Result<TextEmbedding> {
    anyhow::ensure!(intra_threads > 0, "intra_threads must be greater than 0");
    build_embedding_model(Some(intra_threads))
}

pub fn create_embedding_models(
    worker_count: usize,
    intra_threads: usize,
) -> anyhow::Result<Vec<TextEmbedding>> {
    anyhow::ensure!(worker_count > 0, "worker_count must be greater than 0");

    let mut workers = Vec::with_capacity(worker_count);

    for _ in 0..worker_count {
        workers.push(create_embedding_model_with_intra_threads(intra_threads)?);
    }

    Ok(workers)
}
