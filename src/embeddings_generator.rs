use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

pub fn create_function_embedding(texts: Vec<&String>) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut model =
        TextEmbedding::try_new(InitOptions::new(EmbeddingModel::JinaEmbeddingsV2BaseCode))?;
    let embeddings = model.embed(texts, None)?;
    Ok(embeddings)
}

pub fn create_query_embedding(keywords: &String) -> anyhow::Result<Vec<f32>> {
    let mut model =
        TextEmbedding::try_new(InitOptions::new(EmbeddingModel::JinaEmbeddingsV2BaseCode))?;
    let mut keyword_embedding = model.embed(vec![keywords], None)?;
    let query_embedding = keyword_embedding.pop().unwrap();
    Ok(query_embedding)
}
