use fastembed::TextEmbedding;

pub fn create_function_header_embeddings(
    headers: Vec<String>,
) -> anyhow::Result<Vec<FunctionEmbedding>> {
    let mut model = TextEmbedding::try_new(Default::default())?;

    let function_header_embeddings = model.embed(&headers, None)?;

    let result = headers
        .into_iter()
        .zip(function_header_embeddings)
        .map(|(header, embedding)| FunctionEmbedding {
            header,
            header_embedding: embedding,
        })
        .collect();
    Ok(result)
}

pub struct FunctionEmbedding {
    pub(crate) header: String,
    pub(crate) header_embedding: Vec<f32>,
}

pub fn create_embedding(keywords: &String) -> anyhow::Result<Vec<f32>> {
    let mut model = TextEmbedding::try_new(Default::default())?;

    let mut keyword_embedding = model.embed(vec![keywords], None)?;
    let query_embedding = keyword_embedding.pop().unwrap();

    Ok(query_embedding)
}
