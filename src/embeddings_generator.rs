use fastembed::TextEmbedding;

pub fn create_function_header_embeddings(
    headers: Vec<String>,
) -> anyhow::Result<Vec<FunctionEmbedding>> {
    let mut model = TextEmbedding::try_new(Default::default())?;

    let function_header_embeddings = model.embed(&headers, None)?;

    let result = headers
        .into_iter()
        .zip(function_header_embeddings)
        .map(|(header, embedding)| FunctionEmbedding { header, embedding })
        .collect();
    Ok(result)
}

pub struct FunctionEmbedding {
    header: String,
    embedding: Vec<f32>,
}
