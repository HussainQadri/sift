use crate::treesitter_parse::ExtractedFunction;
use fastembed::TextEmbedding;

pub fn create_function_embedding(
    functions: Vec<ExtractedFunction>,
) -> anyhow::Result<Vec<FunctionEmbedding>> {
    let mut model = TextEmbedding::try_new(Default::default())?;

    let function_sources: Vec<&String> = functions.iter().map(|function| &function.source).collect();
    let function_embeddings = model.embed(function_sources, None)?;

    let result = functions
        .into_iter()
        .zip(function_embeddings)
        .map(|(function, embedding)| FunctionEmbedding {
            header: function.header,
            function_embedding: embedding,
        })
        .collect();
    Ok(result)
}

pub struct FunctionEmbedding {
    pub(crate) header: String,
    pub(crate) function_embedding: Vec<f32>,
}

pub fn create_embedding(keywords: &String) -> anyhow::Result<Vec<f32>> {
    let mut model = TextEmbedding::try_new(Default::default())?;

    let mut keyword_embedding = model.embed(vec![keywords], None)?;
    let query_embedding = keyword_embedding.pop().unwrap();

    Ok(query_embedding)
}
