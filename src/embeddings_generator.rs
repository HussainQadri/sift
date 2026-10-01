use crate::era::EraQueryEncoder;
use model2vec_rs::model::StaticModel;

// A global limit as to how much of a function we will read and the query
const MAX_TOKEN_LENGTH: usize = 256;
const MODEL_ID: &str = "minishlab/potion-code-16M-v2";

pub fn create_function_embedding(
    model: &StaticModel,
    texts: Vec<&String>,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let texts: Vec<String> = texts.into_iter().cloned().collect();
    let embeddings = model.encode_with_args(&texts, Some(MAX_TOKEN_LENGTH), 1024);
    Ok(embeddings)
}

pub fn create_query_embedding(model: &StaticModel, keywords: &str) -> anyhow::Result<Vec<f32>> {
    let query_embedding = model
        .encode_with_args(&[keywords.to_owned()], Some(MAX_TOKEN_LENGTH), 1)
        .pop()
        .ok_or_else(|| anyhow::anyhow!("Potion returned no query embedding"))?;
    Ok(query_embedding)
}
pub fn create_embedding_model() -> anyhow::Result<StaticModel> {
    StaticModel::from_pretrained(MODEL_ID, None, Some(true), None)
}

pub enum QueryEncoder {
    Potion(StaticModel),
    Era(EraQueryEncoder),
}

impl QueryEncoder {
    pub fn load(era: bool) -> anyhow::Result<Self> {
        if era {
            Ok(Self::Era(EraQueryEncoder::load()?))
        } else {
            Ok(Self::Potion(create_embedding_model()?))
        }
    }

    pub fn encode_query(&mut self, query: &str) -> anyhow::Result<Vec<f32>> {
        match self {
            Self::Potion(model) => create_query_embedding(model, query),
            Self::Era(encoder) => encoder.encode_query(query),
        }
    }
}
