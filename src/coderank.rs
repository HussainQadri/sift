use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::path::PathBuf;
use tokenizers::Tokenizer;

const CODERANK_QUERY_PREFIX: &str = "search_query: ";
const CODERANK_MAX_TOKENS: usize = 512;

pub struct CodeRankEncoder {
    tokenizer: Tokenizer,
    session: Session,
}

pub fn coderank_model_dir() -> anyhow::Result<PathBuf> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("could not determine system cache directory"))?;

    Ok(cache_dir.join("sift").join("models").join("coderank"))
}

impl CodeRankEncoder {
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let model_path = model_dir.join("model.onnx");
        let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
            Ok(tokenizer) => tokenizer,
            Err(error) => {
                anyhow::bail!(
                    "failed to load tokenizer from {}: {error}",
                    tokenizer_path.display()
                )
            }
        };
        let session = Session::builder()?.commit_from_file(&model_path)?;
        Ok(Self { tokenizer, session })
    }

    pub fn encode_query(&mut self, query: &str) -> anyhow::Result<Vec<f32>> {
        let formatted_query = format!("{CODERANK_QUERY_PREFIX}{query}");
        let encoding = self
            .tokenizer
            .encode(formatted_query, true)
            .map_err(|error| anyhow::anyhow!("failed to tokenize query: {error}"))?;

        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mut attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&mask| mask as i64)
            .collect();
        input_ids.truncate(CODERANK_MAX_TOKENS);
        attention_mask.truncate(CODERANK_MAX_TOKENS);
        let sequence_length = input_ids.len();

        let outputs = self.session.run(ort::inputs![
            "input_ids" => Tensor::from_array(([1, sequence_length], input_ids))?,
            "attention_mask" => Tensor::from_array(([1, sequence_length], attention_mask))?,
        ])?;
        let (_, sentence_embedding) = outputs["sentence_embedding"].try_extract_tensor::<f32>()?;

        let length = sentence_embedding
            .iter()
            .map(|element| element * element)
            .sum::<f32>()
            .sqrt();
        anyhow::ensure!(length > 0.0, "CodeRank returned a zero query embedding");
        Ok(sentence_embedding
            .iter()
            .map(|element| element / length)
            .collect())
    }
}

#[test]

fn successfully_loads_coderank_model() -> anyhow::Result<()> {
    let model_dir = coderank_model_dir()?;
    let _encoder = CodeRankEncoder::load(&model_dir)?;
    Ok(())
}

#[test]
fn query_embedding_matches_reference_vector() -> anyhow::Result<()> {
    let fixture = include_str!("../tests/fixtures/era/reference_vectors.json");
    let fixture: serde_json::Value = serde_json::from_str(fixture)?;
    let mut encoder = CodeRankEncoder::load(&coderank_model_dir()?)?;

    for reference in fixture["queries"].as_array().unwrap() {
        let query = reference["query"].as_str().unwrap();
        let expected: Vec<f32> = reference["q_cr"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        let actual = encoder.encode_query(query)?;
        assert_eq!(actual.len(), expected.len());
        let max_difference = actual
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_difference < 1e-5,
            "{query}: max difference {max_difference}"
        );
    }
    Ok(())
}
