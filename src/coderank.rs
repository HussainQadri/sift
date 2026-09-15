use ort::session::Session;
use std::path::Path;
use std::path::PathBuf;
use tokenizers::Tokenizer;

struct CodeRankEncoder {
    tokenizer: Tokenizer,
    session: Session,
}

fn coderank_model_dir() -> anyhow::Result<PathBuf> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("could not determine system cache directory"))?;

    Ok(cache_dir.join("sift").join("models").join("coderank"))
}

impl CodeRankEncoder {
    fn load(model_dir: &Path) -> anyhow::Result<Self> {
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
}

#[test]

fn successfully_loads_coderank_model() -> anyhow::Result<()> {
    let model_dir = coderank_model_dir()?;
    let _encoder = CodeRankEncoder::load(&model_dir)?;
    Ok(())
}
