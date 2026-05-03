mod similarity;
use fastembed::TextEmbedding;
use similarity::cosine_similarity;
use std::io;
fn main() -> anyhow::Result<()> {
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to readline");
    let input = input.trim();
    let mut model = TextEmbedding::try_new(Default::default())?;
    let documents = vec![input];
    let documents2 = vec!["i love rust"];
    let documents3 = vec!["i hate programming languages"];

    let embeddings = model.embed(documents, None)?;
    let embeddings2 = model.embed(documents2, None)?;
    let embeddings3 = model.embed(documents3, None)?;
    println!(
        "Similarity score: {}",
        cosine_similarity(&embeddings[0], &embeddings2[0])
    );
    println!(
        "Similarity score: {}",
        cosine_similarity(&embeddings[0], &embeddings3[0])
    );
    Ok(())
}
