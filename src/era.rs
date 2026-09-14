const ADAPTER_BYTES: &[u8] = include_bytes!("../assets/era/coderank_to_potion_v1.f32");

fn load_adapter(adapter_bytes: &[u8]) -> Vec<f32> {
    assert!(adapter_bytes.len() == 768 * 256 * 4);

    adapter_bytes
        .chunks_exact(4)
        // Because .chunks_exact(4) guarantees each chunk is 4 bytes long, this conversion will
        // never fail
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn adapt_query_embedding(coderank_embedding: &[f32]) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(
        coderank_embedding.len() == 768,
        "expected 768-dimensional CodeRank embedding, got {}",
        coderank_embedding.len()
    );
    let w = load_adapter(ADAPTER_BYTES);
    // Multiply the coderank_embedding (1 x 768) with W (768 x 256), getting a 1x256 matrix
    let mut output = vec![0.0f32; 256];
    for i in 0..768 {
        let query_value = coderank_embedding[i];
        for j in 0..256 {
            // W is stored as a flat matrix, we have to do some coordinate conversions to treat as
            // 2D matrix
            output[j] += query_value * w[i * 256 + j];
        }
    }

    let squared_sum: f32 = output.iter().map(|element| element * element).sum();
    let length = squared_sum.sqrt();
    for element in &mut output {
        *element /= length;
    }
    Ok(output)
}

#[test]

fn adapter_matches_reference_vector() {
    let fixture = include_str!("../tests/fixtures/era/reference_vectors.json");
    let fixture: serde_json::Value = serde_json::from_str(fixture).unwrap();
    let first_query = &fixture["queries"][0];
    let q_cr: Vec<f32> = first_query["q_cr"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect();
    let q_hat: Vec<f32> = first_query["q_hat"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect();
    let output = adapt_query_embedding(&q_cr).unwrap();
    for (index, (actual_value, expected_value)) in output.iter().zip(q_hat.iter()).enumerate() {
        let difference = (actual_value - expected_value).abs();
        assert!(
            difference < 1e-6,
            "mismatch at index {index}: expected {expected_value}, but got {actual_value}"
        );
    }
}
