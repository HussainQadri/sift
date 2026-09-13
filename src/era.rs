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
