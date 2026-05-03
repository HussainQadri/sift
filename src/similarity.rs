pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot: f32 = 0.0;
    let mut a_sum_squares: f32 = 0.0;
    let mut b_sum_squares: f32 = 0.0;
    for x in 0..a.len() {
        dot += a[x] * b[x];
        a_sum_squares += a[x] * a[x];
        b_sum_squares += b[x] * b[x];
    }

    dot / (a_sum_squares.sqrt() * b_sum_squares.sqrt())
}
