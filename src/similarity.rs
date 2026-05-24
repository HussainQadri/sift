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

#[cfg(test)]
mod tests {
    use super::cosine_similarity;

    #[test]
    fn identical_vectors_have_maximum_similarity() {
        let vector = [1.0, 2.0, 3.0];

        assert!((cosine_similarity(&vector, &vector) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn orthogonal_vectors_have_zero_similarity() {
        let score = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);

        assert_eq!(score, 0.0);
    }
}
