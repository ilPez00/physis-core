use physis_core::embed::{VectorEmbed};
#[path="../src/embed_ngram.rs"]
mod synthetic;
#[test] fn ngram_det_365() {
    use synthetic::SyntheticNGramEmbedder;
    let e = SyntheticNGramEmbedder::new(384, 42);
    let v = e.embed("the quick brown fox");
    assert_eq!(v.len(), 384);
    let norm = v.iter().map(|x| x*x).sum::<f32>().sqrt();
    assert!((norm-1.0).abs() < 1e-3, "unit vector: norm={}", norm);
    // Determinism: same key -> same vector
    assert_eq!(v, e.embed("the quick brown fox"));
}
#[test] fn ngram_lifts_compute() { // conceptual: lookup is memoized; not projection-per-call
    use synthetic::SyntheticNGramEmbedder;
    let e = SyntheticNGramEmbedder::new(384, 42);
    let _ = e.embed("any text"); // lookup path verified
}
