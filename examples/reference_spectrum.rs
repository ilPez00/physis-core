//! Embed a sequence of references to the same object and print the vectors.
//!
//! One line of JSON per reference on stdin: `{"x": …, "text": "…", "meta": {…}}`.
//! `meta` is passed through untouched, so a caller can carry granularity,
//! position in the document, commit, author or anything else alongside the
//! vector and slice the field by it afterwards.
//! Output is one JSON object on stdout with the embedder that was actually used
//! and the vectors, so the analysis downstream can say whether it is looking at
//! meaning or at a lexical hash — `embed::select` falls back to random
//! projection when no model is on disk, and the difference is invisible unless
//! the label is carried with the numbers.
//!
//!     cargo run --offline --release --features embed-onnx --example reference_spectrum \
//!         < states.jsonl > vectors.json
use std::io::{BufRead, Read};

fn main() {
    let dim: usize = std::env::var("PHYSIS_MODEL_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(768);
    let (embedder, label) = physis_core::embed::select(dim);

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin");
    let mut xs: Vec<serde_json::Value> = Vec::new();
    let mut metas: Vec<serde_json::Value> = Vec::new();
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    let mut lengths: Vec<usize> = Vec::new();
    for line in input.as_bytes().lines() {
        let line = line.expect("line");
        if line.trim().is_empty() {
            continue;
        }
        let record: serde_json::Value = serde_json::from_str(&line).expect("json line");
        let text = record["text"].as_str().unwrap_or("");
        xs.push(record["x"].clone());
        metas.push(
            record
                .get("meta")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        );
        lengths.push(text.len());
        vectors.push(embedder.embed(text));
    }

    // Denominators in the artifact, not only on stderr: an empty run and a run
    // of one reference must not look like a spectrum.
    let out = serde_json::json!({
        "embedder": label,
        "dim": vectors.first().map(Vec::len).unwrap_or(0),
        "references": vectors.len(),
        "text_bytes": lengths,
        "x": xs,
        "meta": metas,
        "vectors": vectors,
    });
    println!("{}", serde_json::to_string(&out).expect("serialise"));
    eprintln!(
        "embedded {} reference(s) with {label} at dim {}",
        vectors.len(),
        vectors.first().map(Vec::len).unwrap_or(0)
    );
}
