//! Generic: embed a JSON list of texts with BGE and write the vectors.
//!
//! Reads  {"texts": ["...", "..."]}  from argv[1]
//! Writes argv[2] as raw little-endian f32, 768 per text, in order.
//!
//! Exists so the Python experiments can score candidate ANCHOR TEXTS without a
//! bespoke exporter each time — E13 needs to compare several ways of writing an
//! anchor, and only the text differs.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release \
//!     --example mapper_embed_texts -- in.json out.f32

fn main() {
    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed::VectorEmbed;
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

        let args: Vec<String> = std::env::args().collect();
        // `cargo run -- a b` gives argv = [binary, a, b] — length 3, not 4.
        // The first version checked for 4 and indexed [2]/[3], so it printed
        // usage and wrote nothing.
        if args.len() < 3 {
            println!("usage: ... -- <in.json> <out.f32>");
            return;
        }
        let Some(dir) = ["models/bge-base-en-v1.5", "../models/bge-base-en-v1.5"]
            .iter()
            .find(|d| std::path::Path::new(d).join("onnx/model.onnx").exists())
        else {
            println!("WARNING: BGE not available — aborting.");
            return;
        };
        let e = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 768,
            model_dir: Some(dir.to_string()),
            pooling: PoolingStrategy::Mean,
            ..OnnxConfig::default()
        });
        if !e.is_available() {
            println!("WARNING: embedder unavailable — aborting.");
            return;
        }
        let raw = std::fs::read_to_string(&args[1]).expect("read input");
        let v: serde_json::Value = serde_json::from_str(&raw).expect("parse");
        let texts: Vec<String> = v["texts"]
            .as_array()
            .expect("texts array")
            .iter()
            .filter_map(|x| x.as_str().map(|s| s.to_string()))
            .collect();
        let mut out: Vec<u8> = Vec::with_capacity(texts.len() * 768 * 4);
        for (k, t) in texts.iter().enumerate() {
            for f in e.embed(t) {
                out.extend_from_slice(&f.to_le_bytes());
            }
            if k % 200 == 0 && k > 0 {
                println!("  {k}/{}", texts.len());
            }
        }
        std::fs::write(&args[2], out).expect("write vectors");
        println!("embedded {} texts -> {}", texts.len(), args[2]);
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
