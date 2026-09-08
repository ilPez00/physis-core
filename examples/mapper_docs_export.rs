//! E4 — embed and classify the personal document library for corpus density.
//!
//! Reads `research/mapper/.docs_chunks.jsonl` (produced by `extract_docs.py`,
//! gitignored) and writes a COMMITTABLE artifact that contains no document text
//! and no filenames: only the machine-assigned cell, its score, a coarse
//! top-level directory tag, and the embedding vector.
//!
//! The cells are assigned by physis's own `CellClassifier` and are
//! MACHINE-PRODUCED. As in `mapper_selfcorpus_export`, they are never evaluated
//! against and never used as anchors — evaluation stays on the 658
//! human-authored ontology entries.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example mapper_docs_export

use physis_core::classify::CellClassifier;
use physis_core::ontology::OntologyLoader;

fn jesc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' | '\r' | '\t' => o.push(' '),
            c if (c as u32) < 0x20 => {}
            c => o.push(c),
        }
    }
    o
}

/// Minimal field pull — avoids a serde_json Value allocation per line for a
/// file this size, and the two fields are known.
fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{key}\":");
    let i = line.find(&pat)? + pat.len();
    let rest = &line[i..];
    let start = rest.find('"')? + 1;
    let bytes = rest.as_bytes();
    let mut j = start;
    while j < bytes.len() {
        if bytes[j] == b'"' && bytes[j - 1] != b'\\' {
            return Some(&rest[start..j]);
        }
        j += 1;
    }
    None
}

fn main() {
    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed::VectorEmbed;
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

        let src = "research/mapper/.docs_chunks.jsonl";
        let Ok(body) = std::fs::read_to_string(src) else {
            println!("WARNING: {src} not found — run extract_docs.py first.");
            return;
        };
        let mut items: Vec<(String, String)> = Vec::new();
        for line in body.lines() {
            if let (Some(s), Some(t)) = (field(line, "src"), field(line, "text")) {
                items.push((s.to_string(), t.replace("\\\"", "\"")));
            }
        }
        println!("{} chunks", items.len());

        let cap: usize = std::env::var("MAPPER_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX);
        if items.len() > cap {
            let stride = items.len() / cap + 1;
            items = items.into_iter().step_by(stride).collect();
            println!("strided to {} (cap {cap})", items.len());
        }

        let Some(dir) = ["models/bge-base-en-v1.5", "../models/bge-base-en-v1.5"]
            .iter()
            .find(|d| std::path::Path::new(d).join("onnx/model.onnx").exists())
        else {
            println!("WARNING: BGE not available — aborting.");
            return;
        };
        let embedder = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 768,
            model_dir: Some(dir.to_string()),
            pooling: PoolingStrategy::Mean,
            ..OnnxConfig::default()
        });
        if !embedder.is_available() {
            println!("WARNING: embedder unavailable — aborting.");
            return;
        }

        let ontology = OntologyLoader::load_all();
        let clf = CellClassifier::build(&ontology, &embedder);
        println!("physis classifier: {} populated cells", clf.cell_count());

        let mut meta = String::from(
            "{\n  \"source\": \"mapper_docs_export\",\n  \"privacy\": \"no document text and no filenames: only a machine-assigned cell, its score, and a coarse top-level tag\",\n  \"note\": \"cells are MACHINE-assigned by physis's own CellClassifier; NOT ground truth\",\n  \"chunks\": [\n",
        );
        let mut vecs: Vec<f32> = Vec::with_capacity(items.len() * 768);
        for (k, (tag, text)) in items.iter().enumerate() {
            let emb = embedder.embed(text);
            let (d, m, sc) = clf
                .classify(&emb)
                .first()
                .map(|c| (c.domain.clone(), c.mode.clone(), c.score))
                .unwrap_or_default();
            vecs.extend_from_slice(&emb);
            if k > 0 {
                meta.push_str(",\n");
            }
            meta.push_str(&format!(
                "    {{\"i\": {k}, \"src\": \"{}\", \"cell\": \"{d}/{m}\", \"score\": {sc:.4}, \"len\": {}}}",
                jesc(tag),
                text.len()
            ));
            if k % 1000 == 0 && k > 0 {
                println!("  {k}/{}", items.len());
            }
        }
        meta.push_str("\n  ]\n}\n");
        std::fs::create_dir_all("research/mapper").ok();
        std::fs::write("research/mapper/docs_corpus.json", meta).expect("write meta");
        let mut raw: Vec<u8> = Vec::with_capacity(vecs.len() * 4);
        for v in &vecs {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write("research/mapper/docs_vecs.f32", raw).expect("write vecs");
        println!(
            "wrote research/mapper/docs_corpus.json and docs_vecs.f32 ({} x 768)",
            items.len()
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
