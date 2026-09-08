//! E3 — run the physis codebase through physis itself.
//!
//! E2 concluded that the mapper mechanism failed for want of DENSITY: 658 short
//! entries over 70 cells leaves PPMI too sparse to estimate and gives label
//! propagation almost no unlabelled substrate to diffuse through. That was a
//! diagnosis, not a measurement. This supplies the volume to test it.
//!
//! The codebase is chunked into units, embedded, and classified by physis's OWN
//! `CellClassifier` — the shipped classifier, not a reimplementation.
//!
//! ## The circularity this avoids, stated up front
//!
//! The cells physis assigns here are MACHINE-PRODUCED. They are not ground
//! truth and must never be evaluated against: predicting them would be scoring
//! physis against itself. They are used for two things only, both legitimate:
//!
//!   1. TEXT VOLUME  — to estimate the L1/L2 relational statistics densely,
//!                     which is exactly what E2 could not do at n=658.
//!   2. GRAPH NODES  — extra UNLABELLED vertices for label propagation, which
//!                     is what semi-supervised diffusion actually wants.
//!
//! Evaluation stays on the 658 human-authored entries, same held-out split.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example mapper_selfcorpus_export

use physis_core::classify::CellClassifier;
use physis_core::ontology::OntologyLoader;
use std::collections::HashSet;

fn jesc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

/// Meaningful units, not arbitrary byte slices: markdown sections at headings,
/// source at blank-line-separated blocks. A chunk has to carry enough text to
/// be worth an n-gram profile at all, hence the length floor.
fn chunk(path: &str, body: &str) -> Vec<String> {
    let md = path.ends_with(".md") || path.ends_with(".txt");
    let mut out = Vec::new();
    let mut cur = String::new();
    for line in body.lines() {
        let boundary = if md {
            line.starts_with('#')
        } else {
            line.trim().is_empty()
        };
        if boundary && cur.trim().len() >= 80 {
            out.push(std::mem::take(&mut cur));
        } else if boundary {
            cur.clear();
        }
        if md && line.starts_with('#') {
            cur.push_str(line);
            cur.push(' ');
            continue;
        }
        // strip comment sigils and punctuation noise; keep the words
        let t = line
            .trim()
            .trim_start_matches("//!")
            .trim_start_matches("///")
            .trim_start_matches("//")
            .trim_start_matches('#')
            .trim();
        if !t.is_empty() {
            cur.push_str(t);
            cur.push(' ');
        }
        if cur.len() > 1400 {
            out.push(std::mem::take(&mut cur));
        }
    }
    if cur.trim().len() >= 80 {
        out.push(cur);
    }
    out.into_iter()
        .map(|c| c.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|c| c.len() >= 80 && c.split_whitespace().count() >= 12)
        .collect()
}

fn walk(dir: &std::path::Path, out: &mut Vec<(String, String)>, skip: &HashSet<&str>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.starts_with('.') || skip.contains(name) {
            continue;
        }
        if p.is_dir() {
            walk(&p, out, skip);
        } else if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
            if matches!(ext, "rs" | "md" | "py" | "toml" | "ts" | "sh") {
                if let Ok(body) = std::fs::read_to_string(&p) {
                    if body.len() < 4_000_000 {
                        out.push((p.display().to_string(), body));
                    }
                }
            }
        }
    }
}

fn main() {
    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed::VectorEmbed;
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

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

        let skip: HashSet<&str> = [
            "target",
            "node_modules",
            "models",
            "vendor",
            "shots",
            "benchmark-results",
            "assets",
            "Cargo.lock",
        ]
        .into_iter()
        .collect();
        let roots = [".", "../physis-core"];
        let mut files = Vec::new();
        for r in roots {
            let p = std::path::Path::new(r);
            if p.exists() {
                walk(p, &mut files, &skip);
            }
        }
        println!("{} source/doc files", files.len());

        let mut chunks: Vec<(String, String)> = Vec::new();
        for (path, body) in &files {
            for c in chunk(path, body) {
                chunks.push((path.clone(), c));
            }
        }
        println!("{} chunks after filtering", chunks.len());

        let cap: usize = std::env::var("MAPPER_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6000);
        if chunks.len() > cap {
            // deterministic stride, so the sample spans the whole repo rather
            // than truncating to whatever the walk happened to reach first
            let stride = chunks.len() / cap + 1;
            chunks = chunks.into_iter().step_by(stride).collect();
            println!("strided to {} chunks (cap {cap})", chunks.len());
        }

        // physis's OWN classifier, built from its own ontology.
        let ontology = OntologyLoader::load_all();
        let clf = CellClassifier::build(&ontology, &embedder);
        println!("physis classifier: {} populated cells", clf.cell_count());

        let mut meta = String::from("{\n  \"source\": \"mapper_selfcorpus_export\",\n  \"note\": \"cells are MACHINE-assigned by physis's own CellClassifier; NOT ground truth\",\n  \"chunks\": [\n");
        let mut vecs: Vec<f32> = Vec::with_capacity(chunks.len() * 768);
        for (k, (path, text)) in chunks.iter().enumerate() {
            let emb = embedder.embed(text);
            let scores = clf.classify(&emb);
            let (d, m, sc) = scores
                .first()
                .map(|c| (c.domain.clone(), c.mode.clone(), c.score))
                .unwrap_or_default();
            vecs.extend_from_slice(&emb);
            if k > 0 {
                meta.push_str(",\n");
            }
            let t: String = text.chars().take(400).collect();
            meta.push_str(&format!(
                "    {{\"i\": {k}, \"path\": \"{}\", \"cell\": \"{d}/{m}\", \"score\": {sc:.4}, \"text\": \"{}\"}}",
                jesc(path),
                jesc(&t)
            ));
            if k % 500 == 0 && k > 0 {
                println!("  {k}/{} embedded+classified", chunks.len());
            }
        }
        meta.push_str("\n  ]\n}\n");

        std::fs::create_dir_all("research/mapper").ok();
        std::fs::write("research/mapper/selfcorpus.json", meta).expect("write meta");
        // vectors as raw little-endian f32: 768 per chunk, in chunk order
        let mut raw: Vec<u8> = Vec::with_capacity(vecs.len() * 4);
        for v in &vecs {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write("research/mapper/selfcorpus_vecs.f32", raw).expect("write vecs");
        println!(
            "wrote research/mapper/selfcorpus.json and selfcorpus_vecs.f32 ({} chunks x 768)",
            chunks.len()
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
