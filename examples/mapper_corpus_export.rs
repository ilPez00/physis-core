//! E2 step 0 — dump the corpus once, so the mapper experiments can iterate
//! without re-embedding. EXPERIMENTAL: reads only, writes only to research/.
//!
//! Emits, per classification entry: name, hints, the externally-authored cell
//! (domain/mode), and the BGE-base vector -- BGE because E1 measured it at
//! 0.866 mean AUC against MiniLM's 0.733, so it is the strongest existing
//! representation this repo has and therefore the right ablation ceiling.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example mapper_corpus_export

use physis_core::ontology::OntologyLoader;
use std::collections::HashSet;

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter().map(|x| x / n).collect()
}

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

        // Mode anchors are excluded from the entries, exactly as experiments 38
        // and 47 exclude them, and emitted separately as candidate anchors.
        let anchor_path = ["physis-core/config", "config", "../config"]
            .iter()
            .map(|d| std::path::Path::new(d).join("mode_anchors_ontology.json"))
            .find(|p| p.exists());
        let mut anchor_names: HashSet<String> = HashSet::new();
        if let Some(ap) = &anchor_path {
            if let Ok(txt) = std::fs::read_to_string(ap) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                    for a in v["domains"].as_array().into_iter().flatten() {
                        if let Some(nm) = a["name"].as_str() {
                            anchor_names.insert(nm.to_string());
                        }
                    }
                }
            }
        }

        let ontology = OntologyLoader::load_all();
        let mut out = String::from("{\n  \"source\": \"mapper_corpus_export\",\n  \"embedder\": \"bge-base-en-v1.5 768d mean\",\n  \"entries\": [\n");
        let mut n = 0usize;
        let mut first = true;
        for def in ontology.classification_domains() {
            let (Some(d), Some(m)) = (&def.domain, &def.mode) else {
                continue;
            };
            if anchor_names.contains(&def.name) {
                continue;
            }
            let mut text = def.name.clone();
            for h in &def.hints {
                text.push(' ');
                text.push_str(h);
            }
            let v = normalize(&e.embed(&text));
            // name-only vector too: Stage 13's format control needs it, and
            // computing it here costs one extra pass instead of a rerun.
            let vn = normalize(&e.embed(&def.name));
            let hints: Vec<String> = def
                .hints
                .iter()
                .map(|h| format!("\"{}\"", jesc(h)))
                .collect();
            let vs: Vec<String> = v.iter().map(|x| format!("{x:.5}")).collect();
            let vns: Vec<String> = vn.iter().map(|x| format!("{x:.5}")).collect();
            if !first {
                out.push_str(",\n");
            }
            first = false;
            out.push_str(&format!(
                "    {{\"i\": {n}, \"name\": \"{}\", \"hints\": [{}], \"domain\": \"{d}\", \"mode\": \"{m}\", \"category\": \"{}\", \"axis\": \"{}\", \"vec\": [{}], \"vec_name\": [{}]}}",
                jesc(&def.name),
                hints.join(","),
                jesc(def.category.as_deref().unwrap_or("")),
                jesc(def.axis_name.as_deref().unwrap_or("")),
                vs.join(","),
                vns.join(",")
            ));
            n += 1;
        }
        out.push_str("\n  ]\n}\n");
        let dest = std::env::var("MAPPER_OUT")
            .unwrap_or_else(|_| "research/mapper/corpus.json".to_string());
        if let Some(parent) = std::path::Path::new(&dest).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&dest, out).expect("write corpus");
        println!("wrote {n} entries to {dest}");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
