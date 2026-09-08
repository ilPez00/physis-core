//! E6 — export the 70 authored cell anchors with their vectors.
//!
//! Stage 11's root cause for the DOMAIN axis was that five words had no written
//! definition, so the anchors did not actually separate the axis. E5 found the
//! MODE axis is now the weaker half (kappa 0.418 against domain's 0.552) and
//! that it never received Stage 11's treatment. Testing the same root cause on
//! the mode axis needs the anchors themselves, embedded.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example mapper_anchor_export

fn jesc(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '"' => "\\\"".to_string(),
            '\\' => "\\\\".to_string(),
            '\n' | '\r' | '\t' => " ".to_string(),
            c => c.to_string(),
        })
        .collect()
}

fn main() {
    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed::VectorEmbed;
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

        let Some(ap) = ["physis-core/config", "config", "../config"]
            .iter()
            .map(|d| std::path::Path::new(d).join("mode_anchors_ontology.json"))
            .find(|p| p.exists())
        else {
            println!("WARNING: mode_anchors_ontology.json not found — aborting.");
            return;
        };
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
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&ap).expect("read")).expect("parse");

        let mut out =
            String::from("{\n  \"source\": \"mapper_anchor_export\",\n  \"anchors\": [\n");
        let mut first = true;
        let mut n = 0;
        for a in v["domains"].as_array().into_iter().flatten() {
            let (Some(d), Some(m), Some(nm)) =
                (a["domain"].as_str(), a["mode"].as_str(), a["name"].as_str())
            else {
                continue;
            };
            let hints: Vec<&str> = a["hints"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|h| h.as_str())
                .collect();
            // Two texts on purpose: the anchor as authored (name + hints), and
            // the hints alone. If an axis is defined only by its cell NAMES and
            // not by what the hints say, the two will behave very differently.
            let full = format!("{nm} {}", hints.join(" "));
            let vf: Vec<String> = e.embed(&full).iter().map(|x| format!("{x:.5}")).collect();
            let vh: Vec<String> = e
                .embed(&hints.join(" "))
                .iter()
                .map(|x| format!("{x:.5}"))
                .collect();
            if !first {
                out.push_str(",\n");
            }
            first = false;
            out.push_str(&format!(
                "    {{\"domain\": \"{d}\", \"mode\": \"{m}\", \"name\": \"{}\", \"n_hints\": {}, \"vec\": [{}], \"vec_hints\": [{}]}}",
                jesc(nm), hints.len(), vf.join(","), vh.join(",")
            ));
            n += 1;
        }
        out.push_str("\n  ]\n}\n");
        std::fs::create_dir_all("research/mapper").ok();
        std::fs::write("research/mapper/anchors.json", out).expect("write");
        println!("wrote {n} anchors to research/mapper/anchors.json");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
