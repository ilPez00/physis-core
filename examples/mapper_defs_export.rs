//! E12 — embed the WRITTEN axis definitions, as an alternative to the authored
//! anchors.
//!
//! E11's miner pulled concrete machinery for CONSTRUCT cells (crane, hoist,
//! freight elevator) although `docs/GRID_AXES.md` defines CONSTRUCT as "the
//! durable arrangement other work happens inside or on top of". The authored
//! anchor and the written definition disagree, and the anchor is what the
//! system actually uses.
//!
//! Stage 11 fixed the domain axis by writing a definition and regenerating
//! against it. This exports the definitions in the same embedded form as the
//! anchors so the two can be compared directly, cell by cell.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example mapper_defs_export

const DOMAINS: [(&str, &str); 5] = [
    (
        "HEAL",
        "condition — the state and capacity of something already running",
    ),
    (
        "CONSTRUCT",
        "structure — the durable arrangement other work happens inside or on top of",
    ),
    (
        "FABRICATE",
        "output — discrete artifacts and throughput a process emits",
    ),
    ("BOND", "relation — coupling between agents or components"),
    (
        "STUDY",
        "knowledge — acquiring and reasoning about information",
    ),
];

const MODES: [(&str, &str); 14] = [
    ("LIFT", "peak, intensity, maximum effort or load"),
    ("REST", "recover, idle, pause"),
    ("WALK", "steady flow, ongoing operation at nominal rhythm"),
    ("WORK", "execute, labour, do the routine doing"),
    ("CREATE", "make or design something that did not exist"),
    ("LEARN", "study or practise, self-directed acquisition"),
    ("DESTROY", "tear down, remove, end"),
    ("SENSE", "perceive, measure, observe"),
    ("GUIDE", "lead, mentor, direct others"),
    ("PLAY", "explore or improvise, low stakes"),
    ("BRAINSTORM", "ideate, diverge"),
    ("MAINTAIN", "upkeep, repair, preserve against decay"),
    ("MOVE", "relocate, transport, logistics"),
    ("PLAN", "sequence, organise, converge"),
];

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
        let mut out = String::from("{\n  \"source\": \"mapper_defs_export\",\n  \"cells\": [\n");
        let mut first = true;
        for (d, dd) in DOMAINS {
            for (m, md) in MODES {
                // The cell's definition is the conjunction of its two axis
                // definitions — nothing hand-authored per cell, which is the
                // point of the comparison.
                let text = format!("{d} {dd}. {m} {md}.");
                let v: Vec<String> = e.embed(&text).iter().map(|x| format!("{x:.5}")).collect();
                if !first {
                    out.push_str(",\n");
                }
                first = false;
                out.push_str(&format!(
                    "    {{\"domain\": \"{d}\", \"mode\": \"{m}\", \"vec\": [{}]}}",
                    v.join(",")
                ));
            }
        }
        out.push_str("\n  ]\n}\n");
        std::fs::create_dir_all("research/mapper").ok();
        std::fs::write("research/mapper/definitions.json", out).expect("write");
        println!("wrote research/mapper/definitions.json (70 cells)");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
