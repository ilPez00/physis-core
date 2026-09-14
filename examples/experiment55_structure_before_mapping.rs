//! E55 — structure before mapping, on the frozen WorldState corpus.
//!
//! Embeds the same 72 sentences in two independent representations, derives a
//! mutual-kNN graph in each **without reference to the other**, and asks four
//! arms to recover which node in A is which node in B. Identity is the gold
//! answer and is never shown to any matcher.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx \
//!       --example experiment55_structure_before_mapping
//!
//! Artifact: benchmarks/results/worldstate-e55.json
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use physis_core::embed::{RandomProjectionEmbedder, VectorEmbed};
use physis_core::worldstate as ws;

#[derive(serde::Deserialize)]
struct Rec {
    pos: usize,
    text: String,
    transition: Option<String>,
    split: String,
}

fn find(rel: &str) -> Option<PathBuf> {
    for base in ["", "../", "../../"] {
        let p = PathBuf::from(format!("{base}{rel}"));
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn main() {
    let corpus = find("benchmarks/worldstate/corpus.jsonl")
        .expect("benchmarks/worldstate/corpus.jsonl — run make_corpus.py first");
    let body = std::fs::read_to_string(&corpus).unwrap();
    let sha = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(body.as_bytes());
        format!("{:x}", h.finalize())
    };
    let recs: Vec<Rec> = body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    println!("corpus {} — {} sentences\n", &sha[..12], recs.len());

    // A: whatever `select` resolves to — bge if the model is on disk. B: the
    // word-hash lexical space. Two genuinely different constructions; the run
    // says which A actually was, because a claim made on a lexical hash in both
    // arms is not a cross-representation claim.
    let (embedder_a, kind_a) = physis_core::embed::select(768);
    let embedder_b = RandomProjectionEmbedder::new(384);
    if kind_a == "random-projection" {
        eprintln!(
            "WARNING: A resolved to random-projection — both spaces are then lexical hashes \
             and this run measures nothing about cross-representation structure."
        );
    }

    // Pre-registered defaults. The sweep below overrides them only to report
    // sensitivity; the headline row is k=5, profile=12, fixed before the run.
    let envnum = |n: &str, d: usize| {
        std::env::var(n).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(d)
    };
    let k = envnum("PHYSIS_E55_K", 5);
    let profile = envnum("PHYSIS_E55_PROFILE", 12);
    let n_anchors = 8usize;

    for split in ["all", "train", "heldout"] {
        let idx: Vec<usize> = (0..recs.len())
            .filter(|&i| split == "all" || recs[i].split == split)
            .collect();
        if idx.len() < 12 {
            continue;
        }
        let texts: Vec<&str> = idx.iter().map(|&i| recs[i].text.as_str()).collect();
        let va: Vec<Vec<f32>> = texts.iter().map(|t| embedder_a.embed(t)).collect();
        let vb: Vec<Vec<f32>> = texts.iter().map(|t| embedder_b.embed(t)).collect();

        let sa = ws::sim_matrix(&va);
        let sb = ws::sim_matrix(&vb);
        let aa = ws::mutual_knn(&sa, k);
        let ab = ws::mutual_knn(&sb, k);
        let ari_baseline = ws::ari(&ws::components(&aa), &ws::components(&ab));

        // Supervised labels: the gold transition class. Explicitly a supervised
        // condition — it reads the answer key's classes.
        let mut label_id: HashMap<String, usize> = HashMap::new();
        let labels: Vec<usize> = idx
            .iter()
            .map(|&i| {
                let key = recs[i].transition.clone().unwrap_or_else(|| "none".into());
                let next = label_id.len();
                *label_id.entry(key).or_insert(next)
            })
            .collect();
        let n_labels = label_id.len();

        // Anchors: evenly spaced by position, so the choice is not tuned.
        let step = idx.len() / n_anchors.max(1);
        let anchors: Vec<usize> = (0..n_anchors).map(|i| (i * step).min(idx.len() - 1)).collect();

        let arms = vec![
            ws::score_arm(
                "random",
                &ws::random_candidates(idx.len(), 3, 20260914),
                &aa,
                &ab,
                20260914,
            ),
            ws::score_arm(
                "anchor",
                &ws::rank_candidates(
                    &ws::anchor_descriptor(&sa, &anchors),
                    &ws::anchor_descriptor(&sb, &anchors),
                    3,
                ),
                &aa,
                &ab,
                20260914,
            ),
            ws::score_arm(
                "supervised",
                &ws::rank_candidates(
                    &ws::label_descriptor(&sa, &labels, n_labels),
                    &ws::label_descriptor(&sb, &labels, n_labels),
                    3,
                ),
                &aa,
                &ab,
                20260914,
            ),
            ws::score_arm(
                "structural",
                &ws::rank_candidates(
                    &ws::structural_descriptor(&sa, &aa, profile),
                    &ws::structural_descriptor(&sb, &ab, profile),
                    3,
                ),
                &aa,
                &ab,
                20260914,
            ),
        ];

        let run = ws::Run {
            corpus_sha256: sha.clone(),
            n_items: idx.len(),
            split: split.to_string(),
            embedder_a: kind_a.to_string(),
            embedder_b: "random-projection-384".to_string(),
            k,
            profile,
            anchors: n_anchors,
            ari_baseline,
            arms,
        };
        print!("{}", run.render());
        println!();

        let out = Path::new("benchmarks/results");
        let _ = std::fs::create_dir_all(out);
        let suffix = std::env::var("PHYSIS_E55_TAG").unwrap_or_default();
        let path = out.join(format!("worldstate-e55-{split}{suffix}.json"));
        std::fs::write(&path, serde_json::to_vec_pretty(&run).unwrap()).unwrap();
        println!("artifact -> {}\n", path.display());
    }
}
