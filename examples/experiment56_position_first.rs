//! E56 — position in text as the temporal coordinate.
//!
//! Three legs, each against a null that can fire:
//!   L1  derive transition labels from the state sequence; null = shuffle the
//!       reading order and derive again.
//!   L2  do embedding neighbours sit near each other in the text?
//!   L3  "what preceded this?" — order-blind cosine vs cosine restricted to
//!       earlier positions vs position alone vs the shuffled-position null.
//!
//! Run:
//!   PHYSIS_MODEL_DIR=../models/bge-base-en-v1.5 PHYSIS_MODEL_DIM=768 \
//!   cargo run -p physis-core --release --features embed-onnx \
//!       --example experiment56_position_first
//!
//! Artifact: benchmarks/results/worldstate-e56.json
use std::path::{Path, PathBuf};

use physis_core::embed::VectorEmbed;
use physis_core::worldstate as ws;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(serde::Deserialize)]
struct Rec {
    text: String,
    entities: Vec<String>,
    relation: Option<Vec<String>>,
    transition: Option<String>,
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
    let corpus = find("benchmarks/worldstate/corpus.jsonl").expect("corpus.jsonl");
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
    let n = recs.len();

    let texts: Vec<&str> = recs.iter().map(|r| r.text.as_str()).collect();
    let entities: Vec<Vec<String>> = recs.iter().map(|r| r.entities.clone()).collect();
    let facts: Vec<Option<(String, String, String)>> = recs
        .iter()
        .map(|r| {
            r.relation
                .as_ref()
                .map(|v| (v[0].clone(), v[1].clone(), v[2].clone()))
        })
        .collect();
    let gold: Vec<String> = recs
        .iter()
        .map(|r| r.transition.clone().unwrap_or_else(|| "none".into()))
        .collect();

    // L1 — derivation vs gold, with the shuffled-order null.
    // PHYSIS_E56_GAPSWEEP=1 prints the whole RECUR_GAP curve and exits: the gap
    // decides the `recur` label, and E56 refused to report that label until the
    // constant had been swept rather than chosen.
    if std::env::var("PHYSIS_E56_GAPSWEEP").is_ok() {
        println!("recur_gap  accuracy  null±sd        z      recur-recall");
        for g in [2usize, 3, 4, 6, 8, 10, 12, 16] {
            let t = ws::score_transitions_gap(&entities, &facts, &texts, &gold, 500, 20260914, g);
            let rr = t
                .per_label
                .iter()
                .find(|(l, _, _)| l == "recur")
                .map(|(_, _, r)| *r)
                .unwrap_or(0.0);
            println!(
                "{:>9}  {:.3}     {:.3}±{:.3}  {:+.2}   {:.3}",
                g, t.accuracy, t.shuffled_null_mean, t.shuffled_null_sd, t.z, rr
            );
        }
        return;
    }
    let transitions = ws::score_transitions(&entities, &facts, &texts, &gold, 500, 20260914);

    // L2/L3 need the geometry.
    let (embedder, kind) = physis_core::embed::select(768);
    if kind == "random-projection" {
        eprintln!("WARNING: lexical hash — L2/L3 are then about word overlap, not meaning.");
    }
    let vecs: Vec<Vec<f32>> = texts.iter().map(|t| embedder.embed(t)).collect();
    let sim = ws::sim_matrix(&vecs);
    let k = 5usize;
    let adj = ws::mutual_knn(&sim, k);
    let (loc_obs, loc_null, loc_ratio) = ws::position_locality(&adj, 20260914, 20000);

    // L3 — gold is mechanical: nearest earlier sentence sharing an entity.
    // The gold is "nearest earlier sentence sharing an entity", which is
    // recency-biased by construction: a position-only arm that always answers
    // p-1 is handed a large share of it for free. PHYSIS_E56_MINGAP drops every
    // query whose gold antecedent is nearer than the gap, which is the check
    // that separates a real ordering effect from the artifact.
    let min_gap: usize = std::env::var("PHYSIS_E56_MINGAP")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    let gold_ant: Vec<Option<usize>> = ws::gold_antecedents(&entities)
        .into_iter()
        .enumerate()
        .map(|(p, g)| match g {
            Some(q) if p - q >= min_gap => Some(q),
            _ => None,
        })
        .collect();
    let order_blind = |p: usize| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).filter(|&q| q != p).collect();
        idx.sort_by(|&a, &b| sim[p][b].partial_cmp(&sim[p][a]).unwrap());
        idx.truncate(3);
        idx
    };
    let earlier_only = |p: usize| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..p).collect();
        idx.sort_by(|&a, &b| sim[p][b].partial_cmp(&sim[p][a]).unwrap());
        idx.truncate(3);
        idx
    };
    let position_only = |p: usize| -> Vec<usize> {
        (1..=3).filter(|d| *d <= p).map(|d| p - d).collect()
    };
    // Null: keep the arm, destroy the positions. A permutation is drawn once and
    // reused for every query, so the arm still sees a consistent (wrong) order.
    let perm: Vec<usize> = {
        let mut v: Vec<usize> = (0..n).collect();
        let mut rng = StdRng::seed_from_u64(20260914);
        for i in (1..n).rev() {
            v.swap(i, rng.gen_range(0..=i));
        }
        v
    };
    let shuffled = |p: usize| -> Vec<usize> {
        let fake_p = perm[p];
        let mut idx: Vec<usize> = (0..n).filter(|&q| perm[q] < fake_p).collect();
        idx.sort_by(|&a, &b| sim[p][b].partial_cmp(&sim[p][a]).unwrap());
        idx.truncate(3);
        idx
    };

    let antecedents = vec![
        ws::score_antecedents("cosine (blind)", &gold_ant, order_blind, 20260914),
        ws::score_antecedents("cosine + pos", &gold_ant, earlier_only, 20260914),
        ws::score_antecedents("position only", &gold_ant, position_only, 20260914),
        ws::score_antecedents("shuffled null", &gold_ant, shuffled, 20260914),
    ];

    let run = ws::PositionRun {
        corpus_sha256: sha,
        n_items: n,
        embedder: kind.to_string(),
        k,
        transitions,
        locality_observed: loc_obs,
        locality_null: loc_null,
        locality_ratio: loc_ratio,
        antecedents,
    };
    print!("{}", run.render());

    let out = Path::new("benchmarks/results");
    let _ = std::fs::create_dir_all(out);
    let tag = if min_gap > 0 { format!("-gap{min_gap}") } else { String::new() };
    let path = out.join(format!("worldstate-e56{tag}.json"));
    std::fs::write(&path, serde_json::to_vec_pretty(&run).unwrap()).unwrap();
    println!("\nartifact -> {}", path.display());
}
