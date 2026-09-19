//! E57 — "what does this supersede?", on a gold that recency gets wrong.
//!
//! E56's L3 gold was "nearest earlier sentence sharing an entity", which hands
//! a position-only arm a large share for free. Here the gold is the sentence
//! whose fact the query **supersedes** — same subject and object, opposite
//! predicate — with distractors mentioning the same entity placed *between* the
//! two. Answering "most recent mention" is therefore wrong by construction.
//!
//! The corpus is GENERATED (`corpus-large.jsonl`) and is used for one thing:
//! closing intervals on retrieval arms. No structural claim rests on it.
//!
//! Run:
//!   PHYSIS_MODEL_DIR=../models/bge-base-en-v1.5 PHYSIS_MODEL_DIM=768 \
//!   cargo run -p physis-core --release --features embed-onnx \
//!       --example experiment57_supersession_retrieval
//!
//! Artifact: benchmarks/results/worldstate-e57.json
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use physis_core::embed::VectorEmbed;
use physis_core::worldstate as ws;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(serde::Deserialize)]
struct Rec {
    id: String,
    text: String,
    entities: Vec<String>,
    supersedes: Option<String>,
    gap: Option<usize>,
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
    let corpus = find("benchmarks/worldstate/corpus-large.jsonl")
        .expect("corpus-large.jsonl — run make_corpus_large.py");
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
    let pos_of: HashMap<&str, usize> = recs
        .iter()
        .enumerate()
        .map(|(i, r)| (r.id.as_str(), i))
        .collect();

    let (embedder, kind) = physis_core::embed::select(768);
    if kind == "random-projection" {
        eprintln!("WARNING: lexical hash — this measures word overlap, not meaning.");
    }
    let texts: Vec<&str> = recs.iter().map(|r| r.text.as_str()).collect();
    let vecs: Vec<Vec<f32>> = texts.iter().map(|t| embedder.embed(t)).collect();
    let sim = ws::sim_matrix(&vecs);

    // Gold: query position -> the position it supersedes. Every other position
    // has no gold and is skipped by `score_antecedents`.
    let mut gold: Vec<Option<usize>> = vec![None; n];
    for (i, r) in recs.iter().enumerate() {
        if let Some(g) = &r.supersedes {
            gold[i] = pos_of.get(g.as_str()).copied();
        }
    }

    let entity_of: Vec<&[String]> = recs.iter().map(|r| r.entities.as_slice()).collect();
    let blind = |p: usize| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).filter(|&q| q != p).collect();
        idx.sort_by(|&a, &b| sim[p][b].partial_cmp(&sim[p][a]).unwrap());
        idx.truncate(3);
        idx
    };
    let earlier = |p: usize| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..p).collect();
        idx.sort_by(|&a, &b| sim[p][b].partial_cmp(&sim[p][a]).unwrap());
        idx.truncate(3);
        idx
    };
    let recency = |p: usize| -> Vec<usize> { (1..=3).filter(|d| *d <= p).map(|d| p - d).collect() };
    // The world-model arm: earlier positions, restricted to the query's own
    // entities, ranked by similarity. Structure (entity) + order + geometry.
    let entity_earlier = |p: usize| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..p)
            .filter(|&q| entity_of[q].iter().any(|e| entity_of[p].contains(e)))
            .collect();
        idx.sort_by(|&a, &b| sim[p][b].partial_cmp(&sim[p][a]).unwrap());
        idx.truncate(3);
        idx
    };
    // E63's arms, on the regime that broke order-blind cosine outright: does a
    // word-level search function need the entity abstraction here?
    let owned: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
    let bm25 = physis_core::rag::Bm25Index::build(&owned);
    let terms: Vec<Vec<String>> = owned
        .iter()
        .map(|t| physis_core::rag::bm25_terms(t))
        .collect();
    let bm25_earlier = |p: usize| -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = (0..p)
            .map(|q| (q, bm25.score(q, &terms[p])))
            .filter(|(_, s)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(3);
        scored.into_iter().map(|(q, _)| q).collect()
    };
    let shared_word = |p: usize| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..p)
            .filter(|&q| terms[q].iter().any(|t| terms[p].contains(t)))
            .collect();
        idx.sort_by(|&a, &b| sim[p][b].partial_cmp(&sim[p][a]).unwrap());
        idx.truncate(3);
        idx
    };
    let perm: Vec<usize> = {
        let mut v: Vec<usize> = (0..n).collect();
        let mut rng = StdRng::seed_from_u64(20260914);
        for i in (1..n).rev() {
            v.swap(i, rng.gen_range(0..=i));
        }
        v
    };
    let shuffled = |p: usize| -> Vec<usize> {
        let fake = perm[p];
        let mut idx: Vec<usize> = (0..n).filter(|&q| perm[q] < fake).collect();
        idx.sort_by(|&a, &b| sim[p][b].partial_cmp(&sim[p][a]).unwrap());
        idx.truncate(3);
        idx
    };

    let arms = vec![
        ws::score_antecedents("cosine (blind)", &gold, blind, 20260914),
        ws::score_antecedents("cosine + pos", &gold, earlier, 20260914),
        ws::score_antecedents("recency only", &gold, recency, 20260914),
        ws::score_antecedents("entity+pos+cos", &gold, entity_earlier, 20260914),
        ws::score_antecedents("shuffled null", &gold, shuffled, 20260914),
        ws::score_antecedents("bm25 + position", &gold, bm25_earlier, 20260914),
        ws::score_antecedents("shared word + cos", &gold, shared_word, 20260914),
    ];

    println!(
        "corpus {} (GENERATED, {} sentences)  embedder={}",
        &sha[..12],
        n,
        kind
    );
    println!(
        "{} queries, gold = the superseded sentence, distractors between\n",
        gold.iter().filter(|g| g.is_some()).count()
    );
    println!("arm                 n     top1    top3    top1 95% CI");
    for a in &arms {
        println!(
            "{:<18} {:<5} {:.3}   {:.3}   [{:.3}, {:.3}]",
            a.arm, a.queries, a.top1, a.top3, a.top1_ci.0, a.top1_ci.1
        );
    }

    // Per-distance breakdown for the winning arm shape.
    let mut by_gap: HashMap<usize, (usize, usize)> = HashMap::new();
    for (p, g) in gold.iter().enumerate() {
        let Some(g) = g else { continue };
        let gap = recs[p].gap.unwrap_or(0);
        let hit = usize::from(entity_earlier(p).first() == Some(g));
        let e = by_gap.entry(gap).or_insert((0, 0));
        e.0 += hit;
        e.1 += 1;
    }
    let mut gaps: Vec<_> = by_gap.into_iter().collect();
    gaps.sort();
    println!("\nentity+pos+cos by distance to gold:");
    for (gap, (hit, tot)) in &gaps {
        println!(
            "  gap {gap:>3}  {hit}/{tot}  = {:.3}",
            *hit as f64 / *tot as f64
        );
    }

    let out = Path::new("benchmarks/results");
    let _ = std::fs::create_dir_all(out);
    let path = out.join("worldstate-e57.json");
    let json = serde_json::json!({
        "corpus_sha256": sha,
        "corpus": "corpus-large.jsonl (GENERATED)",
        "n_sentences": n,
        "embedder": kind,
        "arms": arms,
        "by_gap": gaps.iter().map(|(g, (h, t))| serde_json::json!({"gap": g, "hits": h, "queries": t})).collect::<Vec<_>>(),
    });
    std::fs::write(&path, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    println!("\nartifact -> {}", path.display());
}
