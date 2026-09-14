//! E58 — what extraction costs.
//!
//! Every number in E56 and E57 was handed the corpus's gold entity annotation.
//! This run replaces it with links extracted from the text by three declared
//! rules (see `worldstate::extract_entities`) and reports two things:
//!
//!   1. extraction precision / recall / F1 against gold, on both corpora;
//!   2. E57's winning arm re-run on extracted links — the end-to-end number,
//!      with the gold-link run beside it as the upper bound it always was.
//!
//! Run:
//!   PHYSIS_MODEL_DIR=../models/bge-base-en-v1.5 PHYSIS_MODEL_DIM=768 \
//!   cargo run -p physis-core --release --features embed-onnx \
//!       --example experiment58_extraction
//!
//! Artifact: benchmarks/results/worldstate-e58.json
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use physis_core::embed::VectorEmbed;
use physis_core::worldstate as ws;

#[derive(serde::Deserialize)]
struct Rec {
    id: String,
    text: String,
    entities: Vec<String>,
    #[serde(default)]
    supersedes: Option<String>,
    #[serde(default)]
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

fn load(rel: &str) -> (String, Vec<Rec>) {
    let p = find(rel).unwrap_or_else(|| panic!("{rel} missing"));
    let body = std::fs::read_to_string(p).unwrap();
    let sha = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(body.as_bytes());
        format!("{:x}", h.finalize())
    };
    let recs = body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    (sha, recs)
}

fn main() {
    let min_df = 3usize;
    let mut report = serde_json::Map::new();

    // --- leg 1: extraction quality on both corpora -------------------------
    println!("extraction (3 rules, min_df = {min_df})\n");
    println!("corpus                     n     P       R       F1");
    let mut prf_rows = Vec::new();
    for (name, rel) in [
        ("corpus.jsonl (hand)", "benchmarks/worldstate/corpus.jsonl"),
        ("corpus-large (generated)", "benchmarks/worldstate/corpus-large.jsonl"),
    ] {
        let (_, recs) = load(rel);
        let texts: Vec<&str> = recs.iter().map(|r| r.text.as_str()).collect();
        let gold: Vec<Vec<String>> = recs.iter().map(|r| r.entities.clone()).collect();
        let got = ws::extract_entities(&texts, min_df);
        let (p, r, f) = ws::extraction_prf(&got, &gold);
        println!("{name:<26} {:<5} {p:.3}   {r:.3}   {f:.3}", recs.len());
        prf_rows.push(serde_json::json!({
            "corpus": name, "n": recs.len(), "precision": p, "recall": r, "f1": f
        }));
    }
    report.insert("extraction".into(), serde_json::Value::Array(prf_rows));

    // --- leg 1b: which rule costs what ------------------------------------
    // Run after the full extractor scored P = 0.162 on the generated corpus.
    println!("\nrule ablation (id = identifier-shaped, cap = capitalised, df = repeated)");
    println!("corpus                     rules        P       R       F1");
    let mut ablation = Vec::new();
    for (name, rel) in [
        ("corpus.jsonl (hand)", "benchmarks/worldstate/corpus.jsonl"),
        ("corpus-large (generated)", "benchmarks/worldstate/corpus-large.jsonl"),
    ] {
        let (_, recs) = load(rel);
        let texts: Vec<&str> = recs.iter().map(|r| r.text.as_str()).collect();
        let gold: Vec<Vec<String>> = recs.iter().map(|r| r.entities.clone()).collect();
        for (label, id, cap, df) in [
            ("id", true, false, false),
            ("cap", false, true, false),
            ("df", false, false, true),
            ("id+cap", true, true, false),
        ] {
            let got = ws::extract_entities_rules(&texts, min_df, id, cap, df);
            let (p, r, f) = ws::extraction_prf(&got, &gold);
            println!("{name:<26} {label:<12} {p:.3}   {r:.3}   {f:.3}");
            ablation.push(serde_json::json!({
                "corpus": name, "rules": label, "precision": p, "recall": r, "f1": f
            }));
        }
    }
    report.insert("ablation".into(), serde_json::Value::Array(ablation));

    // --- leg 2: E57's arm, gold links vs extracted links --------------------
    let (sha, recs) = load("benchmarks/worldstate/corpus-large.jsonl");
    let n = recs.len();
    let texts: Vec<&str> = recs.iter().map(|r| r.text.as_str()).collect();
    let (embedder, kind) = physis_core::embed::select(768);
    if kind == "random-projection" {
        eprintln!("WARNING: lexical hash — this measures word overlap, not meaning.");
    }
    let vecs: Vec<Vec<f32>> = texts.iter().map(|t| embedder.embed(t)).collect();
    let sim = ws::sim_matrix(&vecs);

    let pos_of: HashMap<&str, usize> = recs
        .iter()
        .enumerate()
        .map(|(i, r)| (r.id.as_str(), i))
        .collect();
    let mut gold_link: Vec<Option<usize>> = vec![None; n];
    for (i, r) in recs.iter().enumerate() {
        if let Some(g) = &r.supersedes {
            gold_link[i] = pos_of.get(g.as_str()).copied();
        }
    }

    let gold_ents: Vec<Vec<String>> = recs
        .iter()
        .map(|r| r.entities.iter().map(|e| ws::entity_key(e)).collect())
        .collect();
    let auto_ents = ws::extract_entities(&texts, min_df);

    let arm = |ents: &Vec<Vec<String>>| {
        let ents = ents.clone();
        let sim = &sim;
        move |p: usize| -> Vec<usize> {
            let mut idx: Vec<usize> = (0..p)
                .filter(|&q| ents[q].iter().any(|e| ents[p].contains(e)))
                .collect();
            idx.sort_by(|&a, &b| sim[p][b].partial_cmp(&sim[p][a]).unwrap());
            idx.truncate(3);
            idx
        }
    };

    let id_ents = ws::extract_entities_rules(&texts, min_df, true, false, false);
    let idcap_ents = ws::extract_entities_rules(&texts, min_df, true, true, false);
    let arms = vec![
        ws::score_antecedents("gold links", &gold_link, arm(&gold_ents), 20260914),
        ws::score_antecedents("extracted (all 3)", &gold_link, arm(&auto_ents), 20260914),
        ws::score_antecedents("extracted (id)", &gold_link, arm(&id_ents), 20260914),
        ws::score_antecedents("extracted (id+cap)", &gold_link, arm(&idcap_ents), 20260914),
    ];
    println!("\nE57 arm (entity + position + cosine), {} queries", arms[0].queries);
    println!("links               top1    top3    top1 95% CI");
    for a in &arms {
        println!(
            "{:<18} {:.3}   {:.3}   [{:.3}, {:.3}]",
            a.arm, a.top1, a.top3, a.top1_ci.0, a.top1_ci.1
        );
    }
    println!("\ncost of extraction, per link source:");
    for a in arms.iter().skip(1) {
        println!("  {:<18} {:+.3} top-1 vs gold links", a.arm, a.top1 - arms[0].top1);
    }

    // Per-distance for the extracted arm, so a loss is not hidden in the mean.
    let f = arm(&id_ents);
    let mut by_gap: HashMap<usize, (usize, usize)> = HashMap::new();
    for (p, g) in gold_link.iter().enumerate() {
        let Some(g) = g else { continue };
        let e = by_gap.entry(recs[p].gap.unwrap_or(0)).or_insert((0, 0));
        e.0 += usize::from(f(p).first() == Some(g));
        e.1 += 1;
    }
    let mut gaps: Vec<_> = by_gap.into_iter().collect();
    gaps.sort();
    println!("\nextracted (id) links by distance to gold:");
    for (gap, (hit, tot)) in &gaps {
        println!("  gap {gap:>3}  {hit}/{tot}  = {:.3}", *hit as f64 / *tot as f64);
    }

    report.insert("corpus_sha256".into(), serde_json::Value::String(sha));
    report.insert("embedder".into(), serde_json::Value::String(kind.to_string()));
    report.insert("min_df".into(), serde_json::json!(min_df));
    report.insert("arms".into(), serde_json::to_value(&arms).unwrap());
    report.insert(
        "by_gap".into(),
        serde_json::json!(gaps
            .iter()
            .map(|(g, (h, t))| serde_json::json!({"gap": g, "hits": h, "queries": t}))
            .collect::<Vec<_>>()),
    );
    let out = Path::new("benchmarks/results");
    let _ = std::fs::create_dir_all(out);
    let path = out.join("worldstate-e58.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&serde_json::Value::Object(report)).unwrap(),
    )
    .unwrap();
    println!("\nartifact -> {}", path.display());
}
