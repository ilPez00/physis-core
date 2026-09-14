//! E65 — does measured redundancy predict which filter wins?
//!
//! Prints the statistic for every corpus in the series beside the arm that
//! actually won on it, so the prediction can be read against the record rather
//! than asserted.
//!
//! Run: cargo run -p physis-core --release --example experiment65_redundancy
//! Artifact: benchmarks/results/worldstate-e65.json
use std::path::PathBuf;

use physis_core::worldstate as ws;

#[derive(serde::Deserialize)]
struct Rec {
    text: String,
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
    // (label, path, the arm that won, its top-1, the losing arm's top-1)
    let corpora = [
        ("hand-written", "benchmarks/worldstate/corpus.jsonl", "entity link 0.357", "bm25 NOT RUN"),
        ("documentation", "benchmarks/worldstate/corpus-docs.jsonl", "bm25+pos 0.535", "entity link 0.209"),
        ("machine log", "benchmarks/worldstate/corpus-log.jsonl", "bm25+pos 0.447", "entity link 0.296"),
        ("generated templates", "benchmarks/worldstate/corpus-large.jsonl", "entity link 0.700", "bm25+pos 0.013"),
    ];

    println!("corpus                 n     redundancy  margin      winner (measured)");
    let mut rows = Vec::new();
    for (label, rel, winner, loser) in corpora {
        let Some(p) = find(rel) else {
            println!("{label:<22} MISSING {rel}");
            continue;
        };
        let body = std::fs::read_to_string(&p).unwrap();
        let recs: Vec<Rec> = body
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let texts: Vec<&str> = recs.iter().map(|r| r.text.as_str()).collect();
        let r = ws::redundancy(&texts, 200);
        let owned: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        let m = ws::lexical_margin(&owned, 150);
        println!(
            "{label:<22} {:<5} {r:.3}       {m:.3}       {winner:<20} (loser: {loser})",
            texts.len()
        );
        rows.push(serde_json::json!({
            "corpus": label, "n": texts.len(), "redundancy": r, "lexical_margin": m,
            "winner": winner, "loser": loser
        }));
    }

    let out = std::path::Path::new("benchmarks/results");
    let _ = std::fs::create_dir_all(out);
    let path = out.join("worldstate-e65.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
    println!("\nartifact -> {}", path.display());
}
