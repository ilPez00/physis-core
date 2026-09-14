//! E59 — noun-phrase links on the corpus that has no identifiers.
//!
//! E58 ended with the pipeline end-to-end complete where entities are machine
//! ids, and untested where they are noun phrases: the `id` rule recovers 30.4%
//! of the hand corpus's entities. This run adds a POS-tagger-free noun-phrase
//! chunker and scores it two ways — against gold entity sets, and through the
//! downstream arm that consumes the links.
//!
//! Downstream task here is E56's L3 with a gap filter (the hand corpus has no
//! supersession field): "what earlier sentence about this thing does this one
//! follow?", restricted to golds at least `gap` positions back so recency
//! cannot win by default.
//!
//! Run:
//!   PHYSIS_MODEL_DIR=../models/bge-base-en-v1.5 PHYSIS_MODEL_DIM=768 \
//!   cargo run -p physis-core --release --features embed-onnx \
//!       --example experiment59_noun_phrases
//!
//! Artifact: benchmarks/results/worldstate-e59.json
use std::path::{Path, PathBuf};

use physis_core::embed::VectorEmbed;
use physis_core::worldstate as ws;

#[derive(serde::Deserialize)]
struct Rec {
    text: String,
    entities: Vec<String>,
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
    let min_df = 3usize;
    let max_len = 3usize;
    let gap = 3usize;

    // PHYSIS_CORPUS points the same two scores at a different corpus. E60 uses
    // it for `corpus-docs.jsonl` — this repository's own documentation, whose
    // gold entity sets are the author's backticked spans, written years before
    // this chunker existed and never with it in mind.
    let rel = std::env::var("PHYSIS_CORPUS")
        .unwrap_or_else(|_| "benchmarks/worldstate/corpus.jsonl".to_string());
    let path = find(&rel).unwrap_or_else(|| panic!("{rel} missing"));
    let body = std::fs::read_to_string(&path).unwrap();
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
    let gold_ents: Vec<Vec<String>> = recs
        .iter()
        .map(|r| r.entities.iter().map(|e| ws::entity_key(e)).collect())
        .collect();

    let id_only = ws::extract_entities_rules(&texts, min_df, true, true, false);
    let df_only = ws::extract_entities_rules(&texts, min_df, false, false, true);
    let np = ws::extract_noun_phrases(&texts, max_len);
    let np_id = ws::merge_entities(&np, &id_only);
    let code = ws::extract_code_identifiers(&texts);
    let np_code = ws::merge_entities(&np_id, &code);

    // PHYSIS_E59_DUMP writes the rule links per sentence, so an external
    // extractor can be asked for exactly the sentences the rules missed.
    if let Ok(dump) = std::env::var("PHYSIS_E59_DUMP") {
        let rows: Vec<serde_json::Value> = (0..n)
            .map(|i| {
                serde_json::json!({"pos": i, "text": texts[i], "rule_links": np_code[i]})
            })
            .collect();
        std::fs::write(&dump, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
        println!("rule links -> {dump} ({} of {n} sentences have none)",
            np_code.iter().filter(|e| e.is_empty()).count());
    }

    // PHYSIS_E59_EXTRA merges an external extractor's links (E61: a local
    // interpreter, asked only where the rules produced nothing).
    let extra: Option<Vec<Vec<String>>> = std::env::var("PHYSIS_E59_EXTRA").ok().map(|path| {
        let body = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{path} missing"));
        let rows: Vec<serde_json::Value> = serde_json::from_str(&body).unwrap();
        let mut out = vec![Vec::new(); n];
        for r in rows {
            let pos = r["pos"].as_u64().unwrap_or(0) as usize;
            if pos < n {
                out[pos] = r["links"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str())
                            .map(ws::entity_key)
                            .filter(|k| k.len() > 2)
                            .collect()
                    })
                    .unwrap_or_default();
            }
        }
        out
    });

    println!("{rel}\ncorpus {} ({} sentences), max_len = {max_len}\n", &sha[..12], n);
    println!("extractor        P       R       F1");
    let mut rows = Vec::new();
    for (label, got) in [
        ("id+cap", &id_only),
        ("df", &df_only),
        ("noun phrase", &np),
        ("np + id+cap", &np_id),
        ("code ids", &code),
        ("np + code", &np_code),
    ]
    .into_iter()
    .chain(extra.iter().flat_map(|e| {
        vec![
            ("interpreter", e as &Vec<Vec<String>>),
            ("rules + interp", Box::leak(Box::new(ws::merge_entities(&np_code, e)))),
        ]
    })) {
        let (p, r, f) = ws::extraction_prf(got, &gold_ents);
        println!("{label:<15} {p:.3}   {r:.3}   {f:.3}");
        rows.push(serde_json::json!({"extractor": label, "precision": p, "recall": r, "f1": f}));
    }

    // Downstream: does the chunker's output carry the arm?
    let (embedder, kind) = physis_core::embed::select(768);
    if kind == "random-projection" {
        eprintln!("WARNING: lexical hash — this measures word overlap, not meaning.");
    }
    // E64 — PHYSIS_E64_REP selects what produces the sentence vectors:
    //   model       the embedder runs on every sentence (the default, E55–E63)
    //   table-model the embedder runs ONCE PER VOCABULARY WORD; sentences are the
    //               mean of their words' looked-up vectors. No model at query time.
    //   table-count a table built by counting alone — random indexing with PPMI
    //               over a co-occurrence window. No model at any time.
    let rep = std::env::var("PHYSIS_E64_REP").unwrap_or_else(|_| "model".into());
    let dim = 768usize;
    let t_build = std::time::Instant::now();
    let vecs: Vec<Vec<f32>> = match rep.as_str() {
        "table-model" => {
            let vocab = ws::vocabulary(&texts, 2);
            let table: std::collections::HashMap<String, Vec<f32>> = vocab
                .iter()
                .map(|w| (w.clone(), embedder.embed(w)))
                .collect();
            eprintln!("table-model: {} words embedded once", table.len());
            texts
                .iter()
                .map(|t| ws::sentence_from_table(&table, t, dim))
                .collect()
        }
        "table-count" => {
            let table = ws::word_table_count(&texts, 512, 4);
            eprintln!("table-count: {} words counted, no model used", table.len());
            texts
                .iter()
                .map(|t| ws::sentence_from_table(&table, t, 512))
                .collect()
        }
        _ => texts.iter().map(|t| embedder.embed(t)).collect(),
    };
    eprintln!("representation {rep} built in {:?}", t_build.elapsed());
    let sim = ws::sim_matrix(&vecs);
    let gold_link: Vec<Option<usize>> = ws::gold_antecedents(&gold_ents)
        .into_iter()
        .enumerate()
        .map(|(p, g)| match g {
            Some(q) if p - q >= gap => Some(q),
            _ => None,
        })
        .collect();

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
    // E63 — the hypothesis that the entity abstraction is unnecessary at this
    // scale, and plain word-level search is enough. Three arms, all lexical:
    // BM25 over earlier positions (a search engine, word per word), a
    // shared-token filter ranked by cosine, and the same restricted to rare
    // tokens (the IDF intuition, made an arm rather than an argument).
    let owned: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
    let bm25 = physis_core::rag::Bm25Index::build(&owned);
    let terms: Vec<Vec<String>> = owned
        .iter()
        .map(|t| physis_core::rag::bm25_terms(t))
        .collect();
    let mut df: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for ts in &terms {
        let mut seen: Vec<&str> = Vec::new();
        for t in ts {
            if !seen.contains(&t.as_str()) {
                seen.push(t.as_str());
                *df.entry(t.as_str()).or_insert(0) += 1;
            }
        }
    }
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
    let rare_word = |p: usize| -> Vec<usize> {
        let rare: Vec<&String> = terms[p]
            .iter()
            .filter(|t| df.get(t.as_str()).copied().unwrap_or(0) <= 3)
            .collect();
        let mut idx: Vec<usize> = (0..p)
            .filter(|&q| terms[q].iter().any(|t| rare.contains(&t)))
            .collect();
        idx.sort_by(|&a, &b| sim[p][b].partial_cmp(&sim[p][a]).unwrap());
        idx.truncate(3);
        idx
    };
    let blind = |p: usize| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).filter(|&q| q != p).collect();
        idx.sort_by(|&a, &b| sim[p][b].partial_cmp(&sim[p][a]).unwrap());
        idx.truncate(3);
        idx
    };

    let arms = vec![
        ws::score_antecedents("gold links", &gold_link, arm(&gold_ents), 20260914),
        ws::score_antecedents("noun phrase", &gold_link, arm(&np), 20260914),
        ws::score_antecedents("np + id+cap", &gold_link, arm(&np_id), 20260914),
        ws::score_antecedents("id+cap only", &gold_link, arm(&id_only), 20260914),
        ws::score_antecedents("df only", &gold_link, arm(&df_only), 20260914),
        ws::score_antecedents("code ids", &gold_link, arm(&code), 20260914),
        ws::score_antecedents("np + code", &gold_link, arm(&np_code), 20260914),
        ws::score_antecedents("cosine (blind)", &gold_link, blind, 20260914),
        ws::score_antecedents("bm25 + position", &gold_link, bm25_earlier, 20260914),
        ws::score_antecedents("shared word + cos", &gold_link, shared_word, 20260914),
        ws::score_antecedents("rare word + cos", &gold_link, rare_word, 20260914),
    ];
    let merged = extra.as_ref().map(|e| ws::merge_entities(&np_code, e));
    let mut arms = arms;
    if let Some(e) = &extra {
        arms.push(ws::score_antecedents("interpreter", &gold_link, arm(e), 20260914));
    }
    if let Some(m) = &merged {
        arms.push(ws::score_antecedents("rules + interp", &gold_link, arm(m), 20260914));
    }
    println!(
        "\ndownstream: antecedent at gap >= {gap}, {} queries",
        arms[0].queries
    );
    println!("links               top1    top3    top1 95% CI");
    for a in &arms {
        println!(
            "{:<18} {:.3}   {:.3}   [{:.3}, {:.3}]",
            a.arm, a.top1, a.top3, a.top1_ci.0, a.top1_ci.1
        );
    }

    let out = Path::new("benchmarks/results");
    let _ = std::fs::create_dir_all(out);
    let tag = std::env::var("PHYSIS_E59_TAG").unwrap_or_default();
    let p = out.join(format!("worldstate-e59{tag}.json"));
    std::fs::write(
        &p,
        serde_json::to_vec_pretty(&serde_json::json!({
            "corpus_sha256": sha, "corpus": rel, "n_sentences": n,
            "embedder": kind, "min_df": min_df, "max_len": max_len, "gap": gap,
            "extraction": rows, "arms": arms,
        }))
        .unwrap(),
    )
    .unwrap();
    println!("\nartifact -> {}", p.display());
}
