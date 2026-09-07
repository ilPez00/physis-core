//! Experiment 36 — the EPISODIC form of the depth-1 nesting proposal.
//!
//! Iteration 34 tested it with the authored `hints` and got nothing (z = +0.62),
//! because hints are inside the embedded text — the embedder had already read
//! every token being scored. Iteration 35 re-ran it with WordNet glosses, a
//! source the embedder never saw, and it held (z = +6.64 same-cell). But a
//! gloss is a DEFINITION. The proposal is episodic:
//!
//!   "the dog ate my homework ... if i then encounter a description of the dog,
//!    which may have large white fangs that ripped through the virgin paper i
//!    write my homework in ... where 'white' is a second-grade link, nested in
//!    both primitives"
//!
//! `white` is not in a dictionary entry for `dog`. It is in THIS episode. So the
//! second-grade elements here come from episodes — real contexts in which the
//! entry's terms actually appeared — drawn from the operational corpus
//! populated from this machine (6234 events, 200,986 words: document
//! paragraphs, machine telemetry, commit bodies).
//!
//! ## Independence, enforced three ways
//!
//! 1. Episode tokens that appear in an entry's own embedded text are dropped,
//!    so the embedding never received them.
//! 2. At pair time a shared token is counted only if it is in NEITHER entry's
//!    embedded text — a token absent from i's text but present in j's is still
//!    something j's embedding got.
//! 3. The entry's own NAME terms are dropped, since those are what matched the
//!    episode in the first place and would corroborate trivially.
//!
//! ## Matching
//!
//! An episode belongs to an entry when it contains at least 2 of the entry's
//! name terms. Single-term matching is too loose — "Data Engineering" and "Data
//! Science" would both match every episode containing "data" and then share all
//! of its tokens for free. Full-name substring matching is reported alongside
//! as a stricter robustness check.
//!
//! Controls are unchanged from 34/35: stratify by cosine so geometry is held
//! ~fixed, IDF-weight so `the`/`and` cannot carry it, report the residual
//! cosine imbalance per stratum, and read the verdict only where that control
//! actually held.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment36_episodic_substructure -- <corpus.json>

use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Deserialize)]
struct Event {
    #[serde(default)]
    subject: String,
    #[serde(default)]
    evidence: Vec<String>,
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter().map(|x| x / n).collect()
}

fn z_prop(p1: f64, n1: f64, p2: f64, n2: f64) -> f64 {
    if n1 <= 0.0 || n2 <= 0.0 {
        return 0.0;
    }
    let pooled = (p1 * n1 + p2 * n2) / (n1 + n2);
    let se = (pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)).sqrt();
    if se <= 0.0 {
        0.0
    } else {
        (p1 - p2) / se
    }
}

fn words(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|w| w.len() > 2)
        .collect()
}

fn main() {
    println!("Experiment 36: episodic second-grade structure — the proposal's actual form\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed::VectorEmbed;
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

        let corpus_path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "operational_corpus.json".to_string());
        let Ok(raw) = std::fs::read_to_string(&corpus_path) else {
            println!("WARNING: cannot read {corpus_path} — run populate_operational first.");
            return;
        };
        let Ok(events) = serde_json::from_str::<Vec<Event>>(&raw) else {
            println!("WARNING: {corpus_path} is not a Vec<Event>.");
            return;
        };
        let episodes: Vec<(String, Vec<String>)> = events
            .iter()
            .map(|e| {
                let text = format!("{} {}", e.subject, e.evidence.join(" ")).to_lowercase();
                let toks = {
                    let mut t = words(&text);
                    t.sort();
                    t.dedup();
                    t
                };
                (text, toks)
            })
            .collect();
        println!("corpus: {} episodes", episodes.len());

        let Some(dir) = ["models", "../models"]
            .iter()
            .find(|d| std::path::Path::new(d).join("model.onnx").exists())
        else {
            println!("WARNING: MiniLM not available — aborting.");
            return;
        };
        let embedder = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 384,
            model_dir: Some(dir.to_string()),
            pooling: PoolingStrategy::Mean,
            ..OnnxConfig::default()
        });
        if !embedder.is_available() {
            println!("WARNING: embedder unavailable — aborting.");
            return;
        }

        let ontology = OntologyLoader::load_all();
        let mut texts = Vec::new();
        let mut cells = Vec::new();
        let mut own: Vec<HashSet<String>> = Vec::new();
        let mut name_terms: Vec<Vec<String>> = Vec::new();
        for def in ontology.classification_domains() {
            let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
            let mut t = def.name.clone();
            for h in &def.hints {
                t.push(' ');
                t.push_str(h);
            }
            own.push(words(&t).into_iter().collect());
            name_terms.push(words(&def.name));
            texts.push(t);
            cells.push((d.clone(), m.clone()));
        }
        let n = texts.len();
        println!("{n} ontology entries, embedding...");
        let emb: Vec<Vec<f32>> = texts.iter().map(|t| normalize(&embedder.embed(t))).collect();

        for (mode_label, strict) in [("MATCH: >=2 name terms", false), ("MATCH: full-name substring", true)] {
            println!("\n############ {mode_label} ############\n");

            // Episode tokens per entry, minus everything the embedding saw.
            let mut sub: Vec<Vec<String>> = Vec::with_capacity(n);
            let mut matched = 0usize;
            let mut total_eps = 0usize;
            for i in 0..n {
                let nm = texts[i]
                    .split_whitespace()
                    .take(1)
                    .collect::<String>()
                    .to_lowercase();
                let name_lc: String = name_terms[i].join(" ");
                let mut bag: HashMap<String, usize> = HashMap::new();
                let mut eps = 0usize;
                for (etext, etoks) in &episodes {
                    let hit = if strict {
                        !name_lc.is_empty() && etext.contains(&name_lc)
                    } else {
                        name_terms[i].len() >= 2
                            && name_terms[i].iter().filter(|w| etoks.binary_search(w).is_ok()).count() >= 2
                    };
                    if !hit {
                        continue;
                    }
                    eps += 1;
                    for t in etoks {
                        // (1) never something the embedding read, and
                        // (3) never one of the entry's own name terms, which
                        // are what matched the episode to begin with.
                        if !own[i].contains(t) && !name_terms[i].contains(t) {
                            *bag.entry(t.clone()).or_default() += 1;
                        }
                    }
                }
                let _ = nm;
                if eps > 0 {
                    matched += 1;
                    total_eps += eps;
                }
                let mut v: Vec<String> = bag.into_keys().collect();
                v.sort();
                sub.push(v);
            }
            println!(
                "  entries with >=1 episode: {matched}/{n} ({:.0}%), mean {:.1} episodes each",
                100.0 * matched as f64 / n as f64,
                total_eps as f64 / matched.max(1) as f64
            );
            if matched < 40 {
                println!("  too few matched entries to test — skipping.");
                continue;
            }

            let mut df: HashMap<&str, usize> = HashMap::new();
            for v in &sub {
                for w in v {
                    *df.entry(w.as_str()).or_default() += 1;
                }
            }
            let idf = |w: &str| -> f64 { (n as f64 / *df.get(w).unwrap_or(&1) as f64).ln() };

            struct Pair {
                cos: f32,
                overlap: f64,
                same_cell: bool,
                same_domain: bool,
            }
            let mut pairs: Vec<Pair> = Vec::new();
            for i in 0..n {
                if sub[i].is_empty() {
                    continue;
                }
                for j in (i + 1)..n {
                    if sub[j].is_empty() {
                        continue;
                    }
                    let (a, b) = (&sub[i], &sub[j]);
                    let (mut x, mut y, mut ov) = (0usize, 0usize, 0.0f64);
                    while x < a.len() && y < b.len() {
                        match a[x].cmp(&b[y]) {
                            std::cmp::Ordering::Less => x += 1,
                            std::cmp::Ordering::Greater => y += 1,
                            std::cmp::Ordering::Equal => {
                                // (2) shared only if NEITHER embedding saw it.
                                if !own[i].contains(&a[x]) && !own[j].contains(&a[x]) {
                                    ov += idf(&a[x]);
                                }
                                x += 1;
                                y += 1;
                            }
                        }
                    }
                    pairs.push(Pair {
                        cos: cosine_sim(&emb[i], &emb[j]),
                        overlap: ov,
                        same_cell: cells[i] == cells[j],
                        same_domain: cells[i].0 == cells[j].0,
                    });
                }
            }
            println!("  {} pairs among matched entries", pairs.len());
            if pairs.len() < 2000 {
                println!("  too few pairs to stratify — skipping.");
                continue;
            }

            let mut order: Vec<usize> = (0..pairs.len()).collect();
            order.sort_by(|&a, &b| pairs[a].cos.partial_cmp(&pairs[b].cos).unwrap());
            const STRATA: usize = 40;
            const MAX_IMB: f64 = 0.01;
            let per = order.len() / STRATA;

            for (label, fine) in [("same-cell ", true), ("same-domain", false)] {
                let (mut wth, mut wnh, mut wtl, mut wnl, mut wk) =
                    (0usize, 0usize, 0usize, 0usize, 0usize);
                let (mut th, mut nh, mut tl, mut nl) = (0usize, 0usize, 0usize, 0usize);
                for s in 0..STRATA {
                    let lo = s * per;
                    let hi = if s == STRATA - 1 { order.len() } else { (s + 1) * per };
                    let mut idx: Vec<usize> = order[lo..hi].to_vec();
                    if idx.len() < 20 {
                        continue;
                    }
                    idx.sort_by(|&a, &b| pairs[a].overlap.partial_cmp(&pairs[b].overlap).unwrap());
                    let mid = idx.len() / 2;
                    let hit = |k: &usize| if fine { pairs[*k].same_cell } else { pairs[*k].same_domain };
                    let (lo_i, hi_i) = idx.split_at(mid);
                    let (hc, lc) = (
                        hi_i.iter().filter(|k| hit(k)).count(),
                        lo_i.iter().filter(|k| hit(k)).count(),
                    );
                    let mc = |v: &[usize]| {
                        v.iter().map(|&k| pairs[k].cos as f64).sum::<f64>() / v.len() as f64
                    };
                    th += hc;
                    nh += hi_i.len();
                    tl += lc;
                    nl += lo_i.len();
                    if (mc(hi_i) - mc(lo_i)).abs() < MAX_IMB {
                        wth += hc;
                        wnh += hi_i.len();
                        wtl += lc;
                        wnl += lo_i.len();
                        wk += 1;
                    }
                }
                let (phr, plr) = (th as f64 / nh as f64, tl as f64 / nl as f64);
                let (wphr, wplr) = (wth as f64 / wnh.max(1) as f64, wtl as f64 / wnl.max(1) as f64);
                println!(
                    "  {label}  all-strata hi {phr:.4} lo {plr:.4} z={:+.2}   |   well-controlled hi {wphr:.4} lo {wplr:.4} z={:+.2} ({wk}/{STRATA})",
                    z_prop(phr, nh as f64, plr, nl as f64),
                    z_prop(wphr, wnh as f64, wplr, wnl as f64)
                );
            }
        }

        println!(
            "\n(Independence is enforced three ways rather than assumed: episode tokens the\n entry's embedding already read are dropped; a shared token counts only when\n NEITHER embedding read it; and the entry's own name terms are dropped, since\n those are what matched the episode and would corroborate for free. Matching on a\n single name term was rejected — \"Data Engineering\" and \"Data Science\" would both\n match every episode containing \"data\" and share its whole bag. What is still NOT\n controlled: two entries can match the same episode for unrelated reasons, and\n this corpus is about ONE domain — physis's own development — so the episodes are\n not a neutral sample of the world.)"
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
