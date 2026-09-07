// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 35 — the same question as Iteration 34, with second-grade
//! structure the embedder never saw.
//!
//! Iteration 34 tested "does a shared inward element corroborate an outward
//! link" using the authored `hints`, and got nothing (z = +0.62 / +1.72 over
//! well-controlled strata). But that test was rigged toward the null: hints are
//! part of the embedded text (`name + hints`), so the embedder had already read
//! every token being scored. Asking whether that overlap adds information
//! beyond the embedding is close to asking whether a model failed to absorb
//! tokens it read.
//!
//! The proposal's real form needs a SEPARATE encounter — "if i then encounter a
//! description of the dog". physis's operational corpus would be the right
//! source, but it is empty on this machine (`nodes: 0`, `edges: 0`).
//!
//! ## What is used instead, and why it qualifies
//!
//! WordNet glosses. A gloss is a description of a word written by
//! lexicographers:
//!
//!   - it is NOT part of the embedded text, so the embedder never read it;
//!   - it was not written by whoever authored physis's hints, so it is not the
//!     same hand labelling twice;
//!   - it is a description of the thing, which is the shape the proposal needs.
//!
//! On top of that, every gloss token that already appears in an entry's own
//! embedded text is DISCARDED. What remains can only be information the
//! embedding did not receive — which is the whole point of the test.
//!
//! ## Where it falls short of the proposal
//!
//! A dictionary gloss is decontextualised. "The dog ate my homework" supplies
//! situational co-occurrence — fangs, paper, white, in THIS episode. A gloss
//! supplies the general definition instead. So this tests "independent
//! description" but not "independent episode", and a null here would leave the
//! episodic version still open.
//!
//! Controls are Iteration 34's, unchanged: stratify by cosine so geometry is
//! held ~fixed, IDF-weight so `the`/`and` cannot carry the signal, print the
//! residual cosine imbalance per stratum, and read the verdict only from strata
//! where that control actually held.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment35_independent_substructure

use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use std::collections::{HashMap, HashSet};

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

/// lemma -> gloss, from a WordNet `data.<pos>` file. First gloss per lemma
/// wins, deterministically (file order is fixed).
fn wordnet_glosses(path: &std::path::Path, out: &mut HashMap<String, String>) {
    let Ok(text) = std::fs::read_to_string(path) else { return };
    for line in text.lines() {
        if line.starts_with("  ") || line.trim().is_empty() {
            continue;
        }
        let Some((head, gloss)) = line.split_once(" | ") else { continue };
        let f: Vec<&str> = head.split_whitespace().collect();
        if f.len() < 5 {
            continue;
        }
        let lemma = f[4].replace('_', " ").to_lowercase();
        out.entry(lemma).or_insert_with(|| gloss.trim().to_string());
    }
}

fn main() {
    println!("Experiment 35: shared substructure from a source the embedder never saw\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed::VectorEmbed;
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

        let Some(dir) = ["models", "../models"]
            .iter()
            .find(|d| std::path::Path::new(d).join("model.onnx").exists())
        else {
            println!("WARNING: MiniLM not available — aborting.");
            return;
        };
        let Some(wn_dir) = ["models/wordnet", "../models/wordnet"]
            .iter()
            .map(std::path::Path::new)
            .find(|d| d.join("data.noun").exists())
        else {
            println!("WARNING: WordNet not found — aborting.");
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

        let mut gloss: HashMap<String, String> = HashMap::new();
        for f in ["data.noun", "data.verb", "data.adj"] {
            wordnet_glosses(&wn_dir.join(f), &mut gloss);
        }
        println!("WordNet lemmas with glosses: {}", gloss.len());

        let ontology = OntologyLoader::load_all();
        let mut texts = Vec::new();
        let mut cells = Vec::new();
        let mut own: Vec<HashSet<String>> = Vec::new(); // tokens the embedder DID see
        for def in ontology.classification_domains() {
            let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
            let mut t = def.name.clone();
            for h in &def.hints {
                t.push(' ');
                t.push_str(h);
            }
            own.push(words(&t).into_iter().collect());
            texts.push(t);
            cells.push((d.clone(), m.clone()));
        }
        let n = texts.len();
        println!("{n} entries, embedding...");
        let emb: Vec<Vec<f32>> = texts.iter().map(|t| normalize(&embedder.embed(t))).collect();

        // Second-grade elements: content tokens of the WordNet glosses of the
        // entry's own NAME words. Every token the embedder already saw for this
        // entry is dropped, so what survives is strictly information the
        // embedding did not receive.
        let mut sub: Vec<Vec<String>> = Vec::with_capacity(n);
        let mut with_any = 0usize;
        for (i, def) in ontology
            .classification_domains()
            .filter(|d| d.domain.is_some() && d.mode.is_some())
            .enumerate()
        {
            let mut v: HashSet<String> = HashSet::new();
            for w in words(&def.name) {
                if let Some(g) = gloss.get(&w) {
                    for t in words(g) {
                        if !own[i].contains(&t) {
                            v.insert(t);
                        }
                    }
                }
            }
            if !v.is_empty() {
                with_any += 1;
            }
            let mut vv: Vec<String> = v.into_iter().collect();
            vv.sort();
            sub.push(vv);
        }
        let mean_sub = sub.iter().map(|v| v.len()).sum::<usize>() as f64 / n as f64;
        println!(
            "entries with >=1 independent second-grade token: {with_any}/{n} ({:.1}%), mean {mean_sub:.1} tokens\n",
            100.0 * with_any as f64 / n as f64
        );

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
        let mut pairs: Vec<Pair> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                // Strictest independence: a shared token is counted only when
                // it appears in NEITHER entry's embedded text. Dropping it from
                // the owner's own text is not enough — a token absent from i's
                // text but present in j's is still something j's embedding
                // received, so it would not be independent of the geometry
                // being controlled for.
                let (a, b) = (&sub[i], &sub[j]);
                let (mut x, mut y, mut ov) = (0usize, 0usize, 0.0f64);
                while x < a.len() && y < b.len() {
                    match a[x].cmp(&b[y]) {
                        std::cmp::Ordering::Less => x += 1,
                        std::cmp::Ordering::Greater => y += 1,
                        std::cmp::Ordering::Equal => {
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
        println!("{} pairs scored", pairs.len());
        let nonzero = pairs.iter().filter(|p| p.overlap > 0.0).count();
        println!(
            "pairs sharing >=1 independent token: {nonzero} ({:.1}%)\n",
            100.0 * nonzero as f64 / pairs.len() as f64
        );

        let mut order: Vec<usize> = (0..pairs.len()).collect();
        order.sort_by(|&a, &b| pairs[a].cos.partial_cmp(&pairs[b].cos).unwrap());
        const STRATA: usize = 40;
        const MAX_IMB: f64 = 0.01;
        let per = order.len() / STRATA;

        for (label, fine) in [
            ("SAME CELL (fine, 70 cells)", true),
            ("SAME DOMAIN (coarse, 5 domains)", false),
        ] {
            println!("=== {label} ===\n");
            let (mut wth, mut wnh, mut wtl, mut wnl, mut wk) = (0usize, 0usize, 0usize, 0usize, 0usize);
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
                let mc = |v: &[usize]| v.iter().map(|&k| pairs[k].cos as f64).sum::<f64>() / v.len() as f64;
                let (mch, mcl) = (mc(hi_i), mc(lo_i));
                th += hc; nh += hi_i.len(); tl += lc; nl += lo_i.len();
                if (mch - mcl).abs() < MAX_IMB {
                    wth += hc; wnh += hi_i.len(); wtl += lc; wnl += lo_i.len(); wk += 1;
                }
            }
            let (phr, plr) = (th as f64 / nh as f64, tl as f64 / nl as f64);
            let (wphr, wplr) = (wth as f64 / wnh.max(1) as f64, wtl as f64 / wnl.max(1) as f64);
            let pz = z_prop(phr, nh as f64, plr, nl as f64);
            let wz = z_prop(wphr, wnh as f64, wplr, wnl as f64);
            println!("  all strata          hi {phr:.4}  lo {plr:.4}   z = {pz:+.2}");
            println!(
                "  well-controlled     hi {wphr:.4}  lo {wplr:.4}   z = {wz:+.2}   ({wk}/{STRATA} strata, |imbalance| < {MAX_IMB})  <- counts"
            );
            println!(
                "  -> {}\n",
                if wz > 1.96 {
                    "independent substructure ADDS predictive power beyond cosine"
                } else if wz < -1.96 {
                    "NEGATIVELY related once cosine is held fixed"
                } else {
                    "no incremental validity over cosine"
                }
            );
        }

        println!(
            "(The independence claim is the point of this experiment, so it is enforced rather\n than assumed: second-grade tokens come from WordNet glosses — written by\n lexicographers, not by whoever authored the hints — and every gloss token that\n already appears in the entry's own embedded text is DISCARDED, so what is scored\n can only be information the embedding never received. Where this still falls\n short of the proposal: a gloss is a DEFINITION, decontextualised, whereas \"the\n dog ate my homework\" supplies a situational episode. A null here leaves the\n episodic version open, and physis's operational corpus — the right source for\n that — is empty on this machine.)"
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
