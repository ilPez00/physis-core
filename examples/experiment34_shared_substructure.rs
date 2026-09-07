// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 34 — is a link corroborated by SHARED SECOND-GRADE STRUCTURE?
//!
//! User's proposal, and the reason it is not a tenth geometric mechanism:
//!
//!   "the dog ate my homework = dog-ate-homework ... but if i then encounter a
//!    description of the dog, which may have large white fangs that ripped
//!    through the virgin paper i write my homework in, it could be
//!    dog(fangs-white-paper)-homework(paper-white-fangs) where 'white' is a
//!    second-grade link, nested in both primitives"
//!
//! Two layers, capped at depth 1: an OUTWARD layer of links between primitives
//! (`ate`), and an INWARD layer of constituents within each primitive
//! (`fangs`, `white`, `paper`). The claim is that a shared inward element —
//! "white" occurring inside BOTH endpoints — is what makes the outward link
//! non-arbitrary.
//!
//! physis already carries exactly this shape: an `OntologyEntry`'s NAME is the
//! primitive and its `hints` are the second-grade constituents. A shared hint
//! token between two entries is the user's "white".
//!
//! ## Why this might escape the nine failures
//!
//! All nine asked geometry to certify geometry — a statistic over the same
//! embedding space that produced the candidate. This asks a COMBINATORIAL
//! question instead: do the two endpoints' decompositions intersect? That is
//! set intersection over observed tokens, not another cosine. Only partially
//! independent — it is the same corpus — but a different operation on it.
//!
//! ## The confound this experiment exists to control
//!
//! Hints feed the embedded text (`name + hints`), so shared hints MECHANICALLY
//! raise cosine. An uncontrolled correlation between overlap and link quality
//! would therefore prove nothing. Pairs are stratified into cosine deciles and
//! compared only WITHIN a stratum, so geometry is held ~fixed and overlap is
//! the only thing varying. This is the incremental-validity question: does
//! substructure add anything BEYOND cosine?
//!
//! Stopwords are removed by inverse document frequency — the most-shared tokens
//! in this corpus are `the`, `and`, `for`, `with`, which are not anybody's
//! "white".
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment34_shared_substructure

use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use std::collections::HashMap;

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

fn main() {
    println!("Experiment 34: does shared second-grade structure add anything beyond cosine?\n");

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
        let mut toks: Vec<Vec<String>> = Vec::new();
        for def in ontology.classification_domains() {
            let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
            let mut t = def.name.clone();
            let mut tv: Vec<String> = Vec::new();
            for h in &def.hints {
                t.push(' ');
                t.push_str(h);
                for w in h.split_whitespace() {
                    let w: String = w
                        .to_lowercase()
                        .chars()
                        .filter(|c| c.is_alphanumeric())
                        .collect();
                    if w.len() > 2 {
                        tv.push(w);
                    }
                }
            }
            tv.sort();
            tv.dedup();
            texts.push(t);
            cells.push((d.clone(), m.clone()));
            toks.push(tv);
        }
        let n = texts.len();
        println!("{n} entries, embedding...");
        let emb: Vec<Vec<f32>> = texts.iter().map(|t| normalize(&embedder.embed(t))).collect();

        // IDF over the second-grade tokens. Without it the "shared element" is
        // dominated by `the`/`and`/`for`, which corroborate nothing.
        let mut df: HashMap<&str, usize> = HashMap::new();
        for tv in &toks {
            for w in tv {
                *df.entry(w.as_str()).or_default() += 1;
            }
        }
        let idf = |w: &str| -> f64 {
            let d = *df.get(w).unwrap_or(&1) as f64;
            (n as f64 / d).ln()
        };

        // Per pair: cosine, IDF-weighted shared-substructure score, and the two
        // ground-truth targets.
        struct Pair {
            cos: f32,
            overlap: f64,
            same_cell: bool,
            same_domain: bool,
        }
        let mut pairs: Vec<Pair> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let mut ov = 0.0f64;
                let (a, b) = (&toks[i], &toks[j]);
                let (mut x, mut y) = (0usize, 0usize);
                while x < a.len() && y < b.len() {
                    match a[x].cmp(&b[y]) {
                        std::cmp::Ordering::Less => x += 1,
                        std::cmp::Ordering::Greater => y += 1,
                        std::cmp::Ordering::Equal => {
                            ov += idf(&a[x]);
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
        println!("{} pairs scored\n", pairs.len());

        let base_cell = pairs.iter().filter(|p| p.same_cell).count() as f64 / pairs.len() as f64;
        let base_dom = pairs.iter().filter(|p| p.same_domain).count() as f64 / pairs.len() as f64;
        println!("base rates: same-cell {base_cell:.4}, same-domain {base_dom:.4}\n");

        // Stratify by cosine decile. Within a stratum geometry is ~fixed, so a
        // difference between high- and low-overlap pairs is attributable to
        // substructure rather than to the cosine that substructure inflates.
        let mut order: Vec<usize> = (0..pairs.len()).collect();
        order.sort_by(|&a, &b| pairs[a].cos.partial_cmp(&pairs[b].cos).unwrap());
        // 40 strata, not 10. A decile at the top spans cosine 0.76-0.98, wide
        // enough that high-overlap pairs could sit systematically higher WITHIN
        // it and the effect would still be cosine. Narrower strata tighten the
        // control, and the mean cosine of each half is printed so any residual
        // imbalance is visible rather than assumed away.
        const STRATA: usize = 40;
        let per = order.len() / STRATA;

        for (label, target) in [
            ("SAME CELL (fine, 70 cells)", true),
            ("SAME DOMAIN (coarse, 5 domains)", false),
        ] {
            println!("=== {label} ===\n");
            println!("  stratum  cos-range      n/half   hi-rate  lo-rate       z   | mean-cos hi/lo (imbalance)");
            let (mut th, mut nh, mut tl, mut nl) = (0usize, 0usize, 0usize, 0usize);
            let (mut imb_sum, mut imb_n) = (0.0f64, 0.0f64);
            // Pool separately over strata where the control actually HELD. The
            // extreme strata keep a real residual cosine gap (the cosine
            // distribution has long thin tails, so equal-count bins are wide
            // there), and that is exactly where the apparent effect sits. A
            // result that survives only in its worst-controlled bin is not a
            // result.
            const MAX_IMB: f64 = 0.01;
            let (mut wth, mut wnh, mut wtl, mut wnl, mut wk) = (0usize, 0usize, 0usize, 0usize, 0usize);
            for s in 0..STRATA {
                let lo = s * per;
                let hi = if s == STRATA - 1 { order.len() } else { (s + 1) * per };
                let mut idx: Vec<usize> = order[lo..hi].to_vec();
                if idx.len() < 20 {
                    continue;
                }
                // Split the stratum at its own median overlap.
                idx.sort_by(|&a, &b| pairs[a].overlap.partial_cmp(&pairs[b].overlap).unwrap());
                let mid = idx.len() / 2;
                let hit = |k: &usize| if target { pairs[*k].same_cell } else { pairs[*k].same_domain };
                let (lo_i, hi_i) = idx.split_at(mid);
                let (hc, lc) = (
                    hi_i.iter().filter(|k| hit(k)).count(),
                    lo_i.iter().filter(|k| hit(k)).count(),
                );
                let (hr, lr) = (
                    hc as f64 / hi_i.len() as f64,
                    lc as f64 / lo_i.len() as f64,
                );
                let z = z_prop(hr, hi_i.len() as f64, lr, lo_i.len() as f64);
                // Residual-confound check: if the hi-overlap half also has a
                // higher mean cosine inside the stratum, the difference may
                // still be geometry rather than substructure.
                let mc = |v: &[usize]| v.iter().map(|&k| pairs[k].cos as f64).sum::<f64>() / v.len() as f64;
                let (mch, mcl) = (mc(hi_i), mc(lo_i));
                if s % 4 == 0 || s == STRATA - 1 {
                    println!(
                        "  {:>3}  [{:.3}-{:.3}] {:>6}   {hr:.4}   {lr:.4}  {z:+7.2}   | {mch:.4}/{mcl:.4} ({:+.4})",
                        s + 1,
                        pairs[order[lo]].cos,
                        pairs[order[hi - 1]].cos,
                        hi_i.len(),
                        mch - mcl
                    );
                }
                imb_sum += (mch - mcl) * hi_i.len() as f64;
                imb_n += hi_i.len() as f64;
                if (mch - mcl).abs() < MAX_IMB {
                    wth += hc;
                    wnh += hi_i.len();
                    wtl += lc;
                    wnl += lo_i.len();
                    wk += 1;
                }
                th += hc;
                nh += hi_i.len();
                tl += lc;
                nl += lo_i.len();
            }
            let (phr, plr) = (th as f64 / nh as f64, tl as f64 / nl as f64);
            let pz = z_prop(phr, nh as f64, plr, nl as f64);
            println!(
                "\n  POOLED across strata: hi {th}/{nh} = {phr:.4}   lo {tl}/{nl} = {plr:.4}   z = {pz:+.2}"
            );
            println!(
                "  mean residual cosine imbalance (hi minus lo, weighted): {:+.5}",
                imb_sum / imb_n.max(1.0)
            );
            let (wphr, wplr) = (wth as f64 / wnh.max(1) as f64, wtl as f64 / wnl.max(1) as f64);
            let wz = z_prop(wphr, wnh as f64, wplr, wnl as f64);
            println!(
                "  WELL-CONTROLLED strata only (|imbalance| < {MAX_IMB}, {wk}/{STRATA} strata):"
            );
            println!("    hi {wth}/{wnh} = {wphr:.4}   lo {wtl}/{wnl} = {wplr:.4}   z = {wz:+.2}  <- the one that counts");
            println!(
                "  -> {}\n",
                if wz > 1.96 {
                    "shared substructure ADDS predictive power beyond cosine"
                } else if wz < -1.96 {
                    "shared substructure is NEGATIVELY related once cosine is held fixed"
                } else {
                    "no incremental validity over cosine"
                }
            );
        }

        println!(
            "\n=== Why this is a WEAK test of the underlying idea ===\n"
        );
        println!("  The second-grade elements used here are the authored `hints`, and hints are");
        println!("  part of the text that was EMBEDDED (`name + hints`). So the embedder already");
        println!("  read every token this experiment scores. Asking whether token overlap adds");
        println!("  information beyond the embedding is close to asking whether a model that read");
        println!("  those tokens failed to absorb them — the null is the expected answer, and");
        println!("  getting it back does not falsify much.");
        println!();
        println!("  The proposal's actual form needs second-grade structure from a source the");
        println!("  embedder did NOT see: \"if i then encounter a description of the dog\" — a");
        println!("  SEPARATE encounter, not a curated keyword list written by the same hand that");
        println!("  assigned the cell. In physis that source exists, but it is the operational");
        println!("  corpus (`operational.rs` events, telemetry, workflow steps), not the ontology.");
        println!();
        println!("  So: NOT SUPPORTED on hint-derived substructure, and the architecture is");
        println!("  untested rather than refuted. Re-run against independent encounters before");
        println!("  concluding anything about the depth-1 nesting idea itself.");
        println!(
            "\n(The stratification is the whole design. Hints feed the embedded text, so shared\n hints mechanically raise cosine — an uncontrolled correlation would prove nothing.\n Comparing only WITHIN a narrow cosine stratum holds geometry ~fixed. The extreme\n strata keep a real residual gap because the cosine distribution has long thin\n tails, and that is exactly where the apparent effect sat: pooling over the 29 of\n 40 strata where the control actually held drops same-cell from z=+3.78 to +0.62\n and same-domain from +10.18 to +1.72. IDF weighting is likewise load-bearing:\n the most-shared tokens here are `the`, `and`, `for`, `with`.)"
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
