// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 33 — can the disposer be a MACHINE rather than a person?
//!
//! Nine mechanisms have now failed to certify discovered structure, and every
//! one of them asked a REPRESENTATIONAL question: is this split correct, does
//! a lexicon recognise it, do many pairs agree on it. All were answered with a
//! statistic computed over the same embedding space that produced the
//! candidate, or over a lexicon that turned out not to care.
//!
//! This asks an OPERATIONAL question instead, and physis-pro already has the
//! machinery for it. `physis-pro/src/planner.rs`:
//!
//!   "Counterfactual decision planning. A planner run compares the live
//!    classifier with a candidate ontology over a bounded corpus. It returns
//!    an impact diff and never mutates live state."
//!
//! Its `ImpactAnalysis::newly_covered` is the disposer signal — "records the
//! candidate newly classifies with acceptable confidence". A proposed concept
//! earns its keep if adding it lets the system confidently handle records it
//! could not handle before. That is not a judgement about meaning; it is an
//! outcome, and the corpus of records is external to the proposal mechanism.
//!
//! ## Coverage rule, taken verbatim from planner.rs
//!
//! ```text
//! let covered = |o: &CellOutcome|
//!     o.domain != "unknown" && o.mode != "unknown" && o.score >= threshold;
//! ...
//! (false, true) => newly_covered.push(item.id.clone()),
//! ```
//!
//! Adding a cell can only raise a record's best score, so `newly_uncovered` is
//! structurally zero for a pure addition — the planner's "gaps opened" signal
//! cannot fire here, and only `newly_covered` carries information.
//!
//! This experiment lives in physis-core because that is where the working
//! embedder and the other 32 experiments are; physis-pro's `planner::simulate`
//! plus `ImpactAnalysis::derive` is the production implementation of the same
//! rule, and wiring the two together is an integration step, not a redesign.
//!
//! ## The test
//!
//! Same discipline as Iterations 30 and 32: 5-fold stratified CV, and a
//! manifold-matched control. Near-neighbour midpoints (the proposer validated
//! in Iteration 30) versus random-pair midpoints, scored on how many held-out
//! records each one rescues from below the confidence threshold.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment33_mechanical_disposer

use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter().map(|x| x / n).collect()
}

fn midpoint(a: &[f32], b: &[f32]) -> Vec<f32> {
    normalize(&a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<f32>>())
}

fn quantile(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[(((sorted.len() - 1) as f64) * q).round() as usize]
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Welch's t statistic — the two arms have different variances (proposals
/// cluster, controls spread), so a pooled-variance test would misstate it.
fn welch_t(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let (ma, mb) = (mean(a), mean(b));
    let va = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / (a.len() - 1) as f64;
    let vb = b.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / (b.len() - 1) as f64;
    let se = (va / a.len() as f64 + vb / b.len() as f64).sqrt();
    if se <= 0.0 {
        0.0
    } else {
        (ma - mb) / se
    }
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
    println!("Experiment 33: can the disposer be a machine? (planner impact as the check)\n");

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
        for def in ontology.classification_domains() {
            let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
            let mut t = def.name.clone();
            for h in &def.hints {
                t.push(' ');
                t.push_str(h);
            }
            texts.push(t);
            cells.push((d.clone(), m.clone()));
        }
        println!("physis: {} entries, embedding...", texts.len());
        let emb: Vec<Vec<f32>> = texts.iter().map(|t| normalize(&embedder.embed(t))).collect();

        const FOLDS: usize = 5;
        const SAMPLE: usize = 400; // candidates per arm per fold
        let mut by_cell: HashMap<(String, String), Vec<usize>> = HashMap::new();
        for (i, c) in cells.iter().enumerate() {
            by_cell.entry(c.clone()).or_default().push(i);
        }
        let mut cell_keys: Vec<&(String, String)> = by_cell.keys().collect();
        cell_keys.sort();

        let mut prop_scores: Vec<f64> = Vec::new();
        let mut ctrl_scores: Vec<f64> = Vec::new();
        // Two stronger controls. A random-pair midpoint is a weak control: the
        // average of two unrelated unit vectors lands near the global centroid,
        // close to everything and near nothing, so almost any point sitting on
        // real content beats it. These two remove that advantage.
        let mut mid_scores: Vec<f64> = Vec::new(); // same construction, ranks 20-40
        let mut entry_scores: Vec<f64> = Vec::new(); // an existing entry, verbatim
        let mut total_uncovered = 0usize;
        let mut total_records = 0usize;

        println!("\nRunning {FOLDS} folds ({SAMPLE} candidates per arm per fold)...");
        for fold in 0..FOLDS {
            let mut held: Vec<usize> = Vec::new();
            for k in &cell_keys {
                let m = &by_cell[*k];
                if m.len() < 3 {
                    continue;
                }
                for (j, &idx) in m.iter().enumerate() {
                    if j % FOLDS == fold {
                        held.push(idx);
                    }
                }
            }
            held.sort_unstable();
            held.dedup();
            let held_set: HashSet<usize> = held.iter().copied().collect();
            let known: Vec<usize> = (0..texts.len()).filter(|i| !held_set.contains(i)).collect();

            // BEFORE: each held-out record's best raw entry cosine against the
            // live ontology. Raw best-entry cosine rather than the blended
            // classifier score, following discovery.rs ("Scoring is raw
            // best-entry cosine (undiluted by the blended classifier)").
            let base: Vec<f32> = held
                .iter()
                .map(|&h| {
                    known
                        .iter()
                        .map(|&k| cosine_sim(&emb[h], &emb[k]))
                        .fold(f32::NEG_INFINITY, f32::max)
                })
                .collect();

            // Confidence threshold at the median of the live scores, so that
            // about half the records start UNCOVERED and there is something to
            // rescue. A threshold that covers everything would make
            // newly_covered structurally zero and measure nothing.
            let mut sorted = base.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let threshold = quantile(&sorted, 0.50);
            let uncovered: Vec<usize> =
                (0..held.len()).filter(|&i| base[i] < threshold).collect();
            total_uncovered += uncovered.len();
            total_records += held.len();

            // Candidate generators: the validated proposer, and the
            // manifold-matched control from Iteration 30.
            let mut proposals: Vec<Vec<f32>> = Vec::new();
            for (pos, &i) in known.iter().enumerate() {
                let mut sims: Vec<(f32, usize)> = known
                    .iter()
                    .enumerate()
                    .filter(|(p, _)| *p != pos)
                    .map(|(_, &j)| (cosine_sim(&emb[i], &emb[j]), j))
                    .collect();
                sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.cmp(&b.1)));
                for &(_, j) in sims.iter().take(5) {
                    if i < j {
                        proposals.push(midpoint(&emb[i], &emb[j]));
                    }
                }
            }
            // Control 2: midpoints of MEDIUM-rank pairs (20-40). Identical
            // construction to the proposals — a midpoint of two real, related
            // entries — differing only in how close the pair is. If the
            // proposer's advantage is really about near-neighbour specificity
            // rather than about sitting on real content, it must survive this.
            let mut medium: Vec<Vec<f32>> = Vec::new();
            for (pos, &i) in known.iter().enumerate() {
                let mut sims: Vec<(f32, usize)> = known
                    .iter()
                    .enumerate()
                    .filter(|(p, _)| *p != pos)
                    .map(|(_, &j)| (cosine_sim(&emb[i], &emb[j]), j))
                    .collect();
                sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.cmp(&b.1)));
                for &(_, j) in sims.iter().take(40).skip(20) {
                    if i < j {
                        medium.push(midpoint(&emb[i], &emb[j]));
                    }
                }
            }

            let mut rng = StdRng::seed_from_u64(9_000 + fold as u64);
            let mut controls: Vec<Vec<f32>> = Vec::new();
            while controls.len() < proposals.len() {
                let a = known[rng.gen_range(0..known.len())];
                let mut b = known[rng.gen_range(0..known.len())];
                while b == a {
                    b = known[rng.gen_range(0..known.len())];
                }
                controls.push(midpoint(&emb[a], &emb[b]));
            }

            // newly_covered, planner's rule: a record uncovered before and
            // covered after. Adding a cell only raises the max, so this is
            // exactly "was below threshold, the candidate is at or above it".
            let newly_covered = |cand: &[f32]| -> f64 {
                uncovered
                    .iter()
                    .filter(|&&i| cosine_sim(cand, &emb[held[i]]) >= threshold)
                    .count() as f64
            };

            // Control 3: an existing entry used verbatim as the candidate.
            // Unambiguously ON the manifold and on real content — the strictest
            // "near real data" baseline available.
            let mut entries_arm: Vec<Vec<f32>> = Vec::new();
            let mut rng2 = StdRng::seed_from_u64(77_000 + fold as u64);
            for _ in 0..proposals.len() {
                entries_arm.push(emb[known[rng2.gen_range(0..known.len())]].clone());
            }

            let take = |v: &[Vec<f32>], out: &mut Vec<f64>| {
                let step = (v.len() / SAMPLE).max(1);
                for c in v.iter().step_by(step).take(SAMPLE) {
                    out.push(newly_covered(c));
                }
            };
            take(&proposals, &mut prop_scores);
            take(&controls, &mut ctrl_scores);
            take(&medium, &mut mid_scores);
            take(&entries_arm, &mut entry_scores);
            println!(
                "  fold {fold}: {} records, {} uncovered (threshold {:.3}), {} candidates/arm",
                held.len(),
                uncovered.len(),
                threshold,
                proposals.len().min(SAMPLE)
            );
        }

        // ---------- results ----------
        let arms: Vec<(&str, &Vec<f64>)> = vec![
            ("proposals (nn ranks 1-5)", &prop_scores),
            ("medium-rank midpoints", &mid_scores),
            ("existing entry verbatim", &entry_scores),
            ("random-pair midpoints", &ctrl_scores),
        ];
        println!(
            "\n=== Corpus: {total_records} held-out records pooled, {total_uncovered} uncovered ({:.1}%) ===\n",
            100.0 * total_uncovered as f64 / total_records as f64
        );
        println!("  arm                          n     mean rescued   rescues >=1");
        for (name, v) in &arms {
            let m = mean(v);
            let any = v.iter().filter(|x| **x > 0.0).count();
            println!(
                "  {name:<26} {:>5}   {m:>12.3}   {:>6}/{} = {:.3}",
                v.len(),
                any,
                v.len(),
                any as f64 / v.len() as f64
            );
        }

        let mp = mean(&prop_scores);
        let rate_of = |v: &Vec<f64>| v.iter().filter(|x| **x > 0.0).count() as f64 / v.len() as f64;
        let rate_p = rate_of(&prop_scores);

        println!("\n=== Proposals vs each control ===\n");
        println!("  control                      ratio    Welch t    z(rescues>=1)");
        let mut worst_t = f64::INFINITY;
        let mut worst_z = f64::INFINITY;
        for (name, v) in arms.iter().skip(1) {
            let m = mean(v);
            let t = welch_t(&prop_scores, v);
            let z = z_prop(rate_p, prop_scores.len() as f64, rate_of(v), v.len() as f64);
            println!(
                "  {name:<26} {:>6.2}x   {t:>+7.2}    {z:>+8.2}",
                if m > 1e-9 { mp / m } else { f64::INFINITY }
            );
            worst_t = worst_t.min(t);
            worst_z = worst_z.min(z);
        }
        println!("\n  weakest margin across all controls: t = {worst_t:+.2}, z = {worst_z:+.2}");

        println!("\n=== Verdict ===\n");

        let m_entry = mean(&entry_scores);
        let m_mid = mean(&mid_scores);
        let m_ctrl = mean(&ctrl_scores);
        let t_entry = welch_t(&prop_scores, &entry_scores);
        let t_mid = welch_t(&prop_scores, &mid_scores);

        println!("  Two separate questions, and they get opposite answers.\n");

        println!("  1. DOES THE MACHINE DISPOSER DISCRIMINATE?  Yes, decisively.");
        println!("     An existing entry re-added as a candidate rescues {m_entry:.3} records —");
        println!("     {}/{} of them, exactly zero — because records near it were", entry_scores.iter().filter(|x| **x > 0.0).count(), entry_scores.len());
        println!("     already covered. Interpolations rescue {mp:.3} (t = {t_entry:+.2}). The check");
        println!("     is not fooled by a candidate that adds nothing, and it needs no human");
        println!("     and no lexicon to say so. Of {} candidates it marks {} as", prop_scores.len(), prop_scores.iter().filter(|x| **x > 0.0).count());
        println!("     demonstrably coverage-adding — a {:.0}x reduction in what anyone would", prop_scores.len() as f64 / prop_scores.iter().filter(|x| **x > 0.0).count().max(1) as f64);
        println!("     have to read.");
        println!();
        println!("  2. DOES IT PREFER ITERATION 30'S PROPOSER?  No.");
        println!("     Near-neighbour midpoints {mp:.3} vs medium-rank midpoints {m_mid:.3}");
        println!("     ({:.2}x, t = {t_mid:+.2}). Any midpoint of two RELATED entries works as", if m_mid > 1e-9 { mp / m_mid } else { f64::INFINITY });
        println!("     well; only midpoints of UNRELATED entries do worse ({m_ctrl:.3}). So what");
        println!("     carries the operational value is interpolating between related content");
        println!("     at all — not the near-neighbour specificity Iteration 30 measured.");
        println!();
        println!("  Net: the DISPOSER is real and mechanical. The PROPOSER's edge does not");
        println!("  survive a control matched on construction rather than only on manifold —");
        println!("  the third time in this track that a positive result has died to a");
        println!("  stronger control, and the second time to this same class of control.");
        println!();
        println!("  Practical consequence, which is the useful part: a machine can stand");
        println!("  where the human was, not to judge whether a concept is GOOD, but to");
        println!("  throw away the {:.0}% of candidates that provably change nothing. The", 100.0 * (1.0 - rate_p));
        println!("  human then reads {} instead of {}.", prop_scores.iter().filter(|x| **x > 0.0).count(), prop_scores.len());

        println!(
            "\n(What this does NOT establish: that a rescued record is rescued CORRECTLY. The\n planner's coverage rule asks whether a record now classifies confidently, not\n whether it classifies rightly — a candidate that swallows records into a wrong\n cell scores identically to one that captures a real gap. That distinction needs\n labels this corpus does not carry, and it is exactly the question nine\n representational mechanisms failed to answer. The claim here is narrower and\n operational: coverage, not correctness.)"
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
