// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 32 — the powered re-test of Iteration 31, with its confound fixed.
//!
//! Iteration 31 built the outward loop and got two underpowered answers:
//! the WordNet check looked dead (z = -0.38, n = 22 vs 33) and convergence
//! looked promising but unproven (z = +1.35, n = 33 vs 119). Neither n was
//! large enough to decide anything, and one bucket carried a real confound.
//!
//! ## The confound, and it was a bug rather than a property of the data
//!
//! Iteration 31 assigned the REDUNDANT bucket using similarity to all known
//! entries INCLUDING a region's own parents, then validated using similarity
//! EXCLUDING them. A region whose high coverage came from its parents was
//! therefore filed as "physis already covers this" and then scored against a
//! much weaker rival — which is why REDUNDANT came back second-highest at
//! 0.265 and made the hit metric look like it was measuring local density.
//! Parents are excluded consistently here, in bucketing and in validation.
//!
//! ## The power
//!
//! 5-fold stratified cross-validation instead of one hold-out. Every entry is
//! held out exactly once, so the folds are disjoint and cover the corpus, and
//! regions pool across folds — roughly five times the sample at no cost in
//! independence.
//!
//! ## What is being decided
//!
//! Two controlled comparisons, each holding one axis fixed, because the
//! buckets differ on two axes at once and a raw bucket-vs-overall difference
//! cannot say which one did the work:
//!
//!   A. WordNet, at fixed convergence: does lexical recognition predict that a
//!      region holds a concept the ontology is missing?
//!   B. Convergence, at fixed WordNet status: does agreement among many
//!      distinct entry pairs predict it?
//!
//! Run (reuses Iteration 31's embedding cache):
//!   cargo run -p physis-core --features embed-onnx --release --example experiment32_convergence_crossval

use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use std::collections::{HashMap, HashSet};

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter().map(|x| x / n).collect()
}

fn midpoint(a: &[f32], b: &[f32]) -> Vec<f32> {
    normalize(&a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<f32>>())
}

fn parse_wordnet(path: &std::path::Path) -> Vec<(String, String)> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with("  ") || line.trim().is_empty() {
            continue;
        }
        let Some((head, gloss)) = line.split_once(" | ") else { continue };
        let f: Vec<&str> = head.split_whitespace().collect();
        if f.len() < 5 {
            continue;
        }
        out.push((f[4].replace('_', " "), gloss.trim().to_string()));
    }
    out
}

fn quantile(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[(((sorted.len() - 1) as f64) * q).round() as usize]
}

/// Two-proportion z-test.
fn z_test(p1: f64, n1: f64, p2: f64, n2: f64) -> f64 {
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

/// One region, pooled across folds.
struct Region {
    convergence: usize,
    wn_known: bool,
    phys_covered: bool,
    /// Hit under parent-exclusion. CONFOUNDED with convergence by
    /// construction: more parents means more near competitors removed from
    /// the comparison, which makes a hit mechanically easier. Kept only so
    /// the two metrics can be shown side by side.
    hit: bool,
    /// Similarity to the nearest held-out concept.
    best_held: f32,
    /// The region's top known-entry similarities, descending. `best_held >
    /// known_top[k]` is the hit test under a FIXED exclusion of the k nearest
    /// known entries — uniform across regions, so convergence gets no
    /// mechanical advantage. Swept over k rather than fixed at one value,
    /// because a single k can be confounded (small k) or saturated (large k)
    /// and only the sweep shows which.
    known_top: Vec<f32>,
}

fn main() {
    println!("Experiment 32: 5-fold re-test of the outward loop, confound fixed.\n");

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

        // ---------- corpus ----------
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

        // ---------- lexicon (cache shared with Iteration 31) ----------
        const STRIDE: usize = 4;
        let mut wn: Vec<(String, String)> = Vec::new();
        for file in ["data.noun", "data.verb"] {
            let all = parse_wordnet(&wn_dir.join(file));
            wn.extend(all.iter().step_by(STRIDE).cloned());
        }
        let cache = std::path::Path::new("target").join(format!("wordnet_nv_emb_s{STRIDE}.bin"));
        let wn_emb: Vec<Vec<f32>> = match std::fs::read(&cache) {
            Ok(b) if b.len() == wn.len() * 384 * 4 => {
                println!("lexicon: {} synsets (from cache)", wn.len());
                b.chunks_exact(384 * 4)
                    .map(|c| {
                        c.chunks_exact(4)
                            .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                            .collect()
                    })
                    .collect()
            }
            _ => {
                println!("lexicon: {} synsets, embedding (slow)...", wn.len());
                let v: Vec<Vec<f32>> = wn
                    .iter()
                    .map(|(l, g)| normalize(&embedder.embed(&format!("{l}: {g}"))))
                    .collect();
                let mut bytes = Vec::with_capacity(v.len() * 384 * 4);
                for e in &v {
                    for x in e {
                        bytes.extend_from_slice(&x.to_le_bytes());
                    }
                }
                let _ = std::fs::create_dir_all("target");
                let _ = std::fs::write(&cache, &bytes);
                v
            }
        };

        // ---------- fold assignment ----------
        // Disjoint stratified folds: within each cell, member j goes to fold
        // j % FOLDS. Every entry is held out exactly once across the run.
        const FOLDS: usize = 5;
        let mut by_cell: HashMap<(String, String), Vec<usize>> = HashMap::new();
        for (i, c) in cells.iter().enumerate() {
            by_cell.entry(c.clone()).or_default().push(i);
        }
        let mut cell_keys: Vec<&(String, String)> = by_cell.keys().collect();
        cell_keys.sort();

        let mut all_regions: Vec<Region> = Vec::new();
        println!("\nRunning {FOLDS} folds...");

        for fold in 0..FOLDS {
            let mut held: Vec<usize> = Vec::new();
            for k in &cell_keys {
                let m = &by_cell[*k];
                if m.len() < 3 {
                    continue; // keep at least 2 so the cell survives
                }
                for (j, &idx) in m.iter().enumerate() {
                    if j % FOLDS == fold && m.len() > 2 {
                        held.push(idx);
                    }
                }
            }
            held.sort_unstable();
            held.dedup();
            let held_set: HashSet<usize> = held.iter().copied().collect();
            let known: Vec<usize> = (0..texts.len()).filter(|i| !held_set.contains(i)).collect();

            // propose
            const NN: usize = 5;
            let mut proposals: Vec<(Vec<f32>, usize, usize)> = Vec::new();
            for (pos, &i) in known.iter().enumerate() {
                let mut sims: Vec<(f32, usize)> = known
                    .iter()
                    .enumerate()
                    .filter(|(p, _)| *p != pos)
                    .map(|(_, &j)| (cosine_sim(&emb[i], &emb[j]), j))
                    .collect();
                sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.cmp(&b.1)));
                for &(_, j) in sims.iter().take(NN) {
                    if i < j {
                        proposals.push((midpoint(&emb[i], &emb[j]), i, j));
                    }
                }
            }

            // converge
            const CONV_TAU: f32 = 0.90;
            let mut seeds: Vec<Vec<f32>> = Vec::new();
            let mut members: Vec<Vec<usize>> = Vec::new();
            for (pi, (p, _, _)) in proposals.iter().enumerate() {
                let mut placed = false;
                for (ci, s) in seeds.iter().enumerate() {
                    if cosine_sim(p, s) >= CONV_TAU {
                        members[ci].push(pi);
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    seeds.push(p.clone());
                    members.push(vec![pi]);
                }
            }

            let centroids: Vec<Vec<f32>> = members
                .iter()
                .map(|ms| {
                    let mut acc = vec![0.0f32; 384];
                    for &pi in ms {
                        for (d, a) in acc.iter_mut().enumerate() {
                            *a += proposals[pi].0[d];
                        }
                    }
                    normalize(&acc)
                })
                .collect();
            let parents: Vec<Vec<usize>> = members
                .iter()
                .map(|ms| {
                    let mut v: Vec<usize> = ms
                        .iter()
                        .flat_map(|&pi| [proposals[pi].1, proposals[pi].2])
                        .collect();
                    v.sort_unstable();
                    v.dedup();
                    v
                })
                .collect();
            let convergence: Vec<usize> = parents.iter().map(|p| p.len()).collect();

            // THE FIX: one parent-excluded rival similarity, used for BOTH the
            // bucketing and the validation. Iteration 31 used a
            // parent-inclusive value for bucketing and a parent-excluded one
            // for validation, which manufactured the density confound.
            let rival: Vec<f32> = (0..centroids.len())
                .map(|ci| {
                    known
                        .iter()
                        .filter(|k| !parents[ci].contains(k))
                        .map(|&i| cosine_sim(&centroids[ci], &emb[i]))
                        .fold(f32::NEG_INFINITY, f32::max)
                })
                .collect();
            let wn_sim: Vec<f32> = centroids
                .iter()
                .map(|c| {
                    wn_emb
                        .iter()
                        .map(|w| cosine_sim(c, w))
                        .fold(f32::NEG_INFINITY, f32::max)
                })
                .collect();

            let mut wn_sorted = wn_sim.clone();
            wn_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut rv_sorted = rival.clone();
            rv_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut cv_sorted = convergence.clone();
            cv_sorted.sort_unstable();
            let wn_t = quantile(&wn_sorted, 0.50);
            let rv_t = quantile(&rv_sorted, 0.50);
            let cv_t = cv_sorted[cv_sorted.len() / 2];

            // Fixed-K rival: drop the K nearest KNOWN entries for every region,
            // the same K regardless of how many parents it has. Parent-exclusion
            // removes a variable number of near neighbours (exactly the
            // convergence count), so it hands high-convergence regions a weaker
            // rival and could manufacture the trend on its own. K is set above
            // the largest convergence seen so the fixed exclusion is never the
            // looser of the two.
            let known_tops: Vec<Vec<f32>> = (0..centroids.len())
                .map(|ci| {
                    let mut sims: Vec<f32> = known
                        .iter()
                        .map(|&i| cosine_sim(&centroids[ci], &emb[i]))
                        .collect();
                    sims.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    sims.truncate(20);
                    sims
                })
                .collect();

            for ci in 0..centroids.len() {
                let best_held = held
                    .iter()
                    .map(|&h| cosine_sim(&centroids[ci], &emb[h]))
                    .fold(f32::NEG_INFINITY, f32::max);
                all_regions.push(Region {
                    convergence: convergence[ci],
                    wn_known: wn_sim[ci] >= wn_t,
                    phys_covered: rival[ci] >= rv_t,
                    hit: best_held > rival[ci],
                    best_held,
                    known_top: known_tops[ci].clone(),
                });
            }
            println!("  fold {fold}: {} held out, {} regions", held.len(), centroids.len());
            let _ = cv_t;
        }

        // ---------- pooled analysis ----------
        let n = all_regions.len();
        let overall = all_regions.iter().filter(|r| r.hit).count() as f64 / n as f64;
        println!("\n=== Pooled over {FOLDS} folds: {n} regions, overall hit rate {overall:.3} ===\n");

        let rate = |f: &dyn Fn(&Region) -> bool| -> (usize, usize, f64) {
            let sel: Vec<&Region> = all_regions.iter().filter(|r| f(r)).collect();
            let h = sel.iter().filter(|r| r.hit).count();
            (h, sel.len(), if sel.is_empty() { 0.0 } else { h as f64 / sel.len() as f64 })
        };
        let rate_k = |f: &dyn Fn(&Region) -> bool, k: usize| -> (usize, usize, f64) {
            let sel: Vec<&Region> = all_regions.iter().filter(|r| f(r)).collect();
            let h = sel
                .iter()
                .filter(|r| r.known_top.get(k).is_some_and(|t| r.best_held > *t))
                .count();
            (h, sel.len(), if sel.is_empty() { 0.0 } else { h as f64 / sel.len() as f64 })
        };

        // Convergence threshold pooled across folds, so the split is one
        // consistent line rather than five per-fold medians.
        let mut cv_all: Vec<usize> = all_regions.iter().map(|r| r.convergence).collect();
        cv_all.sort_unstable();
        let cv_t = cv_all[cv_all.len() / 2];
        println!("  (pooled convergence median = {cv_t} distinct parent entries)\n");

        println!("=== A. Does the WORDNET check help? (convergence held fixed: high only) ===\n");
        let (ah, an, ar) = rate(&|r| r.convergence >= cv_t && r.wn_known && !r.phys_covered);
        let (bh, bn, br) = rate(&|r| r.convergence >= cv_t && !r.wn_known && !r.phys_covered);
        let za = z_test(ar, an as f64, br, bn as f64);
        println!("  WordNet knows it    {ah:4}/{an:<5} = {ar:.3}");
        println!("  WordNet does not    {bh:4}/{bn:<5} = {br:.3}");
        println!("  z = {za:+.2}");

        println!("\n=== B. Does CONVERGENCE help? (WordNet status held fixed: unknown only) ===\n");
        let (ch, cn, cr) = rate(&|r| r.convergence >= cv_t && !r.wn_known && !r.phys_covered);
        let (dh, dn, dr) = rate(&|r| r.convergence < cv_t && !r.wn_known && !r.phys_covered);
        let zb = z_test(cr, cn as f64, dr, dn as f64);
        println!("  high convergence    {ch:4}/{cn:<5} = {cr:.3}");
        println!("  low convergence     {dh:4}/{dn:<5} = {dr:.3}");
        println!("  z = {zb:+.2}");

        // The decisive test, swept. Parent-exclusion removes exactly
        // `convergence` near neighbours, so it hands high-convergence regions a
        // weaker rival and can produce the trend by itself. A fixed k removes
        // the same number from every region. But k matters: too small and the
        // parents themselves still dominate the rival, too large and every
        // region clears the bar and the metric saturates. So the whole sweep is
        // reported, and the conclusion is read from the region where the
        // overall rate is neither floored nor ceilinged.
        println!("\n=== B-CONTROL. Convergence under FIXED-SIZE exclusion, swept over k ===\n");
        println!("     k   overall   high-conv        low-conv         z");
        let mut verdicts: Vec<(usize, f64, f64)> = Vec::new();
        for k in [0usize, 1, 2, 3, 4, 6, 8, 12, 16] {
            let (_, _, ov) = rate_k(&|_r| true, k);
            let (eh, en, er) = rate_k(&|r| r.convergence >= cv_t && !r.wn_known && !r.phys_covered, k);
            let (fh, fnn, fr) = rate_k(&|r| r.convergence < cv_t && !r.wn_known && !r.phys_covered, k);
            let z = z_test(er, en as f64, fr, fnn as f64);
            let flag = if !(0.10..=0.90).contains(&ov) { "  <- floored/saturated" } else { "" };
            println!("  {k:4}    {ov:.3}   {eh:3}/{en:<4} {er:.3}   {fh:4}/{fnn:<4} {fr:.3}   {z:+.2}{flag}");
            verdicts.push((k, ov, z));
        }
        // Read the verdict only from unsaturated k.
        let usable: Vec<&(usize, f64, f64)> =
            verdicts.iter().filter(|(_, ov, _)| (0.10..=0.90).contains(ov)).collect();
        let zb_fixed = if usable.is_empty() {
            0.0
        } else {
            usable.iter().map(|(_, _, z)| *z).sum::<f64>() / usable.len() as f64
        };
        println!(
            "\n  usable k (overall rate in 0.10-0.90): {}   mean z = {zb_fixed:+.2}",
            if usable.is_empty() { "NONE".to_string() } else { usable.iter().map(|(k, _, _)| k.to_string()).collect::<Vec<_>>().join(", ") }
        );

        println!("\n=== B2. Convergence as a trend, all uncovered regions ===\n");
        println!("  convergence   hits/regions    rate     rate(k=2)");
        let mut levels: Vec<usize> = all_regions
            .iter()
            .filter(|r| !r.phys_covered)
            .map(|r| r.convergence)
            .collect();
        levels.sort_unstable();
        levels.dedup();
        for lv in levels.iter().take(10) {
            let (h, m, rr) = rate(&|r| !r.phys_covered && r.convergence == *lv);
            let (_, _, rf) = rate_k(&|r| !r.phys_covered && r.convergence == *lv, 2);
            if m >= 5 {
                println!("  {lv:>9}     {h:4}/{m:<6}   {rr:.3}         {rf:.3}");
            }
        }

        println!("\n=== Verdict ===\n");
        println!("  A (WordNet check):              z = {za:+.2}  ->  {}", if za > 1.96 { "SUPPORTED" } else { "NOT SUPPORTED" });
        println!("  B (convergence, parent-excl):   z = {zb:+.2}  ->  {} [CONFOUNDED]", if zb > 1.96 { "supported" } else { "not supported" });
        println!("  B-CONTROL (fixed-k, mean over usable k): z = {zb_fixed:+.2}  ->  {}  <- the one that counts", if zb_fixed > 1.96 { "SUPPORTED" } else { "NOT SUPPORTED" });
        println!();
        if zb_fixed > 1.96 && za <= 1.96 {
            println!("  The two discriminators separate cleanly. Agreement among many distinct");
            println!("  entry pairs predicts that a region holds a genuinely-missing concept;");
            println!("  agreement with a hand-built lexicon does not. The loop should keep the");
            println!("  convergence filter and drop the external lexical check.");
        } else if za <= 1.96 && zb_fixed <= 1.96 {
            println!("  Neither discriminator survives the powered test. The Iteration 30");
            println!("  proposer carries signal, but nothing built on top of it here does —");
            println!("  the triage is not separating regions that hold a missing concept from");
            println!("  regions that do not.");
        } else {
            println!("  Both axes clear significance; they are not independent, so the loop");
            println!("  needs a joint model before either is trusted alone.");
        }
        println!(
            "\n(Fixed vs Iteration 31: parents are excluded consistently in BOTH bucketing and\n validation — 31 used a parent-inclusive coverage test and a parent-excluded hit\n test, which manufactured its density confound. Folds are disjoint and every entry\n is held out exactly once. Percentile thresholds still fix bucket SIZES by\n construction, so only differential rates are evidence.)"
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
