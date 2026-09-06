//! Experiment 30 — hold-out rediscovery. Does the embedding geometry point at
//! WHERE concepts are missing?
//!
//! ## Why this is not one of the seven that failed
//!
//! Iterations 9-28 all asked a CLOSED question — "is this partition of these
//! items correct?" — which has no referent outside the geometry, so the only
//! available judge was another statistic from the same geometry. All seven
//! mechanisms failed.
//!
//! This asks an OPEN question: "does something exist near here that I don't
//! have?" That has an external referent. Here the referent is a hold-out set:
//! concepts we know are real, and know are absent from the model, because we
//! removed them ourselves.
//!
//! The embedder is therefore a SEARCH HEURISTIC, not a truth criterion. Its
//! cross-embedder instability (ARI ~0.10, Iteration 26b) stops being fatal —
//! a bad heuristic wastes a query, it cannot corrupt an ontology, because
//! nothing is admitted until the external check passes.
//!
//! ## Method
//!
//! Stratified hold-out: ~10% of entries removed per cell, never emptying a
//! cell. Build only from the 90%. Then propose candidate locations for missing
//! concepts, and measure how often a proposal lands near a held-out entry.
//!
//! Proposals are midpoints between near-neighbour pairs — the direct reading
//! of "an embedding's correlates are potential objects to search for" — ranked
//! by gap score (distance to the nearest known entry), so we surface locations
//! that are plausible but unoccupied.
//!
//! ## The baseline that makes it falsifiable
//!
//! In a dense 384-d space, "some real concept is near my proposal" is trivially
//! true. So near-neighbour midpoints are compared against RANDOM-PAIR midpoints
//! — random convex combinations of two known entries. That control sits in the
//! same manifold, at the same scale, and differs only in whether the pair was
//! chosen for proximity. If both score the same, the geometry carries no
//! information about where concepts are missing, and the idea is dead.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment30_holdout_rediscovery

use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter().map(|x| x / n).collect()
}

fn midpoint(a: &[f32], b: &[f32]) -> Vec<f32> {
    normalize(&a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<f32>>())
}

/// Highest cosine from `x` to anything in `set`.
fn nearest_sim(x: &[f32], set: &[usize], all: &[Vec<f32>]) -> f32 {
    set.iter()
        .map(|&i| cosine_sim(x, &all[i]))
        .fold(f32::NEG_INFINITY, f32::max)
}

fn main() {
    println!("Experiment 30: does the geometry point at WHERE concepts are missing?\n");

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

        // Deterministic entry order since 0.1.15.
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
        println!("Loaded {} entries, embedding...", texts.len());
        let emb: Vec<Vec<f32>> = texts.iter().map(|t| normalize(&embedder.embed(t))).collect();

        // ---- stratified hold-out: ~10% per cell, never emptying a cell ----
        let mut by_cell: HashMap<(String, String), Vec<usize>> = HashMap::new();
        for (i, c) in cells.iter().enumerate() {
            by_cell.entry(c.clone()).or_default().push(i);
        }
        let mut cell_keys: Vec<&(String, String)> = by_cell.keys().collect();
        cell_keys.sort(); // never rely on HashMap order (Iteration 26a)

        let mut held: Vec<usize> = Vec::new();
        for k in &cell_keys {
            let members = &by_cell[*k];
            // Keep at least 2 per cell so the cell still exists in the model.
            let n_hold = ((members.len() as f64 * 0.10).round() as usize).min(members.len().saturating_sub(2));
            // Deterministic stride pick — no RNG, so the split is identical
            // across runs and reviewers can reproduce the exact partition.
            for j in 0..n_hold {
                held.push(members[(j * 7 + 3) % members.len()]);
            }
        }
        held.sort_unstable();
        held.dedup();
        let held_set: std::collections::HashSet<usize> = held.iter().copied().collect();
        let known: Vec<usize> = (0..texts.len()).filter(|i| !held_set.contains(i)).collect();
        println!(
            "Stratified hold-out: {} known / {} held out ({:.1}%), across {} cells\n",
            known.len(),
            held.len(),
            100.0 * held.len() as f64 / texts.len() as f64,
            cell_keys.len()
        );

        // ---- proposals: midpoints of near-neighbour pairs among KNOWN ----
        // "An embedding's correlates are potential objects to search for":
        // between two concepts that sit close together, there may be a third.
        const NN: usize = 5;
        let mut proposals: Vec<(Vec<f32>, f32, usize, usize)> = Vec::new(); // (point, gap, i, j)
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
                    let p = midpoint(&emb[i], &emb[j]);
                    // Gap score: how far the midpoint sits from everything known.
                    // Low nearest-sim = plausible but unoccupied.
                    let gap = 1.0 - nearest_sim(&p, &known, &emb);
                    proposals.push((p, gap, i, j));
                }
            }
        }
        proposals.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap()
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.3.cmp(&b.3))
        });
        println!("Generated {} near-neighbour midpoint proposals.\n", proposals.len());

        // ---- generators, compared head to head ----
        // A midpoint between two VERY close entries sits on top of both, so it
        // can only ever rediscover them. Medium-distance pairs are the more
        // honest reading of "between two concepts there may be a third", so
        // both are tested rather than assumed.
        let gen_midpoints = |lo: usize, hi: usize| -> Vec<(Vec<f32>, usize, usize)> {
            let mut out = Vec::new();
            for (pos, &i) in known.iter().enumerate() {
                let mut sims: Vec<(f32, usize)> = known
                    .iter()
                    .enumerate()
                    .filter(|(p, _)| *p != pos)
                    .map(|(_, &j)| (cosine_sim(&emb[i], &emb[j]), j))
                    .collect();
                sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.cmp(&b.1)));
                for &(_, j) in sims.iter().take(hi).skip(lo) {
                    if i < j {
                        out.push((midpoint(&emb[i], &emb[j]), i, j));
                    }
                }
            }
            out
        };
        let near = gen_midpoints(0, 5);
        let mid = gen_midpoints(20, 40);

        // Control: random-pair midpoints — same manifold, same construction,
        // differing only in whether the pair was chosen for proximity.
        let mut rng = StdRng::seed_from_u64(20260906);
        let n_ctrl = near.len().max(mid.len());
        let mut controls: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(n_ctrl);
        for _ in 0..n_ctrl {
            let a = known[rng.gen_range(0..known.len())];
            let mut b = known[rng.gen_range(0..known.len())];
            while b == a {
                b = known[rng.gen_range(0..known.len())];
            }
            controls.push((midpoint(&emb[a], &emb[b]), a, b));
        }

        // ---- the metric, properly normalised ----
        // For each proposal, find its nearest entry among ALL 730 (known AND
        // held-out). What fraction of proposals point at a HELD-OUT one?
        // Held-out entries are 10.0% of the corpus, so 0.100 is the chance
        // rate and any real signal must clear it. Unlike a fixed-tau hit rate,
        // this cannot saturate: it is a competition, not a threshold.
        // The two parent entries are excluded from the competition: a midpoint
        // sits adjacent to both BY CONSTRUCTION, so including them measures the
        // construction rather than the prediction. (First pass did include them
        // and every generator scored 0.000 — an artifact, not a result.)
        let all: Vec<usize> = (0..texts.len()).collect();
        let nearest_excluding = |p: &[f32], x: usize, y: usize| -> usize {
            let mut best = (f32::NEG_INFINITY, usize::MAX);
            for &i in &all {
                if i == x || i == y {
                    continue;
                }
                let s = cosine_sim(p, &emb[i]);
                if s > best.0 {
                    best = (s, i);
                }
            }
            best.1
        };
        let points_at_held = |points: &[(Vec<f32>, usize, usize)]| -> f64 {
            let n = points
                .iter()
                .filter(|(p, x, y)| held_set.contains(&nearest_excluding(p, *x, *y)))
                .count();
            n as f64 / points.len() as f64
        };

        let chance = held.len() as f64 / texts.len() as f64;
        println!("=== Does a proposal point at a MISSING concept more often than chance? ===\n");
        println!("  (nearest entry among all {} is a held-out one; chance = {:.3})\n", texts.len(), chance);
        let n_near = points_at_held(&near);
        let n_mid = points_at_held(&mid);
        let n_ctl = points_at_held(&controls);
        println!("  near-neighbour midpoints (ranks 1-5)    n={:5}   {:.3}   lift {:.2}x", near.len(), n_near, n_near / chance);
        println!("  medium-distance midpoints (ranks 20-40) n={:5}   {:.3}   lift {:.2}x", mid.len(), n_mid, n_mid / chance);
        println!("  random-pair midpoints (control)         n={:5}   {:.3}   lift {:.2}x", controls.len(), n_ctl, n_ctl / chance);

        // ---- recall view: how many held-out concepts get pointed at at all? ----
        let recall = |points: &[(Vec<f32>, usize, usize)]| -> f64 {
            let mut found = std::collections::HashSet::new();
            for (p, x, y) in points {
                let n = nearest_excluding(p, *x, *y);
                if held_set.contains(&n) {
                    found.insert(n);
                }
            }
            found.len() as f64 / held.len() as f64
        };
        println!("\n=== How many of the {} missing concepts get pointed at at all? ===\n", held.len());
        println!("  near-neighbour   {:.3}", recall(&near));
        println!("  medium-distance  {:.3}", recall(&mid));
        println!("  random control   {:.3}", recall(&controls));

        // ---- what the best proposals actually found ----
        println!("\n=== Examples: proposals whose nearest entry is a held-out concept ===\n");
        let short = |s: &str| s.split_whitespace().take(4).collect::<Vec<_>>().join(" ");
        let mut shown = 0;
        for (p, _gap, i, j) in proposals.iter() {
            if shown >= 5 {
                break;
            }
            let nb = nearest_excluding(p, *i, *j);
            let best = (cosine_sim(p, &emb[nb]), nb);
            if held_set.contains(&best.1) {
                println!("  between {:?}", short(&texts[*i]));
                println!("      and {:?}", short(&texts[*j]));
                println!("   -> MISSING {:?}  (cos {:.3})\n", short(&texts[best.1]), best.0);
                shown += 1;
            }
        }
        if shown == 0 {
            println!("  none.\n");
        }

        println!("=== Verdict ===\n");
        let best_lift = (n_near / chance).max(n_mid / chance);
        let ctl_lift = n_ctl / chance;
        if best_lift > 1.3 * ctl_lift.max(1.0) {
            println!("  SUPPORTED: proposals point at genuinely-missing concepts {:.2}x above", best_lift);
            println!("  chance, and {:.2}x above manifold-matched random pairs. The geometry", best_lift / ctl_lift.max(1e-9));
            println!("  carries real information about WHERE concepts are absent — which is");
            println!("  exactly the premise the outward search loop needs.");
        } else {
            println!("  NOT SUPPORTED: best generator reaches {:.2}x chance vs {:.2}x for", best_lift, ctl_lift);
            println!("  manifold-matched random pairs. Choosing pairs by proximity adds nothing");
            println!("  over choosing them at random, so 'correlates point at missing objects'");
            println!("  does not hold here — the external check would carry the whole burden.");
        }
        println!(
            "\n(What is NOT claimed: the hold-out entries were removed from an ontology the\n embedder never trained on, but they are concepts of the SAME authored style — a\n found-in-the-wild concept may be harder. One corpus, one embedder; per Iteration\n 26b nothing here transfers across a model swap. The control is the load-bearing\n part: it is matched for manifold and scale, differing only in pair proximity.)"
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
