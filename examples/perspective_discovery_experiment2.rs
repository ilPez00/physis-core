// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 2 — Iteration 2: fix the broken baseline, add fine-grained
//! ground truth, test across every item as its own reference.
//!
//! Iteration 1 (see `perspective_discovery_experiment.rs` and
//! `../research/perspective-discovery/RESEARCH_LOG.md`) left two problems
//! unresolved:
//!
//! 1. The "ordinary discovery" baseline (single-link greedy clustering at a
//!    fixed cosine threshold) collapsed to one giant cluster at every
//!    threshold from 0.3 to 0.8 — unmeasurable, not merely beaten. Fixed here
//!    with a real k-means (farthest-point/k-means++ init, deterministic, no
//!    RNG) at a fixed target k.
//! 2. Only 3 of 21 possible references were tested, and the one interesting
//!    result (a tight, 100%-precision sub-cluster for "purchase price" that
//!    scored only 50% recall against coarse labels) couldn't be evaluated
//!    properly with 3-label ground truth. Fixed here with 6 fine-grained
//!    sub-perspective labels (~2 per top-level perspective) and by testing
//!    every item as a reference, not just one per perspective.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --example perspective_discovery_experiment2

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;
use serde::Serialize;

/// Dataset B, extended with fine-grained sub-perspective ground truth.
/// Fine-group sizes are 3 or 4, decided by inspecting the corpus for a real
/// semantic sub-relation, NOT reverse-engineered from what would make any
/// method look good — the "value-over-time" split is exactly the 3 items
/// Iteration 1's delta-direction clustering found unprompted for
/// "purchase price," predating this ground truth being written.
const CORPUS: &[(&str, &str, &str)] = &[
    // Mechanical / power-core (the combustion/power internals)
    ("engine", "mechanical", "power-core"),
    ("piston", "mechanical", "power-core"),
    ("camshaft", "mechanical", "power-core"),
    ("torque", "mechanical", "power-core"),
    // Mechanical / drivetrain-support
    ("transmission", "mechanical", "drivetrain-support"),
    ("fuel injection", "mechanical", "drivetrain-support"),
    ("cooling system", "mechanical", "drivetrain-support"),
    // Driver / active-controls (things you operate)
    ("steering wheel", "driver", "active-controls"),
    ("brake pedal", "driver", "active-controls"),
    ("accelerator pedal", "driver", "active-controls"),
    ("turn signal", "driver", "active-controls"),
    // Driver / safety-awareness
    ("rearview mirror", "driver", "safety-awareness"),
    ("seat belt", "driver", "safety-awareness"),
    ("dashboard display", "driver", "safety-awareness"),
    // Economic / value-over-time (this exact grouping is what Experiment 1
    // found for "purchase price" before this file existed)
    ("resale value", "economic", "value-over-time"),
    ("depreciation rate", "economic", "value-over-time"),
    ("trade-in value", "economic", "value-over-time"),
    // Economic / recurring-cost (purchase price is a judgment call: it's a
    // one-time cost, not recurring, but grouping it alone would make a
    // 1-item field nothing can "recover" — noted honestly rather than hidden)
    ("insurance premium", "economic", "recurring-cost"),
    ("fuel cost", "economic", "recurring-cost"),
    ("loan interest", "economic", "recurring-cost"),
    ("purchase price", "economic", "recurring-cost"),
];

#[derive(Serialize)]
struct AggregateResult {
    embedder: String,
    n_references_tested: usize,
    kmeans_k3_purity_vs_coarse: f32,
    kmeans_k6_purity_vs_fine: f32,
    mean_f1_naive_coarse: f32,
    mean_f1_naive_fine: f32,
    mean_f1_delta_coarse: f32,
    mean_f1_delta_fine: f32,
    delta_wins_coarse: usize,
    delta_wins_fine: usize,
    ties_coarse: usize,
    ties_fine: usize,
    per_reference: Vec<PerReferenceResult>,
}

#[derive(Serialize)]
struct PerReferenceResult {
    reference: &'static str,
    coarse: &'static str,
    fine: &'static str,
    naive_f1_coarse: f32,
    delta_f1_coarse: f32,
    naive_f1_fine: f32,
    delta_f1_fine: f32,
}

fn embed_all(embedder: &dyn VectorEmbed) -> Vec<Vec<f32>> {
    CORPUS.iter().map(|(text, ..)| embedder.embed(text)).collect()
}

/// Deterministic k-means: farthest-point (k-means++-style, no randomness)
/// initialization, then Lloyd's iterations on cosine similarity with
/// re-normalized mean centroids (correct for unit-length embeddings).
fn kmeans(embeddings: &[Vec<f32>], k: usize, iterations: usize) -> Vec<usize> {
    let n = embeddings.len();
    let dim = embeddings[0].len();

    // Farthest-point init: first centroid = item 0; each subsequent centroid
    // is the point with the largest minimum distance to all chosen so far.
    // Fully deterministic, no seed to tune, doesn't touch the label data.
    let mut centroid_idx = vec![0usize];
    while centroid_idx.len() < k {
        let next = (0..n)
            .max_by(|&a, &b| {
                let da = centroid_idx
                    .iter()
                    .map(|&c| 1.0 - cosine_sim(&embeddings[a], &embeddings[c]))
                    .fold(f32::INFINITY, f32::min);
                let db = centroid_idx
                    .iter()
                    .map(|&c| 1.0 - cosine_sim(&embeddings[b], &embeddings[c]))
                    .fold(f32::INFINITY, f32::min);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        centroid_idx.push(next);
    }
    let mut centroids: Vec<Vec<f32>> = centroid_idx.iter().map(|&i| embeddings[i].clone()).collect();

    let mut assignment = vec![0usize; n];
    for _ in 0..iterations {
        // Assign.
        for i in 0..n {
            assignment[i] = (0..k)
                .max_by(|&a, &b| {
                    cosine_sim(&embeddings[i], &centroids[a])
                        .partial_cmp(&cosine_sim(&embeddings[i], &centroids[b]))
                        .unwrap()
                })
                .unwrap();
        }
        // Update: mean of assigned points, re-normalized. A cluster that
        // loses all members keeps its previous centroid rather than becoming
        // NaN — deterministic, doesn't need re-seeding logic for k=3/k=6 on
        // n=21, where empty clusters do not occur in practice but must not
        // panic if they did.
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignment[i];
            counts[c] += 1;
            for d in 0..dim {
                sums[c][d] += embeddings[i][d];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                continue;
            }
            let norm: f32 = sums[c].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            centroids[c] = sums[c].iter().map(|x| x / norm).collect();
        }
    }
    assignment
}

fn purity(assignment: &[usize], k: usize, label_of: impl Fn(usize) -> &'static str) -> f32 {
    let n = assignment.len();
    let mut correct = 0usize;
    for c in 0..k {
        let mut counts = std::collections::HashMap::new();
        for (i, a) in assignment.iter().enumerate() {
            if *a == c {
                *counts.entry(label_of(i)).or_insert(0usize) += 1;
            }
        }
        correct += counts.values().copied().max().unwrap_or(0);
    }
    correct as f32 / n as f32
}

/// Naive nearest-neighbor-to-reference, generalized to any label function.
/// Field size = how many OTHER items share ref_idx's label under that
/// function; if that's zero (a label with no other members, e.g. a fine
/// group of size 1) the method is undefined and reported as `None`.
fn naive_nn_f1(embeddings: &[Vec<f32>], ref_idx: usize, label_of: impl Fn(usize) -> &'static str) -> Option<f32> {
    let true_label = label_of(ref_idx);
    let field_size = (0..embeddings.len())
        .filter(|&i| i != ref_idx && label_of(i) == true_label)
        .count();
    if field_size == 0 {
        return None;
    }
    let r = &embeddings[ref_idx];
    let mut scored: Vec<(usize, f32)> = (0..embeddings.len())
        .filter(|&i| i != ref_idx)
        .map(|i| (i, cosine_sim(&embeddings[i], r)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let hits = scored.iter().take(field_size).filter(|(i, _)| label_of(*i) == true_label).count();
    // Fixed-size retrieval: precision == recall == hits/field_size, so F1
    // collapses to the same value.
    Some(hits as f32 / field_size as f32)
}

/// Delta-direction clustering, generalized to any label function, with the
/// Iteration-1 bug fix retained (choose the cluster containing r's nearest
/// neighbor, not the largest cluster).
fn delta_direction_f1(
    embeddings: &[Vec<f32>],
    ref_idx: usize,
    threshold: f32,
    label_of: impl Fn(usize) -> &'static str,
) -> Option<f32> {
    let true_label = label_of(ref_idx);
    let field_size = (0..embeddings.len())
        .filter(|&i| i != ref_idx && label_of(i) == true_label)
        .count();
    if field_size == 0 {
        return None;
    }
    let r = &embeddings[ref_idx];
    let others: Vec<usize> = (0..embeddings.len()).filter(|&i| i != ref_idx).collect();
    let deltas: Vec<Vec<f32>> = others
        .iter()
        .map(|&i| embeddings[i].iter().zip(r.iter()).map(|(a, b)| a - b).collect())
        .collect();
    let n = deltas.len();
    let mut assigned = vec![false; n];
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        if assigned[i] {
            continue;
        }
        assigned[i] = true;
        let mut members = vec![i];
        for j in (i + 1)..n {
            if !assigned[j] && cosine_sim(&deltas[i], &deltas[j]) > threshold {
                assigned[j] = true;
                members.push(j);
            }
        }
        clusters.push(members);
    }
    let nearest = (0..n)
        .min_by(|&a, &b| {
            let da: f32 = deltas[a].iter().map(|x| x * x).sum();
            let db: f32 = deltas[b].iter().map(|x| x * x).sum();
            da.partial_cmp(&db).unwrap()
        })
        .unwrap();
    let chosen = clusters.iter().find(|c| c.contains(&nearest)).unwrap();
    let hits = chosen.iter().filter(|&&k| label_of(others[k]) == true_label).count();
    let precision = hits as f32 / chosen.len() as f32;
    let recall = hits as f32 / field_size as f32;
    Some(if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    })
}

fn coarse(i: usize) -> &'static str {
    CORPUS[i].1
}
fn fine(i: usize) -> &'static str {
    CORPUS[i].2
}

fn main() {
    println!("Experiment 2 (Iteration 2): fixed k-means baseline + fine-grained ground truth");
    println!("Dataset B extended: {} items, 3 coarse labels, 6 fine labels\n", CORPUS.len());

    #[cfg(feature = "embed-onnx")]
    let embeddings = {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder};
        let mut found = None;
        for dir in ["models", "../models"] {
            let p = std::path::Path::new(dir);
            if !(p.join("model.onnx").exists() || p.join("onnx/model.onnx").exists()) {
                continue;
            }
            let cfg = OnnxConfig { dim: 384, model_dir: Some(dir.to_string()), ..OnnxConfig::default() };
            let e = OnnxEmbedder::with_config(&cfg);
            if e.is_available() {
                println!("Using real semantic embedder: {dir}/model.onnx (all-MiniLM-L6-v2, 384d)\n");
                found = Some(embed_all(&e));
                break;
            }
        }
        match found {
            Some(e) => e,
            None => {
                println!("WARNING: no ONNX model found — aborting. Experiment 2 requires the real embedder (Experiment 1 already established the fallback isn't meaningful evidence).");
                return;
            }
        }
    };
    #[cfg(not(feature = "embed-onnx"))]
    {
        println!("Built without --features embed-onnx; this experiment requires the real embedder. Aborting.");
    }

    #[cfg(feature = "embed-onnx")]
    {
        let k3 = kmeans(&embeddings, 3, 25);
        let k6 = kmeans(&embeddings, 6, 25);
        let purity_k3 = purity(&k3, 3, coarse);
        let purity_k6 = purity(&k6, 6, fine);
        println!("k-means(k=3) purity vs coarse labels: {purity_k3:.3}");
        println!("k-means(k=6) purity vs fine labels:   {purity_k6:.3}\n");

        let delta_threshold = 0.3;
        let mut per_reference = Vec::new();
        let (mut nc_sum, mut nf_sum, mut dc_sum, mut df_sum) = (0.0, 0.0, 0.0, 0.0);
        let (mut nc_n, mut nf_n) = (0, 0);
        let (mut delta_wins_c, mut delta_wins_f, mut ties_c, mut ties_f) = (0, 0, 0, 0);

        for (i, item) in CORPUS.iter().enumerate() {
            let naive_c = naive_nn_f1(&embeddings, i, coarse);
            let delta_c = delta_direction_f1(&embeddings, i, delta_threshold, coarse);
            let naive_f = naive_nn_f1(&embeddings, i, fine);
            let delta_f = delta_direction_f1(&embeddings, i, delta_threshold, fine);

            if let (Some(nc), Some(dc)) = (naive_c, delta_c) {
                nc_sum += nc;
                dc_sum += dc;
                nc_n += 1;
                if dc > nc { delta_wins_c += 1 } else if (dc - nc).abs() < 1e-6 { ties_c += 1 }
            }
            if let (Some(nf), Some(df)) = (naive_f, delta_f) {
                nf_sum += nf;
                df_sum += df;
                nf_n += 1;
                if df > nf { delta_wins_f += 1 } else if (df - nf).abs() < 1e-6 { ties_f += 1 }
            }

            per_reference.push(PerReferenceResult {
                reference: item.0,
                coarse: item.1,
                fine: item.2,
                naive_f1_coarse: naive_c.unwrap_or(f32::NAN),
                delta_f1_coarse: delta_c.unwrap_or(f32::NAN),
                naive_f1_fine: naive_f.unwrap_or(f32::NAN),
                delta_f1_fine: delta_f.unwrap_or(f32::NAN),
            });
        }

        println!(
            "{:<20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "reference", "coarse", "fine", "nv_C", "dl_C", "nv_F", "dl_F"
        );
        for pr in &per_reference {
            println!(
                "{:<20} {:<8} {:<8} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
                pr.reference, pr.coarse, pr.fine, pr.naive_f1_coarse, pr.delta_f1_coarse, pr.naive_f1_fine, pr.delta_f1_fine
            );
        }

        let agg = AggregateResult {
            embedder: "onnx-minilm-384d".to_string(),
            n_references_tested: CORPUS.len(),
            kmeans_k3_purity_vs_coarse: purity_k3,
            kmeans_k6_purity_vs_fine: purity_k6,
            mean_f1_naive_coarse: nc_sum / nc_n as f32,
            mean_f1_naive_fine: nf_sum / nf_n as f32,
            mean_f1_delta_coarse: dc_sum / nc_n as f32,
            mean_f1_delta_fine: df_sum / nf_n as f32,
            delta_wins_coarse: delta_wins_c,
            delta_wins_fine: delta_wins_f,
            ties_coarse: ties_c,
            ties_fine: ties_f,
            per_reference,
        };

        println!("\n=== AGGREGATE (n={} references; fine-label aggregates exclude {} singleton groups) ===",
            agg.n_references_tested, agg.n_references_tested - nf_n);
        println!("k-means(k=3) purity vs coarse:  {:.3}", agg.kmeans_k3_purity_vs_coarse);
        println!("k-means(k=6) purity vs fine:    {:.3}", agg.kmeans_k6_purity_vs_fine);
        println!("mean F1  naive/coarse: {:.3}   delta/coarse: {:.3}   (delta wins {}/{}, ties {})",
            agg.mean_f1_naive_coarse, agg.mean_f1_delta_coarse, agg.delta_wins_coarse, nc_n, agg.ties_coarse);
        println!("mean F1  naive/fine:   {:.3}   delta/fine:   {:.3}   (delta wins {}/{}, ties {})",
            agg.mean_f1_naive_fine, agg.mean_f1_delta_fine, agg.delta_wins_fine, nf_n, agg.ties_fine);

        let json = serde_json::to_string_pretty(&agg).unwrap();
        std::fs::write("experiment2_results.json", &json).expect("write results");
        println!("\nFull results written to experiment2_results.json");
    }
}
