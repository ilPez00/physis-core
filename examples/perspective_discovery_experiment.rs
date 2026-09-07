// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 1 — reference-conditioned discovery vs. naive baselines.
//!
//! Tests the hypothesis: "a reference point changes the *relational structure*
//! discovered in a shared embedding space, not just which items rank near the
//! top of a similarity search." Section 7 of the research mission explicitly
//! warns that `Δ(x,r) = embed(x) - embed(r)` ranked by magnitude is suspected
//! to be mathematically identical to plain `cosine(x,r)` ranking for
//! L2-normalized vectors — this experiment verifies that numerically and then
//! tests a method that is NOT reducible to a single-point similarity search:
//! clustering items by the *direction* of their delta from the reference
//! (`cos(Δ(x,r), Δ(y,r))`) rather than by the raw embedding similarity
//! `cos(x,y)` or the naive `cos(x,r)`.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --example perspective_discovery_experiment
//!
//! Falls back to `RandomProjectionEmbedder` (with a loud warning) if
//! `models/model.onnx` is not found relative to the workspace root — results
//! under that fallback are not meaningful evidence for or against the
//! hypothesis (see README: it is character-n-gram hashing, not semantics) and
//! are labeled as such in the output.

use physis_core::embed::{RandomProjectionEmbedder, VectorEmbed};
use physis_core::models::cosine_sim;
use serde::Serialize;

/// Dataset B — multiple perspectives on one topic (CAR), ground-truth labeled.
/// 7 items per perspective so a "field" (perspective minus the reference
/// itself) is a fixed size of 6 — precision and recall coincide, simplifying
/// the metric to a single hit-rate number.
const CORPUS: &[(&str, &str)] = &[
    // Mechanical
    ("engine", "mechanical"),
    ("transmission", "mechanical"),
    ("torque", "mechanical"),
    ("piston", "mechanical"),
    ("fuel injection", "mechanical"),
    ("cooling system", "mechanical"),
    ("camshaft", "mechanical"),
    // Driver
    ("steering wheel", "driver"),
    ("brake pedal", "driver"),
    ("accelerator pedal", "driver"),
    ("rearview mirror", "driver"),
    ("seat belt", "driver"),
    ("dashboard display", "driver"),
    ("turn signal", "driver"),
    // Economic
    ("purchase price", "economic"),
    ("insurance premium", "economic"),
    ("resale value", "economic"),
    ("depreciation rate", "economic"),
    ("fuel cost", "economic"),
    ("loan interest", "economic"),
    ("trade-in value", "economic"),
];

/// One reference per perspective — the term whose "field" we try to recover.
const REFERENCES: &[&str] = &["engine", "steering wheel", "purchase price"];

const FIELD_SIZE: usize = 6; // 7 per perspective, minus the reference itself.

#[derive(Serialize)]
struct RunResult {
    embedder: String,
    embedder_is_semantic: bool,
    baseline_flat_clustering_purity: f32,
    baseline_flat_num_clusters: usize,
    delta_magnitude_vs_cosine_rank_agreement: f32,
    per_reference: Vec<ReferenceResult>,
}

#[derive(Serialize)]
struct ReferenceResult {
    reference: &'static str,
    true_perspective: &'static str,
    naive_nn_hit_rate: f32,
    delta_direction_precision: f32,
    delta_direction_recall: f32,
    delta_direction_f1: f32,
}

fn embed_all(embedder: &dyn VectorEmbed) -> Vec<Vec<f32>> {
    CORPUS.iter().map(|(text, _)| embedder.embed(text)).collect()
}

/// Greedy single-link clustering at a fixed cosine threshold — the same shape
/// of algorithm `physis_core::discovery::cluster_at` uses, applied directly to
/// raw embeddings with no reference at all. This is "ordinary discovery."
fn flat_cluster(embeddings: &[Vec<f32>], threshold: f32) -> Vec<Vec<usize>> {
    let n = embeddings.len();
    let mut assigned = vec![false; n];
    let mut clusters = Vec::new();
    for i in 0..n {
        if assigned[i] {
            continue;
        }
        assigned[i] = true;
        let mut members = vec![i];
        for j in (i + 1)..n {
            if !assigned[j] && cosine_sim(&embeddings[i], &embeddings[j]) > threshold {
                assigned[j] = true;
                members.push(j);
            }
        }
        clusters.push(members);
    }
    clusters
}

/// Majority-vote purity: for each cluster, the fraction of members sharing
/// its most common true label, weighted by cluster size and averaged.
fn cluster_purity(clusters: &[Vec<usize>]) -> f32 {
    let n: usize = clusters.iter().map(|c| c.len()).sum();
    if n == 0 {
        return 0.0;
    }
    let mut correct = 0usize;
    for c in clusters {
        let mut counts = std::collections::HashMap::new();
        for &i in c {
            *counts.entry(CORPUS[i].1).or_insert(0usize) += 1;
        }
        correct += counts.values().copied().max().unwrap_or(0);
    }
    correct as f32 / n as f32
}

/// Naive nearest-neighbor-to-reference: rank all other items by cosine(x,r),
/// take the top FIELD_SIZE, report the hit-rate against the reference's true
/// perspective. This is exactly the "recursive nearest-neighbor search" the
/// mission (Section 7) says does NOT constitute reference conditioning.
fn naive_nn_hit_rate(embeddings: &[Vec<f32>], ref_idx: usize) -> f32 {
    let r = &embeddings[ref_idx];
    let true_label = CORPUS[ref_idx].1;
    let mut scored: Vec<(usize, f32)> = (0..embeddings.len())
        .filter(|&i| i != ref_idx)
        .map(|i| (i, cosine_sim(&embeddings[i], r)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let hits = scored
        .iter()
        .take(FIELD_SIZE)
        .filter(|(i, _)| CORPUS[*i].1 == true_label)
        .count();
    hits as f32 / FIELD_SIZE as f32
}

/// Delta-direction clustering: compute Δ(x,r) for every other item, cluster
/// THOSE vectors by cosine similarity to each other (not by raw cos(x,y) and
/// not by cos(x,r)), take the largest resulting cluster as "the field of r",
/// report its hit-rate. This is the one method that is not reducible to a
/// single-point similarity search against r.
/// Cluster the delta vectors Δ(x,r) among themselves and identify "r's own
/// cluster" as the one containing the item with the smallest |Δ| (i.e. the
/// item closest to r in the original embedding — r's own nearest neighbor).
/// Returns (precision, recall, f1, member terms). This fixes a real bug found
/// in the first version of this experiment: picking the *largest* cluster
/// instead of the one closest to r let a near-total-corpus blob "win" and
/// silently credited it with a perfect score just because the true field
/// (being a subset of everything) was trivially contained in it — precision
/// was never checked. Confirmed by inspection: at threshold=0.3, two of three
/// references produced one 18-20-member cluster out of 20 possible, and the
/// old metric (recall against a fixed denominator, no precision term) scored
/// that as ~1.0 whenever the 6 true members happened to be swept in with
/// everything else.
fn delta_direction_diagnose(
    embeddings: &[Vec<f32>],
    ref_idx: usize,
    threshold: f32,
) -> (f32, f32, f32, Vec<&'static str>) {
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
    // r's own nearest neighbor by |Δ| (equivalently, by cosine — see the
    // magnitude/cosine equivalence proof above) anchors which cluster is
    // "r's field," instead of just taking whichever cluster happens biggest.
    let nearest = (0..n)
        .min_by(|&a, &b| {
            let da: f32 = deltas[a].iter().map(|x| x * x).sum();
            let db: f32 = deltas[b].iter().map(|x| x * x).sum();
            da.partial_cmp(&db).unwrap()
        })
        .unwrap();
    let chosen = clusters.iter().find(|c| c.contains(&nearest)).unwrap();

    let true_label = CORPUS[ref_idx].1;
    let hits = chosen.iter().filter(|&&k| CORPUS[others[k]].1 == true_label).count();
    let precision = hits as f32 / chosen.len() as f32;
    let recall = hits as f32 / FIELD_SIZE as f32;
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let members: Vec<&'static str> = chosen.iter().map(|&k| CORPUS[others[k]].0).collect();
    (precision, recall, f1, members)
}

fn delta_direction_f1(embeddings: &[Vec<f32>], ref_idx: usize, threshold: f32) -> f32 {
    delta_direction_diagnose(embeddings, ref_idx, threshold).2
}

/// Numerically verify (or falsify) the Section-7 suspicion: for L2-normalized
/// vectors, |Δ(x,r)|² = 2 - 2·cos(x,r), so ranking by ascending |Δ| must be
/// identical to ranking by descending cos(x,r). Returns the fraction of
/// adjacent-pair orderings that agree between the two rankings (1.0 = fully
/// equivalent, matching the algebra).
fn delta_magnitude_vs_cosine_rank_agreement(embeddings: &[Vec<f32>], ref_idx: usize) -> f32 {
    let r = &embeddings[ref_idx];
    let mut by_cosine: Vec<usize> = (0..embeddings.len()).filter(|&i| i != ref_idx).collect();
    let mut by_delta = by_cosine.clone();
    by_cosine.sort_by(|&a, &b| {
        cosine_sim(&embeddings[b], r)
            .partial_cmp(&cosine_sim(&embeddings[a], r))
            .unwrap()
    });
    by_delta.sort_by(|&a, &b| {
        let da: f32 = embeddings[a].iter().zip(r.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        let db: f32 = embeddings[b].iter().zip(r.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        da.partial_cmp(&db).unwrap()
    });
    let n = by_cosine.len();
    let agree = by_cosine.iter().zip(by_delta.iter()).filter(|(a, b)| a == b).count();
    agree as f32 / n as f32
}

fn run(embedder: &dyn VectorEmbed, embedder_name: &str, is_semantic: bool) -> RunResult {
    let embeddings = embed_all(embedder);

    let baseline_threshold = 0.5;
    let clusters = flat_cluster(&embeddings, baseline_threshold);
    let purity = cluster_purity(&clusters);

    let delta_threshold = 0.3;
    let per_reference: Vec<ReferenceResult> = REFERENCES
        .iter()
        .map(|&r| {
            let ref_idx = CORPUS.iter().position(|(t, _)| *t == r).unwrap();
            let (precision, recall, f1, _) = delta_direction_diagnose(&embeddings, ref_idx, delta_threshold);
            ReferenceResult {
                reference: r,
                true_perspective: CORPUS[ref_idx].1,
                naive_nn_hit_rate: naive_nn_hit_rate(&embeddings, ref_idx),
                delta_direction_precision: precision,
                delta_direction_recall: recall,
                delta_direction_f1: f1,
            }
        })
        .collect();

    let rank_agreement = REFERENCES
        .iter()
        .map(|&r| {
            let ref_idx = CORPUS.iter().position(|(t, _)| *t == r).unwrap();
            delta_magnitude_vs_cosine_rank_agreement(&embeddings, ref_idx)
        })
        .sum::<f32>()
        / REFERENCES.len() as f32;

    RunResult {
        embedder: embedder_name.to_string(),
        embedder_is_semantic: is_semantic,
        baseline_flat_clustering_purity: purity,
        baseline_flat_num_clusters: clusters.len(),
        delta_magnitude_vs_cosine_rank_agreement: rank_agreement,
        per_reference,
    }
}

fn main() {
    println!("Experiment 1: reference-conditioned discovery vs. naive baselines");
    println!("Dataset B: CAR, 3 perspectives x 7 terms = {} items\n", CORPUS.len());

    let mut results = Vec::new();

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder};
        for dir in ["models", "../models"] {
            let p = std::path::Path::new(dir);
            if !(p.join("model.onnx").exists() || p.join("onnx/model.onnx").exists()) {
                continue;
            }
            let cfg = OnnxConfig {
                dim: 384,
                model_dir: Some(dir.to_string()),
                ..OnnxConfig::default()
            };
            let e = OnnxEmbedder::with_config(&cfg);
            if e.is_available() {
                println!("Using real semantic embedder: {dir}/model.onnx (all-MiniLM-L6-v2, 384d)\n");
                results.push(run(&e, "onnx-minilm-384d", true));
            }
            break;
        }
    }

    if results.is_empty() {
        println!(
            "WARNING: no ONNX model found (checked models/, ../models/). Falling back to \
             RandomProjectionEmbedder — this is character-n-gram hashing, NOT semantics. \
             Results below are a negative-control sanity check only, not evidence for or \
             against the hypothesis.\n"
        );
        let e = RandomProjectionEmbedder::new(384);
        results.push(run(&e, "random-projection-384d (NOT semantic)", false));
    }

    for r in &results {
        println!("== {} (semantic={}) ==", r.embedder, r.embedder_is_semantic);
        println!(
            "baseline flat clustering (no reference): {} clusters, purity={:.3}",
            r.baseline_flat_num_clusters, r.baseline_flat_clustering_purity
        );
        println!(
            "delta-magnitude vs cosine rank agreement: {:.3} (1.000 = mathematically identical, as predicted by |Δ|²=2-2cos for unit vectors)",
            r.delta_magnitude_vs_cosine_rank_agreement
        );
        for pr in &r.per_reference {
            println!(
                "  reference={:<16} true_perspective={:<10} naive_nn_hit_rate={:.3}  delta_direction[P={:.3} R={:.3} F1={:.3}]  delta_beats_naive={}",
                pr.reference,
                pr.true_perspective,
                pr.naive_nn_hit_rate,
                pr.delta_direction_precision,
                pr.delta_direction_recall,
                pr.delta_direction_f1,
                pr.delta_direction_f1 > pr.naive_nn_hit_rate
            );
        }
        println!();
    }

    // Diagnose the failure/success modes: what did delta-direction actually
    // retrieve for each reference, and how sensitive is it to the threshold?
    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder};
        let cfg = OnnxConfig {
            dim: 384,
            model_dir: Some("models".to_string()),
            ..OnnxConfig::default()
        };
        let e = OnnxEmbedder::with_config(&cfg);
        if e.is_available() {
            let embeddings = embed_all(&e);
            println!("== diagnosis: delta-direction cluster contents (r's own cluster, not the largest) ==");
            for &r in REFERENCES {
                let ref_idx = CORPUS.iter().position(|(t, _)| *t == r).unwrap();
                let (p, rec, f1, members) = delta_direction_diagnose(&embeddings, ref_idx, 0.3);
                println!("  reference={r:<16} P={p:.3} R={rec:.3} F1={f1:.3} retrieved={members:?}");
            }
            println!();
            println!("== threshold sweep: delta-direction F1 vs threshold ==");
            for &r in REFERENCES {
                let ref_idx = CORPUS.iter().position(|(t, _)| *t == r).unwrap();
                let sweep: Vec<String> = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
                    .iter()
                    .map(|&t| {
                        let rate = delta_direction_f1(&embeddings, ref_idx, t);
                        format!("{t:.2}->{rate:.2}")
                    })
                    .collect();
                println!("  reference={r:<16} {}", sweep.join("  "));
            }
            println!();
            println!("== threshold sweep: baseline flat clustering purity vs threshold ==");
            for &t in &[0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8] {
                let clusters = flat_cluster(&embeddings, t);
                println!(
                    "  threshold={t:.2} num_clusters={:<3} purity={:.3}",
                    clusters.len(),
                    cluster_purity(&clusters)
                );
            }
            println!();
        }
    }

    let json = serde_json::to_string_pretty(&results).unwrap();
    std::fs::write("experiment1_results.json", &json).expect("write results");
    println!("Full results written to experiment1_results.json");
}
