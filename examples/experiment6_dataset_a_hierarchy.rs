// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 6 — replicate Iteration 5's finding on a SECOND, structurally
//! different dataset: mission Dataset A (a real taxonomic hierarchy), not
//! Dataset B's cross-cutting perspectives-on-one-object.
//!
//! Iteration 5 found: contextual sentences dramatically help REFERENCE-FREE
//! k-means clustering (purity 0.476 -> 0.810) but leave per-reference
//! methods (naive NN, Δ-direction) flat or worse, on the CAR corpus. A
//! single-corpus finding is weak evidence for a general claim about
//! language and context — this experiment tests whether the same pattern
//! holds on a domain with a completely different structure: a strict
//! two-level taxonomy (Animal -> Mammal/Bird -> Domestic/Predator/Flying/
//! Flightless) instead of one object viewed from unrelated perspectives.
//!
//! Same evaluation as Iterations 1-2-5 (naive NN F1, Δ-direction F1,
//! k-means purity, coarse+fine), same "no category-name leakage" rule for
//! the contextual sentences (no "mammal"/"bird"/"predator"/"domestic"/
//! "flying"/"flightless" appears in any sentence).
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment6_dataset_a_hierarchy

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;
use serde::Serialize;

/// (bare term, contextual sentence, coarse [Mammal/Bird], fine [sub-group])
const CORPUS: &[(&str, &str, &str, &str)] = &[
    ("dog", "The dog wagged its tail and waited by the door for its owner to come home.", "mammal", "domestic"),
    ("cat", "The cat curled up on the windowsill and purred in the afternoon sun.", "mammal", "domestic"),
    ("horse", "The horse trotted around the paddock, its mane flowing in the breeze.", "mammal", "domestic"),
    ("sheep", "The sheep grazed quietly in the pasture, following the rest of the flock.", "mammal", "domestic"),
    ("lion", "The lion stalked its prey across the savanna before launching a sudden charge.", "mammal", "predator"),
    ("wolf", "The wolf howled at dusk, calling the rest of its pack to the hunt.", "mammal", "predator"),
    ("bear", "The bear caught a salmon in its claws as the fish leapt upstream.", "mammal", "predator"),
    ("tiger", "The tiger prowled silently through the tall grass, stripes blending with the shadows.", "mammal", "predator"),
    ("eagle", "The eagle soared high above the canyon, scanning the ground for movement.", "bird", "flying"),
    ("sparrow", "The sparrow hopped along the branch before darting off between the leaves.", "bird", "flying"),
    ("owl", "The owl turned its head silently, watching for the faintest movement in the dark.", "bird", "flying"),
    ("swan", "The swan glided smoothly across the lake, barely rippling the water.", "bird", "flying"),
    ("penguin", "The penguin waddled across the ice before diving into the frigid water.", "bird", "flightless"),
    ("ostrich", "The ostrich sprinted across the plain on powerful legs, kicking up dust.", "bird", "flightless"),
    ("kiwi", "The kiwi foraged in the undergrowth at night, sniffing out insects with its long beak.", "bird", "flightless"),
];

fn coarse(i: usize) -> &'static str { CORPUS[i].2 }
fn fine(i: usize) -> &'static str { CORPUS[i].3 }

fn naive_nn_f1(embeddings: &[Vec<f32>], ref_idx: usize, label_of: impl Fn(usize) -> &'static str) -> Option<f32> {
    let true_label = label_of(ref_idx);
    let field_size = (0..embeddings.len()).filter(|&i| i != ref_idx && label_of(i) == true_label).count();
    if field_size == 0 { return None; }
    let r = &embeddings[ref_idx];
    let mut scored: Vec<(usize, f32)> = (0..embeddings.len()).filter(|&i| i != ref_idx).map(|i| (i, cosine_sim(&embeddings[i], r))).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let hits = scored.iter().take(field_size).filter(|(i, _)| label_of(*i) == true_label).count();
    Some(hits as f32 / field_size as f32)
}

fn delta_direction_f1(embeddings: &[Vec<f32>], ref_idx: usize, threshold: f32, label_of: impl Fn(usize) -> &'static str) -> Option<f32> {
    let true_label = label_of(ref_idx);
    let field_size = (0..embeddings.len()).filter(|&i| i != ref_idx && label_of(i) == true_label).count();
    if field_size == 0 { return None; }
    let r = &embeddings[ref_idx];
    let others: Vec<usize> = (0..embeddings.len()).filter(|&i| i != ref_idx).collect();
    let deltas: Vec<Vec<f32>> = others.iter().map(|&i| embeddings[i].iter().zip(r.iter()).map(|(a, b)| a - b).collect()).collect();
    let n = deltas.len();
    let mut assigned = vec![false; n];
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        if assigned[i] { continue; }
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
    let nearest = (0..n).min_by(|&a, &b| {
        let da: f32 = deltas[a].iter().map(|x| x * x).sum();
        let db: f32 = deltas[b].iter().map(|x| x * x).sum();
        da.partial_cmp(&db).unwrap()
    }).unwrap();
    let chosen = clusters.iter().find(|c| c.contains(&nearest)).unwrap();
    let hits = chosen.iter().filter(|&&k| label_of(others[k]) == true_label).count();
    let precision = hits as f32 / chosen.len() as f32;
    let recall = hits as f32 / field_size as f32;
    Some(if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 })
}

fn kmeans(embeddings: &[Vec<f32>], k: usize, iterations: usize) -> Vec<usize> {
    let n = embeddings.len();
    let dim = embeddings[0].len();
    let mut centroid_idx = vec![0usize];
    while centroid_idx.len() < k {
        let next = (0..n).max_by(|&a, &b| {
            let da = centroid_idx.iter().map(|&c| 1.0 - cosine_sim(&embeddings[a], &embeddings[c])).fold(f32::INFINITY, f32::min);
            let db = centroid_idx.iter().map(|&c| 1.0 - cosine_sim(&embeddings[b], &embeddings[c])).fold(f32::INFINITY, f32::min);
            da.partial_cmp(&db).unwrap()
        }).unwrap();
        centroid_idx.push(next);
    }
    let mut centroids: Vec<Vec<f32>> = centroid_idx.iter().map(|&i| embeddings[i].clone()).collect();
    let mut assignment = vec![0usize; n];
    for _ in 0..iterations {
        for i in 0..n {
            assignment[i] = (0..k).max_by(|&a, &b| cosine_sim(&embeddings[i], &centroids[a]).partial_cmp(&cosine_sim(&embeddings[i], &centroids[b])).unwrap()).unwrap();
        }
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignment[i];
            counts[c] += 1;
            for d in 0..dim { sums[c][d] += embeddings[i][d]; }
        }
        for c in 0..k {
            if counts[c] == 0 { continue; }
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
            if *a == c { *counts.entry(label_of(i)).or_insert(0usize) += 1; }
        }
        correct += counts.values().copied().max().unwrap_or(0);
    }
    correct as f32 / n as f32
}

#[derive(Serialize)]
struct Comparison {
    representation: String,
    mean_f1_naive_coarse: f32,
    mean_f1_naive_fine: f32,
    mean_f1_delta_coarse: f32,
    mean_f1_delta_fine: f32,
    kmeans_k2_purity: f32,
    kmeans_k4_purity: f32,
}

fn evaluate(embeddings: &[Vec<f32>], label: &str) -> Comparison {
    let delta_threshold = 0.3;
    let (mut nc_sum, mut nf_sum, mut dc_sum, mut df_sum) = (0.0, 0.0, 0.0, 0.0);
    let (mut nc_n, mut nf_n) = (0, 0);
    for i in 0..CORPUS.len() {
        if let Some(v) = naive_nn_f1(embeddings, i, coarse) { nc_sum += v; nc_n += 1; }
        if let Some(v) = naive_nn_f1(embeddings, i, fine) { nf_sum += v; nf_n += 1; }
        if let Some(v) = delta_direction_f1(embeddings, i, delta_threshold, coarse) { dc_sum += v; }
        if let Some(v) = delta_direction_f1(embeddings, i, delta_threshold, fine) { df_sum += v; }
    }
    let k2 = kmeans(embeddings, 2, 25);
    let k4 = kmeans(embeddings, 4, 25);
    Comparison {
        representation: label.to_string(),
        mean_f1_naive_coarse: nc_sum / nc_n as f32,
        mean_f1_naive_fine: nf_sum / nf_n as f32,
        mean_f1_delta_coarse: dc_sum / nc_n as f32,
        mean_f1_delta_fine: df_sum / nf_n as f32,
        kmeans_k2_purity: purity(&k2, 2, coarse),
        kmeans_k4_purity: purity(&k4, 4, fine),
    }
}

fn main() {
    println!("Experiment 6: Dataset A (taxonomic hierarchy) — replicate Iteration 5's finding on different structure\n");
    println!("15 animals: mammal/bird coarse (8/7), domestic/predator/flying/flightless fine (4/4/4/3)\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder};
        let mut embedder = None;
        for dir in ["models", "../models"] {
            let p = std::path::Path::new(dir);
            if !(p.join("model.onnx").exists() || p.join("onnx/model.onnx").exists()) { continue; }
            let cfg = OnnxConfig { dim: 384, model_dir: Some(dir.to_string()), ..OnnxConfig::default() };
            let e = OnnxEmbedder::with_config(&cfg);
            if e.is_available() {
                println!("Using real semantic embedder: {dir}/model.onnx\n");
                embedder = Some(e);
                break;
            }
        }
        let embedder = match embedder { Some(e) => e, None => { println!("WARNING: no ONNX model — aborting."); return; } };

        let bare: Vec<Vec<f32>> = CORPUS.iter().map(|(term, ..)| embedder.embed(term)).collect();
        let ctx: Vec<Vec<f32>> = CORPUS.iter().map(|(_, s, ..)| embedder.embed(s)).collect();

        let bare_result = evaluate(&bare, "bare-term");
        let ctx_result = evaluate(&ctx, "contextual-sentence");

        println!("{:<24} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}", "representation", "nv_C", "nv_F", "dl_C", "dl_F", "km2_pur", "km4_pur");
        for r in [&bare_result, &ctx_result] {
            println!("{:<24} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>10.3} {:>10.3}",
                r.representation, r.mean_f1_naive_coarse, r.mean_f1_naive_fine, r.mean_f1_delta_coarse, r.mean_f1_delta_fine, r.kmeans_k2_purity, r.kmeans_k4_purity);
        }
        println!("\n=== DELTA (contextual - bare) ===");
        println!("naive/coarse:  {:+.3}", ctx_result.mean_f1_naive_coarse - bare_result.mean_f1_naive_coarse);
        println!("naive/fine:    {:+.3}", ctx_result.mean_f1_naive_fine - bare_result.mean_f1_naive_fine);
        println!("delta/coarse:  {:+.3}", ctx_result.mean_f1_delta_coarse - bare_result.mean_f1_delta_coarse);
        println!("delta/fine:    {:+.3}", ctx_result.mean_f1_delta_fine - bare_result.mean_f1_delta_fine);
        println!("kmeans coarse: {:+.3}", ctx_result.kmeans_k2_purity - bare_result.kmeans_k2_purity);
        println!("kmeans fine:   {:+.3}", ctx_result.kmeans_k4_purity - bare_result.kmeans_k4_purity);

        println!("\n=== Cross-corpus comparison (CAR = Iteration 5, ANIMAL = this experiment) ===");
        println!("k-means coarse purity delta:  CAR +0.333   ANIMAL {:+.3}", ctx_result.kmeans_k2_purity - bare_result.kmeans_k2_purity);
        println!("naive NN fine F1 delta:       CAR -0.183   ANIMAL {:+.3}", ctx_result.mean_f1_naive_fine - bare_result.mean_f1_naive_fine);

        let json = serde_json::to_string_pretty(&[&bare_result, &ctx_result]).unwrap();
        std::fs::write("experiment6_results.json", &json).expect("write");
        println!("\nFull results written to experiment6_results.json");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
