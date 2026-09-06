//! Experiment 9 — does a cheap "certification" signal actually distinguish
//! a good split from a bad one, before wiring it into recursion?
//!
//! Prompted directly by the user's proposed architecture: multi-membership
//! (Iteration 8) + coherence gating + certified recursion (Iteration 7).
//! Iteration 7 showed recursion works well GIVEN a correct top-level split
//! (oracle, 0.625-0.75 purity) and badly given a wrong one (real, a 13-vs-2
//! imbalance instead of the true ~8-vs-7, 0.462 purity). "Certified
//! recursion" means: only recurse when some internal signal (no ground
//! truth available in production) predicts the split is trustworthy. That
//! signal has never been tested in this track — physis-core's existing
//! `coherence_score` is a NODE-level density metric (mean cosine to
//! nearest neighbours), not a SPLIT-level quality metric, so it cannot be
//! assumed to transfer without checking.
//!
//! This experiment computes two candidate certification signals — a
//! silhouette-style score (within-cluster cohesion minus between-cluster
//! separation) and a size-balance ratio — on the exact two splits Iteration
//! 7 already scored against ground truth, and checks whether either signal
//! actually ranks the known-good split above the known-bad one. If it
//! doesn't, "coherence gating" needs a different signal before it goes into
//! any architecture, not just an assumption that gating helps.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment9_certification_gate

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;

const CORPUS: &[(&str, &str, &str)] = &[
    ("dog", "The dog wagged its tail and waited by the door for its owner to come home.", "mammal"),
    ("cat", "The cat curled up on the windowsill and purred in the afternoon sun.", "mammal"),
    ("horse", "The horse trotted around the paddock, its mane flowing in the breeze.", "mammal"),
    ("sheep", "The sheep grazed quietly in the pasture, following the rest of the flock.", "mammal"),
    ("lion", "The lion stalked its prey across the savanna before launching a sudden charge.", "mammal"),
    ("wolf", "The wolf howled at dusk, calling the rest of its pack to the hunt.", "mammal"),
    ("bear", "The bear caught a salmon in its claws as the fish leapt upstream.", "mammal"),
    ("tiger", "The tiger prowled silently through the tall grass, stripes blending with the shadows.", "mammal"),
    ("eagle", "The eagle soared high above the canyon, scanning the ground for movement.", "bird"),
    ("sparrow", "The sparrow hopped along the branch before darting off between the leaves.", "bird"),
    ("owl", "The owl turned its head silently, watching for the faintest movement in the dark.", "bird"),
    ("swan", "The swan glided smoothly across the lake, barely rippling the water.", "bird"),
    ("penguin", "The penguin waddled across the ice before diving into the frigid water.", "bird"),
    ("ostrich", "The ostrich sprinted across the plain on powerful legs, kicking up dust.", "bird"),
    ("kiwi", "The kiwi foraged in the undergrowth at night, sniffing out insects with its long beak.", "bird"),
];

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

/// Mean silhouette-style score for a 2-way partition: for each item, cosine
/// similarity to its own cluster's centroid minus similarity to the other
/// cluster's centroid, averaged. Higher = more cleanly separated.
fn silhouette_like(embeddings: &[Vec<f32>], assignment: &[usize]) -> f32 {
    let dim = embeddings[0].len();
    let mut centroids = vec![vec![0.0f32; dim]; 2];
    let mut counts = [0usize; 2];
    for (i, e) in embeddings.iter().enumerate() {
        let c = assignment[i];
        counts[c] += 1;
        for d in 0..dim { centroids[c][d] += e[d]; }
    }
    for c in 0..2 {
        if counts[c] == 0 { continue; }
        let norm: f32 = centroids[c].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for d in 0..dim { centroids[c][d] /= norm; }
    }
    let mut total = 0.0;
    for (i, e) in embeddings.iter().enumerate() {
        let own = assignment[i];
        let other = 1 - own;
        total += cosine_sim(e, &centroids[own]) - cosine_sim(e, &centroids[other]);
    }
    total / embeddings.len() as f32
}

/// Size-balance ratio: min(cluster sizes) / max(cluster sizes). 1.0 = perfectly
/// balanced, near 0 = one cluster absorbed almost everything (exactly
/// Iteration 7's 13-vs-2 failure mode).
fn balance_ratio(assignment: &[usize]) -> f32 {
    let c0 = assignment.iter().filter(|&&a| a == 0).count();
    let c1 = assignment.iter().filter(|&&a| a == 1).count();
    let (lo, hi) = (c0.min(c1), c0.max(c1));
    if hi == 0 { 0.0 } else { lo as f32 / hi as f32 }
}

fn true_coarse(i: usize) -> &'static str { CORPUS[i].2 }

fn purity(assignment: &[usize]) -> f32 {
    let n = assignment.len();
    let mut correct = 0usize;
    for c in 0..2 {
        let mut counts = std::collections::HashMap::new();
        for i in 0..n {
            if assignment[i] == c { *counts.entry(true_coarse(i)).or_insert(0usize) += 1; }
        }
        correct += counts.values().copied().max().unwrap_or(0);
    }
    correct as f32 / n as f32
}

fn main() {
    println!("Experiment 9: does a certification signal distinguish a good split from a bad one?\n");

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

        let embeddings: Vec<Vec<f32>> = CORPUS.iter().map(|(_, s, _)| embedder.embed(s)).collect();

        // The BAD split: unconstrained k-means k=2 (Iteration 7's actual 13-vs-2 result).
        let bad = kmeans(&embeddings, 2, 25);

        // The GOOD split: true mammal/bird labels directly (Iteration 7's "oracle").
        let good: Vec<usize> = (0..CORPUS.len()).map(|i| if true_coarse(i) == "mammal" { 0 } else { 1 }).collect();

        for (name, assignment) in [("BAD (k-means depth-1, Iteration 7)", &bad), ("GOOD (true mammal/bird)", &good)] {
            let sizes = (assignment.iter().filter(|&&a| a == 0).count(), assignment.iter().filter(|&&a| a == 1).count());
            println!("=== {name} ===");
            println!("  sizes: {:?}", sizes);
            println!("  purity vs ground truth: {:.3}", purity(assignment));
            println!("  balance ratio (candidate certification signal 1): {:.3}", balance_ratio(assignment));
            println!("  silhouette-like score (candidate certification signal 2): {:.3}\n", silhouette_like(&embeddings, assignment));
        }

        let bad_balance = balance_ratio(&bad);
        let good_balance = balance_ratio(&good);
        let bad_sil = silhouette_like(&embeddings, &bad);
        let good_sil = silhouette_like(&embeddings, &good);

        println!("=== Verdict ===");
        println!(
            "balance ratio correctly ranks GOOD above BAD: {} ({:.3} vs {:.3})",
            good_balance > bad_balance, good_balance, bad_balance
        );
        println!(
            "silhouette-like score correctly ranks GOOD above BAD: {} ({:.3} vs {:.3})",
            good_sil > bad_sil, good_sil, bad_sil
        );
        println!(
            "\n(if a signal ranks GOOD above BAD here, it's a viable certification gate candidate to threshold on before recursing; if not, 'coherence gating' needs a different signal before being trusted in the architecture)"
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
