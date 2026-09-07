// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 7 — mission Steps 8-9: recursive exploration, and stress-testing it.
//!
//! Every prior iteration tested a single, flat level of discovery. This
//! tests whether RECURSING actually recovers nested structure — the
//! mission's original hypothesis (`scan(A) -> D,E,F -> scan(D) -> ...`).
//!
//! Built on k-means, not Δ-direction: Iterations 1-4 falsified Δ-direction
//! (and the trained predictor) as improvements over baseline at every scale
//! tested. Recursing a mechanism already shown not to beat its own baseline
//! would not produce a winning result — recursing the one mechanism that
//! DID show real (if inconsistent) signal, plain k-means clustering, is the
//! only choice that could honestly test the recursive hypothesis rather
//! than just re-confirm the earlier negative result one level deeper.
//!
//! Dataset A (animal taxonomy, Experiment 6) is used because it has a real,
//! independently-verified two-level ground truth (Mammal/Bird at depth 1,
//! Domestic/Predator/Flying/Flightless at depth 2) — unlike Dataset B's
//! perspectives, which are not truly hierarchical.
//!
//! Two pipelines are measured separately, on purpose:
//!   - REAL: recurse into whatever k-means actually discovered at depth 1
//!     (errors from depth 1 propagate into depth 2 — this is what a real
//!     recursive system would experience).
//!   - ORACLE: recurse into the TRUE depth-1 groups instead of the
//!     discovered ones — isolates whether recursion itself adds value,
//!     independent of how good the top-level split was.
//!
//! Then a stress test (mission Section 20, "branch explosion" /
//! "generic-node collapse"): recurse ONE level past where real structure
//! ends (splitting each already-fine 3-4 item leaf group again) and check
//! whether the result is stable across bare-term vs contextual embeddings —
//! instability here would mean recursion doesn't know when to stop and is
//! fitting noise, not signal, past the point where real structure runs out.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment7_recursion

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;

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

fn kmeans(embeddings: &[Vec<f32>], k: usize, iterations: usize) -> Vec<usize> {
    let n = embeddings.len();
    if n < k {
        return (0..n).collect();
    }
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

fn purity_of(members: &[usize], assignment: &[usize], k: usize, label_of: impl Fn(usize) -> &'static str) -> f32 {
    let n = members.len();
    if n == 0 { return 0.0; }
    let mut correct = 0usize;
    for c in 0..k {
        let mut counts = std::collections::HashMap::new();
        for (local_i, &global_i) in members.iter().enumerate() {
            if assignment[local_i] == c { *counts.entry(label_of(global_i)).or_insert(0usize) += 1; }
        }
        correct += counts.values().copied().max().unwrap_or(0);
    }
    correct as f32 / n as f32
}

fn coarse(i: usize) -> &'static str { CORPUS[i].2 }
fn fine(i: usize) -> &'static str { CORPUS[i].3 }

/// Jaccard similarity between two 2-way partitions of the same item set —
/// used as the stability metric for the depth-3 stress test (no ground
/// truth exists there, so stability across representations is the only
/// available signal).
fn partition_agreement(a: &[usize], b: &[usize]) -> f32 {
    // Best matching under the 2 possible label permutations for k=2.
    let n = a.len();
    let direct = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    let flipped = a.iter().zip(b.iter()).filter(|(x, y)| **x == 1 - **y).count();
    direct.max(flipped) as f32 / n as f32
}

fn run_pipeline(embeddings: &[Vec<f32>], label: &str) {
    println!("--- {label} ---");

    // Depth 1: coarse split (k=2), against true mammal/bird.
    let depth1 = kmeans(embeddings, 2, 25);
    let depth1_purity = purity_of(&(0..CORPUS.len()).collect::<Vec<_>>(), &depth1, 2, coarse);
    println!("depth-1 (k=2) purity vs mammal/bird: {depth1_purity:.3}");

    // REAL pipeline: recurse into whatever depth-1 actually found.
    println!("REAL pipeline (recurse into discovered depth-1 clusters):");
    for c in 0..2 {
        let members: Vec<usize> = (0..CORPUS.len()).filter(|&i| depth1[i] == c).collect();
        if members.len() < 2 {
            println!("  cluster {c}: only {} member(s), cannot recurse", members.len());
            continue;
        }
        let sub_embeddings: Vec<Vec<f32>> = members.iter().map(|&i| embeddings[i].clone()).collect();
        let depth2 = kmeans(&sub_embeddings, 2, 25);
        let p = purity_of(&members, &depth2, 2, fine);
        let majority_coarse = {
            let mut counts = std::collections::HashMap::new();
            for &i in &members { *counts.entry(coarse(i)).or_insert(0usize) += 1; }
            counts.into_iter().max_by_key(|(_, c)| *c).unwrap().0
        };
        println!(
            "  cluster {c} (n={}, majority true coarse label={majority_coarse}): depth-2 purity vs fine label = {p:.3}",
            members.len()
        );
    }

    // ORACLE pipeline: recurse into the TRUE mammal/bird groups instead.
    println!("ORACLE pipeline (recurse into TRUE mammal/bird groups):");
    for true_coarse in ["mammal", "bird"] {
        let members: Vec<usize> = (0..CORPUS.len()).filter(|&i| coarse(i) == true_coarse).collect();
        let sub_embeddings: Vec<Vec<f32>> = members.iter().map(|&i| embeddings[i].clone()).collect();
        let depth2 = kmeans(&sub_embeddings, 2, 25);
        let p = purity_of(&members, &depth2, 2, fine);
        println!("  {true_coarse} (n={}): depth-2 purity vs fine label = {p:.3}", members.len());
    }

    // STRESS TEST: recurse ONE level past where real structure ends. Take
    // the "domestic" leaf group (4 items: dog, cat, horse, sheep) - there is
    // no known sub-structure here - and force a k=2 split. Compare the split
    // obtained from bare-term vs contextual embeddings of JUST those 4
    // items: if recursion has any business continuing here, the split
    // should be at least stable/motivated; if it's fitting noise, the two
    // representations will likely disagree on how to split a set with no
    // real substructure.
    println!("STRESS TEST (recurse past real structure — 'domestic' leaf group, no known sub-labels):");
}

fn main() {
    println!("Experiment 7: recursive discovery (mission Steps 8-9)\n");

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

        run_pipeline(&bare, "bare-term");
        println!();
        run_pipeline(&ctx, "contextual-sentence");

        // Stress test detail, run once with both representations for the
        // "domestic" leaf group specifically.
        let domestic: Vec<usize> = (0..CORPUS.len()).filter(|&i| fine(i) == "domestic").collect();
        let bare_sub: Vec<Vec<f32>> = domestic.iter().map(|&i| bare[i].clone()).collect();
        let ctx_sub: Vec<Vec<f32>> = domestic.iter().map(|&i| ctx[i].clone()).collect();
        let split_bare = kmeans(&bare_sub, 2, 25);
        let split_ctx = kmeans(&ctx_sub, 2, 25);
        let names: Vec<&str> = domestic.iter().map(|&i| CORPUS[i].0).collect();
        println!("\n=== Stress test: forcing a k=2 split of the 'domestic' leaf group (dog, cat, horse, sheep) ===");
        println!("items: {names:?}");
        println!("bare-term split:      {split_bare:?}");
        println!("contextual split:     {split_ctx:?}");
        println!(
            "agreement between the two representations' forced splits: {:.3} (1.0 = identical split, 0.5 = no better than chance for n=4)",
            partition_agreement(&split_bare, &split_ctx)
        );
        println!("(no ground truth exists for this split by construction — agreement/disagreement is the only available signal, and low agreement here would mean recursion is inventing structure past where real signal ends)");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
