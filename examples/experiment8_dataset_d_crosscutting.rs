// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 8 — mission Dataset D: cross-cutting concepts.
//!
//! Every dataset used so far (B: perspectives, A: taxonomy) assumed each
//! item belongs to EXACTLY ONE coarse field — the one structural assumption
//! that happens to be exactly what hard clustering (k-means) is built for,
//! and exactly what would make a global-clustering baseline look
//! artificially strong relative to per-reference methods. Dataset D tests
//! the case the mission specifically calls out (Section 5): concepts that
//! genuinely belong to MULTIPLE fields ("fuel: CAR, ENGINE, ECONOMICS,
//! ENVIRONMENT"). This is the one place per-reference retrieval has a real
//! STRUCTURAL advantage over hard clustering — a k-means assignment is
//! mechanically incapable of giving an item two labels; a per-reference
//! similarity score is not. The question is whether the embedding actually
//! carries that dual signal in practice, not just in principle.
//!
//! Corpus: the 21-item CAR corpus from Iterations 1/2/5 (contextual
//! sentences, one true coarse field each: mechanical/driver/economic) plus
//! 4 NEW items deliberately written to genuinely span two fields each
//! (fuel efficiency: mechanical+economic; hybrid battery: mechanical+
//! economic; driver assistance system: mechanical+driver; warranty
//! coverage: driver+economic) — no field-name keywords leaked into any
//! sentence, same discipline as Iteration 5.
//!
//! Two tests:
//!   1. Hard clustering (k-means, k=3, all 25 items): each cross-cutting
//!      item gets exactly one cluster. Report which, and whether it matches
//!      EITHER true label (best-case credit) — demonstrates the structural
//!      ceiling empirically, not just by definition.
//!   2. Per-reference dual-signal test: compute each cross-cutting item's
//!      cosine similarity to the CENTROID of each of the 3 pure fields
//!      (computed from the 21 single-labeled items only). "Dual margin" =
//!      min(sim to its two true fields) minus (sim to its one false field).
//!      Positive means the embedding places the item closer to BOTH its
//!      true fields than to the unrelated one — real dual-membership
//!      signal a hard assignment cannot express by construction.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment8_dataset_d_crosscutting

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;

/// The 21 single-field CAR items (Iteration 5's contextual sentences).
const PURE: &[(&str, &str, &str)] = &[
    ("engine", "The engine converts fuel into motion through repeated combustion cycles.", "mechanical"),
    ("piston", "Each piston slides within its cylinder, driven by the pressure of combustion.", "mechanical"),
    ("camshaft", "The camshaft rotates in time with the crankshaft, opening and closing the valves.", "mechanical"),
    ("torque", "Torque measures the twisting force the crankshaft delivers to the wheels.", "mechanical"),
    ("transmission", "The transmission shifts gears to match engine speed with road speed.", "mechanical"),
    ("fuel injection", "Fuel injection sprays a precise mist of gasoline into each cylinder.", "mechanical"),
    ("cooling system", "The cooling system circulates coolant to keep the engine block from overheating.", "mechanical"),
    ("steering wheel", "The driver turns the steering wheel to change the car's direction.", "driver"),
    ("brake pedal", "Pressing the brake pedal squeezes the pads against the rotors to slow down.", "driver"),
    ("accelerator pedal", "The accelerator pedal opens the throttle to increase speed.", "driver"),
    ("turn signal", "Flicking the turn signal alerts other drivers before changing lanes.", "driver"),
    ("rearview mirror", "Glancing at the rearview mirror shows traffic approaching from behind.", "driver"),
    ("seat belt", "Buckling the seat belt keeps the passenger secured during a sudden stop.", "driver"),
    ("dashboard display", "The dashboard display shows speed, fuel level, and warning lights.", "driver"),
    ("resale value", "A well-maintained car keeps a higher resale value after years of use.", "economic"),
    ("depreciation rate", "New cars lose a large share of their value in the first year, a steep drop.", "economic"),
    ("trade-in value", "The dealer offered a trade-in based on mileage and condition.", "economic"),
    ("insurance premium", "Younger drivers usually pay a higher monthly amount for coverage.", "economic"),
    ("fuel cost", "Long commutes add up at the pump over a year.", "economic"),
    ("loan interest", "Financing a car means paying extra on top of the sticker price over time.", "economic"),
    ("purchase price", "The sticker price is negotiated before taxes and fees are added.", "economic"),
];

/// 4 cross-cutting items, each genuinely spanning two fields, with no
/// field-name keyword leakage.
const CROSSCUTTING: &[(&str, &str, &str, &str)] = &[
    (
        "fuel efficiency",
        "Better fuel efficiency means the engine burns less gasoline per mile, which adds up to real savings at the pump.",
        "mechanical",
        "economic",
    ),
    (
        "hybrid battery",
        "The hybrid battery works alongside the engine to cut down on gasoline use, though replacing it years later can be an expensive repair.",
        "mechanical",
        "economic",
    ),
    (
        "driver assistance system",
        "Sensors built into the car watch the road and can nudge the wheel if it starts drifting out of the lane.",
        "mechanical",
        "driver",
    ),
    (
        "warranty coverage",
        "A multi-year warranty means a sudden breakdown won't leave the owner facing a large repair bill out of pocket.",
        "driver",
        "economic",
    ),
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

fn centroid(embeddings: &[&Vec<f32>]) -> Vec<f32> {
    let dim = embeddings[0].len();
    let mut sum = vec![0.0f32; dim];
    for e in embeddings {
        for d in 0..dim { sum[d] += e[d]; }
    }
    let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    sum.iter().map(|x| x / norm).collect()
}

fn main() {
    println!("Experiment 8: Dataset D (cross-cutting concepts)\n");

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

        let pure_embeddings: Vec<Vec<f32>> = PURE.iter().map(|(_, s, _)| embedder.embed(s)).collect();
        let cross_embeddings: Vec<Vec<f32>> = CROSSCUTTING.iter().map(|(_, s, ..)| embedder.embed(s)).collect();

        // ── Test 1: hard clustering, all 25 items together ──
        println!("=== Test 1: hard k-means (k=3) clustering, all 25 items ===");
        let mut all_embeddings = pure_embeddings.clone();
        all_embeddings.extend(cross_embeddings.iter().cloned());
        let assignment = kmeans(&all_embeddings, 3, 30);

        // Name each cluster by its majority true label among the 21 pure items.
        let mut cluster_label = vec!["?"; 3];
        for (c, slot) in cluster_label.iter_mut().enumerate() {
            let mut counts = std::collections::HashMap::new();
            for i in 0..PURE.len() {
                if assignment[i] == c { *counts.entry(PURE[i].2).or_insert(0usize) += 1; }
            }
            if let Some((label, _)) = counts.into_iter().max_by_key(|(_, n)| *n) {
                *slot = label;
            }
        }
        println!("cluster -> majority true label among pure items: {cluster_label:?}\n");

        let mut best_case_hits = 0;
        for (idx, (name, _, l1, l2)) in CROSSCUTTING.iter().enumerate() {
            let global_idx = PURE.len() + idx;
            let assigned_cluster = assignment[global_idx];
            let assigned_label = cluster_label[assigned_cluster];
            let matches_either = assigned_label == *l1 || assigned_label == *l2;
            if matches_either { best_case_hits += 1; }
            println!(
                "  {name:<28} true=({l1}, {l2})  hard-assigned to cluster labeled '{assigned_label}'  (captures ONE of its TWO true fields: {matches_either}; the other is structurally lost)"
            );
        }
        println!(
            "\nHard clustering best-case single-label credit: {}/{} — by construction, it can NEVER represent both true fields for any item, regardless of embedding quality.\n",
            best_case_hits, CROSSCUTTING.len()
        );

        // ── Test 2: per-field-centroid dual-signal test ──
        println!("=== Test 2: does the embedding itself carry dual-membership signal? ===");
        let mechanical: Vec<&Vec<f32>> = (0..PURE.len()).filter(|&i| PURE[i].2 == "mechanical").map(|i| &pure_embeddings[i]).collect();
        let driver: Vec<&Vec<f32>> = (0..PURE.len()).filter(|&i| PURE[i].2 == "driver").map(|i| &pure_embeddings[i]).collect();
        let economic: Vec<&Vec<f32>> = (0..PURE.len()).filter(|&i| PURE[i].2 == "economic").map(|i| &pure_embeddings[i]).collect();
        let centroids: std::collections::HashMap<&str, Vec<f32>> = [
            ("mechanical", centroid(&mechanical)),
            ("driver", centroid(&driver)),
            ("economic", centroid(&economic)),
        ].into_iter().collect();
        let all_fields = ["mechanical", "driver", "economic"];

        let mut positive_margins = 0;
        for (idx, (name, _, l1, l2)) in CROSSCUTTING.iter().enumerate() {
            let e = &cross_embeddings[idx];
            let sim1 = cosine_sim(e, &centroids[l1]);
            let sim2 = cosine_sim(e, &centroids[l2]);
            let other_field = all_fields.iter().find(|f| *f != l1 && *f != l2).unwrap();
            let sim_other = cosine_sim(e, &centroids[other_field]);
            let margin = sim1.min(sim2) - sim_other;
            if margin > 0.0 { positive_margins += 1; }
            println!(
                "  {name:<28} sim({l1})={sim1:.3}  sim({l2})={sim2:.3}  sim({other_field}, unrelated)={sim_other:.3}  dual_margin={margin:+.3}"
            );
        }
        println!(
            "\nItems with positive dual margin (closer to BOTH true fields than the unrelated one): {}/{}",
            positive_margins, CROSSCUTTING.len()
        );
        println!("(positive margin = real dual-membership signal the embedding carries on its own; a hard clustering assignment could never express this even when it IS present in the embedding)");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
