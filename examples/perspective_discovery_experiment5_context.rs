// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 5 — does perspective-framing context in the embedding INPUT
//! matter more than the discovery MECHANISM?
//!
//! Prompted by a specific claim: bare noun phrases ("engine", "steering
//! wheel") carry no relational context, so a static embedding of one has no
//! way to encode "this, considered mechanically" vs "this, considered
//! economically" — language is linear/sequential, a 1-3 word phrase commits
//! to nothing. If that's right, Iterations 1-2 never gave ANY method (naive
//! cosine, Δ-direction, k-means) a fair chance: the input embeddings
//! themselves may have been too impoverished to carry perspectival
//! structure, regardless of which geometric or learned mechanism was then
//! applied on top.
//!
//! Test: replace each of the 21 items with one natural sentence that
//! surrounds it with genuinely co-occurring, perspective-relevant
//! vocabulary — WITHOUT literally naming the perspective category anywhere
//! ("mechanical"/"driver"/"economic" do not appear in any sentence below;
//! that would be keyword leakage, a different and cheaper confound than the
//! one being tested). E.g. "engine" becomes "The engine converts fuel into
//! motion through repeated combustion cycles" — the co-occurring words
//! (fuel, motion, combustion) are what a real sentence naturally supplies
//! and a bare noun phrase cannot.
//!
//! Re-runs the EXACT SAME evaluation as Iterations 1-2 (naive NN F1,
//! Δ-direction F1, k-means purity, at both coarse and fine granularity) on
//! these sentence embeddings, so the numbers are directly comparable to the
//! already-published ones. If context changes which method wins, or lifts
//! the absolute numbers meaningfully, that's real evidence for the
//! "language needs context" claim, separable from anything about which
//! mechanism is best.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --example perspective_discovery_experiment5_context

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;
use serde::Serialize;

/// (bare term [for reference/printing only], contextual sentence, coarse, fine)
const CORPUS: &[(&str, &str, &str, &str)] = &[
    ("engine", "The engine converts fuel into motion through repeated combustion cycles.", "mechanical", "power-core"),
    ("piston", "Each piston slides within its cylinder, driven by the pressure of combustion.", "mechanical", "power-core"),
    ("camshaft", "The camshaft rotates in time with the crankshaft, opening and closing the valves.", "mechanical", "power-core"),
    ("torque", "Torque measures the twisting force the crankshaft delivers to the wheels.", "mechanical", "power-core"),
    ("transmission", "The transmission shifts gears to match engine speed with road speed.", "mechanical", "drivetrain-support"),
    ("fuel injection", "Fuel injection sprays a precise mist of gasoline into each cylinder.", "mechanical", "drivetrain-support"),
    ("cooling system", "The cooling system circulates coolant to keep the engine block from overheating.", "mechanical", "drivetrain-support"),
    ("steering wheel", "The driver turns the steering wheel to change the car's direction.", "driver", "active-controls"),
    ("brake pedal", "Pressing the brake pedal squeezes the pads against the rotors to slow down.", "driver", "active-controls"),
    ("accelerator pedal", "The accelerator pedal opens the throttle to increase speed.", "driver", "active-controls"),
    ("turn signal", "Flicking the turn signal alerts other drivers before changing lanes.", "driver", "active-controls"),
    ("rearview mirror", "Glancing at the rearview mirror shows traffic approaching from behind.", "driver", "safety-awareness"),
    ("seat belt", "Buckling the seat belt keeps the passenger secured during a sudden stop.", "driver", "safety-awareness"),
    ("dashboard display", "The dashboard display shows speed, fuel level, and warning lights.", "driver", "safety-awareness"),
    ("resale value", "A well-maintained car keeps a higher resale value after years of use.", "economic", "value-over-time"),
    ("depreciation rate", "New cars lose a large share of their value in the first year, a steep drop.", "economic", "value-over-time"),
    ("trade-in value", "The dealer offered a trade-in based on mileage and condition.", "economic", "value-over-time"),
    ("insurance premium", "Younger drivers usually pay a higher monthly amount for coverage.", "economic", "recurring-cost"),
    ("fuel cost", "Long commutes add up at the pump over a year.", "economic", "recurring-cost"),
    ("loan interest", "Financing a car means paying extra on top of the sticker price over time.", "economic", "recurring-cost"),
    ("purchase price", "The sticker price is negotiated before taxes and fees are added.", "economic", "recurring-cost"),
];

fn embed_all(embedder: &dyn VectorEmbed) -> Vec<Vec<f32>> {
    CORPUS.iter().map(|(_, sentence, ..)| embedder.embed(sentence)).collect()
}

fn coarse(i: usize) -> &'static str {
    CORPUS[i].2
}
fn fine(i: usize) -> &'static str {
    CORPUS[i].3
}

fn naive_nn_f1(embeddings: &[Vec<f32>], ref_idx: usize, label_of: impl Fn(usize) -> &'static str) -> Option<f32> {
    let true_label = label_of(ref_idx);
    let field_size = (0..embeddings.len()).filter(|&i| i != ref_idx && label_of(i) == true_label).count();
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
    Some(hits as f32 / field_size as f32)
}

fn delta_direction_f1(
    embeddings: &[Vec<f32>],
    ref_idx: usize,
    threshold: f32,
    label_of: impl Fn(usize) -> &'static str,
) -> Option<f32> {
    let true_label = label_of(ref_idx);
    let field_size = (0..embeddings.len()).filter(|&i| i != ref_idx && label_of(i) == true_label).count();
    if field_size == 0 {
        return None;
    }
    let r = &embeddings[ref_idx];
    let others: Vec<usize> = (0..embeddings.len()).filter(|&i| i != ref_idx).collect();
    let deltas: Vec<Vec<f32>> = others.iter().map(|&i| embeddings[i].iter().zip(r.iter()).map(|(a, b)| a - b).collect()).collect();
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
    Some(if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 })
}

fn kmeans(embeddings: &[Vec<f32>], k: usize, iterations: usize) -> Vec<usize> {
    let n = embeddings.len();
    let dim = embeddings[0].len();
    let mut centroid_idx = vec![0usize];
    while centroid_idx.len() < k {
        let next = (0..n)
            .max_by(|&a, &b| {
                let da = centroid_idx.iter().map(|&c| 1.0 - cosine_sim(&embeddings[a], &embeddings[c])).fold(f32::INFINITY, f32::min);
                let db = centroid_idx.iter().map(|&c| 1.0 - cosine_sim(&embeddings[b], &embeddings[c])).fold(f32::INFINITY, f32::min);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        centroid_idx.push(next);
    }
    let mut centroids: Vec<Vec<f32>> = centroid_idx.iter().map(|&i| embeddings[i].clone()).collect();
    let mut assignment = vec![0usize; n];
    for _ in 0..iterations {
        for i in 0..n {
            assignment[i] = (0..k)
                .max_by(|&a, &b| cosine_sim(&embeddings[i], &centroids[a]).partial_cmp(&cosine_sim(&embeddings[i], &centroids[b])).unwrap())
                .unwrap();
        }
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

#[derive(Serialize)]
struct Comparison {
    representation: String,
    mean_f1_naive_coarse: f32,
    mean_f1_naive_fine: f32,
    mean_f1_delta_coarse: f32,
    mean_f1_delta_fine: f32,
    kmeans_k3_purity: f32,
    kmeans_k6_purity: f32,
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
    let k3 = kmeans(embeddings, 3, 25);
    let k6 = kmeans(embeddings, 6, 25);
    Comparison {
        representation: label.to_string(),
        mean_f1_naive_coarse: nc_sum / nc_n as f32,
        mean_f1_naive_fine: nf_sum / nf_n as f32,
        mean_f1_delta_coarse: dc_sum / nc_n as f32,
        mean_f1_delta_fine: df_sum / nf_n as f32,
        kmeans_k3_purity: purity(&k3, 3, coarse),
        kmeans_k6_purity: purity(&k6, 6, fine),
    }
}

fn main() {
    println!("Experiment 5: perspective-framing CONTEXT vs. bare terms\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder};
        let mut embedder = None;
        for dir in ["models", "../models"] {
            let p = std::path::Path::new(dir);
            if !(p.join("model.onnx").exists() || p.join("onnx/model.onnx").exists()) {
                continue;
            }
            let cfg = OnnxConfig { dim: 384, model_dir: Some(dir.to_string()), ..OnnxConfig::default() };
            let e = OnnxEmbedder::with_config(&cfg);
            if e.is_available() {
                println!("Using real semantic embedder: {dir}/model.onnx (all-MiniLM-L6-v2, 384d)\n");
                embedder = Some(e);
                break;
            }
        }
        let embedder = match embedder {
            Some(e) => e,
            None => { println!("WARNING: no ONNX model found — aborting."); return; }
        };

        let bare_terms: Vec<Vec<f32>> = CORPUS.iter().map(|(term, ..)| embedder.embed(term)).collect();
        let sentences = embed_all(&embedder);

        let bare_result = evaluate(&bare_terms, "bare-term (Iteration 1-2 style)");
        let ctx_result = evaluate(&sentences, "contextual-sentence (Experiment 5)");

        println!(
            "{:<32} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "representation", "nv_C", "nv_F", "dl_C", "dl_F", "km3_pur", "km6_pur"
        );
        for r in [&bare_result, &ctx_result] {
            println!(
                "{:<32} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                r.representation, r.mean_f1_naive_coarse, r.mean_f1_naive_fine,
                r.mean_f1_delta_coarse, r.mean_f1_delta_fine, r.kmeans_k3_purity, r.kmeans_k6_purity
            );
        }

        println!("\n=== DELTA (contextual - bare) ===");
        println!("naive/coarse:  {:+.3}", ctx_result.mean_f1_naive_coarse - bare_result.mean_f1_naive_coarse);
        println!("naive/fine:    {:+.3}", ctx_result.mean_f1_naive_fine - bare_result.mean_f1_naive_fine);
        println!("delta/coarse:  {:+.3}", ctx_result.mean_f1_delta_coarse - bare_result.mean_f1_delta_coarse);
        println!("delta/fine:    {:+.3}", ctx_result.mean_f1_delta_fine - bare_result.mean_f1_delta_fine);
        println!("kmeans k3:     {:+.3}", ctx_result.kmeans_k3_purity - bare_result.kmeans_k3_purity);
        println!("kmeans k6:     {:+.3}", ctx_result.kmeans_k6_purity - bare_result.kmeans_k6_purity);

        // Sanity check: are the reused bare-term numbers actually reproducing
        // Iterations 1-2's published values, or has something drifted?
        println!("\n(sanity check against published Iteration 2 numbers: naive/coarse=0.587 naive/fine=0.468 delta/coarse=0.469 delta/fine=0.226 km3=0.476 km6=0.571)");

        let json = serde_json::to_string_pretty(&[&bare_result, &ctx_result]).unwrap();
        std::fs::write("experiment5_results.json", &json).expect("write results");
        println!("\nFull results written to experiment5_results.json");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
