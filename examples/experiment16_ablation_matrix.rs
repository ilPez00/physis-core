// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 16 — the mission's Section 18 formal ablation matrix,
//! finally run end-to-end: does each component (multi-membership,
//! certification, recursion) actually earn its complexity over the
//! baseline, using Iteration 13's FIXED local/split calibration rather
//! than Iteration 12's broken flat one? Direct test of Section 19's
//! fairness rule: "if raw cosine plus thresholding performs equally
//! well, say so."
//!
//! Same Vehicle ontology as Iterations 12-13 (3 coarse fields x 2 fine
//! subfields, 15 pure + 5 cross-cutting items). Six configurations,
//! collapsing the mission's 7 combinations to the ones that are
//! semantically distinct for this architecture (certification without
//! recursion is a no-op — there is nothing for it to gate):
//!
//!   1. flat (coarse-only, single-label, no MM, no recursion)
//!   2. flat + multi-membership (coarse-level dual membership only)
//!   3. + blind recursion (always recurse into fine sub-structure)
//!   4. + certified recursion (only recurse if balance-ratio gate passes)
//!   5. multi-membership + blind recursion
//!   6. FULL: multi-membership + certified recursion
//!
//! Coarse assignment is IDENTICAL across all 6 configs (real argmax
//! cosine against 3 known coarse centroids, not oracle) — isolating
//! exactly what each additional mechanism contributes on top of a fixed,
//! realistic starting point, which is the actual point of an ablation.
//!
//! Per-item "final accuracy": if a config does not recurse for an item
//! (no recursion at all, or the gate rejected that item's branch),
//! correctness is scored at the COARSE level; if it does recurse,
//! correctness is scored at the FINE level (stricter — a bad recursion
//! can only make an item's score worse than stopping at coarse would
//! have, which is exactly the risk/reward tradeoff certification exists
//! to manage). Averaged over all 20 items, run for both bare and
//! contextual representations (Section 15).
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment16_ablation_matrix

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;
use std::collections::HashMap;

fn kmeans(embeddings: &[Vec<f32>], k: usize, iterations: usize) -> Vec<usize> {
    let n = embeddings.len();
    if n < k { return (0..n).collect(); }
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
    for e in embeddings { for d in 0..dim { sum[d] += e[d]; } }
    let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    sum.iter().map(|x| x / norm).collect()
}

fn balance_ratio_k(assignment: &[usize], k: usize) -> f32 {
    let counts: Vec<usize> = (0..k).map(|c| assignment.iter().filter(|&&a| a == c).count()).collect();
    let lo = *counts.iter().min().unwrap_or(&0);
    let hi = *counts.iter().max().unwrap_or(&1);
    if hi == 0 { 0.0 } else { lo as f32 / hi as f32 }
}

const PURE: &[(&str, &str, &str, &str, &str)] = &[
    ("engine", "engine", "The engine converts fuel into motion through repeated combustion cycles.", "mechanical", "propulsion"),
    ("transmission", "transmission", "The transmission shifts gears to match engine speed with road speed.", "mechanical", "propulsion"),
    ("turbocharger", "turbocharger", "The turbocharger forces extra compressed air into the cylinders to boost power.", "mechanical", "propulsion"),
    ("fuel_tank", "fuel tank", "The fuel tank stores gasoline until it is pumped toward the engine.", "mechanical", "energy"),
    ("spark_plug", "spark plug", "The spark plug ignites the compressed air-fuel mixture inside the cylinder.", "mechanical", "energy"),
    ("steering_wheel", "steering wheel", "The driver turns the steering wheel to change the car's direction.", "driver", "controls"),
    ("accelerator_pedal", "accelerator pedal", "The accelerator pedal opens the throttle to increase speed.", "driver", "controls"),
    ("turn_signal", "turn signal", "Flicking the turn signal alerts other drivers before changing lanes.", "driver", "controls"),
    ("cruise_control", "cruise control", "Cruise control holds a steady speed on the highway without the driver's foot on the pedal.", "driver", "assistance"),
    ("blind_spot_monitor", "blind spot monitor", "A light flashes in the mirror when another car lingers just out of view.", "driver", "assistance"),
    ("insurance_premium", "insurance premium", "Younger drivers usually pay a higher monthly amount for coverage.", "economic", "ownership"),
    ("maintenance_cost", "maintenance cost", "Regular oil changes and tune-ups add up over the years of ownership.", "economic", "ownership"),
    ("resale_value", "resale value", "A well-maintained car keeps a higher resale value after years of use.", "economic", "ownership"),
    ("purchase_price", "purchase price", "The sticker price is negotiated before taxes and fees are added.", "economic", "acquisition"),
    ("trade_in_value", "trade-in value", "The dealer offered a trade-in based on mileage and condition.", "economic", "acquisition"),
];

const CROSS: &[(&str, &str, &str, &str, &str, &str)] = &[
    ("hybrid_battery", "hybrid battery", "The hybrid battery works alongside the engine to reduce gasoline use, though replacing it years later can be a costly repair.", "mechanical", "economic", "mechanical"),
    ("fuel_efficiency", "fuel efficiency", "Better fuel efficiency means burning less gasoline per mile, which adds up to real savings at the pump over time.", "mechanical", "economic", "mechanical"),
    ("driver_assistance_system", "driver assistance system", "Sensors built into the car watch the road and can nudge the wheel if it starts drifting out of its lane.", "mechanical", "driver", "driver"),
    ("regenerative_braking", "regenerative braking", "Easing off the pedal on the hybrid gently slows the car while feeding energy back into the battery instead of wasting it as heat.", "mechanical", "driver", "driver"),
    ("warranty_coverage", "warranty coverage", "A multi-year warranty means a sudden breakdown on the road won't leave the owner facing a large repair bill out of pocket.", "driver", "economic", "economic"),
];

const FINE_CATS: [&str; 6] = ["propulsion", "energy", "controls", "assistance", "ownership", "acquisition"];
fn fine_to_coarse(fine: &str) -> &'static str {
    match fine {
        "propulsion" | "energy" => "mechanical",
        "controls" | "assistance" => "driver",
        "ownership" | "acquisition" => "economic",
        _ => unreachable!(),
    }
}
fn sibling_of(fine: &str) -> &'static str {
    match fine {
        "propulsion" => "energy", "energy" => "propulsion",
        "controls" => "assistance", "assistance" => "controls",
        "ownership" => "acquisition", "acquisition" => "ownership",
        _ => unreachable!(),
    }
}
const COARSE_CATS: [&str; 3] = ["mechanical", "driver", "economic"];

struct ConfigResult {
    name: &'static str,
    final_accuracy: f32,
    coarse_mm_recall: Option<(usize, usize)>, // (hits, total) if multi-membership on
    coarse_mm_fpr: Option<f32>,
    fine_recursed_branches: usize,
}

fn run(representation: &str, embedder: &impl VectorEmbed) -> Vec<ConfigResult> {
    let n_pure = PURE.len();
    let n_cross = CROSS.len();
    let pure_emb: Vec<Vec<f32>> = PURE.iter().map(|(_, bare, ctx, ..)| embedder.embed(if representation == "bare" { bare } else { ctx })).collect();
    let cross_emb: Vec<Vec<f32>> = CROSS.iter().map(|(_, bare, ctx, ..)| embedder.embed(if representation == "bare" { bare } else { ctx })).collect();
    let mut all_emb = pure_emb.clone();
    all_emb.extend(cross_emb.iter().cloned());
    let n = n_pure + n_cross;

    let true_coarses = |i: usize| -> Vec<&'static str> {
        if i < n_pure { vec![PURE[i].3] } else { vec![CROSS[i - n_pure].3, CROSS[i - n_pure].4] }
    };
    let true_fines = |i: usize| -> Vec<&'static str> {
        if i < n_pure { vec![PURE[i].4] } else {
            let (_, _, _, f1, f2, _) = CROSS[i - n_pure];
            FINE_CATS.iter().filter(|c| fine_to_coarse(c) == f1 || fine_to_coarse(c) == f2).copied().collect()
        }
    };

    // ── Shared, real (non-oracle) coarse assignment: argmax cosine vs 3 known coarse centroids ──
    let coarse_centroids: HashMap<&'static str, Vec<f32>> = COARSE_CATS.iter().map(|&c| {
        let members: Vec<&Vec<f32>> = (0..n_pure).filter(|&i| PURE[i].3 == c).map(|i| &pure_emb[i]).collect();
        (c, centroid(&members))
    }).collect();
    let coarse_sims_for = |e: &Vec<f32>| -> HashMap<&'static str, f32> {
        COARSE_CATS.iter().map(|&c| (c, cosine_sim(e, &coarse_centroids[c]))).collect()
    };
    let all_coarse_sims: Vec<HashMap<&'static str, f32>> = all_emb.iter().map(coarse_sims_for).collect();
    let real_coarse_argmax: Vec<&'static str> = all_coarse_sims.iter().map(|s| *s.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0).collect();
    let coarse_correct: Vec<bool> = (0..n).map(|i| true_coarses(i).contains(&real_coarse_argmax[i])).collect();

    // ── Coarse-level multi-membership calibration (single delta, only 3 classes so no split needed) ──
    let coarse_membership_set = |sims: &HashMap<&'static str, f32>, delta: f32| -> Vec<&'static str> {
        let top = sims.values().cloned().fold(f32::NEG_INFINITY, f32::max);
        COARSE_CATS.iter().filter(|c| sims[*c] >= top - delta).copied().collect()
    };
    let coarse_fpr_at = |delta: f32| -> f32 {
        (0..n_pure).filter(|&i| coarse_membership_set(&all_coarse_sims[i], delta).len() > 1).count() as f32 / n_pure as f32
    };
    let coarse_recall_at = |delta: f32| -> usize {
        (n_pure..n).filter(|&i| {
            let ms = coarse_membership_set(&all_coarse_sims[i], delta);
            true_coarses(i).iter().all(|f| ms.contains(f))
        }).count()
    };
    let coarse_sweep: Vec<(f32, f32, usize)> = (0..=30).map(|i| { let d = i as f32 * 0.01; (d, coarse_fpr_at(d), coarse_recall_at(d)) }).collect();
    let max_recall = coarse_sweep.iter().map(|(_, _, r)| *r).max().unwrap();
    let (coarse_delta, coarse_fpr, coarse_recall) = *coarse_sweep.iter().find(|(_, _, r)| *r == max_recall).unwrap();

    // ── Fine-level local/split calibration (Iteration 13's fix), computed once, reused wherever recursion happens ──
    let fine_centroids: HashMap<&'static str, Vec<f32>> = FINE_CATS.iter().map(|&cat| {
        let members: Vec<&Vec<f32>> = (0..n_pure).filter(|&i| PURE[i].4 == cat).map(|i| &pure_emb[i]).collect();
        (cat, centroid(&members))
    }).collect();
    let fine_sims_for = |e: &Vec<f32>| -> HashMap<&'static str, f32> {
        FINE_CATS.iter().map(|&c| (c, cosine_sim(e, &fine_centroids[c]))).collect()
    };
    let all_fine_sims: Vec<HashMap<&'static str, f32>> = all_emb.iter().map(fine_sims_for).collect();
    let sibling_fpr_at = |delta: f32| -> f32 {
        (0..n_pure).filter(|&i| { let own = PURE[i].4; all_fine_sims[i][own] - all_fine_sims[i][sibling_of(own)] < delta }).count() as f32 / n_pure as f32
    };
    let sibling_delta = (0..=30).map(|i| i as f32 * 0.01).find(|&d| sibling_fpr_at(d) == 0.0).unwrap_or(0.30);
    let cross_branch_fpr_at = |delta: f32| -> f32 {
        (0..n_pure).filter(|&i| {
            let own = PURE[i].4;
            let own_sim = all_fine_sims[i][own];
            let other_max = FINE_CATS.iter().filter(|c| fine_to_coarse(c) != fine_to_coarse(own)).map(|c| all_fine_sims[i][*c]).fold(f32::NEG_INFINITY, f32::max);
            other_max >= own_sim - delta
        }).count() as f32 / n_pure as f32
    };
    let cross_branch_recall_at = |delta: f32| -> usize {
        (n_pure..n).filter(|&i| {
            let (_, _, _, f1, f2, _) = CROSS[i - n_pure];
            let (&top_cat, &top_val) = all_fine_sims[i].iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
            let top_coarse = fine_to_coarse(top_cat);
            let other_field = if top_coarse == f1 { f2 } else { f1 };
            FINE_CATS.iter().filter(|c| fine_to_coarse(c) == other_field).any(|c| all_fine_sims[i][*c] >= top_val - delta)
        }).count()
    };
    let cb_sweep: Vec<(f32, f32, usize)> = (0..=30).map(|i| { let d = i as f32 * 0.01; (d, cross_branch_fpr_at(d), cross_branch_recall_at(d)) }).collect();
    let cb_max_recall = cb_sweep.iter().map(|(_, _, r)| *r).max().unwrap();
    let (cross_branch_delta, ..) = *cb_sweep.iter().find(|(_, _, r)| *r == cb_max_recall).unwrap();

    let fine_membership_set = |i: usize| -> Vec<&'static str> {
        let sims = &all_fine_sims[i];
        let top = sims.values().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut ms: Vec<&'static str> = vec![];
        for &c in &FINE_CATS {
            let is_top = sims[c] == top;
            let is_sibling_of_top = FINE_CATS.iter().any(|t| sims[*t] == top && sibling_of(t) == c);
            let within_sibling = is_sibling_of_top && (top - sims[c] < sibling_delta);
            let within_cross_branch = !is_sibling_of_top && (top - sims[c] < cross_branch_delta);
            if is_top || within_sibling || within_cross_branch { ms.push(c); }
        }
        ms
    };

    // ── Recursion: for each REAL coarse group, k=2 fine sub-split + balance ratio ──
    struct Branch { coarse: &'static str, members: Vec<usize>, split: Vec<usize>, balance: f32, leaf_labels: [String; 2] }
    let mut branches = Vec::new();
    for &coarse in &COARSE_CATS {
        let members: Vec<usize> = (0..n).filter(|&i| real_coarse_argmax[i] == coarse).collect();
        if members.len() < 2 { continue; }
        let sub_emb: Vec<Vec<f32>> = members.iter().map(|&i| all_emb[i].clone()).collect();
        let split = kmeans(&sub_emb, 2, 30);
        let balance = balance_ratio_k(&split, 2);
        let mut leaf_labels = [String::from("?"), String::from("?")];
        for (c, slot) in leaf_labels.iter_mut().enumerate() {
            let mut counts: HashMap<&str, usize> = HashMap::new();
            for (li, &gi) in members.iter().enumerate() { if split[li] == c && gi < n_pure { *counts.entry(PURE[gi].4).or_insert(0) += 1; } }
            if let Some((label, _)) = counts.into_iter().max_by_key(|(_, c)| *c) { *slot = label.to_string(); }
        }
        branches.push(Branch { coarse, members, split, balance, leaf_labels });
    }
    let fine_recursed_branches_blind = branches.len();
    let fine_recursed_branches_certified = branches.iter().filter(|b| b.balance >= 0.5).count();

    let fine_correct_for_item = |i: usize| -> bool {
        for b in &branches {
            if let Some(li) = b.members.iter().position(|&m| m == i) {
                let leaf_label = &b.leaf_labels[b.split[li]];
                return true_fines(i).iter().any(|f| f == leaf_label);
            }
        }
        false
    };
    let branch_certified = |coarse: &str| -> bool {
        branches.iter().find(|b| b.coarse == coarse).map(|b| b.balance >= 0.5).unwrap_or(false)
    };

    let mut results = Vec::new();

    // Config 1: flat, coarse-only
    let acc1 = (0..n).filter(|&i| coarse_correct[i]).count() as f32 / n as f32;
    results.push(ConfigResult { name: "1: flat (coarse-only, single-label)", final_accuracy: acc1, coarse_mm_recall: None, coarse_mm_fpr: None, fine_recursed_branches: 0 });

    // Config 2: flat + coarse multi-membership (accuracy unaffected — MM only adds a second label, doesn't change primary correctness; report MM stats separately)
    results.push(ConfigResult { name: "2: flat + multi-membership (coarse)", final_accuracy: acc1, coarse_mm_recall: Some((coarse_recall, n_cross)), coarse_mm_fpr: Some(coarse_fpr), fine_recursed_branches: 0 });

    // Config 3: + blind recursion (always trust fine split, no MM)
    let acc3 = (0..n).filter(|&i| fine_correct_for_item(i)).count() as f32 / n as f32;
    results.push(ConfigResult { name: "3: + blind recursion (no gate, no MM)", final_accuracy: acc3, coarse_mm_recall: None, coarse_mm_fpr: None, fine_recursed_branches: fine_recursed_branches_blind });

    // Config 4: + certified recursion (gate blocks bad branches -> those items scored at coarse level instead)
    let acc4 = (0..n).filter(|&i| {
        if branch_certified(real_coarse_argmax[i]) { fine_correct_for_item(i) } else { coarse_correct[i] }
    }).count() as f32 / n as f32;
    results.push(ConfigResult { name: "4: + certified recursion (gate, no MM)", final_accuracy: acc4, coarse_mm_recall: None, coarse_mm_fpr: None, fine_recursed_branches: fine_recursed_branches_certified });

    // Config 5: multi-membership + blind recursion (fine correctness now also checks fine MM recall as a secondary stat)
    results.push(ConfigResult { name: "5: multi-membership + blind recursion", final_accuracy: acc3, coarse_mm_recall: Some((coarse_recall, n_cross)), coarse_mm_fpr: Some(coarse_fpr), fine_recursed_branches: fine_recursed_branches_blind });

    // Config 6: FULL — multi-membership + certified recursion
    results.push(ConfigResult { name: "6: FULL (multi-membership + certified recursion)", final_accuracy: acc4, coarse_mm_recall: Some((coarse_recall, n_cross)), coarse_mm_fpr: Some(coarse_fpr), fine_recursed_branches: fine_recursed_branches_certified });

    // Fine-level MM recall/FPR (using Iteration 13's local/split calibration), reported once for configs 5/6.
    let fine_mm_recall = (n_pure..n).filter(|&i| {
        let ms = fine_membership_set(i);
        let coarses: std::collections::HashSet<&str> = ms.iter().map(|c| fine_to_coarse(c)).collect();
        true_coarses(i).iter().all(|f| coarses.contains(f))
    }).count();
    let fine_mm_fpr = (0..n_pure).filter(|&i| fine_membership_set(i).len() > 1).count() as f32 / n_pure as f32;

    println!("\n--- {representation} ---");
    println!("{:<50} {:>10} {:>16} {:>10}", "config", "final-acc", "coarse-MM r/fpr", "#branches recursed");
    for r in &results {
        let mm = match (r.coarse_mm_recall, r.coarse_mm_fpr) {
            (Some((h, t)), Some(fpr)) => format!("{h}/{t} @ fpr={fpr:.2}"),
            _ => "-".to_string(),
        };
        println!("{:<50} {:>10.3} {:>16} {:>10}", r.name, r.final_accuracy, mm, r.fine_recursed_branches);
    }
    println!("(fine-level MM, Iteration 13 local/split calibration, configs 5-6): recall={fine_mm_recall}/{n_cross}, FPR={fine_mm_fpr:.3} (sibling_delta={sibling_delta:.2}, cross_branch_delta={cross_branch_delta:.2})");
    println!("(coarse-level MM calibrated delta: {coarse_delta:.2})");
    println!("real coarse argmax assignment: {}/{} items correct (best-case credit for cross items)", (0..n).filter(|&i| coarse_correct[i]).count(), n);

    results
}

fn main() {
    println!("Experiment 16: formal ablation matrix (Section 18) — does each component earn its complexity?\n");

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

        let bare_results = run("bare", &embedder);
        let ctx_results = run("contextual", &embedder);

        println!("\n=== Section 19 fairness verdict ===");
        for (label, results) in [("bare", &bare_results), ("contextual", &ctx_results)] {
            let flat = results[0].final_accuracy;
            let full = results[5].final_accuracy;
            println!(
                "{label}: flat baseline={flat:.3}, FULL system={full:.3}, delta={:+.3} — {}",
                full - flat,
                if full > flat { "FULL system beats the flat baseline" }
                else if full == flat { "FULL system performs EXACTLY the same as the flat baseline — the added machinery earned nothing on final accuracy here" }
                else { "FULL system performs WORSE than the flat baseline — recursion+certification actively hurt on this run" }
            );
        }
        println!("\n(multi-membership's value is NOT visible in final_accuracy by design — it adds a second label without changing which single label is scored as primary. Its value is the separately-reported cross-link recall/FPR, i.e. whether it captures information hard clustering structurally cannot.)");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
