// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 19 — Roadmap item 15 (user-proposed): use a SECOND,
//! architecturally different embedder not to pick a winner (Iteration
//! 18 already showed BGE and MiniLM land in the same rough range), but
//! to CORROBORATE specific multi-membership candidates independently.
//!
//! Refined from the plan's blunter "do top-1 picks disagree" framing to
//! a more precise test: for every candidate secondary category MiniLM's
//! margin-based detector flags for an item, check whether BGE's OWN,
//! independently-computed similarity ranking for that item ALSO ranks
//! that specific category as a plausible candidate. A false positive
//! driven by one architecture's idiosyncrasy is unlikely to be
//! independently reproduced by a differently-trained model; a genuine
//! cross-cutting concept should show up as a plausible candidate in
//! more than one reasonable embedding space.
//!
//! Part A (rigorous, ground-truth-based): the Vehicle ontology from
//! Iterations 12-13, where true multi-membership IS known. Measures
//! whether requiring BGE corroboration reduces Iteration 13's already
//! locally-calibrated false-positive rate (6.7%/13.3%) without losing
//! its full 5/5 recall.
//!
//! Part B (descriptive, no ground truth exists): the real 730-entry
//! ontology. Reports how many candidate flags survive corroboration,
//! plus specific example text so the reduction can be sanity-checked by
//! reading, not just trusted as a number — this track's standing
//! discipline.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment19_cross_embedder_corroboration

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use std::collections::HashMap;
#[cfg(feature = "embed-onnx")]
use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

fn centroid(embeddings: &[&Vec<f32>]) -> Vec<f32> {
    let dim = embeddings[0].len();
    let mut sum = vec![0.0f32; dim];
    for e in embeddings { for d in 0..dim { sum[d] += e[d]; } }
    let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    sum.iter().map(|x| x / norm).collect()
}

// ═══════════════════════════════ Part A: Vehicle ontology (ground truth known) ═══════════════════════════════

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

/// Iteration 13's local/split calibration: sibling-confusion and
/// cross-branch checks calibrated separately on ONE embedder's data.
/// Returns, for every item, the set of flagged fine categories (its own
/// top plus anything within the appropriate calibrated tolerance).
fn local_split_membership_sets(all_emb: &[Vec<f32>], n_pure: usize) -> Vec<Vec<&'static str>> {
    let fine_centroids: HashMap<&'static str, Vec<f32>> = FINE_CATS.iter().map(|&cat| {
        let members: Vec<&Vec<f32>> = (0..n_pure).filter(|&i| PURE[i].4 == cat).map(|i| &all_emb[i]).collect();
        (cat, centroid(&members))
    }).collect();
    let sims_for = |e: &Vec<f32>| -> HashMap<&'static str, f32> {
        FINE_CATS.iter().map(|&c| (c, cosine_sim(e, &fine_centroids[c]))).collect()
    };
    let all_sims: Vec<HashMap<&'static str, f32>> = all_emb.iter().map(sims_for).collect();

    let sibling_fpr_at = |delta: f32| -> f32 {
        (0..n_pure).filter(|&i| { let own = PURE[i].4; all_sims[i][own] - all_sims[i][sibling_of(own)] < delta }).count() as f32 / n_pure as f32
    };
    let sibling_delta = (0..=30).map(|i| i as f32 * 0.01).find(|&d| sibling_fpr_at(d) == 0.0).unwrap_or(0.30);

    let cross_branch_fpr_at = |delta: f32| -> f32 {
        (0..n_pure).filter(|&i| {
            let own = PURE[i].4;
            let own_sim = all_sims[i][own];
            let other_max = FINE_CATS.iter().filter(|c| fine_to_coarse(c) != fine_to_coarse(own)).map(|c| all_sims[i][*c]).fold(f32::NEG_INFINITY, f32::max);
            other_max >= own_sim - delta
        }).count() as f32 / n_pure as f32
    };
    let cross_branch_recall_at = |delta: f32| -> usize {
        (n_pure..all_emb.len()).filter(|&i| {
            let (_, _, _, f1, f2, _) = CROSS[i - n_pure];
            let (&top_cat, &top_val) = all_sims[i].iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
            let top_coarse = fine_to_coarse(top_cat);
            let other_field = if top_coarse == f1 { f2 } else { f1 };
            FINE_CATS.iter().filter(|c| fine_to_coarse(c) == other_field).any(|c| all_sims[i][*c] >= top_val - delta)
        }).count()
    };
    let cb_sweep: Vec<(f32, usize)> = (0..=30).map(|i| { let d = i as f32 * 0.01; (d, cross_branch_recall_at(d)) }).collect();
    let cb_max_recall = cb_sweep.iter().map(|(_, r)| *r).max().unwrap();
    let cross_branch_delta = cb_sweep.iter().find(|(_, r)| *r == cb_max_recall).unwrap().0;
    let _ = cross_branch_fpr_at; // used for reporting only, kept for parity with Iteration 13

    (0..all_emb.len()).map(|i| {
        let sims = &all_sims[i];
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
    }).collect()
}

#[cfg(feature = "embed-onnx")]
fn run_vehicle_test(minilm: &OnnxEmbedder, bge: &OnnxEmbedder, representation: &str) {
    let n_pure = PURE.len();
    let n_cross = CROSS.len();
    let text_of = |i: usize| -> &str {
        if i < n_pure { if representation == "bare" { PURE[i].1 } else { PURE[i].2 } }
        else { if representation == "bare" { CROSS[i - n_pure].1 } else { CROSS[i - n_pure].2 } }
    };
    let n = n_pure + n_cross;

    let minilm_emb: Vec<Vec<f32>> = (0..n).map(|i| minilm.embed(text_of(i))).collect();
    let bge_emb: Vec<Vec<f32>> = (0..n).map(|i| bge.embed(text_of(i))).collect();

    let minilm_sets = local_split_membership_sets(&minilm_emb, n_pure);
    let bge_sets = local_split_membership_sets(&bge_emb, n_pure);

    // Baseline: MiniLM alone (Iteration 13's local/split calibration).
    let baseline_fp = (0..n_pure).filter(|&i| minilm_sets[i].len() > 1).count();
    let baseline_recall = (n_pure..n).filter(|&i| {
        let (_, _, _, f1, f2, _) = CROSS[i - n_pure];
        let coarses: std::collections::HashSet<&str> = minilm_sets[i].iter().map(|c| fine_to_coarse(c)).collect();
        coarses.contains(f1) && coarses.contains(f2)
    }).count();

    // Corroborated: keep a flagged secondary category only if BGE's OWN
    // independent set for that item ALSO contains it.
    let corroborated_sets: Vec<Vec<&str>> = (0..n).map(|i| {
        let own_top = { // recompute own top fine cat for this item under MiniLM for reference labeling
            minilm_sets[i][0]
        };
        minilm_sets[i].iter().filter(|&&c| c == own_top || bge_sets[i].contains(&c)).copied().collect()
    }).collect();
    let corroborated_fp = (0..n_pure).filter(|&i| corroborated_sets[i].len() > 1).count();
    let corroborated_recall = (n_pure..n).filter(|&i| {
        let (_, _, _, f1, f2, _) = CROSS[i - n_pure];
        let coarses: std::collections::HashSet<&str> = corroborated_sets[i].iter().map(|c| fine_to_coarse(c)).collect();
        coarses.contains(f1) && coarses.contains(f2)
    }).count();

    println!("\n--- {representation} ---");
    println!("MiniLM alone (Iteration 13 local/split calibration): FPR={:.3} ({}/{}), recall={}/{}", baseline_fp as f32 / n_pure as f32, baseline_fp, n_pure, baseline_recall, n_cross);
    println!("+ BGE corroboration filter:                          FPR={:.3} ({}/{}), recall={}/{}", corroborated_fp as f32 / n_pure as f32, corroborated_fp, n_pure, corroborated_recall, n_cross);

    if baseline_fp > 0 {
        println!("pure items still falsely flagged after corroboration:");
        for i in 0..n_pure {
            if corroborated_sets[i].len() > 1 {
                println!("  {:<20} MiniLM-alone={:?}  after-corroboration={:?}", PURE[i].0, minilm_sets[i], corroborated_sets[i]);
            }
        }
    }
}

// ═══════════════════════════════ Part B: real 730-entry ontology (descriptive, no ground truth) ═══════════════════════════════

#[cfg(feature = "embed-onnx")]
fn run_real_ontology_test(minilm: &OnnxEmbedder, bge: &OnnxEmbedder) {
    println!("\n\n########## Part B: real 730-entry ontology (descriptive — no ground truth exists) ##########");
    let ontology = OntologyLoader::load_all();
    let mut texts = Vec::new();
    let mut domains = Vec::new();
    for def in ontology.classification_domains() {
        let (Some(d), Some(_m)) = (&def.domain, &def.mode) else { continue };
        let mut text = def.name.clone();
        for hint in &def.hints { text.push(' '); text.push_str(hint); }
        texts.push(text);
        domains.push(d.clone());
    }
    let mut domain_ids: HashMap<String, usize> = HashMap::new();
    let labels: Vec<usize> = domains.iter().map(|d| { let next = domain_ids.len(); *domain_ids.entry(d.clone()).or_insert(next) }).collect();
    let mut domain_names = vec![String::new(); domain_ids.len()];
    for (name, &id) in &domain_ids { domain_names[id] = name.clone(); }
    let n_classes = domain_ids.len();
    let n = texts.len();

    let minilm_emb: Vec<Vec<f32>> = texts.iter().map(|t| minilm.embed(t)).collect();
    let bge_emb: Vec<Vec<f32>> = texts.iter().map(|t| bge.embed(t)).collect();

    let centroids_for = |embs: &[Vec<f32>]| -> Vec<Vec<f32>> {
        (0..n_classes).map(|c| {
            let members: Vec<&Vec<f32>> = (0..n).filter(|&i| labels[i] == c).map(|i| &embs[i]).collect();
            centroid(&members)
        }).collect()
    };
    let minilm_centroids = centroids_for(&minilm_emb);
    let bge_centroids = centroids_for(&bge_emb);

    // Same delta as Iteration 15/18's domain-level analysis: sweep for the smallest delta
    // that still yields SOME candidates, then report the corroboration reduction at a few deltas.
    let sims_for = |e: &Vec<f32>, centroids: &[Vec<f32>]| -> Vec<f32> { centroids.iter().map(|c| cosine_sim(e, c)).collect() };
    let minilm_sims: Vec<Vec<f32>> = minilm_emb.iter().map(|e| sims_for(e, &minilm_centroids)).collect();
    let bge_sims: Vec<Vec<f32>> = bge_emb.iter().map(|e| sims_for(e, &bge_centroids)).collect();

    let candidate_set = |sims: &[f32], delta: f32| -> Vec<usize> {
        let top = sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        (0..sims.len()).filter(|&c| sims[c] >= top - delta).collect()
    };

    for delta in [0.02, 0.03, 0.05] {
        let minilm_flagged: Vec<usize> = (0..n).filter(|&i| candidate_set(&minilm_sims[i], delta).len() > 1).collect();
        let corroborated_count = minilm_flagged.iter().filter(|&&i| {
            let m_set = candidate_set(&minilm_sims[i], delta);
            let own_top = labels[i];
            let bge_set = candidate_set(&bge_sims[i], delta);
            m_set.iter().any(|&c| c != own_top && bge_set.contains(&c))
        }).count();
        println!(
            "delta={delta:.2}: MiniLM flags {}/{} entries as multi-domain candidates; {} ({:.1}%) survive BGE corroboration",
            minilm_flagged.len(), n, corroborated_count,
            100.0 * corroborated_count as f32 / minilm_flagged.len().max(1) as f32
        );
    }

    // Qualitative spot-check at delta=0.03: a few examples, before and after.
    let delta = 0.03;
    println!("\nQualitative spot-check at delta=0.03 (first 8 MiniLM-flagged entries, with corroboration verdict):");
    let mut shown = 0;
    for i in 0..n {
        if shown >= 8 { break; }
        let m_set = candidate_set(&minilm_sims[i], delta);
        if m_set.len() <= 1 { continue; }
        let own_top = labels[i];
        let bge_set = candidate_set(&bge_sims[i], delta);
        let corroborated: Vec<&str> = m_set.iter().filter(|&&c| c != own_top && bge_set.contains(&c)).map(|&c| domain_names[c].as_str()).collect();
        println!(
            "  '{}'  true={}  MiniLM-flags={:?}  BGE-corroborates={:?}",
            texts[i].chars().take(70).collect::<String>(),
            domain_names[own_top],
            m_set.iter().map(|&c| domain_names[c].as_str()).collect::<Vec<_>>(),
            corroborated
        );
        shown += 1;
    }
}

fn main() {
    println!("Experiment 19: cross-embedder corroboration for multi-membership (roadmap item 15)\n");

    #[cfg(feature = "embed-onnx")]
    {
        let minilm_dir = ["models", "../models"].iter().find(|d| std::path::Path::new(d).join("model.onnx").exists());
        let minilm = minilm_dir.map(|dir| OnnxEmbedder::with_config(&OnnxConfig { dim: 384, model_dir: Some(dir.to_string()), pooling: PoolingStrategy::Mean, ..OnnxConfig::default() }));
        let bge_dir = ["models/bge-base-en-v1.5", "../models/bge-base-en-v1.5"].iter().find(|d| std::path::Path::new(d).join("onnx/model.onnx").exists());
        let bge = bge_dir.map(|dir| OnnxEmbedder::with_config(&OnnxConfig { dim: 768, model_dir: Some(dir.to_string()), pooling: PoolingStrategy::Mean, ..OnnxConfig::default() }));

        let (minilm, bge) = match (minilm, bge) {
            (Some(m), Some(b)) if m.is_available() && b.is_available() => (m, b),
            _ => { println!("WARNING: MiniLM or BGE not available — aborting."); return; }
        };
        println!("Both embedders loaded.\n");

        println!("########## Part A: Vehicle ontology (ground truth known, Iterations 12-13's dataset) ##########");
        run_vehicle_test(&minilm, &bge, "bare");
        run_vehicle_test(&minilm, &bge, "contextual");

        run_real_ontology_test(&minilm, &bge);
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
