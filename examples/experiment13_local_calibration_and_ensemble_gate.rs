//! Experiment 13 — the two next-experiments flagged after Iteration 12:
//!
//! Part A: does LOCAL, split calibration fix the granularity-degradation
//! problem (Iteration 12: a single flat delta across all 6 fine cells
//! needed FPR=0.267-0.533 for full cross-item recall)? Hypothesis: the
//! flat delta conflates two different problems — (1) confusing an item
//! with its SAME-BRANCH sibling fine category (close together, needs a
//! tight delta) and (2) detecting a genuine CROSS-BRANCH membership
//! (further apart, needs a looser delta) — forcing one shared threshold
//! to serve both is what produces the bad FPR. Splitting into two
//! separately-calibrated checks should do better.
//!
//! Part B: does an ENSEMBLE gate (balance ratio AND adaptive
//! size-adjusted kNN-consistency, Iteration 11's closest single
//! candidate) catch the "balanced but wrong" failure mode across ALL
//! FOUR now-known real instances (Iteration 10's bird branch, two from
//! Iteration 12's vehicle ontology, plus sanity-check bad-imbalanced
//! cases) while still accepting the two known-good cases (one balanced,
//! one deliberately imbalanced)? Iteration 11 showed neither signal
//! alone works; this tests whether requiring BOTH to pass does better.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment13_local_calibration_and_ensemble_gate

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

fn balance_ratio(assignment: &[usize]) -> f32 {
    let c0 = assignment.iter().filter(|&&a| a == 0).count();
    let c1 = assignment.iter().filter(|&&a| a == 1).count();
    let (lo, hi) = (c0.min(c1), c0.max(c1));
    if hi == 0 { 0.0 } else { lo as f32 / hi as f32 }
}

/// Iteration 11's closest single certification candidate: raw kNN
/// consistency (fraction of an item's k nearest neighbors sharing its
/// cluster) minus the base-rate-expected value for its own cluster size,
/// with k capped per item at (own_cluster_size - 1) so small clusters
/// aren't mechanically penalized.
fn knn_consistency_adaptive(embeddings: &[Vec<f32>], assignment: &[usize], max_k: usize) -> f32 {
    let n = embeddings.len();
    let mut total = 0.0;
    for i in 0..n {
        let own_size = assignment.iter().filter(|&&a| a == assignment[i]).count();
        let k = max_k.min(own_size.saturating_sub(1)).max(1);
        let mut sims: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine_sim(&embeddings[i], &embeddings[j])))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let kk = k.min(sims.len());
        let same = sims.iter().take(kk).filter(|(j, _)| assignment[*j] == assignment[i]).count();
        let observed = same as f32 / kk as f32;
        let expected = if n > 1 { (own_size.saturating_sub(1)) as f32 / (n - 1) as f32 } else { 0.0 };
        total += observed - expected;
    }
    total / n as f32
}

// ═══════════════════════════════ Part A: local vs. global calibration ═══════════════════════════════

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

fn run_calibration_comparison<F: Fn(&str) -> Vec<f32>>(representation: &str, embed: F) {
    let n_pure = PURE.len();
    let pure_emb: Vec<Vec<f32>> = PURE.iter().map(|(_, bare, ctx, ..)| embed(if representation == "bare" { bare } else { ctx })).collect();
    let cross_emb: Vec<Vec<f32>> = CROSS.iter().map(|(_, bare, ctx, ..)| embed(if representation == "bare" { bare } else { ctx })).collect();

    let fine_centroids: HashMap<&'static str, Vec<f32>> = FINE_CATS.iter().map(|&cat| {
        let members: Vec<&Vec<f32>> = (0..n_pure).filter(|&i| PURE[i].4 == cat).map(|i| &pure_emb[i]).collect();
        (cat, centroid(&members))
    }).collect();
    let sims_for = |e: &Vec<f32>| -> HashMap<&'static str, f32> {
        FINE_CATS.iter().map(|&c| (c, cosine_sim(e, &fine_centroids[c]))).collect()
    };
    let pure_sims: Vec<HashMap<&'static str, f32>> = pure_emb.iter().map(sims_for).collect();
    let cross_sims: Vec<HashMap<&'static str, f32>> = cross_emb.iter().map(sims_for).collect();

    // ── GLOBAL (Iteration 12's approach): margin vs the best of all 5 OTHER categories ──
    let global_fpr_at = |delta: f32| -> f32 {
        let membership_set = |sims: &HashMap<&'static str, f32>| -> usize {
            let top = sims.values().cloned().fold(f32::NEG_INFINITY, f32::max);
            FINE_CATS.iter().filter(|c| sims[*c] >= top - delta).count()
        };
        pure_sims.iter().filter(|s| membership_set(s) > 1).count() as f32 / n_pure as f32
    };
    let coarse_covered = |sims: &HashMap<&'static str, f32>, delta: f32, f1: &str, f2: &str| -> bool {
        let top = sims.values().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ms: Vec<&str> = FINE_CATS.iter().filter(|c| sims[**c] >= top - delta).copied().collect();
        let coarses: std::collections::HashSet<&str> = ms.iter().map(|c| fine_to_coarse(c)).collect();
        coarses.contains(f1) && coarses.contains(f2)
    };
    let global_recall_at = |delta: f32| -> usize {
        cross_sims.iter().enumerate().filter(|(i, s)| { let (_, _, _, f1, f2, _) = CROSS[*i]; coarse_covered(s, delta, f1, f2) }).count()
    };
    let global_best = (0..=30).map(|i| i as f32 * 0.01)
        .map(|d| (d, global_fpr_at(d), global_recall_at(d)))
        .filter(|&(_, _, r)| r == CROSS.len())
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap_or((0.30, global_fpr_at(0.30), global_recall_at(0.30)));

    // ── LOCAL/SPLIT: (1) sibling-only margin within the same branch, (2) cross-branch-only margin ──
    // Sibling check: does item i's similarity to its OWN fine category exceed its SAME-BRANCH
    // sibling's similarity by at least delta_sib? If not, it's flagged as sibling-confused.
    let sibling_fpr_at = |delta: f32| -> f32 {
        let mut flagged = 0usize;
        for i in 0..n_pure {
            let own = PURE[i].4;
            let sib = sibling_of(own);
            let margin = pure_sims[i][own] - pure_sims[i][sib];
            if margin < delta { flagged += 1; }
        }
        flagged as f32 / n_pure as f32
    };
    // No pure item's true second field is a same-branch sibling in this dataset (all 5 cross
    // items span two DIFFERENT coarse branches), so sibling recall has nothing to trade off —
    // the sibling threshold can be pushed as tight as possible without losing any recall.
    let sibling_best_delta = (0..=30).map(|i| i as f32 * 0.01)
        .map(|d| (d, sibling_fpr_at(d)))
        .find(|&(_, fpr)| fpr == 0.0)
        .unwrap_or_else(|| { let d = 0.30; (d, sibling_fpr_at(d)) });

    // Cross-branch check: does item i's similarity to some OTHER-branch category exceed its
    // own-category similarity minus delta_cross, considering ONLY other-branch categories
    // (excluding the same-branch sibling entirely)?
    let cross_branch_fpr_at = |delta: f32| -> f32 {
        let mut flagged = 0usize;
        for i in 0..n_pure {
            let own = PURE[i].4;
            let own_sim = pure_sims[i][own];
            let other_branch_max = FINE_CATS.iter()
                .filter(|c| fine_to_coarse(c) != fine_to_coarse(own))
                .map(|c| pure_sims[i][*c])
                .fold(f32::NEG_INFINITY, f32::max);
            if other_branch_max >= own_sim - delta { flagged += 1; }
        }
        flagged as f32 / n_pure as f32
    };
    // Recall: the cross item's PRIMARY coarse field owns its global top-1 cell by construction
    // (that's what "primary" means here); this checks whether the true SECOND coarse field has
    // at least one fine cell within delta of that top-1 value — directly analogous to the FPR
    // check's "is there a same/other-branch cell within delta of my best cell", but specific to
    // the actually-true second field rather than "any other branch" (which would be a weaker,
    // wrong test — an irrelevant category being close would wrongly count as a hit).
    let cross_branch_recall_at = |delta: f32| -> usize {
        cross_sims.iter().enumerate().filter(|(i, s)| {
            let (_, _, _, f1, f2, _) = CROSS[*i];
            let (&top_cat, &top_val) = s.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
            let top_coarse = fine_to_coarse(top_cat);
            let other_field = if top_coarse == f1 { f2 } else { f1 };
            FINE_CATS.iter().filter(|c| fine_to_coarse(c) == other_field).any(|c| s[*c] >= top_val - delta)
        }).count()
    };
    let cross_branch_best = (0..=30).map(|i| i as f32 * 0.01)
        .map(|d| (d, cross_branch_fpr_at(d), cross_branch_recall_at(d)))
        .filter(|&(_, _, r)| r == CROSS.len())
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap_or_else(|| { let d = 0.30; (d, cross_branch_fpr_at(d), cross_branch_recall_at(d)) });

    println!("\n--- {representation} ---");
    println!("GLOBAL (Iteration 12, single flat delta over all 6 cells): delta={:.2} FPR={:.3} recall={}/{}", global_best.0, global_best.1, global_best.2, CROSS.len());
    println!("LOCAL/SPLIT sibling-confusion check: delta={:.2} FPR={:.3} (no recall tradeoff exists in this dataset — 0 of 5 cross items are same-branch sibling pairs)", sibling_best_delta.0, sibling_best_delta.1);
    println!("LOCAL/SPLIT cross-branch check: delta={:.2} FPR={:.3} recall={}/{}", cross_branch_best.0, cross_branch_best.1, cross_branch_best.2, CROSS.len());
    let combined_fpr_upper_bound = (sibling_best_delta.1 + cross_branch_best.1).min(1.0); // union bound, honest worst case
    println!("LOCAL/SPLIT combined FPR (union upper bound, an item could be flagged by either check): <= {combined_fpr_upper_bound:.3}, vs GLOBAL's {:.3}", global_best.1);
}

// ═══════════════════════════════ Part B: ensemble certification gate ═══════════════════════════════

const DATASET_A: &[(&str, &str, &str)] = &[
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

const BIRD_BRANCH: &[(&str, &str, &str)] = &[
    ("eagle", "The eagle soared high above the canyon, scanning the ground for movement.", "flying"),
    ("sparrow", "The sparrow hopped along the branch before darting off between the leaves.", "flying"),
    ("owl", "The owl turned its head silently, watching for the faintest movement in the dark.", "flying"),
    ("swan", "The swan glided smoothly across the lake, barely rippling the water.", "flying"),
    ("penguin", "The penguin waddled across the ice before diving into the frigid water.", "flightless"),
    ("ostrich", "The ostrich sprinted across the plain on powerful legs, kicking up dust.", "flightless"),
    ("kiwi", "The kiwi foraged in the undergrowth at night, sniffing out insects with its long beak.", "flightless"),
    ("farm_duck", "Raised alongside the rest of the poultry on the farm, the duck paddled across the pond each morning and burst into the air whenever the fox padded too close.", "flying"),
    ("cassowary", "The cassowary paced through the rainforest undergrowth on powerful legs, stubby wings folded uselessly at its sides, ready to slash at anything that startled it with the dagger-like claw on each foot.", "flightless"),
    ("farm_chicken", "Kept in a pen behind the farmhouse, the chicken spent its days pecking at the ground, only managing a clumsy flap up to the low roost at dusk.", "flightless"),
];

const VEHICLES: &[(&str, &str, &str)] = &[
    ("motorcycle", "The motorcycle leaned into the curve as the rider downshifted before the hairpin turn.", "two"),
    ("bicycle", "The bicycle wobbled slightly as the child pedaled up the steep driveway.", "two"),
    ("car", "The car merged smoothly onto the highway, accelerating past the slower traffic in the right lane.", "many"),
    ("truck", "The truck rumbled down the gravel road, its heavy bed loaded with stacked lumber.", "many"),
    ("bus", "The bus pulled up to the curb and opened its doors for the line of waiting passengers.", "many"),
    ("van", "The van idled outside the warehouse while the driver checked the delivery manifest.", "many"),
    ("suv", "The vehicle climbed the rocky trail with ease, its high clearance keeping the underside clear of the boulders.", "many"),
    ("pickup", "It towed the boat trailer down the ramp and into the shallow water at the marina.", "many"),
    ("minivan", "Its sliding doors opened automatically as the family unloaded groceries in the driveway.", "many"),
    ("tractor", "The tractor crawled across the muddy field, dragging the plow through the freshly turned soil.", "many"),
    ("forklift", "The forklift lifted the pallet of crates and carried it carefully across the warehouse floor.", "many"),
    ("golf_cart", "The golf cart puttered along the cart path, carrying clubs and coolers toward the ninth hole.", "many"),
];

struct GateCase { name: &'static str, correct: bool, embeddings: Vec<Vec<f32>>, assignment: Vec<usize> }

fn main() {
    println!("Experiment 13: local calibration + ensemble certification gate\n");

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
                println!("Using real semantic embedder: {dir}/model.onnx");
                embedder = Some(e);
                break;
            }
        }
        let embedder = match embedder { Some(e) => e, None => { println!("WARNING: no ONNX model — aborting."); return; } };

        println!("\n########## Part A: local/split calibration vs Iteration 12's global flat delta ##########");
        run_calibration_comparison("bare", |s| embedder.embed(s));
        run_calibration_comparison("contextual", |s| embedder.embed(s));

        println!("\n\n########## Part B: ensemble certification gate across all known real cases ##########");

        let a_emb: Vec<Vec<f32>> = DATASET_A.iter().map(|(_, s, _)| embedder.embed(s)).collect();
        let a_labels: Vec<&str> = DATASET_A.iter().map(|(_, _, l)| *l).collect();
        let a_true: Vec<usize> = a_labels.iter().map(|&l| if l == "mammal" { 0 } else { 1 }).collect();
        let a_real = kmeans(&a_emb, 2, 30);

        let bird_emb: Vec<Vec<f32>> = BIRD_BRANCH.iter().map(|(_, s, _)| embedder.embed(s)).collect();
        let bird_real = kmeans(&bird_emb, 2, 30);

        let veh_emb: Vec<Vec<f32>> = VEHICLES.iter().map(|(_, s, ..)| embedder.embed(s)).collect();
        let veh_true: Vec<usize> = VEHICLES.iter().map(|v| if v.2 == "two" { 0 } else { 1 }).collect();

        // Iteration 12's economic branch, real k-means fine split (bare representation), a
        // second independently-discovered "balanced but wrong" instance.
        const ECONOMIC_BRANCH: &[(&str, &str)] = &[
            ("insurance_premium", "insurance premium"), ("maintenance_cost", "maintenance cost"),
            ("resale_value", "resale value"), ("purchase_price", "purchase price"),
            ("trade_in_value", "trade-in value"), ("warranty_coverage", "warranty coverage"),
        ];
        let econ_emb: Vec<Vec<f32>> = ECONOMIC_BRANCH.iter().map(|(_, s)| embedder.embed(s)).collect();
        let econ_real = kmeans(&econ_emb, 2, 30);

        // Iteration 12's mechanical branch (bare), a known bad-imbalanced sanity check (0.167 balance).
        const MECHANICAL_BRANCH: &[(&str, &str)] = &[
            ("engine", "engine"), ("transmission", "transmission"), ("turbocharger", "turbocharger"),
            ("fuel_tank", "fuel tank"), ("spark_plug", "spark plug"),
            ("hybrid_battery", "hybrid battery"), ("fuel_efficiency", "fuel efficiency"),
        ];
        let mech_emb: Vec<Vec<f32>> = MECHANICAL_BRANCH.iter().map(|(_, s)| embedder.embed(s)).collect();
        let mech_real = kmeans(&mech_emb, 2, 30);

        let cases = vec![
            GateCase { name: "1: GOOD-balanced (Dataset A true mammal/bird, 8v7)", correct: true, embeddings: a_emb.clone(), assignment: a_true.clone() },
            GateCase { name: "2: BAD-imbalanced (Dataset A real k-means, 13v2)", correct: false, embeddings: a_emb, assignment: a_real },
            GateCase { name: "3: BAD-balanced (Iter.10 bird branch real, 6v4)", correct: false, embeddings: bird_emb, assignment: bird_real },
            GateCase { name: "4: GOOD-imbalanced (true wheel count, 2v10)", correct: true, embeddings: veh_emb, assignment: veh_true },
            GateCase { name: "5: BAD-balanced (Iter.12 economic branch real, bare)", correct: false, embeddings: econ_emb, assignment: econ_real },
            GateCase { name: "6: BAD-imbalanced (Iter.12 mechanical branch real, bare)", correct: false, embeddings: mech_emb, assignment: mech_real },
        ];

        println!("{:<50} {:>10} {:>12}", "case", "balance", "knn-adapt");
        let mut scored: Vec<(bool, f32, f32)> = Vec::new();
        for c in &cases {
            let bal = balance_ratio(&c.assignment);
            let knn = knn_consistency_adaptive(&c.embeddings, &c.assignment, 3);
            println!("{:<50} {bal:>10.3} {knn:>12.3}", c.name);
            scored.push((c.correct, bal, knn));
        }

        println!("\n=== Ensemble rules tested: certify iff BOTH balance >= T_bal AND knn_adapt >= T_knn ===");
        // Sweep both thresholds; report whether ANY combination perfectly separates good from bad.
        let bal_thresholds: Vec<f32> = (0..=10).map(|i| i as f32 * 0.1).collect();
        let knn_thresholds: Vec<f32> = (-10..=30).map(|i| i as f32 * 0.01).collect();
        let mut best: Option<(f32, f32, usize, usize)> = None; // (t_bal, t_knn, good_pass, bad_reject)
        let mut perfect_found = false;
        for &t_bal in &bal_thresholds {
            for &t_knn in &knn_thresholds {
                let good_pass = scored.iter().filter(|(c, b, k)| *c && *b >= t_bal && *k >= t_knn).count();
                let good_total = scored.iter().filter(|(c, ..)| *c).count();
                let bad_reject = scored.iter().filter(|(c, b, k)| !*c && !(*b >= t_bal && *k >= t_knn)).count();
                let bad_total = scored.iter().filter(|(c, ..)| !*c).count();
                if good_pass == good_total && bad_reject == bad_total {
                    perfect_found = true;
                    if best.is_none() { best = Some((t_bal, t_knn, good_pass, bad_reject)); }
                }
            }
        }
        if perfect_found {
            let (t_bal, t_knn, gp, br) = best.unwrap();
            println!("PERFECT SEPARATION FOUND: T_bal={t_bal:.2}, T_knn={t_knn:.2} -> {gp}/2 good certified, {br}/4 bad rejected");
        } else {
            // Report the best achievable (max total correct decisions) honestly.
            let mut best_score = 0usize;
            let mut best_params = (0.0f32, 0.0f32);
            for &t_bal in &bal_thresholds {
                for &t_knn in &knn_thresholds {
                    let good_pass = scored.iter().filter(|(c, b, k)| *c && *b >= t_bal && *k >= t_knn).count();
                    let bad_reject = scored.iter().filter(|(c, b, k)| !*c && !(*b >= t_bal && *k >= t_knn)).count();
                    let score = good_pass + bad_reject;
                    if score > best_score { best_score = score; best_params = (t_bal, t_knn); }
                }
            }
            println!(
                "NO perfect separation exists for this ensemble rule over the swept thresholds. Best achievable: {best_score}/6 total correct decisions at T_bal={:.2}, T_knn={:.2}.",
                best_params.0, best_params.1
            );
            let (t_bal, t_knn) = best_params;
            for (name, (c, b, k)) in cases.iter().map(|c| c.name).zip(scored.iter()) {
                let certified = *b >= t_bal && *k >= t_knn;
                let outcome = if *c == certified { "correct" } else { "WRONG" };
                println!("  {name:<50} correct={c} certified={certified} [{outcome}]");
            }
        }
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
