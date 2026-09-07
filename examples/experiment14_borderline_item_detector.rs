// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 14 — a certification signal targeting the SPECIFIC
//! character of "balanced but wrong" splits directly, instead of another
//! whole-cluster aggregate combined into an ensemble (Iteration 13 ruled
//! that out: balance ratio AND adaptive kNN-consistency cannot separate
//! good from bad across the 6 known real cases, because the failure is
//! structural, not a tuning problem).
//!
//! The two known bad-balanced cases (Iteration 10's bird branch, 6-vs-4,
//! mislabeling 2 of 10 items; Iteration 12's economic branch) both have
//! a specific character never directly tested before: a SMALL MINORITY
//! of items are genuinely misplaced while the MAJORITY are correctly
//! grouped. This is structurally different from "uniformly mediocre",
//! which is what Iteration 11's good-imbalanced control (vehicle wheel
//! count) looks like — every item sits on a similarly weak axis, not a
//! few outliers within an otherwise-strong cluster. Three margin-
//! distribution-SHAPE candidates are tested, none of them a simple
//! rename of balance ratio or the mean-margin silhouette metric that
//! already failed:
//!
//!   - misfit fraction: how many items have NEGATIVE margin (i.e. sit
//!     closer to the OTHER cluster's centroid than their own, relative
//!     to the split's OWN centroids, not ground truth) — a "few items
//!     wrong" split should show a small nonzero misfit fraction; a
//!     "uniformly weak" split should show zero (every item still edges
//!     out toward its own side, just not by much).
//!   - worst-item margin: the single most negative margin in the split —
//!     sharper than a fraction, catches even one badly placed item.
//!   - margin standard deviation: a split with a distinct minority of
//!     mismatched items should show HIGHER spread (bimodal: most items
//!     confidently positive, a few confidently negative) than a
//!     uniformly-weak-everywhere split.
//!
//! Tested against the same 6 real cases used in Iteration 13's ensemble
//! sweep (2 good, 4 bad, spanning five iterations) — reusing established
//! ground truth rather than inventing a new synthetic test.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment14_borderline_item_detector

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;

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

fn cluster_centroids(embeddings: &[Vec<f32>], assignment: &[usize]) -> [Vec<f32>; 2] {
    let dim = embeddings[0].len();
    let mut centroids = [vec![0.0f32; dim], vec![0.0f32; dim]];
    let mut counts = [0usize; 2];
    for (i, e) in embeddings.iter().enumerate() {
        let c = assignment[i];
        counts[c] += 1;
        for d in 0..dim { centroids[c][d] += e[d]; }
    }
    for c in 0..2 {
        if counts[c] == 0 { continue; }
        let norm: f32 = centroids[c].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for v in &mut centroids[c] { *v /= norm; }
    }
    centroids
}

fn margins(embeddings: &[Vec<f32>], assignment: &[usize]) -> Vec<f32> {
    let centroids = cluster_centroids(embeddings, assignment);
    embeddings.iter().enumerate().map(|(i, e)| {
        let own = assignment[i];
        let other = 1 - own;
        cosine_sim(e, &centroids[own]) - cosine_sim(e, &centroids[other])
    }).collect()
}

fn balance_ratio(assignment: &[usize]) -> f32 {
    let c0 = assignment.iter().filter(|&&a| a == 0).count();
    let c1 = assignment.iter().filter(|&&a| a == 1).count();
    let (lo, hi) = (c0.min(c1), c0.max(c1));
    if hi == 0 { 0.0 } else { lo as f32 / hi as f32 }
}

// ─────────────────────────── The 6 known real cases (from Iteration 13) ───────────────────────────

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

const ECONOMIC_BRANCH: &[(&str, &str)] = &[
    ("insurance_premium", "insurance premium"), ("maintenance_cost", "maintenance cost"),
    ("resale_value", "resale value"), ("purchase_price", "purchase price"),
    ("trade_in_value", "trade-in value"), ("warranty_coverage", "warranty coverage"),
];

const MECHANICAL_BRANCH: &[(&str, &str)] = &[
    ("engine", "engine"), ("transmission", "transmission"), ("turbocharger", "turbocharger"),
    ("fuel_tank", "fuel tank"), ("spark_plug", "spark plug"),
    ("hybrid_battery", "hybrid battery"), ("fuel_efficiency", "fuel efficiency"),
];

struct GateCase { name: &'static str, correct: bool, embeddings: Vec<Vec<f32>>, assignment: Vec<usize>, item_names: Vec<&'static str> }

/// A genuinely different mechanism from margin-based signals: does
/// forcibly bisecting EACH cluster of the split (k=2 sub-cluster) reveal
/// a small, internally-tight minority subgroup? Hypothesis: "a few items
/// wrong within an otherwise-consistent cluster" should show up as an
/// especially IMBALANCED and TIGHT sub-bisection of at least one of the
/// two parent clusters (the misfits isolate into a small, cohesive
/// sub-group); a genuinely homogeneous cluster forced to bisect should
/// show either a roughly balanced sub-split or an imbalanced one that
/// ISN'T especially tighter than the rest (just noise). Returns, for the
/// whole split, the worst (smallest) sub-bisection balance ratio found in
/// either parent cluster, and whether that minority sub-cluster is
/// tighter (higher mean pairwise cosine) than the majority remainder.
fn worst_subcluster_signal(embeddings: &[Vec<f32>], assignment: &[usize]) -> (f32, bool) {
    let mut worst_balance = 1.0f32;
    let mut worst_is_tighter = false;
    for parent in 0..2 {
        let members: Vec<usize> = (0..embeddings.len()).filter(|&i| assignment[i] == parent).collect();
        if members.len() < 4 { continue; } // too small to meaningfully sub-bisect
        let sub_emb: Vec<Vec<f32>> = members.iter().map(|&i| embeddings[i].clone()).collect();
        let sub_split = kmeans(&sub_emb, 2, 30);
        let c0 = sub_split.iter().filter(|&&a| a == 0).count();
        let c1 = sub_split.iter().filter(|&&a| a == 1).count();
        let (minority, majority) = if c0 <= c1 { (0, 1) } else { (1, 0) };
        let bal = c0.min(c1) as f32 / c0.max(c1) as f32;
        let mean_pairwise = |group: usize| -> f32 {
            let idx: Vec<usize> = (0..sub_emb.len()).filter(|&i| sub_split[i] == group).collect();
            if idx.len() < 2 { return 1.0; }
            let mut total = 0.0;
            let mut count = 0;
            for a in 0..idx.len() {
                for b in (a + 1)..idx.len() {
                    total += cosine_sim(&sub_emb[idx[a]], &sub_emb[idx[b]]);
                    count += 1;
                }
            }
            total / count as f32
        };
        let minority_tightness = mean_pairwise(minority);
        let majority_tightness = mean_pairwise(majority);
        if bal < worst_balance {
            worst_balance = bal;
            worst_is_tighter = minority_tightness > majority_tightness;
        }
    }
    (worst_balance, worst_is_tighter)
}

fn stddev(values: &[f32]) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    var.sqrt()
}

fn main() {
    println!("Experiment 14: a borderline-item detector targeting 'balanced but wrong' directly\n");

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

        let a_emb: Vec<Vec<f32>> = DATASET_A.iter().map(|(_, s, _)| embedder.embed(s)).collect();
        let a_names: Vec<&str> = DATASET_A.iter().map(|(n, ..)| *n).collect();
        let a_true: Vec<usize> = DATASET_A.iter().map(|(_, _, l)| if *l == "mammal" { 0 } else { 1 }).collect();
        let a_real = kmeans(&a_emb, 2, 30);

        let bird_emb: Vec<Vec<f32>> = BIRD_BRANCH.iter().map(|(_, s, _)| embedder.embed(s)).collect();
        let bird_names: Vec<&str> = BIRD_BRANCH.iter().map(|(n, ..)| *n).collect();
        let bird_real = kmeans(&bird_emb, 2, 30);

        let veh_emb: Vec<Vec<f32>> = VEHICLES.iter().map(|(_, s, ..)| embedder.embed(s)).collect();
        let veh_names: Vec<&str> = VEHICLES.iter().map(|(n, ..)| *n).collect();
        let veh_true: Vec<usize> = VEHICLES.iter().map(|v| if v.2 == "two" { 0 } else { 1 }).collect();

        let econ_emb: Vec<Vec<f32>> = ECONOMIC_BRANCH.iter().map(|(_, s)| embedder.embed(s)).collect();
        let econ_names: Vec<&str> = ECONOMIC_BRANCH.iter().map(|(n, _)| *n).collect();
        let econ_real = kmeans(&econ_emb, 2, 30);

        let mech_emb: Vec<Vec<f32>> = MECHANICAL_BRANCH.iter().map(|(_, s)| embedder.embed(s)).collect();
        let mech_names: Vec<&str> = MECHANICAL_BRANCH.iter().map(|(n, _)| *n).collect();
        let mech_real = kmeans(&mech_emb, 2, 30);

        let cases = vec![
            GateCase { name: "1: GOOD-balanced (Dataset A true 8v7)", correct: true, embeddings: a_emb, assignment: a_true, item_names: a_names },
            GateCase { name: "2: BAD-imbalanced (Dataset A real, 13v2)", correct: false, embeddings: DATASET_A.iter().map(|(_, s, _)| embedder.embed(s)).collect(), assignment: a_real, item_names: DATASET_A.iter().map(|(n, ..)| *n).collect() },
            GateCase { name: "3: BAD-balanced (Iter.10 bird branch, 6v4)", correct: false, embeddings: bird_emb, assignment: bird_real, item_names: bird_names },
            GateCase { name: "4: GOOD-imbalanced (true wheel count, 2v10)", correct: true, embeddings: veh_emb, assignment: veh_true, item_names: veh_names },
            GateCase { name: "5: BAD-balanced (Iter.12 economic branch)", correct: false, embeddings: econ_emb, assignment: econ_real, item_names: econ_names },
            GateCase { name: "6: BAD-imbalanced (Iter.12 mechanical branch)", correct: false, embeddings: mech_emb, assignment: mech_real, item_names: mech_names },
        ];

        println!("{:<48} {:>8} {:>10} {:>10} {:>10} {:>12} {:>8}", "case", "balance", "misfit%", "worst-marg", "margin-std", "sub-bal", "tighter");
        let mut misfit_scores: Vec<(bool, f32)> = Vec::new();
        let mut worst_scores: Vec<(bool, f32)> = Vec::new();
        let mut std_scores: Vec<(bool, f32)> = Vec::new();
        let mut subcluster_scores: Vec<(bool, f32)> = Vec::new();
        for c in &cases {
            let m = margins(&c.embeddings, &c.assignment);
            let misfit_count = m.iter().filter(|&&v| v < 0.0).count();
            let misfit_frac = misfit_count as f32 / m.len() as f32;
            let worst = m.iter().cloned().fold(f32::INFINITY, f32::min);
            let sd = stddev(&m);
            let bal = balance_ratio(&c.assignment);
            let (sub_bal, sub_tighter) = worst_subcluster_signal(&c.embeddings, &c.assignment);
            println!("{:<48} {bal:>8.3} {:>10.3} {worst:>10.3} {sd:>10.3} {sub_bal:>12.3} {sub_tighter:>8}", c.name, misfit_frac);
            if misfit_count > 0 {
                let misfit_names: Vec<&str> = c.item_names.iter().zip(m.iter()).filter(|(_, &v)| v < 0.0).map(|(n, _)| *n).collect();
                println!("    misfit items: {misfit_names:?}");
            }
            // trustworthiness convention: higher = more trustworthy, so invert misfit and worst-margin-is-bad
            misfit_scores.push((c.correct, 1.0 - misfit_frac));
            worst_scores.push((c.correct, worst)); // more negative = worse, so raw worst is already "higher = better"
            std_scores.push((c.correct, -sd)); // hypothesis: LOWER std = more trustworthy (uniform), so invert for "higher=better" convention — tested both directions below
            // hypothesis: a TIGHT, IMBALANCED sub-bisection (low sub_bal AND tighter=true) signals
            // contamination — score LOW sub_bal as bad (less trustworthy) only when tighter=true;
            // otherwise an imbalanced-but-not-tighter sub-split is just noise, not evidence.
            let contamination_score = if sub_tighter { sub_bal } else { 1.0 };
            subcluster_scores.push((c.correct, contamination_score));
        }

        fn separates(scores: &[(bool, f32)]) -> bool {
            let min_good = scores.iter().filter(|(c, _)| *c).map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
            let max_bad = scores.iter().filter(|(c, _)| !*c).map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
            min_good > max_bad
        }

        println!("\n=== Verdict: does min(correct-case scores) > max(wrong-case scores)? ===");
        println!("misfit fraction (inverted, higher=better):        {}", separates(&misfit_scores));
        println!("worst-item margin (higher=better, less negative): {}", separates(&worst_scores));
        println!("margin std-dev, LOWER=better hypothesis:          {}", separates(&std_scores));
        let std_scores_higher_better: Vec<(bool, f32)> = std_scores.iter().map(|(c, s)| (*c, -s)).collect();
        println!("margin std-dev, HIGHER=better hypothesis:         {}", separates(&std_scores_higher_better));
        println!("sub-bisection contamination signal (genuinely different mechanism): {}", separates(&subcluster_scores));
        println!("\n(a signal that passes tracks the specific 'few items wrong' character directly; Iteration 13 already ruled out balance+adaptive-kNN ensembles, so this tests genuinely new signals, not a recombination)");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
