//! Experiment 11 — find a certification signal that catches "balanced but
//! wrong" AND correctly accepts "unbalanced but right", closing Iteration
//! 10's open problem: size-balance ratio (validated in Iteration 9) missed
//! exactly the case where a split is well-sized but semantically wrong
//! (Iteration 10's bird depth-2 split, 6-vs-4, balance 0.667, mislabels
//! two flightless birds as flying).
//!
//! Prior tests (Iterations 9-10) only ever varied ONE axis at a time
//! (correctness OR balance), so it was never actually checked whether any
//! candidate signal tracks correctness independent of size. This builds a
//! proper 2x2: correct/wrong crossed with balanced/imbalanced, using real
//! or previously-established ground truth in all four cells — no
//! synthetic "obviously wrong" split invented just to make a point.
//!
//! | | balanced | imbalanced |
//! |---|---|---|
//! | correct | Case 1: Dataset A true mammal/bird (8-vs-7) | Case 4: NEW vehicle corpus, true wheel-count split (2-vs-10) |
//! | wrong | Case 3: Iteration 10 bird depth-2 real k-means (6-vs-4) | Case 2: Dataset A real depth-1 k-means (13-vs-2) |
//!
//! Candidates tested, all computable without ground truth (production
//! usable):
//!   - balance ratio (Iteration 9's validated signal — expected to fail
//!     Case 4, by construction, since it only ever checks size)
//!   - silhouette-like cohesion-minus-separation (Iteration 9's falsified
//!     signal — expected to fail again)
//!   - raw kNN-consistency (k=3): fraction of each item's 3 nearest
//!     neighbors that share its own cluster label, averaged
//!   - size-adjusted kNN-consistency: raw minus the value expected by
//!     cluster-size base rate alone, i.e. correcting for the fact that a
//!     larger cluster has more same-cluster neighbors available just by
//!     construction — this is the one candidate designed specifically to
//!     be size-invariant, the way Iteration 9's silhouette-like metric was
//!     NOT.
//!   - confident-item fraction: 1 minus the fraction of items whose
//!     own-cluster-centroid-minus-other-cluster-centroid margin is below
//!     a small epsilon (0.02) — a per-item ambiguity-rate variant of the
//!     silhouette idea, distinct from the mean-margin version that failed.
//!
//! Pass criterion: min(score over the 2 correct cases) > max(score over
//! the 2 wrong cases). Anything else means the signal is still tracking
//! balance (or something else) rather than correctness.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment11_generalized_certification

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
        for d in 0..dim { centroids[c][d] /= norm; }
    }
    centroids
}

fn balance_ratio(assignment: &[usize]) -> f32 {
    let c0 = assignment.iter().filter(|&&a| a == 0).count();
    let c1 = assignment.iter().filter(|&&a| a == 1).count();
    let (lo, hi) = (c0.min(c1), c0.max(c1));
    if hi == 0 { 0.0 } else { lo as f32 / hi as f32 }
}

fn silhouette_like(embeddings: &[Vec<f32>], assignment: &[usize]) -> f32 {
    let centroids = cluster_centroids(embeddings, assignment);
    let mut total = 0.0;
    for (i, e) in embeddings.iter().enumerate() {
        let own = assignment[i];
        let other = 1 - own;
        total += cosine_sim(e, &centroids[own]) - cosine_sim(e, &centroids[other]);
    }
    total / embeddings.len() as f32
}

/// Fraction of each item's k nearest neighbors (by cosine, excluding
/// itself) that share its own cluster label, averaged over all items.
fn knn_consistency_raw(embeddings: &[Vec<f32>], assignment: &[usize], k: usize) -> f32 {
    let n = embeddings.len();
    let mut total = 0.0;
    for i in 0..n {
        let mut sims: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine_sim(&embeddings[i], &embeddings[j])))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let kk = k.min(sims.len());
        let same = sims.iter().take(kk).filter(|(j, _)| assignment[*j] == assignment[i]).count();
        total += same as f32 / kk as f32;
    }
    total / n as f32
}

/// Raw kNN-consistency minus the value expected by cluster-size base rate
/// alone: if labels were random (preserving cluster sizes), an item in a
/// cluster of size s (n total) would expect (s-1)/(n-1) of its neighbors
/// to share its label just by chance — a larger cluster inflates raw
/// consistency for free. Subtracting this out is the size-invariance fix,
/// analogous to how an adjusted Rand index corrects plain agreement.
fn knn_consistency_adjusted(embeddings: &[Vec<f32>], assignment: &[usize], k: usize) -> f32 {
    let n = embeddings.len();
    let mut total = 0.0;
    for i in 0..n {
        let mut sims: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine_sim(&embeddings[i], &embeddings[j])))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let kk = k.min(sims.len());
        let same = sims.iter().take(kk).filter(|(j, _)| assignment[*j] == assignment[i]).count();
        let observed = same as f32 / kk as f32;
        let own_size = assignment.iter().filter(|&&a| a == assignment[i]).count();
        let expected = if n > 1 { (own_size.saturating_sub(1)) as f32 / (n - 1) as f32 } else { 0.0 };
        total += observed - expected;
    }
    total / n as f32
}

/// Same as knn_consistency_adjusted, but k is capped per-item at
/// (own_cluster_size - 1) instead of a fixed constant. With a fixed k=3, a
/// 2-item cluster's members can have at most 1 same-cluster neighbor by
/// arithmetic (only 1 other same-cluster item exists) — the metric is then
/// forced to count 2 off-cluster items as "neighbors" for the remaining
/// slots, mechanically capping the achievable score at 1/3 regardless of
/// embedding quality. Adapting k removes this size-driven ceiling.
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

/// 1 minus the fraction of items whose own-cluster-minus-other-cluster
/// centroid margin is below epsilon — an ambiguity-RATE variant of
/// Iteration 9's mean-margin (silhouette-like) metric, which failed.
fn confident_item_fraction(embeddings: &[Vec<f32>], assignment: &[usize], epsilon: f32) -> f32 {
    let centroids = cluster_centroids(embeddings, assignment);
    let mut low = 0usize;
    for (i, e) in embeddings.iter().enumerate() {
        let own = assignment[i];
        let other = 1 - own;
        let margin = cosine_sim(e, &centroids[own]) - cosine_sim(e, &centroids[other]);
        if margin < epsilon { low += 1; }
    }
    1.0 - (low as f32 / embeddings.len() as f32)
}

// ─────────────────────────── Case 1 & 2: Dataset A (mammal/bird) ───────────────────────────
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

// ─────────────────────────── Case 3: bird branch (from Iteration 10's 19-item corpus) ───────────────────────────
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

// ─────────────────────────── Case 4: NEW vehicle corpus (wheel count, deliberately imbalanced ground truth) ───────────────────────────
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

fn purity(assignment: &[usize], labels: &[&str]) -> f32 {
    let n = assignment.len();
    let mut correct = 0usize;
    for c in 0..2 {
        let mut counts = std::collections::HashMap::new();
        for i in 0..n {
            if assignment[i] == c { *counts.entry(labels[i]).or_insert(0usize) += 1; }
        }
        correct += counts.values().copied().max().unwrap_or(0);
    }
    correct as f32 / n as f32
}

struct Case {
    name: &'static str,
    correct: bool,
    embeddings: Vec<Vec<f32>>,
    assignment: Vec<usize>,
}

fn main() {
    println!("Experiment 11: finding a certification signal robust to balance, sensitive to correctness\n");

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

        // Case 1 & 2 share the same embeddings (Dataset A).
        let a_emb: Vec<Vec<f32>> = DATASET_A.iter().map(|(_, s, _)| embedder.embed(s)).collect();
        let a_labels: Vec<&str> = DATASET_A.iter().map(|(_, _, l)| *l).collect();
        let a_true: Vec<usize> = a_labels.iter().map(|&l| if l == "mammal" { 0 } else { 1 }).collect();
        let a_real = kmeans(&a_emb, 2, 30);
        println!("Case 2 sanity check — real k-means purity vs true mammal/bird: {:.3} (expect ~0.667, the known 13-vs-2 split)", purity(&a_real, &a_labels));

        // Case 3: bird branch, real k-means depth-2 split.
        let bird_emb: Vec<Vec<f32>> = BIRD_BRANCH.iter().map(|(_, s, _)| embedder.embed(s)).collect();
        let bird_labels: Vec<&str> = BIRD_BRANCH.iter().map(|(_, _, l)| *l).collect();
        let bird_real = kmeans(&bird_emb, 2, 30);
        println!("Case 3 sanity check — real k-means purity vs true flying/flightless: {:.3} (expect a mislabeled split, not close to 1.0)\n", purity(&bird_real, &bird_labels));

        // Case 4: vehicles, TRUE wheel-count partition (2-vs-10, deliberately imbalanced).
        let veh_emb: Vec<Vec<f32>> = VEHICLES.iter().map(|(_, s, ..)| embedder.embed(s)).collect();
        let veh_true: Vec<usize> = VEHICLES.iter().map(|v| if v.2 == "two" { 0 } else { 1 }).collect();

        let cases = vec![
            Case { name: "1: GOOD-balanced (true mammal/bird, 8v7)", correct: true, embeddings: a_emb.clone(), assignment: a_true.clone() },
            Case { name: "2: BAD-imbalanced (real k-means, 13v2)", correct: false, embeddings: a_emb.clone(), assignment: a_real.clone() },
            Case { name: "3: BAD-balanced (Iter.10 bird depth-2 real, 6v4)", correct: false, embeddings: bird_emb.clone(), assignment: bird_real.clone() },
            Case { name: "4: GOOD-imbalanced (true wheel count, 2v10)", correct: true, embeddings: veh_emb.clone(), assignment: veh_true.clone() },
        ];

        println!("{:<48} {:>8} {:>8} {:>10} {:>10} {:>10}", "case", "balance", "sil-like", "knn-raw", "knn-adj", "knn-adapt");
        let mut balance_scores = vec![];
        let mut sil_scores = vec![];
        let mut knn_raw_scores = vec![];
        let mut knn_adj_scores = vec![];
        let mut knn_adapt_scores = vec![];
        for case in &cases {
            let bal = balance_ratio(&case.assignment);
            let sil = silhouette_like(&case.embeddings, &case.assignment);
            let knn_raw = knn_consistency_raw(&case.embeddings, &case.assignment, 3);
            let knn_adj = knn_consistency_adjusted(&case.embeddings, &case.assignment, 3);
            let knn_adapt = knn_consistency_adaptive(&case.embeddings, &case.assignment, 3);
            println!("{:<48} {bal:>8.3} {sil:>8.3} {knn_raw:>10.3} {knn_adj:>10.3} {knn_adapt:>10.3}", case.name);
            balance_scores.push((case.correct, bal));
            sil_scores.push((case.correct, sil));
            knn_raw_scores.push((case.correct, knn_raw));
            knn_adj_scores.push((case.correct, knn_adj));
            knn_adapt_scores.push((case.correct, knn_adapt));
        }

        // Sweep epsilon for confident-item fraction rather than assume 0.02
        // is meaningful — a fixed value that happens to saturate at 1.000
        // for every case (as 0.02 did on the first run of this experiment)
        // is not calibrated, it's uninspected.
        println!("\nconfident-item-fraction epsilon sweep (1 - fraction of items with margin < epsilon):");
        println!("{:<10} {:>8} {:>8} {:>8} {:>8}", "epsilon", "case1", "case2", "case3", "case4");
        let mut best_conf_eps: Option<(f32, bool)> = None;
        for i in 0..=30 {
            let eps = i as f32 * 0.01;
            let vals: Vec<f32> = cases.iter().map(|c| confident_item_fraction(&c.embeddings, &c.assignment, eps)).collect();
            println!("{eps:<10.2} {:>8.3} {:>8.3} {:>8.3} {:>8.3}", vals[0], vals[1], vals[2], vals[3]);
            let scored: Vec<(bool, f32)> = cases.iter().zip(vals.iter()).map(|(c, v)| (c.correct, *v)).collect();
            let min_good = scored.iter().filter(|(c, _)| *c).map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
            let max_bad = scored.iter().filter(|(c, _)| !*c).map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
            if best_conf_eps.is_none() && min_good > max_bad {
                best_conf_eps = Some((eps, true));
            }
        }

        fn separates(scores: &[(bool, f32)]) -> bool {
            let min_good = scores.iter().filter(|(c, _)| *c).map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
            let max_bad = scores.iter().filter(|(c, _)| !*c).map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
            min_good > max_bad
        }

        println!("\n=== Verdict: does min(correct-case scores) > max(wrong-case scores)? ===");
        println!("balance ratio:              {}", separates(&balance_scores));
        println!("silhouette-like:            {}", separates(&sil_scores));
        println!("kNN-consistency (raw):      {}", separates(&knn_raw_scores));
        println!("kNN-consistency (adjusted, fixed k=3): {}", separates(&knn_adj_scores));
        println!("kNN-consistency (adjusted, adaptive k): {}", separates(&knn_adapt_scores));
        match best_conf_eps {
            Some((eps, _)) => println!("confident-item fraction:    TRUE at epsilon={eps:.2} (smallest epsilon that separates)"),
            None => println!("confident-item fraction:    false at every epsilon in [0.00, 0.30]"),
        }
        println!("\n(a signal that passes here tracks correctness independent of cluster-size balance, which is what Iteration 10's 'balanced-but-wrong' failure mode requires; one that fails is still confounded with size, imbalance, or something else, the same way balance-ratio and silhouette-like each are by construction here)");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
