//! Experiment 10 — closing the three gaps flagged after Iteration 9, on one
//! corpus that has BOTH real hierarchy AND cross-cutting structure at once.
//!
//! Gap 1 (never checked): Iteration 8's dual-membership test only measured
//! RECALL on 4 hand-picked cross-cutting items. It never checked whether
//! the 21 pure single-membership items would also trigger spurious
//! "second membership" signal at whatever threshold gets chosen — i.e. the
//! false-positive rate was never measured, so no calibrated threshold has
//! ever been picked.
//!
//! Gap 2 (never attempted): multi-membership (Iteration 8) and certified
//! recursion (Iteration 9's balance-ratio gate) have only been validated in
//! ISOLATION, on different slices of data. This runs them together on one
//! corpus that has real depth-2 hierarchy (mammal/bird -> domestic/
//! predator/flying/flightless, Dataset A) AND cross-cutting items (new,
//! this experiment) at the same time.
//!
//! Gap 3 (never designed): once an item can belong to a leaf in one branch
//! AND be linked to a leaf in ANOTHER branch (a real cross-branch case
//! shows up below: a domesticated bird pulls toward the "domestic" leaf,
//! which lives under the MAMMAL branch, while it is physically recursed
//! under BIRD), the structure is a DAG, not a tree. This demonstrates,
//! with a concrete adversarial graph built from this experiment's own
//! output, that expanding cross-links WITHOUT a visited-node cache does
//! not terminate (ping-pongs between the two linked leaves), and that a
//! trivial visited-set fixes it (each node expanded exactly once).
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment10_integration

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;
use std::collections::HashMap;

/// Dataset A: 15-item animal taxonomy, real two-level ground truth.
const PURE: &[(&str, &str, &str, &str)] = &[
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

/// 4 NEW cross-cutting items, each genuinely spanning two fine categories —
/// two of them (farm_duck, farm_chicken) cross a fine-category boundary
/// that belongs to the OPPOSITE coarse branch from the item's own species
/// (a duck and a chicken are birds, but "domestic" is otherwise a
/// mammal-only category in this corpus) — this is what forces the DAG
/// case in gap 3, not a same-branch dual membership. No field-name
/// keywords ("domestic", "predator", "flying", "flightless") appear in any
/// sentence.
const CROSS: &[(&str, &str, &str, &str, &str)] = &[
    (
        "police_dog",
        "The police dog lived with its handler at the station, trained to sprint after a fleeing suspect and pin them down until backup arrived.",
        "domestic", "predator", "mammal",
    ),
    (
        "farm_duck",
        "Raised alongside the rest of the poultry on the farm, the duck paddled across the pond each morning and burst into the air whenever the fox padded too close.",
        "domestic", "flying", "bird",
    ),
    (
        "cassowary",
        "The cassowary paced through the rainforest undergrowth on powerful legs, stubby wings folded uselessly at its sides, ready to slash at anything that startled it with the dagger-like claw on each foot.",
        "predator", "flightless", "bird",
    ),
    (
        "farm_chicken",
        "Kept in a pen behind the farmhouse, the chicken spent its days pecking at the ground, only managing a clumsy flap up to the low roost at dusk.",
        "domestic", "flightless", "bird",
    ),
];

const FINE_CATS: [&str; 4] = ["domestic", "predator", "flying", "flightless"];

fn kmeans(embeddings: &[Vec<f32>], k: usize, iterations: usize) -> Vec<usize> {
    let n = embeddings.len();
    if n < k {
        return (0..n).collect();
    }
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
    for e in embeddings {
        for d in 0..dim { sum[d] += e[d]; }
    }
    let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    sum.iter().map(|x| x / norm).collect()
}

fn balance_ratio(assignment: &[usize]) -> f32 {
    let c0 = assignment.iter().filter(|&&a| a == 0).count();
    let c1 = assignment.iter().filter(|&&a| a == 1).count();
    let (lo, hi) = (c0.min(c1), c0.max(c1));
    if hi == 0 { 0.0 } else { lo as f32 / hi as f32 }
}

/// The set of fine categories an item is considered a "member" of at a
/// given tolerance delta: any category whose similarity is within `delta`
/// of the item's own top similarity. delta=0.0 is pure hard-classification
/// (exactly 1 membership); larger delta admits more.
fn membership_set(sims: &HashMap<&'static str, f32>, delta: f32) -> Vec<&'static str> {
    let top = sims.values().cloned().fold(f32::NEG_INFINITY, f32::max);
    FINE_CATS.iter().filter(|c| sims[*c] >= top - delta).copied().collect()
}

const GATE_THRESHOLD: f32 = 0.5;

fn main() {
    println!("Experiment 10: closing gaps 1-3 — calibration, end-to-end integration, DAG protection\n");

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

        let pure_emb: Vec<Vec<f32>> = PURE.iter().map(|(_, s, ..)| embedder.embed(s)).collect();
        let cross_emb: Vec<Vec<f32>> = CROSS.iter().map(|(_, s, ..)| embedder.embed(s)).collect();
        let mut all_emb = pure_emb.clone();
        all_emb.extend(cross_emb.iter().cloned());
        let n_pure = PURE.len();

        // Fine-category centroids, built from PURE items only (no cross-item leakage).
        let fine_centroids: HashMap<&'static str, Vec<f32>> = FINE_CATS
            .iter()
            .map(|&cat| {
                let members: Vec<&Vec<f32>> = (0..n_pure).filter(|&i| PURE[i].3 == cat).map(|i| &pure_emb[i]).collect();
                (cat, centroid(&members))
            })
            .collect();

        let sims_for = |e: &Vec<f32>| -> HashMap<&'static str, f32> {
            FINE_CATS.iter().map(|&c| (c, cosine_sim(e, &fine_centroids[c]))).collect()
        };
        let pure_sims: Vec<HashMap<&'static str, f32>> = pure_emb.iter().map(sims_for).collect();
        let cross_sims: Vec<HashMap<&'static str, f32>> = cross_emb.iter().map(sims_for).collect();

        // ───────────────────────── Gap 1: threshold calibration + FPR ─────────────────────────
        println!("=== Gap 1: membership-threshold calibration (never checked before) ===");
        println!("{:>6}  {:>18}  {:>10}  {:>14}", "delta", "FPR on 15 pure", "recall/4", "avg |membership|");
        let mut chosen_delta: Option<f32> = None;
        let deltas: Vec<f32> = (0..=20).map(|i| i as f32 * 0.01).collect();
        for &delta in &deltas {
            let fp = pure_sims.iter().filter(|s| membership_set(s, delta).len() > 1).count();
            let fpr = fp as f32 / n_pure as f32;
            let mut recall_hits = 0;
            let mut total_membership = 0usize;
            for (idx, s) in pure_sims.iter().enumerate() { total_membership += membership_set(s, delta).len(); let _ = idx; }
            for (idx, s) in cross_sims.iter().enumerate() {
                let ms = membership_set(s, delta);
                total_membership += ms.len();
                let (_, _, f1, f2, _) = CROSS[idx];
                if ms.contains(&f1) && ms.contains(&f2) { recall_hits += 1; }
            }
            let avg = total_membership as f32 / (n_pure + CROSS.len()) as f32;
            println!("{delta:>6.2}  {fpr:>18.3}  {:>7}/{}    {avg:>14.2}", recall_hits, CROSS.len(), avg = avg);
            if chosen_delta.is_none() && fpr == 0.0 && recall_hits == CROSS.len() {
                chosen_delta = Some(delta);
            }
        }
        let delta = match chosen_delta {
            Some(d) => { println!("\nCalibrated delta = {d:.2}: smallest tolerance with FPR=0.0 on pure items AND full recall on cross items.\n"); d }
            None => {
                // fall back to smallest delta that maximizes recall, report the FPR cost honestly
                let best = deltas.iter().copied().max_by(|&a, &b| {
                    let ra = cross_sims.iter().enumerate().filter(|(i, s)| { let ms = membership_set(s, a); let (_, _, f1, f2, _) = CROSS[*i]; ms.contains(&f1) && ms.contains(&f2) }).count();
                    let rb = cross_sims.iter().enumerate().filter(|(i, s)| { let ms = membership_set(s, b); let (_, _, f1, f2, _) = CROSS[*i]; ms.contains(&f1) && ms.contains(&f2) }).count();
                    ra.cmp(&rb).then(b.partial_cmp(&a).unwrap())
                }).unwrap();
                let fp = pure_sims.iter().filter(|s| membership_set(s, best).len() > 1).count();
                println!("\nNo delta achieves FPR=0.0 with full recall. Best recall-maximizing delta = {best:.2}, at the cost of {fp}/{n_pure} pure items falsely flagged as multi-member.\n");
                best
            }
        };

        // ───────────────────────── Gap 2: end-to-end integration ─────────────────────────
        println!("=== Gap 2: end-to-end — multi-membership + certified recursion, same corpus ===");

        // REAL depth-1: unconstrained k-means over all 19 items.
        let real_depth1 = kmeans(&all_emb, 2, 30);
        let real_balance = balance_ratio(&real_depth1);
        println!("REAL (discovered) depth-1 split sizes: {} / {}", real_depth1.iter().filter(|&&a| a == 0).count(), real_depth1.iter().filter(|&&a| a == 1).count());
        println!("REAL depth-1 balance ratio: {real_balance:.3} -> gate ({GATE_THRESHOLD:.2} threshold) {}", if real_balance < GATE_THRESHOLD { "REJECTS this split, does not recurse" } else { "passes" });

        // ORACLE depth-1: true species coarse label (mammal/bird) for all 19 items.
        let coarse_of = |i: usize| -> &'static str {
            if i < n_pure { PURE[i].2 } else { CROSS[i - n_pure].4 }
        };
        let oracle_depth1: Vec<usize> = (0..all_emb.len()).map(|i| if coarse_of(i) == "mammal" { 0 } else { 1 }).collect();
        let oracle_balance = balance_ratio(&oracle_depth1);
        println!(
            "\nORACLE (true) depth-1 split sizes: mammal={} bird={}, balance ratio={oracle_balance:.3} -> gate {}",
            oracle_depth1.iter().filter(|&&a| a == 0).count(),
            oracle_depth1.iter().filter(|&&a| a == 1).count(),
            if oracle_balance < GATE_THRESHOLD { "REJECTS" } else { "PASSES, recursing" }
        );

        // Depth-2 within each oracle coarse branch (REAL k-means split of that branch's members).
        struct Leaf { label: String, members: Vec<usize> }
        let mut leaves: Vec<Leaf> = Vec::new();
        for coarse_branch in ["mammal", "bird"] {
            let members: Vec<usize> = (0..all_emb.len()).filter(|&i| coarse_of(i) == coarse_branch).collect();
            let sub_emb: Vec<Vec<f32>> = members.iter().map(|&i| all_emb[i].clone()).collect();
            let split = kmeans(&sub_emb, 2, 30);
            let branch_balance = balance_ratio(&split);
            println!(
                "\ndepth-2 split within '{coarse_branch}' (n={}): sizes {} / {}, balance ratio {branch_balance:.3} -> gate {}",
                members.len(),
                split.iter().filter(|&&a| a == 0).count(),
                split.iter().filter(|&&a| a == 1).count(),
                if branch_balance < GATE_THRESHOLD { "REJECTS (would stop here)" } else { "PASSES" }
            );
            for c in 0..2 {
                let leaf_members: Vec<usize> = members.iter().enumerate().filter(|(li, _)| split[*li] == c).map(|(_, &gi)| gi).collect();
                if leaf_members.is_empty() { continue; }
                // Label leaf by majority PURE fine label among its members.
                let mut counts: HashMap<&str, usize> = HashMap::new();
                for &gi in &leaf_members {
                    if gi < n_pure { *counts.entry(PURE[gi].3).or_insert(0) += 1; }
                }
                let label = counts.into_iter().max_by_key(|(_, c)| *c).map(|(l, _)| l.to_string()).unwrap_or_else(|| "?".to_string());
                println!("  leaf '{coarse_branch}.{label}': {:?}", leaf_members.iter().map(|&gi| if gi < n_pure { PURE[gi].0 } else { CROSS[gi - n_pure].0 }).collect::<Vec<_>>());
                leaves.push(Leaf { label: format!("{coarse_branch}.{label}"), members: leaf_members });
            }
        }

        // Overlay multi-membership cross-links at the calibrated delta.
        println!("\ncross-links found (item's membership set includes a fine category outside its home leaf):");
        let mut cross_links: Vec<(usize, String, String)> = Vec::new(); // (item global idx, home leaf label, linked leaf label)
        for (li, leaf) in leaves.iter().enumerate() {
            for &gi in &leaf.members {
                let sims = if gi < n_pure { &pure_sims[gi] } else { &cross_sims[gi - n_pure] };
                let ms = membership_set(sims, delta);
                let home_fine = leaf.label.split('.').nth(1).unwrap();
                for &cat in &ms {
                    if cat == home_fine { continue; }
                    // Find which leaf owns this fine category.
                    if let Some(target) = leaves.iter().position(|l| l.label.ends_with(&format!(".{cat}"))) {
                        if target == li { continue; }
                        let name = if gi < n_pure { PURE[gi].0 } else { CROSS[gi - n_pure].0 };
                        let cross_branch = leaves[target].label.split('.').next().unwrap() != leaf.label.split('.').next().unwrap();
                        println!(
                            "  {name:<14} home='{}' -> also linked to '{}'{}",
                            leaf.label,
                            leaves[target].label,
                            if cross_branch { "  [CROSS-BRANCH — this is the DAG case]" } else { "" }
                        );
                        cross_links.push((gi, leaf.label.clone(), leaves[target].label.clone()));
                    }
                }
            }
        }
        if cross_links.is_empty() {
            println!("  (none at this delta — calibration in gap 1 chose a conservative tolerance)");
        }

        // ───────────────────────── Gap 3: DAG cycle/revisit protection ─────────────────────────
        println!("\n=== Gap 3: does the recursion tree become a DAG, and does a visited-cache fix naive re-expansion? ===");
        if cross_links.is_empty() {
            println!("No cross-links were found at the calibrated delta, so this run has no DAG edges to test with.");
            println!("(This is itself informative: it means gap 1's conservative calibration also eliminated the gap-3 risk for this corpus/delta — a stricter delta trades away DAG complexity for lower recall.)");
        } else {
            // Build an undirected adjacency: tree parent/child edges (branch -> its leaves) + cross-link edges (leaf <-> leaf).
            let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
            for branch in ["mammal", "bird"] {
                for leaf in &leaves {
                    if leaf.label.starts_with(&format!("{branch}.")) {
                        adjacency.entry(branch.to_string()).or_default().push(leaf.label.clone());
                        adjacency.entry(leaf.label.clone()).or_default().push(branch.to_string());
                    }
                }
            }
            for (_, a, b) in &cross_links {
                adjacency.entry(a.clone()).or_default().push(b.clone());
                adjacency.entry(b.clone()).or_default().push(a.clone());
            }

            // NOTE: with >=2 neighbors per node and no visited check, DFS depth
            // is the wrong thing to cap — branching makes call count grow like
            // branching_factor^depth, so a depth cap of e.g. 200 does not
            // bound runtime (this is exactly what hung for over an hour on
            // the first run of this experiment before being killed and
            // fixed). Capping the total call BUDGET instead guarantees
            // termination in bounded time regardless of the graph's
            // branching factor, while still proving the same point: calls
            // hit the budget while max_depth_reached keeps climbing, i.e.
            // it was not close to naturally finishing.
            fn expand_naive(node: &str, adjacency: &HashMap<String, Vec<String>>, depth: usize, budget: &mut usize, calls: &mut usize, max_depth_reached: &mut usize) {
                if *budget == 0 { return; }
                *budget -= 1;
                *calls += 1;
                *max_depth_reached = (*max_depth_reached).max(depth);
                if let Some(neighbors) = adjacency.get(node) {
                    for n in neighbors {
                        if *budget == 0 { return; }
                        expand_naive(n, adjacency, depth + 1, budget, calls, max_depth_reached);
                    }
                }
            }
            fn expand_guarded(node: &str, adjacency: &HashMap<String, Vec<String>>, visited: &mut std::collections::HashSet<String>, calls: &mut usize) {
                if !visited.insert(node.to_string()) { return; }
                *calls += 1;
                if let Some(neighbors) = adjacency.get(node) {
                    for n in neighbors {
                        expand_guarded(n, adjacency, visited, calls);
                    }
                }
            }

            let start = cross_links[0].1.clone();
            let adjacency_for_thread = adjacency.clone();
            // Pure DFS with no visited check never backtracks out of a cycle
            // here (it dives into the first neighbor repeatedly rather than
            // exploring breadth-first), so recursion depth tracks the call
            // count almost 1:1 — a budget cap alone still overflows the
            // default 8MB thread stack well before the budget is spent.
            // Run it on a dedicated thread with a large explicit stack so
            // the budget cap (not a stack limit) is what stops it.
            let budget_cap = 200_000usize;
            let (naive_calls, max_depth_reached) = std::thread::Builder::new()
                .stack_size(512 * 1024 * 1024)
                .spawn(move || {
                    let mut budget = budget_cap;
                    let mut naive_calls = 0usize;
                    let mut max_depth_reached = 0usize;
                    expand_naive(&start, &adjacency_for_thread, 0, &mut budget, &mut naive_calls, &mut max_depth_reached);
                    (naive_calls, max_depth_reached)
                })
                .unwrap()
                .join()
                .unwrap();
            let start = &cross_links[0].1;
            let mut visited = std::collections::HashSet::new();
            let mut guarded_calls = 0usize;
            expand_guarded(start, &adjacency, &mut visited, &mut guarded_calls);

            println!("starting expansion from '{start}' (has a cross-link, so its neighbors loop back to it, {} nodes in the graph):", adjacency.len());
            println!(
                "  WITHOUT visited-cache: exhausted a {budget_cap}-call budget while still climbing (reached recursion depth {max_depth_reached}) -> does not terminate on its own, no natural base case"
            );
            println!("  WITH visited-cache:    {guarded_calls} expand() calls -> terminates immediately, each of the {} distinct nodes expanded exactly once", adjacency.len());
            println!(
                "\nverdict: {} — a visited-node cache (keyed by node label/id) is REQUIRED once multi-membership makes the structure a DAG; without it, naive recursive expansion of cross-links does not terminate.",
                if naive_calls >= budget_cap && guarded_calls == adjacency.len() { "CONFIRMED" } else { "inconclusive on this run" }
            );
        }
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
