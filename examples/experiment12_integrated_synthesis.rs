//! Experiment 12 — the integrated end-to-end synthesis mission: does
//! "preserve full membership, certify, recurse only when trustworthy"
//! outperform properly-calibrated raw-cosine baselines on a corpus with
//! BOTH real hierarchy and cross-cutting concepts, built specifically for
//! this test (not reusing Dataset A or D alone)?
//!
//! Scope note, stated up front rather than silently: this experiment
//! covers the mission's Sections 4-13, 15, and 17-19 (integrated dataset,
//! three baselines including the critical multi-membership-cosine
//! fairness baseline, membership calibration, certification, certified
//! recursion, DAG protection, ARI/NMI, context ablation, a stopping
//! test). It deliberately SKIPS Section 16 (genuine self-supervised JEPA)
//! per the mission's own instruction not to spend the majority of effort
//! there before this simpler synthesis is tested, and Section 14
//! (reference selection) does not map cleanly onto this architecture: the
//! current design computes field centroids from ALL pure exemplars, not
//! a single reference item, so there is no single "which reference"
//! parameter of the kind Iterations 1-3 falsified. That question was
//! already answered for the per-reference mechanism (falsified); it does
//! not re-open for a centroid-based design without inventing a new
//! mechanism, which is out of scope here.
//!
//! Dataset: a 20-item Vehicle ontology — 3 coarse fields (Mechanical,
//! Driver, Economic), 2 fine subfields each (6 total), 15 pure
//! single-membership items (matching Dataset A's scale) plus 5 NEW
//! cross-cutting concepts spanning all 3 possible coarse-field pairs, no
//! field-name keywords in any sentence.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment12_integrated_synthesis

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;
use std::collections::HashMap;

// ─────────────────────────────── Dataset ───────────────────────────────

/// (name, bare term, contextual sentence, coarse, fine)
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

/// (name, bare term, contextual sentence, field1, field2, primary_coarse_for_tree_placement)
const CROSS: &[(&str, &str, &str, &str, &str, &str)] = &[
    ("hybrid_battery", "hybrid battery", "The hybrid battery works alongside the engine to reduce gasoline use, though replacing it years later can be a costly repair.", "mechanical", "economic", "mechanical"),
    ("fuel_efficiency", "fuel efficiency", "Better fuel efficiency means burning less gasoline per mile, which adds up to real savings at the pump over time.", "mechanical", "economic", "mechanical"),
    ("driver_assistance_system", "driver assistance system", "Sensors built into the car watch the road and can nudge the wheel if it starts drifting out of its lane.", "mechanical", "driver", "driver"),
    ("regenerative_braking", "regenerative braking", "Easing off the pedal on the hybrid gently slows the car while feeding energy back into the battery instead of wasting it as heat.", "mechanical", "driver", "driver"),
    ("warranty_coverage", "warranty coverage", "A multi-year warranty means a sudden breakdown on the road won't leave the owner facing a large repair bill out of pocket.", "driver", "economic", "economic"),
];

const FINE_CATS: [&str; 6] = ["propulsion", "energy", "controls", "assistance", "ownership", "acquisition"];

/// CROSS items' ground truth is stated at the COARSE level (e.g.
/// "mechanical, economic" — matching the mission's own framing of
/// cross-cutting concepts as spanning coarse fields, not a specific fine
/// subcategory), but membership_set operates on the 6 FINE categories.
/// Recall must therefore check whether the membership set's fine
/// categories, mapped up to their coarse parent, cover both true coarse
/// fields — comparing fine-category strings directly against coarse
/// labels (an earlier bug in this file) can never match and silently
/// forces recall to 0 regardless of what the embedding actually encodes.
fn fine_to_coarse(fine: &str) -> &'static str {
    match fine {
        "propulsion" | "energy" => "mechanical",
        "controls" | "assistance" => "driver",
        "ownership" | "acquisition" => "economic",
        _ => unreachable!(),
    }
}

// ─────────────────────────────── Shared primitives ───────────────────────────────

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

fn purity_k(assignment: &[usize], k: usize, labels: &[&str]) -> f32 {
    let n = assignment.len();
    let mut correct = 0usize;
    for c in 0..k {
        let mut counts = HashMap::new();
        for i in 0..n { if assignment[i] == c { *counts.entry(labels[i]).or_insert(0usize) += 1; } }
        correct += counts.values().copied().max().unwrap_or(0);
    }
    correct as f32 / n as f32
}

fn comb2(x: usize) -> f64 { if x < 2 { 0.0 } else { (x as f64) * ((x - 1) as f64) / 2.0 } }

/// Adjusted Rand Index between two partitions (arbitrary cluster counts).
fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len();
    let mut table: HashMap<(usize, usize), usize> = HashMap::new();
    let mut row: HashMap<usize, usize> = HashMap::new();
    let mut col: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        *table.entry((a[i], b[i])).or_insert(0) += 1;
        *row.entry(a[i]).or_insert(0) += 1;
        *col.entry(b[i]).or_insert(0) += 1;
    }
    let sum_table: f64 = table.values().map(|&v| comb2(v)).sum();
    let sum_row: f64 = row.values().map(|&v| comb2(v)).sum();
    let sum_col: f64 = col.values().map(|&v| comb2(v)).sum();
    let comb_n = comb2(n);
    if comb_n == 0.0 { return 1.0; }
    let expected = sum_row * sum_col / comb_n;
    let max_index = 0.5 * (sum_row + sum_col);
    if (max_index - expected).abs() < 1e-12 { return 1.0; }
    (sum_table - expected) / (max_index - expected)
}

/// Normalized Mutual Information (sqrt-normalized) between two partitions.
fn normalized_mutual_info(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len();
    let nf = n as f64;
    let mut table: HashMap<(usize, usize), usize> = HashMap::new();
    let mut row: HashMap<usize, usize> = HashMap::new();
    let mut col: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        *table.entry((a[i], b[i])).or_insert(0) += 1;
        *row.entry(a[i]).or_insert(0) += 1;
        *col.entry(b[i]).or_insert(0) += 1;
    }
    let mut mi = 0.0;
    for (&(ai, bj), &nij) in table.iter() {
        let nij = nij as f64;
        let ac = row[&ai] as f64;
        let bc = col[&bj] as f64;
        mi += (nij / nf) * ((nij * nf) / (ac * bc)).ln();
    }
    let h_a: f64 = row.values().map(|&v| { let p = v as f64 / nf; -p * p.ln() }).sum();
    let h_b: f64 = col.values().map(|&v| { let p = v as f64 / nf; -p * p.ln() }).sum();
    if h_a <= 1e-12 || h_b <= 1e-12 { return if mi.abs() < 1e-9 { 1.0 } else { 0.0 }; }
    (mi / (h_a * h_b).sqrt()).max(0.0)
}

fn label_ids(labels: &[&str]) -> Vec<usize> {
    let mut map: HashMap<&str, usize> = HashMap::new();
    labels.iter().map(|&l| { let next = map.len(); *map.entry(l).or_insert(next) }).collect()
}

// ─────────────────────────────── Pipeline ───────────────────────────────

struct PipelineResult {
    representation: &'static str,
    baseline_a_purity: f32,
    flat_kmeans_purity: f32,
    flat_kmeans_ari: f64,
    flat_kmeans_nmi: f64,
    calibration_delta: f32,
    calibration_fpr: f32,
    calibration_recall: usize,
    branch_reports: Vec<(String, usize, f32, f32, f64, f64, bool)>, // name, n, balance, purity, ari, nmi, gate_pass
    cross_links: Vec<(String, String, String)>,
    dag_nodes: usize,
    dag_naive_calls: usize,
    dag_guarded_calls: usize,
    dag_max_depth_reached: usize,
}

fn run_pipeline<F: Fn(&str) -> Vec<f32>>(representation: &'static str, embed: F) -> PipelineResult {
    let n_pure = PURE.len();
    let n_cross = CROSS.len();
    let pure_emb: Vec<Vec<f32>> = PURE.iter().map(|(_, bare, ctx, ..)| embed(if representation == "bare" { bare } else { ctx })).collect();
    let cross_emb: Vec<Vec<f32>> = CROSS.iter().map(|(_, bare, ctx, ..)| embed(if representation == "bare" { bare } else { ctx })).collect();
    let mut all_emb = pure_emb.clone();
    all_emb.extend(cross_emb.iter().cloned());
    let n = n_pure + n_cross;

    let fine_true: Vec<&str> = PURE.iter().map(|(_, _, _, _, f)| *f).collect();
    let coarse_of = |i: usize| -> &'static str {
        if i < n_pure { PURE[i].3 } else { CROSS[i - n_pure].5 }
    };

    // Fine centroids built from the 15 pure items only.
    let fine_centroids: HashMap<&'static str, Vec<f32>> = FINE_CATS.iter().map(|&cat| {
        let members: Vec<&Vec<f32>> = (0..n_pure).filter(|&i| PURE[i].4 == cat).map(|i| &pure_emb[i]).collect();
        (cat, centroid(&members))
    }).collect();
    let sims_for = |e: &Vec<f32>| -> HashMap<&'static str, f32> {
        FINE_CATS.iter().map(|&c| (c, cosine_sim(e, &fine_centroids[c]))).collect()
    };
    let pure_sims: Vec<HashMap<&'static str, f32>> = pure_emb.iter().map(sims_for).collect();
    let cross_sims: Vec<HashMap<&'static str, f32>> = cross_emb.iter().map(sims_for).collect();

    if std::env::var("DEBUG_SIMS").is_ok() {
        for i in [0usize, 5, 10] {
            let mut v: Vec<(&str, f32)> = pure_sims[i].iter().map(|(k, v)| (*k, *v)).collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!("DEBUG [{representation}] PURE  {:<20} true={:<12} sims={v:?}", PURE[i].0, PURE[i].4);
        }
        for i in 0..cross_sims.len() {
            let mut v: Vec<(&str, f32)> = cross_sims[i].iter().map(|(k, v)| (*k, *v)).collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!("DEBUG [{representation}] CROSS {:<20} true=({},{}) sims={v:?}", CROSS[i].0, CROSS[i].3, CROSS[i].4);
        }
    }

    // ── Baseline A: raw cosine argmax over the 6 known fine cells ──
    let argmax_cat = |sims: &HashMap<&'static str, f32>| -> &'static str {
        *sims.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    };
    let mut a_hits = 0usize;
    for (i, s) in pure_sims.iter().enumerate() { if argmax_cat(s) == fine_true[i] { a_hits += 1; } }
    for (i, s) in cross_sims.iter().enumerate() {
        let (_, _, _, f1, f2, _) = CROSS[i];
        let top = argmax_cat(s);
        if top == f1 || top == f2 { a_hits += 1; } // best-case credit, Iteration 8 style
    }
    let baseline_a_purity = a_hits as f32 / n as f32;

    // ── Baseline B: flat hard k-means (k=6) over ALL 20 items, no hierarchy given ──
    let flat_assignment = kmeans(&all_emb, 6, 40);
    let mut flat_labels: Vec<&str> = fine_true.clone();
    flat_labels.extend(CROSS.iter().map(|(_, _, _, f1, ..)| *f1)); // credit toward first true field only, for a single-partition metric
    let flat_true_ids = label_ids(&flat_labels);
    let flat_kmeans_purity = purity_k(&flat_assignment, 6, &flat_labels);
    let flat_kmeans_ari = adjusted_rand_index(&flat_assignment, &flat_true_ids);
    let flat_kmeans_nmi = normalized_mutual_info(&flat_assignment, &flat_true_ids);

    // ── Baseline C: calibrated multi-membership cosine against the 6 KNOWN fine cells ──
    let membership_set = |sims: &HashMap<&'static str, f32>, delta: f32| -> Vec<&'static str> {
        let top = sims.values().cloned().fold(f32::NEG_INFINITY, f32::max);
        FINE_CATS.iter().filter(|c| sims[*c] >= top - delta).copied().collect()
    };
    let coarse_covered = |ms: &[&'static str], f1: &str, f2: &str| -> bool {
        let coarses: std::collections::HashSet<&str> = ms.iter().map(|c| fine_to_coarse(c)).collect();
        coarses.contains(f1) && coarses.contains(f2)
    };
    // Sweep, print every point (auditable, not a black-box pick), then
    // choose the SMALLEST delta achieving the best recall observed across
    // the whole sweep — not the largest, which would pick the loosest
    // (worst-FPR) threshold among ties and was a real bug in an earlier
    // version of this experiment.
    let sweep: Vec<(f32, f32, usize)> = (0..=30).map(|i| {
        let delta = i as f32 * 0.01;
        let fp = pure_sims.iter().filter(|s| membership_set(s, delta).len() > 1).count();
        let fpr = fp as f32 / n_pure as f32;
        let recall = cross_sims.iter().enumerate().filter(|(i, s)| {
            let ms = membership_set(s, delta);
            let (_, _, _, f1, f2, _) = CROSS[*i];
            coarse_covered(&ms, f1, f2)
        }).count();
        (delta, fpr, recall)
    }).collect();
    if std::env::var("DEBUG_SIMS").is_ok() {
        println!("DEBUG [{representation}] calibration sweep (delta, fpr, recall):");
        for (d, fpr, recall) in &sweep { println!("  {d:.2}  fpr={fpr:.3}  recall={recall}/{n_cross}"); }
    }
    let max_recall = sweep.iter().map(|(_, _, r)| *r).max().unwrap();
    let (calibration_delta, calibration_fpr, calibration_recall) = sweep.iter()
        .find(|(_, _, r)| *r == max_recall)
        .map(|&(d, fpr, r)| (d, fpr, r))
        .unwrap();

    // ── Full pipeline: oracle coarse (3 known top-level cells) -> REAL fine discovery -> gate -> multi-membership overlay ──
    struct Leaf { label: String, coarse: &'static str, members: Vec<usize> }
    let mut leaves: Vec<Leaf> = Vec::new();
    let mut branch_reports = Vec::new();
    for coarse in ["mechanical", "driver", "economic"] {
        let members: Vec<usize> = (0..n).filter(|&i| coarse_of(i) == coarse).collect();
        let sub_emb: Vec<Vec<f32>> = members.iter().map(|&i| all_emb[i].clone()).collect();
        let split = kmeans(&sub_emb, 2, 40);
        let bal = balance_ratio_k(&split, 2);
        let gate_pass = bal >= 0.5;

        // purity / ARI / NMI on the PURE-only subset of this branch (cross items excluded — best-case credit used only for the printed purity, not ARI/NMI which need a single ground-truth partition)
        let pure_members: Vec<usize> = members.iter().copied().filter(|&i| i < n_pure).collect();
        let pure_local_assignment: Vec<usize> = pure_members.iter().map(|&gi| {
            let local_i = members.iter().position(|&m| m == gi).unwrap();
            split[local_i]
        }).collect();
        let pure_local_labels: Vec<&str> = pure_members.iter().map(|&gi| PURE[gi].4).collect();
        let branch_purity = if pure_members.is_empty() { 0.0 } else { purity_k(&pure_local_assignment, 2, &pure_local_labels) };
        let pure_true_ids = label_ids(&pure_local_labels);
        let ari = if pure_members.len() > 1 { adjusted_rand_index(&pure_local_assignment, &pure_true_ids) } else { 1.0 };
        let nmi = if pure_members.len() > 1 { normalized_mutual_info(&pure_local_assignment, &pure_true_ids) } else { 1.0 };

        branch_reports.push((coarse.to_string(), members.len(), bal, branch_purity, ari, nmi, gate_pass));

        for c in 0..2 {
            let leaf_members: Vec<usize> = members.iter().enumerate().filter(|(li, _)| split[*li] == c).map(|(_, &gi)| gi).collect();
            if leaf_members.is_empty() { continue; }
            let mut counts: HashMap<&str, usize> = HashMap::new();
            for &gi in &leaf_members { if gi < n_pure { *counts.entry(PURE[gi].4).or_insert(0) += 1; } }
            let label = counts.into_iter().max_by_key(|(_, c)| *c).map(|(l, _)| l.to_string()).unwrap_or_else(|| "?".to_string());
            leaves.push(Leaf { label: format!("{coarse}.{label}"), coarse, members: leaf_members });
        }
    }

    // Cross-link detection at the calibrated delta.
    let mut cross_links: Vec<(String, String, String)> = Vec::new();
    for (li, leaf) in leaves.iter().enumerate() {
        for &gi in &leaf.members {
            let sims = if gi < n_pure { &pure_sims[gi] } else { &cross_sims[gi - n_pure] };
            let ms = membership_set(sims, calibration_delta);
            let home_fine = leaf.label.split('.').nth(1).unwrap();
            for &cat in &ms {
                if cat == home_fine { continue; }
                if let Some(target) = leaves.iter().position(|l| l.label.ends_with(&format!(".{cat}"))) {
                    if target == li { continue; }
                    let name = if gi < n_pure { PURE[gi].0 } else { CROSS[gi - n_pure].0 };
                    cross_links.push((name.to_string(), leaf.label.clone(), leaves[target].label.clone()));
                }
            }
        }
    }

    // DAG check: adjacency = coarse->leaf tree edges + cross-link edges.
    let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
    for leaf in &leaves {
        adjacency.entry(leaf.coarse.to_string()).or_default().push(leaf.label.clone());
        adjacency.entry(leaf.label.clone()).or_default().push(leaf.coarse.to_string());
    }
    for (_, a, b) in &cross_links {
        adjacency.entry(a.clone()).or_default().push(b.clone());
        adjacency.entry(b.clone()).or_default().push(a.clone());
    }
    let dag_nodes = adjacency.len();

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
            for n in neighbors { expand_guarded(n, adjacency, visited, calls); }
        }
    }

    let (dag_naive_calls, dag_max_depth_reached, dag_guarded_calls) = if let Some((_, start, _)) = cross_links.first() {
        let start = start.clone();
        let adjacency_for_thread = adjacency.clone();
        let budget_cap = 200_000usize;
        let (naive_calls, max_depth) = std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(move || {
                let mut budget = budget_cap;
                let mut calls = 0usize;
                let mut depth = 0usize;
                expand_naive(&start, &adjacency_for_thread, 0, &mut budget, &mut calls, &mut depth);
                (calls, depth)
            })
            .unwrap().join().unwrap();
        let start2 = cross_links.first().unwrap().1.clone();
        let mut visited = std::collections::HashSet::new();
        let mut guarded_calls = 0usize;
        expand_guarded(&start2, &adjacency, &mut visited, &mut guarded_calls);
        (naive_calls, max_depth, guarded_calls)
    } else {
        (0, 0, 0)
    };

    PipelineResult {
        representation,
        baseline_a_purity,
        flat_kmeans_purity,
        flat_kmeans_ari,
        flat_kmeans_nmi,
        calibration_delta,
        calibration_fpr,
        calibration_recall,
        branch_reports,
        cross_links,
        dag_nodes,
        dag_naive_calls,
        dag_guarded_calls,
        dag_max_depth_reached,
    }
}

fn print_report(r: &PipelineResult) {
    println!("\n########## Representation: {} ##########", r.representation);
    println!("Baseline A (raw cosine argmax, best-case credit for cross items): purity = {:.3}", r.baseline_a_purity);
    println!("Baseline B (flat hard k-means, k=6, no hierarchy given): purity = {:.3}, ARI = {:.3}, NMI = {:.3}", r.flat_kmeans_purity, r.flat_kmeans_ari, r.flat_kmeans_nmi);
    println!(
        "Baseline C (calibrated multi-membership cosine vs 6 known cells): delta={:.2}, FPR on 15 pure = {:.3}, recall on 5 cross = {}/{}",
        r.calibration_delta, r.calibration_fpr, r.calibration_recall, CROSS.len()
    );
    println!("\nFull pipeline — oracle coarse (3 known cells) -> REAL fine discovery -> certification gate:");
    for (name, n, bal, purity, ari, nmi, gate) in &r.branch_reports {
        println!(
            "  {name:<12} n={n:<3} balance={bal:.3} purity(pure-only)={purity:.3} ARI={ari:.3} NMI={nmi:.3} gate={}",
            if *gate { "PASS" } else { "REJECT" }
        );
    }
    println!("cross-links discovered ({} total):", r.cross_links.len());
    for (name, home, linked) in &r.cross_links {
        println!("  {name:<24} home='{home}' -> also linked to '{linked}'");
    }
    println!(
        "DAG check: {} nodes; naive expansion without visited-cache: {} calls (depth {} reached, budget-capped) vs guarded: {} calls",
        r.dag_nodes, r.dag_naive_calls, r.dag_max_depth_reached, r.dag_guarded_calls
    );
}

fn main() {
    println!("Experiment 12: integrated synthesis mission — multi-membership + certification + certified recursion vs calibrated baselines\n");

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

        let bare_result = run_pipeline("bare", |s| embedder.embed(s));
        let ctx_result = run_pipeline("contextual", |s| embedder.embed(s));

        print_report(&bare_result);
        print_report(&ctx_result);

        println!("\n########## Context ablation (Section 15): bare vs contextual, same pipeline ##########");
        println!("{:<45} {:>10} {:>12}", "metric", "bare", "contextual");
        println!("{:<45} {:>10.3} {:>12.3}", "Baseline A purity", bare_result.baseline_a_purity, ctx_result.baseline_a_purity);
        println!("{:<45} {:>10.3} {:>12.3}", "Baseline B (flat k-means) purity", bare_result.flat_kmeans_purity, ctx_result.flat_kmeans_purity);
        println!("{:<45} {:>10.3} {:>12.3}", "Baseline B ARI", bare_result.flat_kmeans_ari, ctx_result.flat_kmeans_ari);
        println!("{:<45} {:>10.3} {:>12.3}", "Baseline B NMI", bare_result.flat_kmeans_nmi, ctx_result.flat_kmeans_nmi);
        println!("{:<45} {:>10.2} {:>12.2}", "Baseline C calibrated delta", bare_result.calibration_delta, ctx_result.calibration_delta);
        println!("{:<45} {:>10} {:>12}", "Baseline C recall (of 5 cross items)", bare_result.calibration_recall, ctx_result.calibration_recall);
        println!("{:<45} {:>10} {:>12}", "cross-links discovered", bare_result.cross_links.len(), ctx_result.cross_links.len());
        let gates_bare: Vec<bool> = bare_result.branch_reports.iter().map(|b| b.6).collect();
        let gates_ctx: Vec<bool> = ctx_result.branch_reports.iter().map(|b| b.6).collect();
        println!("{:<45} {:>10?} {:>12?}", "per-branch gate decisions", gates_bare, gates_ctx);

        // ── Ablation: does the certification gate matter here? ──
        println!("\n########## Ablation: does the certification gate ever actually reject a branch? ##########");
        for r in [&bare_result, &ctx_result] {
            let rejected: Vec<&String> = r.branch_reports.iter().filter(|b| !b.6).map(|b| &b.0).collect();
            if rejected.is_empty() {
                println!("{}: no branch rejected (balance >= 0.5 everywhere) — on THIS dataset the gate made no observed difference; report honestly rather than claim it mattered.", r.representation);
            } else {
                println!("{}: gate rejected {:?} — recursion into these branches would be blocked in production.", r.representation, rejected);
            }
        }

        // ── Ablation: does multi-membership recover real dual-field concepts, and at what FPR cost? ──
        println!("\n########## Ablation: multi-membership on vs off ##########");
        for r in [&bare_result, &ctx_result] {
            println!(
                "{}: WITHOUT multi-membership, all {} cross-cutting items would silently keep only their single primary field. WITH it (delta={:.2}): {}/{} cross-links recovered, {:.3} false-positive rate on the 15 pure items.",
                r.representation, CROSS.len(), r.calibration_delta, r.calibration_recall, CROSS.len(), r.calibration_fpr
            );
        }

        println!("\n########## Section 19 fairness check: does the full pipeline beat calibrated multi-membership cosine (Baseline C) alone? ##########");
        println!("This is not an apples-to-apples 'which wins' comparison: Baseline C answers 'which of the 6 ALREADY-KNOWN fine cells does this item belong to', assuming those 6 cells are given in advance (matching how CellClassifier already works against a fixed grid). The full pipeline answers a different, harder question: 'starting from only the 3 known COARSE cells, can the 6 fine cells be DISCOVERED unsupervised, and is that discovery trustworthy enough to certify?' Baseline C cannot even ask that question — there is nothing to score against before the fine cells exist. The honest comparison is therefore: how much purity/ARI/NMI does unsupervised fine discovery (full pipeline) lose relative to knowing the fine cells in advance (Baseline C's implicit ceiling)? See the branch-level ARI/NMI above for that gap, per representation.");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
