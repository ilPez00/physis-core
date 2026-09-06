//! Experiment 17 — a genuinely self-supervised JEPA-style predictor,
//! this time targeting the DISCOVERED ontology structure's own
//! representations (train-only cluster centroids) instead of human
//! labels — the fix for Iteration 3's mistake (a supervised predictor
//! that used perspective labels and was never a real JEPA test). Two
//! questions, both new to this track:
//!
//!   1. Does a lightweight predictor trained to bind an item's embedding
//!      to its discovered ontology node's representation GENERALIZE to
//!      held-out items not used to build that node's centroid — i.e. is
//!      the discovered structure a real regularity in the embedding
//!      space, or just a description of the specific items that produced
//!      it?
//!   2. Does held-out generalization (or training loss, or the
//!      generalization gap) separate the "good" splits from the "bad"
//!      ones across the same 6 known real cases used in Iterations
//!      13-14 — a certification signal genuinely different in kind from
//!      the four that already failed (Iterations 9, 11, 13, 14 all
//!      analyzed properties of ONE static clustering; this asks whether
//!      the split is a learnable, generalizing regularity instead).
//!
//! Design discipline, directly answering the mission's JEPA warnings:
//! targets are TRAIN-ONLY cluster centroids (stop-gradient), never a
//! human label; held-out items are never used to build the centroids
//! they're evaluated against; collapse diagnostics (loss, output norm,
//! pairwise cosine) are reported and checked BEFORE trusting any
//! accuracy number, exactly as Iteration 4 established.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment17_jepa_ontology_binding

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

// ─────────────────────────── rank-K low-rank predictor, hand-derived gradients ───────────────────────────

struct Predictor { u: Vec<Vec<f32>>, v: Vec<Vec<f32>>, dim: usize, k: usize }

impl Predictor {
    /// Deterministic init: no RNG, matching this codebase's established style.
    fn new(dim: usize, k: usize) -> Self {
        let mut u = vec![vec![0.0f32; k]; dim];
        let mut v = vec![vec![0.0f32; k]; dim];
        for i in 0..k.min(dim) {
            u[i][i] = 0.1;
            v[i][i] = 0.1;
        }
        Predictor { u, v, dim, k }
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // h = U^T x  (k-dim)
        let mut h = vec![0.0f32; self.k];
        for kk in 0..self.k {
            let mut s = 0.0;
            for d in 0..self.dim { s += self.u[d][kk] * x[d]; }
            h[kk] = s;
        }
        // pred_raw = V h (dim-dim)
        let mut pred_raw = vec![0.0f32; self.dim];
        for d in 0..self.dim {
            let mut s = 0.0;
            for kk in 0..self.k { s += self.v[d][kk] * h[kk]; }
            pred_raw[d] = s;
        }
        (pred_raw, h)
    }

    /// One training step over the full batch (full-batch gradient descent, deterministic).
    /// Returns mean loss for diagnostics.
    fn train_step(&mut self, xs: &[Vec<f32>], own_centroids: &[Vec<f32>], other_centroids: &[Vec<f32>], tau: f32, lr: f32) -> f32 {
        let n = xs.len();
        let mut grad_u = vec![vec![0.0f32; self.k]; self.dim];
        let mut grad_v = vec![vec![0.0f32; self.k]; self.dim];
        let mut total_loss = 0.0f32;

        for i in 0..n {
            let (pred_raw, h) = self.forward(&xs[i]);
            let z_own: f32 = pred_raw.iter().zip(&own_centroids[i]).map(|(a, b)| a * b).sum::<f32>() / tau;
            let z_other: f32 = pred_raw.iter().zip(&other_centroids[i]).map(|(a, b)| a * b).sum::<f32>() / tau;
            let m = z_own.max(z_other);
            let p_own = (z_own - m).exp() / ((z_own - m).exp() + (z_other - m).exp());
            total_loss += -(p_own.max(1e-8)).ln();

            // grad_pred_raw = (1 - p_own)/tau * (other_centroid - own_centroid)
            let coef = (1.0 - p_own) / tau;
            let grad_pred_raw: Vec<f32> = (0..self.dim).map(|d| coef * (other_centroids[i][d] - own_centroids[i][d])).collect();

            // grad_V += grad_pred_raw outer h^T ; grad_h = V^T grad_pred_raw
            let mut grad_h = vec![0.0f32; self.k];
            for d in 0..self.dim {
                for kk in 0..self.k {
                    grad_v[d][kk] += grad_pred_raw[d] * h[kk];
                    grad_h[kk] += self.v[d][kk] * grad_pred_raw[d];
                }
            }
            // grad_U += x outer grad_h^T
            for d in 0..self.dim {
                for kk in 0..self.k {
                    grad_u[d][kk] += xs[i][d] * grad_h[kk];
                }
            }
        }

        let scale = lr / n as f32;
        for d in 0..self.dim {
            for kk in 0..self.k {
                self.u[d][kk] -= scale * grad_u[d][kk];
                self.v[d][kk] -= scale * grad_v[d][kk];
            }
        }
        total_loss / n as f32
    }
}

impl Predictor {
    /// N-way generalization for real data with more than 2 classes:
    /// standard softmax cross-entropy over all N train centroids.
    fn train_step_nway(&mut self, xs: &[Vec<f32>], class_ids: &[usize], centroids: &[Vec<f32>], tau: f32, lr: f32) -> f32 {
        let n = xs.len();
        let n_classes = centroids.len();
        let mut grad_u = vec![vec![0.0f32; self.k]; self.dim];
        let mut grad_v = vec![vec![0.0f32; self.k]; self.dim];
        let mut total_loss = 0.0f32;

        for i in 0..n {
            let (pred_raw, h) = self.forward(&xs[i]);
            let logits: Vec<f32> = centroids.iter().map(|c| pred_raw.iter().zip(c).map(|(a, b)| a * b).sum::<f32>() / tau).collect();
            let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|z| (z - m).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();
            total_loss += -(probs[class_ids[i]].max(1e-8)).ln();

            // dL/dz_c = (p_c - 1{c=own}) / tau ; dL/dpred_raw = sum_c dL/dz_c * centroid_c
            let mut grad_pred_raw = vec![0.0f32; self.dim];
            for c in 0..n_classes {
                let coef = (probs[c] - if c == class_ids[i] { 1.0 } else { 0.0 }) / tau;
                for d in 0..self.dim { grad_pred_raw[d] += coef * centroids[c][d]; }
            }
            let mut grad_h = vec![0.0f32; self.k];
            for d in 0..self.dim {
                for kk in 0..self.k {
                    grad_v[d][kk] += grad_pred_raw[d] * h[kk];
                    grad_h[kk] += self.v[d][kk] * grad_pred_raw[d];
                }
            }
            for d in 0..self.dim {
                for kk in 0..self.k { grad_u[d][kk] += xs[i][d] * grad_h[kk]; }
            }
        }
        let scale = lr / n as f32;
        for d in 0..self.dim {
            for kk in 0..self.k {
                self.u[d][kk] -= scale * grad_u[d][kk];
                self.v[d][kk] -= scale * grad_v[d][kk];
            }
        }
        total_loss / n as f32
    }
}

fn nway_argmax_correct(pred_raw: &[f32], centroids: &[Vec<f32>], true_class: usize) -> bool {
    let best = (0..centroids.len()).max_by(|&a, &b| {
        let za: f32 = pred_raw.iter().zip(&centroids[a]).map(|(x, y)| x * y).sum();
        let zb: f32 = pred_raw.iter().zip(&centroids[b]).map(|(x, y)| x * y).sum();
        za.partial_cmp(&zb).unwrap()
    }).unwrap();
    best == true_class
}

fn run_real_domain_case(embedder: &impl VectorEmbed) {
    use physis_core::ontology::OntologyLoader;
    use std::collections::HashMap;

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
    println!("\n########## Real-data extension: physis-core's actual 5-domain ontology (n={}) ##########", texts.len());
    println!("Motivation: the 6 toy cases above have only 2-5 held-out items each — far too coarse");
    println!("(20-50% resolution per item) to detect a subtle generalization signal. This tests the same");
    println!("mechanism (Question 1: does binding to discovered structure generalize at all?) at real scale.\n");

    let embeddings: Vec<Vec<f32>> = texts.iter().map(|t| embedder.embed(t)).collect();
    let mut domain_ids: HashMap<String, usize> = HashMap::new();
    let labels: Vec<usize> = domains.iter().map(|d| { let next = domain_ids.len(); *domain_ids.entry(d.clone()).or_insert(next) }).collect();
    let n_classes = domain_ids.len();
    let dim = embeddings[0].len();
    let n = embeddings.len();

    // Stratified ~80/20 split: every 5th item (by within-class order) goes to held-out.
    let mut per_class_seen = vec![0usize; n_classes];
    let held_out: Vec<bool> = (0..n).map(|i| {
        let c = labels[i];
        let seen = per_class_seen[c];
        per_class_seen[c] += 1;
        seen % 5 == 0
    }).collect();
    let train_idx: Vec<usize> = (0..n).filter(|&i| !held_out[i]).collect();
    let holdout_idx: Vec<usize> = (0..n).filter(|&i| held_out[i]).collect();
    println!("train={}, held-out={}", train_idx.len(), holdout_idx.len());

    let train_centroids: Vec<Vec<f32>> = (0..n_classes).map(|c| {
        let members: Vec<&Vec<f32>> = train_idx.iter().filter(|&&i| labels[i] == c).map(|&i| &embeddings[i]).collect();
        centroid(&members)
    }).collect();

    let raw_hits = holdout_idx.iter().filter(|&&i| nway_argmax_correct(&embeddings[i], &train_centroids, labels[i])).count();
    println!("raw-embedding held-out accuracy (baseline, no predictor): {:.3}", raw_hits as f32 / holdout_idx.len() as f32);

    let mut predictor = Predictor::new(dim, 16.min(dim));
    let train_xs: Vec<Vec<f32>> = train_idx.iter().map(|&i| embeddings[i].clone()).collect();
    let train_labels: Vec<usize> = train_idx.iter().map(|&i| labels[i]).collect();
    let mut final_loss = 0.0;
    for epoch in 0..800 {
        final_loss = predictor.train_step_nway(&train_xs, &train_labels, &train_centroids, 0.1, 0.3);
        if epoch % 100 == 0 || epoch == 799 { println!("  epoch {epoch}: loss={final_loss:.4}"); }
    }

    let all_pred: Vec<Vec<f32>> = embeddings.iter().map(|e| predictor.forward(e).0).collect();
    let mean_pairwise = |vecs: &[Vec<f32>]| -> f32 {
        let n = vecs.len().min(200); // cap for O(n^2) cost on 730 items
        let mut total = 0.0;
        let mut count = 0;
        for a in 0..n { for b in (a + 1)..n { total += cosine_sim(&vecs[a], &vecs[b]); count += 1; } }
        if count == 0 { 0.0 } else { total / count as f32 }
    };
    println!("collapse check: predicted pairwise cosine (first 200 items) = {:.3}, raw pairwise cosine = {:.3}", mean_pairwise(&all_pred), mean_pairwise(&embeddings));

    let train_hits = train_idx.iter().filter(|&&i| nway_argmax_correct(&all_pred[i], &train_centroids, labels[i])).count();
    let trained_holdout_hits = holdout_idx.iter().filter(|&&i| nway_argmax_correct(&all_pred[i], &train_centroids, labels[i])).count();
    println!(
        "trained-predictor: train accuracy={:.3}, held-out accuracy={:.3}, generalization gap={:.3}",
        train_hits as f32 / train_idx.len() as f32,
        trained_holdout_hits as f32 / holdout_idx.len() as f32,
        (train_hits as f32 / train_idx.len() as f32) - (trained_holdout_hits as f32 / holdout_idx.len() as f32)
    );
    println!("(this only answers Question 1 — does binding generalize at all on real true-labeled data — not Question 2, since no matched real 'known-bad' split exists to compare against)");
}

fn nearest_centroid_correct(pred_raw: &[f32], own_centroid: &[f32], other_centroid: &[f32]) -> bool {
    let z_own: f32 = pred_raw.iter().zip(own_centroid).map(|(a, b)| a * b).sum();
    let z_other: f32 = pred_raw.iter().zip(other_centroid).map(|(a, b)| a * b).sum();
    z_own > z_other
}

/// Deterministic ~30% held-out split, guaranteeing at least one held-out
/// AND one train item per cluster even for very small clusters.
fn holdout_split(assignment: &[usize]) -> Vec<bool> {
    let n = assignment.len();
    let mut held_out = vec![false; n];
    for i in 0..n { if i % 3 == 0 { held_out[i] = true; } }
    for c in 0..2 {
        let members: Vec<usize> = (0..n).filter(|&i| assignment[i] == c).collect();
        let ho_count = members.iter().filter(|&&i| held_out[i]).count();
        if ho_count == 0 && members.len() > 1 { held_out[members[0]] = true; }
        if ho_count == members.len() { held_out[*members.last().unwrap()] = false; }
    }
    held_out
}

struct CaseResult { name: &'static str, correct: bool, raw_holdout_acc: f32, trained_holdout_acc: f32, final_loss: f32, gen_gap: f32, output_norm: f32, pairwise_cos: f32, raw_pairwise_cos: f32 }

fn run_case(name: &'static str, correct: bool, embeddings: Vec<Vec<f32>>, assignment: Vec<usize>) -> CaseResult {
    let dim = embeddings[0].len();
    let held_out = holdout_split(&assignment);
    let train_idx: Vec<usize> = (0..embeddings.len()).filter(|&i| !held_out[i]).collect();
    let holdout_idx: Vec<usize> = (0..embeddings.len()).filter(|&i| held_out[i]).collect();

    let train_centroids: [Vec<f32>; 2] = [0, 1].map(|c| {
        let members: Vec<&Vec<f32>> = train_idx.iter().filter(|&&i| assignment[i] == c).map(|&i| &embeddings[i]).collect();
        centroid(&members)
    });

    // Raw-embedding baseline: nearest TRAIN centroid, no predictor at all.
    let raw_holdout_hits = holdout_idx.iter().filter(|&&i| {
        let own = assignment[i];
        let other = 1 - own;
        nearest_centroid_correct(&embeddings[i], &train_centroids[own], &train_centroids[other])
    }).count();
    let raw_holdout_acc = raw_holdout_hits as f32 / holdout_idx.len() as f32;

    // Train the predictor on TRAIN items only, targets = train-only centroids (stop-gradient).
    let mut predictor = Predictor::new(dim, 8.min(dim));
    let train_xs: Vec<Vec<f32>> = train_idx.iter().map(|&i| embeddings[i].clone()).collect();
    let train_own: Vec<Vec<f32>> = train_idx.iter().map(|&i| train_centroids[assignment[i]].clone()).collect();
    let train_other: Vec<Vec<f32>> = train_idx.iter().map(|&i| train_centroids[1 - assignment[i]].clone()).collect();
    let tau = 0.1;
    let lr = 0.5;
    let mut final_loss = 0.0;
    for _ in 0..150 {
        final_loss = predictor.train_step(&train_xs, &train_own, &train_other, tau, lr);
    }

    // Collapse diagnostics (Iteration 4 discipline): output norm + pairwise cosine among
    // predicted embeddings, compared to raw embeddings' own pairwise cosine.
    let all_pred: Vec<Vec<f32>> = embeddings.iter().map(|e| predictor.forward(e).0).collect();
    let output_norm: f32 = all_pred.iter().map(|p| p.iter().map(|x| x * x).sum::<f32>().sqrt()).sum::<f32>() / all_pred.len() as f32;
    let mean_pairwise = |vecs: &[Vec<f32>]| -> f32 {
        let n = vecs.len();
        let mut total = 0.0;
        let mut count = 0;
        for a in 0..n { for b in (a + 1)..n { total += cosine_sim(&vecs[a], &vecs[b]); count += 1; } }
        if count == 0 { 0.0 } else { total / count as f32 }
    };
    let pairwise_cos = mean_pairwise(&all_pred);
    let raw_pairwise_cos = mean_pairwise(&embeddings);

    // Train accuracy (for the generalization gap) and held-out accuracy, using the TRAINED predictor.
    let train_hits = train_idx.iter().filter(|&&i| {
        let (pred_raw, _) = predictor.forward(&embeddings[i]);
        let own = assignment[i];
        nearest_centroid_correct(&pred_raw, &train_centroids[own], &train_centroids[1 - own])
    }).count();
    let train_acc = train_hits as f32 / train_idx.len() as f32;
    let trained_holdout_hits = holdout_idx.iter().filter(|&&i| {
        let (pred_raw, _) = predictor.forward(&embeddings[i]);
        let own = assignment[i];
        nearest_centroid_correct(&pred_raw, &train_centroids[own], &train_centroids[1 - own])
    }).count();
    let trained_holdout_acc = trained_holdout_hits as f32 / holdout_idx.len() as f32;

    CaseResult {
        name, correct, raw_holdout_acc, trained_holdout_acc, final_loss,
        gen_gap: train_acc - trained_holdout_acc, output_norm, pairwise_cos, raw_pairwise_cos,
    }
}

// ─────────────────────────── the 6 known real cases (Iterations 13-14) ───────────────────────────

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

fn main() {
    println!("Experiment 17: genuine JEPA-style ontology-binding predictor as a certification signal\n");

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
        let a_true: Vec<usize> = DATASET_A.iter().map(|(_, _, l)| if *l == "mammal" { 0 } else { 1 }).collect();
        let a_real = kmeans(&a_emb, 2, 30);

        let bird_emb: Vec<Vec<f32>> = BIRD_BRANCH.iter().map(|(_, s, _)| embedder.embed(s)).collect();
        let bird_real = kmeans(&bird_emb, 2, 30);

        let veh_emb: Vec<Vec<f32>> = VEHICLES.iter().map(|(_, s, ..)| embedder.embed(s)).collect();
        let veh_true: Vec<usize> = VEHICLES.iter().map(|v| if v.2 == "two" { 0 } else { 1 }).collect();

        let econ_emb: Vec<Vec<f32>> = ECONOMIC_BRANCH.iter().map(|(_, s)| embedder.embed(s)).collect();
        let econ_real = kmeans(&econ_emb, 2, 30);

        let mech_emb: Vec<Vec<f32>> = MECHANICAL_BRANCH.iter().map(|(_, s)| embedder.embed(s)).collect();
        let mech_real = kmeans(&mech_emb, 2, 30);

        println!("balance ratios (context): case1={:.3} case2={:.3} case3={:.3} case4={:.3} case5={:.3} case6={:.3}\n",
            balance_ratio(&a_true), balance_ratio(&a_real), balance_ratio(&bird_real), balance_ratio(&veh_true), balance_ratio(&econ_real), balance_ratio(&mech_real));

        let results = vec![
            run_case("1: GOOD-balanced (Dataset A true 8v7)", true, a_emb, a_true),
            run_case("2: BAD-imbalanced (Dataset A real, 13v2)", false, DATASET_A.iter().map(|(_, s, _)| embedder.embed(s)).collect(), a_real),
            run_case("3: BAD-balanced (Iter.10 bird branch, 6v4)", false, bird_emb, bird_real),
            run_case("4: GOOD-imbalanced (true wheel count, 2v10)", true, veh_emb, veh_true),
            run_case("5: BAD-balanced (Iter.12 economic branch)", false, econ_emb, econ_real),
            run_case("6: BAD-imbalanced (Iter.12 mechanical branch)", false, mech_emb, mech_real),
        ];

        println!("{:<48} {:>8} {:>10} {:>10} {:>9} {:>10} {:>10} {:>10}", "case", "raw-acc", "train-acc", "held-acc", "gen-gap", "loss", "out-norm", "pair-cos");
        for r in &results {
            println!(
                "{:<48} {:>8.3} {:>10} {:>10.3} {:>9.3} {:>10.3} {:>10.3} {:>10.3}",
                r.name, r.raw_holdout_acc, "-", r.trained_holdout_acc, r.gen_gap, r.final_loss, r.output_norm, r.pairwise_cos
            );
            println!("    raw pairwise cosine (for collapse comparison): {:.3}", r.raw_pairwise_cos);
        }

        fn separates(scores: &[(bool, f32)]) -> bool {
            let min_good = scores.iter().filter(|(c, _)| *c).map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
            let max_bad = scores.iter().filter(|(c, _)| !*c).map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
            min_good > max_bad
        }
        let raw_scores: Vec<(bool, f32)> = results.iter().map(|r| (r.correct, r.raw_holdout_acc)).collect();
        let trained_scores: Vec<(bool, f32)> = results.iter().map(|r| (r.correct, r.trained_holdout_acc)).collect();
        let loss_scores: Vec<(bool, f32)> = results.iter().map(|r| (r.correct, -r.final_loss)).collect(); // lower loss = better, invert
        let gap_scores: Vec<(bool, f32)> = results.iter().map(|r| (r.correct, -r.gen_gap)).collect(); // smaller gap = better, invert

        println!("\n=== Verdict: does min(correct-case scores) > max(wrong-case scores)? ===");
        println!("raw-embedding held-out accuracy (baseline, no predictor): {}", separates(&raw_scores));
        println!("trained-predictor held-out accuracy:                      {}", separates(&trained_scores));
        println!("final training loss (lower=better, inverted):             {}", separates(&loss_scores));
        println!("generalization gap (smaller=better, inverted):            {}", separates(&gap_scores));
        println!("\n(collapse check: if pair-cos is near 1.0 for the trained predictor while raw pairwise cosine is much lower, the predictor has collapsed to a near-constant output and NONE of the accuracy numbers above should be trusted)");

        run_real_domain_case(&embedder);
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
