// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 4 — does a trained (JEPA-style) predictor improve classification
//! of physis-core's REAL, existing 70-cell ontology, over the raw cosine
//! similarity `CellClassifier` actually uses in production?
//!
//! Prompted directly by the user, after Iteration 3 (toy CAR corpus) showed
//! supervised metric learning can meaningfully beat raw cosine when there is
//! a real label taxonomy and enough data per class — but that toy corpus
//! (21 hand-picked items) had neither the scale nor the real stakes to mean
//! anything for the actual product. This experiment reuses the identical,
//! already-validated predictor architecture and training code from
//! `perspective_discovery_experiment3.rs` (rank-K asymmetric predictor,
//! stop-gradient target, hand-derived InfoNCE gradients — the same design
//! whose shuffled-label control ruled out pure memorization), applied to:
//!
//!   - **Domain task**: 730 real ontology entries, 5 classes (118-169 each) —
//!     the well-powered, honest test.
//!   - **Cell task**: the subset of entries whose (domain,mode) cell has >=5
//!     members (31 of 70 cells; the other 35 cells have exactly 1 entry each
//!     and cannot support a train/test split at all) — the harder, more
//!     realistic test, on real data instead of a hand-built toy taxonomy.
//!
//! Evaluation: stratified 5-fold cross-validation, 1-nearest-neighbor
//! classification — for each held-out test item, find its nearest TRAINING
//! item by cosine similarity (raw embedding for the baseline, predictor
//! output for the trained method) and check whether that neighbor's true
//! label matches. This mirrors `CellClassifier`'s actual production
//! mechanism ("max cosine to any member entry") exactly, so a win here is
//! directly relevant to the real classifier, not an artifact of a
//! differently-shaped toy metric.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example ontology_metric_learning_experiment

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::collections::HashMap;

const RANK: usize = 16;
const EPOCHS: usize = 150;
const LR: f32 = 0.15;
const TAU: f32 = 0.15;
const LAMBDA: f32 = 1e-3;
const K_FOLDS: usize = 5;

struct Predictor {
    dim: usize,
    u: Vec<f32>,
    v: Vec<f32>,
}

impl Predictor {
    fn init(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = 1.0 / (dim as f32).sqrt();
        let mut rnd = |n: usize| (0..n).map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale).collect();
        Predictor { dim, u: rnd(dim * RANK), v: rnd(dim * RANK) }
    }
    fn project(&self, r: &[f32]) -> Vec<f32> {
        let mut z = vec![0.0f32; RANK];
        for (d, rd) in r.iter().enumerate() {
            for (k, zk) in z.iter_mut().enumerate() {
                *zk += self.u[d * RANK + k] * rd;
            }
        }
        z
    }
    fn predict_from_z(&self, z: &[f32]) -> Vec<f32> {
        let mut p = vec![0.0f32; self.dim];
        for (d, pd) in p.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (k, zk) in z.iter().enumerate() {
                acc += self.v[d * RANK + k] * zk;
            }
            *pd = acc;
        }
        p
    }
    fn predict(&self, r: &[f32]) -> Vec<f32> {
        self.predict_from_z(&self.project(r))
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Identical training procedure to Experiment 3 (see that file for the
/// derivation of the gradient), parameterized by an arbitrary label slice
/// instead of a closure, since this is called many times across folds/tasks.
fn train_predictor(train_embeddings: &[Vec<f32>], labels: &[usize], seed: u64) -> Predictor {
    let dim = train_embeddings[0].len();
    let n = train_embeddings.len();
    let mut pred = Predictor::init(dim, seed);

    // Pre-group indices by label for faster positive lookup across epochs.
    let mut by_label: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &l) in labels.iter().enumerate() {
        by_label.entry(l).or_default().push(i);
    }

    for _epoch in 0..EPOCHS {
        let mut grad_u = vec![0.0f32; dim * RANK];
        let mut grad_v = vec![0.0f32; dim * RANK];

        for qi in 0..n {
            let positives: Vec<usize> = by_label[&labels[qi]].iter().copied().filter(|&i| i != qi).collect();
            if positives.is_empty() {
                continue;
            }
            let z = pred.project(&train_embeddings[qi]);
            let p = pred.predict_from_z(&z);

            let candidates: Vec<usize> = (0..n).filter(|&i| i != qi).collect();
            let scores: Vec<f32> = candidates.iter().map(|&i| dot(&p, &train_embeddings[i]) / TAU).collect();
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum::<f32>().max(1e-8);
            let probs: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

            // Mean over positives, not sum: with domain classes up to ~170
            // members, a sum-over-positives gradient (fine at the toy
            // corpus's max of 6) blows up ~30x here and diverges to NaN
            // (confirmed by inspection — this is the fix for that, not a
            // cosmetic change). Standard multi-positive contrastive
            // formulation (SupCon-style): both terms below are
            // convex-combination-of-unit-vectors, so the gradient stays
            // bounded regardless of class size.
            let mut dloss_dp = vec![0.0f32; dim];
            for (ci, &cand_idx) in candidates.iter().enumerate() {
                let coeff = probs[ci];
                for d in 0..dim {
                    dloss_dp[d] += coeff * train_embeddings[cand_idx][d];
                }
            }
            let inv_pos = 1.0 / positives.len() as f32;
            for &pos_idx in &positives {
                for (d, dp) in dloss_dp.iter_mut().enumerate() {
                    *dp -= inv_pos * train_embeddings[pos_idx][d];
                }
            }
            for dp in dloss_dp.iter_mut() {
                *dp /= TAU;
            }

            for d in 0..dim {
                for (k, zk) in z.iter().enumerate() {
                    grad_v[d * RANK + k] += dloss_dp[d] * zk;
                }
            }
            let mut dloss_dz = [0.0f32; RANK];
            for (d, g) in dloss_dp.iter().enumerate() {
                for (k, dz) in dloss_dz.iter_mut().enumerate() {
                    *dz += pred.v[d * RANK + k] * g;
                }
            }
            let r = &train_embeddings[qi];
            for d in 0..dim {
                let rd = r[d];
                for k in 0..RANK {
                    grad_u[d * RANK + k] += rd * dloss_dz[k];
                }
            }
        }

        let scale = 1.0 / n as f32;
        for i in 0..dim * RANK {
            let gu = grad_u[i] * scale + 2.0 * LAMBDA * pred.u[i];
            let gv = grad_v[i] * scale + 2.0 * LAMBDA * pred.v[i];
            pred.u[i] -= LR * gu;
            pred.v[i] -= LR * gv;
        }
    }
    pred
}

/// Mean multi-positive softmax loss over the training set — a sanity check
/// that training actually descended, not just a number to trust blindly.
fn mean_loss(pred: &Predictor, train_embeddings: &[Vec<f32>], labels: &[usize]) -> f32 {
    let n = train_embeddings.len();
    let mut by_label: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &l) in labels.iter().enumerate() {
        by_label.entry(l).or_default().push(i);
    }
    let mut total = 0.0;
    let mut count = 0;
    for qi in 0..n {
        let positives: Vec<usize> = by_label[&labels[qi]].iter().copied().filter(|&i| i != qi).collect();
        if positives.is_empty() {
            continue;
        }
        let p = pred.predict(&train_embeddings[qi]);
        let candidates: Vec<usize> = (0..n).filter(|&i| i != qi).collect();
        let scores: Vec<f32> = candidates.iter().map(|&i| dot(&p, &train_embeddings[i]) / TAU).collect();
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = scores.iter().map(|&s| (s - max_s).exp()).sum::<f32>().max(1e-8);
        for &pos in &positives {
            let ci = candidates.iter().position(|&c| c == pos).unwrap();
            let logp = (scores[ci] - max_s) - sum_exp.ln();
            total += -logp;
            count += 1;
        }
    }
    total / count as f32
}

/// Stratified fold assignment: round-robin within each class so every fold
/// gets a proportional share of every class (critical — a fold with zero
/// examples of some class would make training/eval on it meaningless).
fn stratified_folds(labels: &[usize], k: usize) -> Vec<usize> {
    let mut by_label: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &l) in labels.iter().enumerate() {
        by_label.entry(l).or_default().push(i);
    }
    let mut fold_of = vec![0usize; labels.len()];
    for members in by_label.values() {
        for (rank, &idx) in members.iter().enumerate() {
            fold_of[idx] = rank % k;
        }
    }
    fold_of
}

#[derive(Serialize)]
struct TaskResult {
    task: String,
    n_items: usize,
    n_classes: usize,
    baseline_1nn_accuracy: f32,
    jepa_1nn_accuracy: f32,
}

fn run_task(
    task_name: &str,
    embeddings: &[Vec<f32>],
    labels: &[usize],
    n_classes: usize,
) -> TaskResult {
    let n = embeddings.len();
    let fold_of = stratified_folds(labels, K_FOLDS);

    let (mut baseline_correct, mut jepa_correct, mut total) = (0usize, 0usize, 0usize);

    for fold in 0..K_FOLDS {
        let train_idx: Vec<usize> = (0..n).filter(|&i| fold_of[i] != fold).collect();
        let test_idx: Vec<usize> = (0..n).filter(|&i| fold_of[i] == fold).collect();
        if train_idx.is_empty() || test_idx.is_empty() {
            continue;
        }
        let train_embeddings: Vec<Vec<f32>> = train_idx.iter().map(|&i| embeddings[i].clone()).collect();
        let train_labels: Vec<usize> = train_idx.iter().map(|&i| labels[i]).collect();

        let seed = 4000 + fold as u64;
        let pred = train_predictor(&train_embeddings, &train_labels, seed);

        if fold == 0 {
            // Diagnostics: did training even converge, and did the
            // predictor collapse to near-identical outputs regardless of
            // input (the classic contrastive-learning failure mode)?
            let loss = mean_loss(&pred, &train_embeddings, &train_labels);
            let sample: Vec<Vec<f32>> = train_embeddings.iter().take(20).map(|e| pred.predict(e)).collect();
            let norms: Vec<f32> = sample.iter().map(|p| p.iter().map(|x| x * x).sum::<f32>().sqrt()).collect();
            let mean_norm = norms.iter().sum::<f32>() / norms.len() as f32;
            let mut pairwise = Vec::new();
            for i in 0..sample.len() {
                for j in (i + 1)..sample.len() {
                    pairwise.push(cosine_sim(&sample[i], &sample[j]));
                }
            }
            let mean_pairwise_cos = pairwise.iter().sum::<f32>() / pairwise.len() as f32;
            // Same diagnostic on RAW embeddings for comparison.
            let raw_sample: Vec<&Vec<f32>> = train_embeddings.iter().take(20).collect();
            let mut raw_pairwise = Vec::new();
            for i in 0..raw_sample.len() {
                for j in (i + 1)..raw_sample.len() {
                    raw_pairwise.push(cosine_sim(raw_sample[i], raw_sample[j]));
                }
            }
            let mean_raw_pairwise_cos = raw_pairwise.iter().sum::<f32>() / raw_pairwise.len() as f32;
            println!(
                "  [diag fold0] final_train_loss={loss:.4}  pred_mean_norm={mean_norm:.4}  pred_mean_pairwise_cos={mean_pairwise_cos:.4} (raw_mean_pairwise_cos={mean_raw_pairwise_cos:.4}) [near 1.0 pairwise = collapse]"
            );
        }

        for &ti in &test_idx {
            let true_label = labels[ti];

            // Baseline: raw cosine 1-NN against the training set.
            let (mut best_j, mut best_s) = (0usize, f32::NEG_INFINITY);
            for (j, te) in train_embeddings.iter().enumerate() {
                let s = cosine_sim(&embeddings[ti], te);
                if s > best_s {
                    best_s = s;
                    best_j = j;
                }
            }
            if train_labels[best_j] == true_label {
                baseline_correct += 1;
            }

            // Trained: predictor output vs raw training embeddings.
            let p = pred.predict(&embeddings[ti]);
            let (mut best_j2, mut best_s2) = (0usize, f32::NEG_INFINITY);
            for (j, te) in train_embeddings.iter().enumerate() {
                let s = cosine_sim(&p, te);
                if s > best_s2 {
                    best_s2 = s;
                    best_j2 = j;
                }
            }
            if train_labels[best_j2] == true_label {
                jepa_correct += 1;
            }
            total += 1;
        }
        println!(
            "  fold {fold}: train={} test={} (running) baseline_acc={:.3} jepa_acc={:.3}",
            train_idx.len(),
            test_idx.len(),
            baseline_correct as f32 / total as f32,
            jepa_correct as f32 / total as f32
        );
    }

    TaskResult {
        task: task_name.to_string(),
        n_items: n,
        n_classes,
        baseline_1nn_accuracy: baseline_correct as f32 / total as f32,
        jepa_1nn_accuracy: jepa_correct as f32 / total as f32,
    }
}

fn main() {
    println!("Experiment 4: trained metric vs. raw cosine on physis-core's REAL ontology\n");

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
            None => {
                println!("WARNING: no ONNX model found — aborting.");
                return;
            }
        };

        let ontology = OntologyLoader::load_all();
        let mut texts = Vec::new();
        let mut domains = Vec::new();
        let mut cells = Vec::new();
        for def in ontology.classification_domains() {
            let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
            let mut text = def.name.clone();
            for hint in &def.hints {
                text.push(' ');
                text.push_str(hint);
            }
            texts.push(text);
            domains.push(d.clone());
            cells.push(format!("{d}/{m}"));
        }
        println!("Loaded {} ontology entries. Embedding...", texts.len());
        let embeddings: Vec<Vec<f32>> = texts.iter().map(|t| embedder.embed(t)).collect();
        println!("Done.\n");

        // ── Domain task: all entries, 5 classes ──
        let mut domain_ids: HashMap<String, usize> = HashMap::new();
        let domain_labels: Vec<usize> = domains
            .iter()
            .map(|d| {
                let next = domain_ids.len();
                *domain_ids.entry(d.clone()).or_insert(next)
            })
            .collect();
        println!("=== Domain task: {} entries, {} classes ===", embeddings.len(), domain_ids.len());
        let domain_result = run_task("domain (5-way, n=730)", &embeddings, &domain_labels, domain_ids.len());

        // ── Cell task: only cells with >=5 entries ──
        let mut cell_counts: HashMap<String, usize> = HashMap::new();
        for c in &cells {
            *cell_counts.entry(c.clone()).or_default() += 1;
        }
        let keep: Vec<usize> = (0..cells.len()).filter(|&i| cell_counts[&cells[i]] >= 5).collect();
        let cell_embeddings: Vec<Vec<f32>> = keep.iter().map(|&i| embeddings[i].clone()).collect();
        let mut cell_ids: HashMap<String, usize> = HashMap::new();
        let cell_labels: Vec<usize> = keep
            .iter()
            .map(|&i| {
                let next = cell_ids.len();
                *cell_ids.entry(cells[i].clone()).or_insert(next)
            })
            .collect();
        println!(
            "\n=== Cell task: {} entries (cells with >=5 members), {} classes ===",
            cell_embeddings.len(),
            cell_ids.len()
        );
        let cell_result = run_task(
            "cell (well-populated subset)",
            &cell_embeddings,
            &cell_labels,
            cell_ids.len(),
        );

        println!("\n=== AGGREGATE ===");
        for r in [&domain_result, &cell_result] {
            println!(
                "{:<30} n={:<5} classes={:<4} baseline_1NN={:.3}  trained_1NN={:.3}  delta={:+.3}",
                r.task,
                r.n_items,
                r.n_classes,
                r.baseline_1nn_accuracy,
                r.jepa_1nn_accuracy,
                r.jepa_1nn_accuracy - r.baseline_1nn_accuracy
            );
        }

        let json = serde_json::to_string_pretty(&[&domain_result, &cell_result]).unwrap();
        std::fs::write("experiment4_results.json", &json).expect("write results");
        println!("\nFull results written to experiment4_results.json");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
