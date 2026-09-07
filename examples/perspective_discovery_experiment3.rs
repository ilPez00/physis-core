// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 3 — JEPA-style predictive latent vs. static cosine similarity.
//!
//! Iterations 1-2 (see `../research/perspective-discovery/RESEARCH_LOG.md`)
//! falsified two *geometric* reference-conditioning mechanisms (Δ-magnitude:
//! proven identical to cosine; Δ-direction clustering: beaten by plain
//! nearest-neighbor at n=21). This experiment tests a mechanism of a
//! genuinely different kind, per the mission's secondary hypothesis
//! (Section 1): instead of transforming a *frozen* embedding geometrically,
//! train a small predictor that maps a reference's embedding toward its
//! field members' embeddings, and use prediction quality (not raw cosine
//! similarity) as the relationship signal.
//!
//! Architecture (deliberately the smallest version that is still a real
//! predictive-latent model, per mission Section 15):
//!   - Frozen encoder: the same all-MiniLM-L6-v2 ONNX embedder as Iterations
//!     1-2 (`embed(x)`).
//!   - Predictor: a rank-K linear map `pred(r) = V (Uᵀ r)`, U,V ∈ ℝ^(D×K).
//!     Low-rank by design: with only ~20 training items, a full D×D matrix
//!     (384×384 ≈ 147k parameters) would trivially memorize the training set
//!     and prove nothing. K=8 (≈6k parameters) is still more parameters than
//!     data points — the honest reading of any result here is "does the
//!     mechanism show a signal at all," not "this generalizes," and the log
//!     says so explicitly.
//!   - Loss: multi-positive InfoNCE (softmax cross-entropy with all of a
//!     reference's field members as positives, all other training items as
//!     in-batch negatives), trained by hand-derived gradient descent — no
//!     autodiff/ML crate dependency.
//!
//! Evaluation: strict leave-one-out cross-validation. For each of the 21
//! items held out in turn, the predictor is trained from scratch on the
//! OTHER 20 (the held-out item does not appear anywhere in training — not as
//! a query, not as a candidate, not as a positive or negative), then used to
//! rank the 20 training items from the held-out item's embedding. This is
//! the same shape of test naive-NN and delta-direction were scored on in
//! Experiments 1-2, so the numbers are directly comparable.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --example perspective_discovery_experiment3

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

const CORPUS: &[(&str, &str, &str)] = &[
    ("engine", "mechanical", "power-core"),
    ("piston", "mechanical", "power-core"),
    ("camshaft", "mechanical", "power-core"),
    ("torque", "mechanical", "power-core"),
    ("transmission", "mechanical", "drivetrain-support"),
    ("fuel injection", "mechanical", "drivetrain-support"),
    ("cooling system", "mechanical", "drivetrain-support"),
    ("steering wheel", "driver", "active-controls"),
    ("brake pedal", "driver", "active-controls"),
    ("accelerator pedal", "driver", "active-controls"),
    ("turn signal", "driver", "active-controls"),
    ("rearview mirror", "driver", "safety-awareness"),
    ("seat belt", "driver", "safety-awareness"),
    ("dashboard display", "driver", "safety-awareness"),
    ("resale value", "economic", "value-over-time"),
    ("depreciation rate", "economic", "value-over-time"),
    ("trade-in value", "economic", "value-over-time"),
    ("insurance premium", "economic", "recurring-cost"),
    ("fuel cost", "economic", "recurring-cost"),
    ("loan interest", "economic", "recurring-cost"),
    ("purchase price", "economic", "recurring-cost"),
];

const RANK: usize = 8;
const EPOCHS: usize = 400;
const LR: f32 = 0.15;
const TAU: f32 = 0.15;
const LAMBDA: f32 = 1e-3;

/// Low-rank predictor: pred(r) = V (Uᵀ r), U,V both D×RANK, row-major.
struct Predictor {
    dim: usize,
    u: Vec<f32>, // dim x RANK
    v: Vec<f32>, // dim x RANK
}

impl Predictor {
    fn init(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = 1.0 / (dim as f32).sqrt();
        let mut rnd = |n: usize| (0..n).map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale).collect();
        Predictor { dim, u: rnd(dim * RANK), v: rnd(dim * RANK) }
    }

    /// z = Uᵀ r  (RANK,)
    fn project(&self, r: &[f32]) -> Vec<f32> {
        let mut z = vec![0.0f32; RANK];
        for (d, rd) in r.iter().enumerate() {
            for (k, zk) in z.iter_mut().enumerate() {
                *zk += self.u[d * RANK + k] * rd;
            }
        }
        z
    }

    /// pred = V z  (dim,)
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

/// Train a predictor on `train_embeddings` (a subset that must NOT contain
/// the held-out item), using `field_of` restricted to indices within that
/// subset. Every item in the subset is used as a query reference in turn;
/// its positives are the other subset items sharing its label.
fn train_predictor(
    train_embeddings: &[Vec<f32>],
    labels: &[&'static str],
    seed: u64,
) -> Predictor {
    let dim = train_embeddings[0].len();
    let n = train_embeddings.len();
    let mut pred = Predictor::init(dim, seed);

    for _epoch in 0..EPOCHS {
        // Accumulate gradients over all queries (full-batch gradient descent).
        let mut grad_u = vec![0.0f32; dim * RANK];
        let mut grad_v = vec![0.0f32; dim * RANK];

        for qi in 0..n {
            let positives: Vec<usize> = (0..n).filter(|&i| i != qi && labels[i] == labels[qi]).collect();
            if positives.is_empty() {
                continue;
            }
            let z = pred.project(&train_embeddings[qi]);
            let p = pred.predict_from_z(&z);

            // scores over all candidates except the query itself.
            let candidates: Vec<usize> = (0..n).filter(|&i| i != qi).collect();
            let scores: Vec<f32> = candidates
                .iter()
                .map(|&i| dot(&p, &train_embeddings[i]) / TAU)
                .collect();
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum::<f32>().max(1e-8);
            let probs: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

            // dLoss/dp = (1/tau) * [ |positives| * E_probs[x] - sum_{x in positives} x ]
            let mut dloss_dp = vec![0.0f32; dim];
            for (ci, &cand_idx) in candidates.iter().enumerate() {
                let coeff = positives.len() as f32 * probs[ci];
                for d in 0..dim {
                    dloss_dp[d] += coeff * train_embeddings[cand_idx][d];
                }
            }
            for &pos_idx in &positives {
                for (d, dp) in dloss_dp.iter_mut().enumerate() {
                    *dp -= train_embeddings[pos_idx][d];
                }
            }
            for dp in dloss_dp.iter_mut() {
                *dp /= TAU;
            }

            // dLoss/dV[d,k] += dloss_dp[d] * z[k]
            for d in 0..dim {
                for (k, zk) in z.iter().enumerate() {
                    grad_v[d * RANK + k] += dloss_dp[d] * zk;
                }
            }
            // dLoss/dz[k] = sum_d V[d,k] * dloss_dp[d]
            let mut dloss_dz = [0.0f32; RANK];
            for (d, g) in dloss_dp.iter().enumerate() {
                for (k, dz) in dloss_dz.iter_mut().enumerate() {
                    *dz += pred.v[d * RANK + k] * g;
                }
            }
            // dLoss/dU[d,k] += r[d] * dloss_dz[k]
            let r = &train_embeddings[qi];
            for d in 0..dim {
                let rd = r[d];
                for k in 0..RANK {
                    grad_u[d * RANK + k] += rd * dloss_dz[k];
                }
            }
        }

        // L2 regularization + SGD step.
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

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Training-loss sanity check (not used for evaluation): confirms the
/// hand-derived gradients actually descend, rather than trusting the numbers
/// blindly. Returns the mean multi-positive softmax loss over all queries.
fn mean_loss(pred: &Predictor, train_embeddings: &[Vec<f32>], labels: &[&'static str]) -> f32 {
    let n = train_embeddings.len();
    let mut total = 0.0;
    let mut count = 0;
    for qi in 0..n {
        let positives: Vec<usize> = (0..n).filter(|&i| i != qi && labels[i] == labels[qi]).collect();
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

/// Leave-one-out: hold out item `held_idx` entirely (not even as a
/// candidate), train on the rest, evaluate the held-out item's predicted
/// ranking of the remaining 20 against its true field (restricted to that
/// remaining set). Returns F1 (== precision == recall, fixed-size
/// retrieval), or None if the field has zero other members in the
/// remaining set.
fn loo_eval(
    embeddings: &[Vec<f32>],
    label_of: impl Fn(usize) -> &'static str,
    held_idx: usize,
    seed: u64,
) -> Option<(f32, f32)> {
    let train_idx: Vec<usize> = (0..embeddings.len()).filter(|&i| i != held_idx).collect();
    let train_embeddings: Vec<Vec<f32>> = train_idx.iter().map(|&i| embeddings[i].clone()).collect();
    let train_labels: Vec<&'static str> = train_idx.iter().map(|&i| label_of(i)).collect();

    let true_label = label_of(held_idx);
    let field_size = train_idx.iter().filter(|&&i| label_of(i) == true_label).count();
    if field_size == 0 {
        return None;
    }

    let pred = train_predictor(&train_embeddings, &train_labels, seed);
    let final_loss = mean_loss(&pred, &train_embeddings, &train_labels);

    let p_held = pred.predict(&embeddings[held_idx]);
    let mut scored: Vec<(usize, f32)> = train_idx
        .iter()
        .map(|&i| (i, cosine_sim(&p_held, &embeddings[i])))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let hits = scored.iter().take(field_size).filter(|(i, _)| label_of(*i) == true_label).count();
    Some((hits as f32 / field_size as f32, final_loss))
}

fn coarse(i: usize) -> &'static str {
    CORPUS[i].1
}
fn fine(i: usize) -> &'static str {
    CORPUS[i].2
}

#[derive(Serialize)]
struct Result3 {
    reference: &'static str,
    jepa_f1_coarse: f32,
    jepa_f1_fine: f32,
    final_train_loss_coarse: f32,
    final_train_loss_fine: f32,
}

fn main() {
    println!("Experiment 3: JEPA-style predictive latent (rank-{RANK} predictor) vs. static cosine baselines");
    println!("Leave-one-out cross-validation over all {} items\n", CORPUS.len());

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder};
        let mut embeddings = None;
        for dir in ["models", "../models"] {
            let p = std::path::Path::new(dir);
            if !(p.join("model.onnx").exists() || p.join("onnx/model.onnx").exists()) {
                continue;
            }
            let cfg = OnnxConfig { dim: 384, model_dir: Some(dir.to_string()), ..OnnxConfig::default() };
            let e = OnnxEmbedder::with_config(&cfg);
            if e.is_available() {
                println!("Using real semantic embedder: {dir}/model.onnx (all-MiniLM-L6-v2, 384d)\n");
                embeddings = Some(CORPUS.iter().map(|(t, ..)| e.embed(t)).collect::<Vec<_>>());
                break;
            }
        }
        let embeddings = match embeddings {
            Some(e) => e,
            None => {
                println!("WARNING: no ONNX model found — aborting (fallback embedder isn't meaningful evidence, see Iteration 1).");
                return;
            }
        };

        println!(
            "{:<20} {:>10} {:>10} {:>12} {:>12}",
            "reference", "jepa_C", "jepa_F", "loss_C", "loss_F"
        );
        let mut results = Vec::new();
        let (mut c_sum, mut f_sum) = (0.0, 0.0);
        let (mut c_n, mut f_n) = (0, 0);

        for (i, item) in CORPUS.iter().enumerate() {
            let seed = 1000 + i as u64; // deterministic, distinct per fold
            let (fc, lc) = loo_eval(&embeddings, coarse, i, seed).unwrap_or((f32::NAN, f32::NAN));
            let (ff, lf) = loo_eval(&embeddings, fine, i, seed).unwrap_or((f32::NAN, f32::NAN));
            if fc.is_finite() { c_sum += fc; c_n += 1; }
            if ff.is_finite() { f_sum += ff; f_n += 1; }
            println!(
                "{:<20} {:>10.3} {:>10.3} {:>12.4} {:>12.4}",
                item.0, fc, ff, lc, lf
            );
            results.push(Result3 {
                reference: CORPUS[i].0,
                jepa_f1_coarse: fc,
                jepa_f1_fine: ff,
                final_train_loss_coarse: lc,
                final_train_loss_fine: lf,
            });
        }

        println!("\n=== AGGREGATE (leave-one-out, n={}) ===", CORPUS.len());
        println!("mean F1 JEPA/coarse: {:.3}   (naive baseline: 0.587, delta-direction: 0.469)", c_sum / c_n as f32);
        println!("mean F1 JEPA/fine:   {:.3}   (naive baseline: 0.468, delta-direction: 0.226)", f_sum / f_n as f32);

        // Control: does the exact same architecture also "learn" to predict
        // SHUFFLED (meaningless) coarse labels well? If yes, the coarse
        // result above is memorization of a small training set by a
        // high-capacity-relative-to-data model, not evidence of real
        // structure. Fixed deterministic shuffle, no cherry-picking.
        let mut perm: Vec<usize> = (0..CORPUS.len()).collect();
        {
            let mut rng = StdRng::seed_from_u64(777);
            for i in (1..perm.len()).rev() {
                let j = rng.gen_range(0..=i);
                perm.swap(i, j);
            }
        }
        let shuffled = |i: usize| coarse(perm[i]);
        let mut shuf_sum = 0.0;
        let mut shuf_n = 0;
        for i in 0..CORPUS.len() {
            let seed = 2000 + i as u64;
            if let Some((f1, _)) = loo_eval(&embeddings, shuffled, i, seed) {
                shuf_sum += f1;
                shuf_n += 1;
            }
        }
        println!(
            "CONTROL mean F1 JEPA/shuffled-coarse-labels: {:.3}  (chance floor ~0.33; if this is also high, the real result above is memorization, not signal)",
            shuf_sum / shuf_n as f32
        );

        let json = serde_json::to_string_pretty(&results).unwrap();
        std::fs::write("experiment3_results.json", &json).expect("write results");
        println!("\nFull results written to experiment3_results.json");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; this experiment requires the real embedder. Aborting.");
}
