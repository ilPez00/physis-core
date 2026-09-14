//! E55 — structure before mapping.
//!
//! ## The question
//!
//! Two representations of the same corpus. Derive structure in each one
//! *independently*, then try to recover which node in A is which node in B
//! **from structure alone**. Identity is the gold answer and is never shown to
//! the matcher, so the accuracy of the recovered correspondence is a number
//! that can come back at chance.
//!
//! ## Why it is shaped this way
//!
//! Three prior results in this repository constrain the design, and skipping
//! any one of them re-runs a settled experiment:
//!
//! 1. **Cross-embedder ARI ≈ 0.10.** Comparing *partitions* across embedders
//!    was measured and is negative. So the primary object here is a **graph**,
//!    not a clustering, and ARI is carried only as the continuity column.
//! 2. **`propose` works where `judge` failed** (top-3 0.712 vs null 0.136).
//!    So a correspondence is emitted as a **ranked candidate list**, and top-3
//!    is reported beside top-1.
//! 3. **No cheap signal certifies structure** (every scorer at chance on real
//!    misfilings). So nothing here scores its own output for confidence; the
//!    test is recovery of held-out identity against a permutation null.
//!
//! ## The arms
//!
//! | arm | descriptor | what it uses |
//! |---|---|---|
//! | `random` | — | a seeded permutation. The null. |
//! | `anchor` | similarity to K anchor items | **weak supervision**: K known cross-space pairs |
//! | `supervised` | similarity to gold-label centroids | **supervision**: gold labels |
//! | `structural` | sorted similarity profile + graph invariants | nothing but Γ |
//!
//! `anchor` and `supervised` are labelled conditions, not competitors: they are
//! upper references that say how much of the gap supervision buys.
//!
//! ## What a positive would mean, and what it would not
//!
//! A `structural` arm above `random` means neighbourhood geometry survives a
//! change of representation well enough to identify nodes. It does **not** mean
//! the two spaces agree, and it does not license calling either space correct.
//!
//! ## RESULT 2026-09-14 — the structural arm is at chance. Read before reusing.
//!
//! On the frozen 72-sentence corpus (`benchmarks/worldstate/corpus.jsonl`,
//! sha256 `264c40a07303…`), A = bge-base-en-v1.5 via ONNX, B = the 384-d word
//! hash:
//!
//! | split | random (null) | anchor (8 pairs) | supervised | **structural** |
//! |---|---|---|---|---|
//! | all (72) | 0.028 | 0.236 | 0.083 | **0.014** |
//! | train (49) | 0.020 | 0.306 | 0.082 | **0.020** |
//! | heldout (23) | 0.130 | 0.609 | 0.174 | **0.043** |
//!
//! Top-1 correspondence accuracy. **The hypothesis arm sits on its null in
//! every split**, and a six-point sweep over k ∈ {3,5,10} × profile ∈ {6,24}
//! keeps it in 0.014–0.056 — it is not a hyperparameter. Eight known anchor
//! pairs do recover the correspondence, so the cross-space information exists;
//! sorted-neighbourhood descriptors do not find it. The descriptor is not
//! broken: `identical_spaces_recover_identity` holds at top1 > 0.99.
//!
//! Full record with the sweep and what it does and does not license:
//! `research/E55_STRUCTURE_BEFORE_MAPPING.md`. Do not re-run this arm as a
//! retune; the next legitimate attempt at H1 is a seeded propagation matcher,
//! pre-registered separately.

use std::collections::HashMap;

use rand::rngs::StdRng;
use sha2::{Digest, Sha256};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Cosine between two L2-normalised-ish vectors. Defensive against zero norm.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let d = (na.sqrt() * nb.sqrt()).max(1e-8);
    dot / d
}

/// Full pairwise similarity matrix, diagonal forced to 0 so a node is never
/// its own neighbour.
pub fn sim_matrix(vecs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = vecs.len();
    let mut m = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let s = cosine(&vecs[i], &vecs[j]);
            m[i][j] = s;
            m[j][i] = s;
        }
    }
    m
}

/// Mutual k-nearest-neighbour adjacency: an edge exists only if each node is in
/// the other's top-k. Mutual, because one-directional kNN makes hubs adjacent
/// to everything and inflates every agreement metric that follows.
pub fn mutual_knn(sim: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    let n = sim.len();
    let topk: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let mut idx: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            idx.sort_by(|&a, &b| sim[i][b].partial_cmp(&sim[i][a]).unwrap());
            idx.truncate(k);
            idx
        })
        .collect();
    (0..n)
        .map(|i| {
            topk[i]
                .iter()
                .copied()
                .filter(|&j| topk[j].contains(&i))
                .collect()
        })
        .collect()
}

/// Connected components of an adjacency list, as a partition label per node.
pub fn components(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    let mut label = vec![usize::MAX; n];
    let mut next = 0;
    for start in 0..n {
        if label[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        label[start] = next;
        while let Some(u) = stack.pop() {
            for &v in &adj[u] {
                if label[v] == usize::MAX {
                    label[v] = next;
                    stack.push(v);
                }
            }
        }
        next += 1;
    }
    label
}

/// Adjusted Rand Index between two partitions of the same node set.
///
/// Carried for continuity with the settled negative (cross-embedder ARI ≈ 0.10)
/// and for no other reason: it is a partition statistic, and the hypothesis
/// under test is about graphs.
pub fn ari(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mut cont: HashMap<(usize, usize), f64> = HashMap::new();
    let mut ra: HashMap<usize, f64> = HashMap::new();
    let mut rb: HashMap<usize, f64> = HashMap::new();
    for i in 0..a.len() {
        *cont.entry((a[i], b[i])).or_insert(0.0) += 1.0;
        *ra.entry(a[i]).or_insert(0.0) += 1.0;
        *rb.entry(b[i]).or_insert(0.0) += 1.0;
    }
    let c2 = |x: f64| x * (x - 1.0) / 2.0;
    let sum_ij: f64 = cont.values().map(|&v| c2(v)).sum();
    let sum_a: f64 = ra.values().map(|&v| c2(v)).sum();
    let sum_b: f64 = rb.values().map(|&v| c2(v)).sum();
    let total = c2(n);
    let expected = sum_a * sum_b / total;
    let max = 0.5 * (sum_a + sum_b);
    if (max - expected).abs() < 1e-12 {
        return 0.0;
    }
    (sum_ij - expected) / (max - expected)
}

/// Per-node descriptor from structure alone: the node's own similarity profile,
/// sorted descending and truncated, plus three graph invariants.
///
/// Sorting is what makes it comparable across spaces — it discards *which*
/// nodes are near and keeps *how* near-ness is distributed, which is the only
/// part of a neighbourhood that can be compared without already knowing the
/// correspondence. Each descriptor is z-scored within itself so two spaces with
/// different similarity scales stay comparable.
pub fn structural_descriptor(sim: &[Vec<f32>], adj: &[Vec<usize>], profile: usize) -> Vec<Vec<f32>> {
    let n = sim.len();
    (0..n)
        .map(|i| {
            let mut prof: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| sim[i][j]).collect();
            prof.sort_by(|a, b| b.partial_cmp(a).unwrap());
            prof.truncate(profile);
            let deg = adj[i].len() as f32;
            let mean_nb_deg = if adj[i].is_empty() {
                0.0
            } else {
                adj[i].iter().map(|&j| adj[j].len() as f32).sum::<f32>() / deg
            };
            let mut tri = 0.0f32;
            for (x, &u) in adj[i].iter().enumerate() {
                for &v in adj[i].iter().skip(x + 1) {
                    if adj[u].contains(&v) {
                        tri += 1.0;
                    }
                }
            }
            let pairs = deg * (deg - 1.0) / 2.0;
            let clustering = if pairs > 0.0 { tri / pairs } else { 0.0 };
            let mut d = prof;
            d.push(deg / n as f32);
            d.push(mean_nb_deg / n as f32);
            d.push(clustering);
            zscore(&d)
        })
        .collect()
}

/// Descriptor from similarity to K anchor nodes whose correspondence is
/// **given**. A labelled weak-supervision condition, not a competitor.
pub fn anchor_descriptor(sim: &[Vec<f32>], anchors: &[usize]) -> Vec<Vec<f32>> {
    (0..sim.len())
        .map(|i| zscore(&anchors.iter().map(|&a| sim[i][a]).collect::<Vec<f32>>()))
        .collect()
}

/// Descriptor from similarity to gold-label centroids. A labelled **supervised**
/// condition: it reads the answer key's classes, though not the identities.
pub fn label_descriptor(sim: &[Vec<f32>], labels: &[usize], n_labels: usize) -> Vec<Vec<f32>> {
    (0..sim.len())
        .map(|i| {
            let mut v = vec![0.0f32; n_labels];
            let mut c = vec![0.0f32; n_labels];
            for j in 0..sim.len() {
                if j == i {
                    continue;
                }
                v[labels[j]] += sim[i][j];
                c[labels[j]] += 1.0;
            }
            for l in 0..n_labels {
                if c[l] > 0.0 {
                    v[l] /= c[l];
                }
            }
            zscore(&v)
        })
        .collect()
}

fn zscore(v: &[f32]) -> Vec<f32> {
    let n = v.len().max(1) as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let sd = var.sqrt().max(1e-8);
    v.iter().map(|x| (x - mean) / sd).collect()
}

/// Ranked correspondence candidates: for each node of A, the nodes of B ordered
/// by descriptor similarity. `propose`, not `judge` — the whole reason top-3 is
/// reported.
pub fn rank_candidates(da: &[Vec<f32>], db: &[Vec<f32>], top: usize) -> Vec<Vec<usize>> {
    da.iter()
        .map(|x| {
            let mut idx: Vec<usize> = (0..db.len()).collect();
            idx.sort_by(|&p, &q| {
                cosine(x, &db[q])
                    .partial_cmp(&cosine(x, &db[p]))
                    .unwrap()
            });
            idx.truncate(top);
            idx
        })
        .collect()
}

/// A seeded random permutation: the null every arm is read against.
pub fn random_candidates(n: usize, top: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut picks: Vec<usize> = Vec::with_capacity(top);
            while picks.len() < top {
                let c = rng.gen_range(0..n);
                if !picks.contains(&c) {
                    picks.push(c);
                }
            }
            picks
        })
        .collect()
}

/// One arm's score. Identity is the gold correspondence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmScore {
    pub arm: String,
    pub n: usize,
    pub top1: f64,
    pub top3: f64,
    /// Fraction of A's mutual-kNN edges whose endpoints are still adjacent in B
    /// after the arm's top-1 correspondence is applied.
    pub knn_preservation: f64,
    /// Bootstrap 95% interval on top-1 over nodes.
    pub top1_ci: (f64, f64),
}

/// Score one arm's ranked candidates against identity.
pub fn score_arm(
    arm: &str,
    cands: &[Vec<usize>],
    adj_a: &[Vec<usize>],
    adj_b: &[Vec<usize>],
    seed: u64,
) -> ArmScore {
    let n = cands.len();
    let hits1: Vec<f64> = (0..n)
        .map(|i| f64::from(u8::from(cands[i].first() == Some(&i))))
        .collect();
    let hits3: Vec<f64> = (0..n)
        .map(|i| f64::from(u8::from(cands[i].iter().take(3).any(|&c| c == i))))
        .collect();
    let map: Vec<usize> = (0..n).map(|i| *cands[i].first().unwrap_or(&i)).collect();
    let mut edges = 0.0;
    let mut kept = 0.0;
    for i in 0..n {
        for &j in &adj_a[i] {
            if j <= i {
                continue;
            }
            edges += 1.0;
            if adj_b[map[i]].contains(&map[j]) {
                kept += 1.0;
            }
        }
    }
    ArmScore {
        arm: arm.to_string(),
        n,
        top1: mean(&hits1),
        top3: mean(&hits3),
        knn_preservation: if edges > 0.0 { kept / edges } else { 0.0 },
        top1_ci: bootstrap_ci(&hits1, 2000, seed),
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Percentile bootstrap over items. Seeded, so a reported interval is
/// regenerable rather than merely plausible.
pub fn bootstrap_ci(values: &[f64], iters: usize, seed: u64) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut means: Vec<f64> = (0..iters)
        .map(|_| {
            let s: f64 = (0..values.len())
                .map(|_| values[rng.gen_range(0..values.len())])
                .sum();
            s / values.len() as f64
        })
        .collect();
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = means[(iters as f64 * 0.025) as usize];
    let hi = means[((iters as f64 * 0.975) as usize).min(iters - 1)];
    (lo, hi)
}

/// The full E55 result, written to `benchmarks/results/worldstate-e55.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub corpus_sha256: String,
    pub n_items: usize,
    pub split: String,
    pub embedder_a: String,
    pub embedder_b: String,
    pub k: usize,
    pub profile: usize,
    pub anchors: usize,
    /// Partition agreement between the two spaces with no mapping at all —
    /// the continuity column against the settled ARI ≈ 0.10.
    pub ari_baseline: f64,
    pub arms: Vec<ArmScore>,
}

impl Run {
    pub fn render(&self) -> String {
        let mut s = format!(
            "corpus {} ({} items, split={})\nA={} B={} k={} profile={} anchors={}\nARI(no mapping) = {:.3}\n\n",
            &self.corpus_sha256[..12],
            self.n_items,
            self.split,
            self.embedder_a,
            self.embedder_b,
            self.k,
            self.profile,
            self.anchors,
            self.ari_baseline
        );
        s.push_str("arm           top1    top3    knn-pres   top1 95% CI\n");
        for a in &self.arms {
            s.push_str(&format!(
                "{:<12}  {:.3}   {:.3}   {:.3}      [{:.3}, {:.3}]\n",
                a.arm, a.top1, a.top3, a.knn_preservation, a.top1_ci.0, a.top1_ci.1
            ));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_identical_is_one() {
        let v = vec![0.3, 0.4, 0.5];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn mutual_knn_is_symmetric() {
        let vecs = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];
        let adj = mutual_knn(&sim_matrix(&vecs), 1);
        for (i, nb) in adj.iter().enumerate() {
            for &j in nb {
                assert!(adj[j].contains(&i), "edge {i}-{j} not mutual");
            }
        }
    }

    #[test]
    fn ari_is_one_for_identical_partitions_and_zero_for_constant() {
        let a = vec![0, 0, 1, 1, 2, 2];
        assert!((ari(&a, &a) - 1.0).abs() < 1e-9);
        let flat = vec![0, 0, 0, 0, 0, 0];
        assert!(ari(&a, &flat).abs() < 1e-9);
    }

    #[test]
    fn identical_spaces_recover_identity() {
        // The sanity floor: if A and B are the same space, structure must
        // recover the correspondence perfectly. A version of this file that
        // scored below 1.0 here had a descriptor bug, not a finding.
        let vecs: Vec<Vec<f32>> = (0..12)
            .map(|i| {
                let t = i as f32 * 0.37;
                vec![t.cos(), t.sin(), (t * 2.0).cos()]
            })
            .collect();
        let sim = sim_matrix(&vecs);
        let adj = mutual_knn(&sim, 3);
        let d = structural_descriptor(&sim, &adj, 6);
        let cands = rank_candidates(&d, &d, 3);
        let s = score_arm("self", &cands, &adj, &adj, 7);
        assert!(s.top1 > 0.99, "self-match top1 = {}", s.top1);
        assert!(s.knn_preservation > 0.99);
    }

    #[test]
    fn random_arm_sits_at_chance() {
        let n = 60;
        let cands = random_candidates(n, 3, 11);
        let adj = vec![vec![]; n];
        let s = score_arm("random", &cands, &adj, &adj, 3);
        assert!(s.top1 < 0.15, "random top1 = {}", s.top1);
    }

    #[test]
    fn bootstrap_interval_brackets_the_mean() {
        let v: Vec<f64> = (0..50).map(|i| f64::from(u8::from(i % 4 == 0))).collect();
        let (lo, hi) = bootstrap_ci(&v, 500, 5);
        let m = mean(&v);
        assert!(lo <= m && m <= hi, "mean {m} outside [{lo}, {hi}]");
    }
}

// ---------------------------------------------------------------------------
// E56 — position in text as the temporal coordinate.
//
// The state sequence is `W_0 … W_n` where `p` is the sentence's ordinal
// position, not a clock. Three legs, each with a null that can fire:
//
//   L1  transition derivation   — do set operations over the state sequence
//                                 recover the gold transition labels, and does
//                                 the recovery collapse when order is shuffled?
//   L2  position in geometry    — do embedding neighbours sit near each other
//                                 in the text, or is |Δpos| at its null?
//   L3  antecedent retrieval    — "what preceded this?" Gold is mechanical: the
//                                 nearest earlier sentence mentioning the same
//                                 entity. Arms are order-blind cosine, cosine
//                                 restricted to earlier positions, position
//                                 alone, and the shuffled-position null.
//
// L1 and L3 are the two questions a file browser cannot answer and a world
// model should.

/// A derived transition between adjacent positions.
pub const TRANSITIONS: [&str; 6] = [
    "emerge",
    "persist",
    "change",
    "recur",
    "disappear",
    "none",
];

/// Lexical cues for disappearance. Declared rather than learned: there are six
/// of them, they are in the open, and a reader can see exactly how much of L1
/// is a word list. Everything else in the derivation is set operations.
const GONE_CUES: [&str; 6] = [
    "no longer",
    "is removed",
    "was removed",
    "is drained",
    "stops",
    "stands idle",
];

/// Gap, in positions, after which a re-mention counts as `recur` rather than
/// `persist`.
pub const RECUR_GAP: usize = 6;

/// Derive a transition label per position from the state sequence alone.
///
/// Reads only: the entity set, the typed relation, and the text (for the six
/// cues above). **It never reads the gold transition label** — that is what it
/// is scored against.
pub fn derive_transitions(
    entities: &[Vec<String>],
    facts: &[Option<(String, String, String)>],
    texts: &[&str],
) -> Vec<&'static str> {
    derive_transitions_gap(entities, facts, texts, RECUR_GAP)
}

/// As [`derive_transitions`], with the re-mention gap exposed.
///
/// The gap decides the whole `recur` label, so it is a parameter and gets swept
/// rather than a constant that quietly sets a result (E56 reported `recur` as
/// NOT MEASURED until this existed).
pub fn derive_transitions_gap(
    entities: &[Vec<String>],
    facts: &[Option<(String, String, String)>],
    texts: &[&str],
    recur_gap: usize,
) -> Vec<&'static str> {
    let mut last_seen: HashMap<String, usize> = HashMap::new();
    let mut seen_fact: HashMap<(String, String, String), usize> = HashMap::new();
    let mut pair_pred: HashMap<(String, String), String> = HashMap::new();
    let mut out = Vec::with_capacity(entities.len());
    for p in 0..entities.len() {
        let lower = texts[p].to_lowercase();
        let gone = GONE_CUES.iter().any(|c| lower.contains(c));
        let novel = entities[p].iter().any(|e| !last_seen.contains_key(e));
        let stale = entities[p]
            .iter()
            .filter_map(|e| last_seen.get(e))
            .map(|&q| p - q)
            .min()
            .map(|d| d > recur_gap)
            .unwrap_or(false);

        let label = if gone {
            "disappear"
        } else if novel {
            "emerge"
        } else if let Some(f) = &facts[p] {
            let key = (f.0.clone(), f.1.clone(), f.2.clone());
            let pair = (f.0.clone(), f.2.clone());
            if seen_fact.contains_key(&key) {
                if stale {
                    "recur"
                } else {
                    "persist"
                }
            } else if pair_pred.get(&pair).map(|q| q != &f.1).unwrap_or(false) {
                "change"
            } else if stale {
                "recur"
            } else {
                "change"
            }
        } else if stale {
            "recur"
        } else if entities[p].is_empty() {
            "none"
        } else {
            "persist"
        };

        for e in &entities[p] {
            last_seen.insert(e.clone(), p);
        }
        if let Some(f) = &facts[p] {
            seen_fact.insert((f.0.clone(), f.1.clone(), f.2.clone()), p);
            pair_pred.insert((f.0.clone(), f.2.clone()), f.1.clone());
        }
        out.push(label);
    }
    out
}

/// Agreement between derived and gold labels, plus the majority-class floor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionScore {
    pub n: usize,
    #[serde(default)]
    pub recur_gap: usize,
    pub accuracy: f64,
    pub majority_baseline: f64,
    pub shuffled_null_mean: f64,
    pub shuffled_null_sd: f64,
    pub z: f64,
    pub per_label: Vec<(String, usize, f64)>,
}

/// Score a derivation against gold, with a shuffled-order null.
///
/// The null re-runs the *whole derivation* on a permuted reading order and
/// scores it against the gold labels in their original order. If position
/// carries nothing, the two agree.
pub fn score_transitions(
    entities: &[Vec<String>],
    facts: &[Option<(String, String, String)>],
    texts: &[&str],
    gold: &[String],
    perms: usize,
    seed: u64,
) -> TransitionScore {
    score_transitions_gap(entities, facts, texts, gold, perms, seed, RECUR_GAP)
}

/// As [`score_transitions`], with the re-mention gap exposed for the sweep.
#[allow(clippy::too_many_arguments)]
pub fn score_transitions_gap(
    entities: &[Vec<String>],
    facts: &[Option<(String, String, String)>],
    texts: &[&str],
    gold: &[String],
    perms: usize,
    seed: u64,
    recur_gap: usize,
) -> TransitionScore {
    let derived = derive_transitions_gap(entities, facts, texts, recur_gap);
    let n = gold.len();
    let acc = |d: &[&str]| -> f64 {
        d.iter()
            .zip(gold.iter())
            .filter(|(a, b)| **a == b.as_str())
            .count() as f64
            / n as f64
    };
    let accuracy = acc(&derived);

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for g in gold {
        *counts.entry(g.as_str()).or_insert(0) += 1;
    }
    let majority = counts.values().copied().max().unwrap_or(0) as f64 / n as f64;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut nulls = Vec::with_capacity(perms);
    for _ in 0..perms {
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, rng.gen_range(0..=i));
        }
        let pe: Vec<Vec<String>> = order.iter().map(|&i| entities[i].clone()).collect();
        let pf: Vec<Option<(String, String, String)>> =
            order.iter().map(|&i| facts[i].clone()).collect();
        let pt: Vec<&str> = order.iter().map(|&i| texts[i]).collect();
        let d = derive_transitions_gap(&pe, &pf, &pt, recur_gap);
        // Put each derived label back at the position its sentence really has,
        // then score against gold in the original order.
        let mut restored = vec!["none"; n];
        for (slot, &orig) in order.iter().enumerate() {
            restored[orig] = d[slot];
        }
        nulls.push(acc(&restored));
    }
    let m = nulls.iter().sum::<f64>() / perms.max(1) as f64;
    let sd = (nulls.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / perms.max(1) as f64).sqrt();
    let z = if sd > 1e-9 { (accuracy - m) / sd } else { 0.0 };

    let mut per_label: Vec<(String, usize, f64)> = TRANSITIONS
        .iter()
        .map(|&t| {
            let idx: Vec<usize> = (0..n).filter(|&i| gold[i] == t).collect();
            let hit = idx.iter().filter(|&&i| derived[i] == t).count();
            (
                t.to_string(),
                idx.len(),
                if idx.is_empty() {
                    0.0
                } else {
                    hit as f64 / idx.len() as f64
                },
            )
        })
        .collect();
    per_label.retain(|(_, c, _)| *c > 0);

    TransitionScore {
        n,
        recur_gap,
        accuracy,
        majority_baseline: majority,
        shuffled_null_mean: m,
        shuffled_null_sd: sd,
        z,
        per_label,
    }
}

/// L2 — mean |Δpos| over mutual-kNN edges against a random-pair null.
///
/// If semantic neighbours are also textual neighbours the first number is the
/// smaller one. Reported as a ratio so it does not depend on corpus length.
pub fn position_locality(adj: &[Vec<usize>], seed: u64, samples: usize) -> (f64, f64, f64) {
    let n = adj.len();
    let mut deltas = Vec::new();
    for (i, nb) in adj.iter().enumerate() {
        for &j in nb {
            if j > i {
                deltas.push((j - i) as f64);
            }
        }
    }
    let observed = if deltas.is_empty() {
        0.0
    } else {
        deltas.iter().sum::<f64>() / deltas.len() as f64
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut null = 0.0;
    for _ in 0..samples {
        let a = rng.gen_range(0..n);
        let b = rng.gen_range(0..n);
        null += (a as f64 - b as f64).abs();
    }
    null /= samples.max(1) as f64;
    let ratio = if null > 0.0 { observed / null } else { 1.0 };
    (observed, null, ratio)
}

/// L3 — "what preceded this?" scored per arm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntecedentScore {
    pub arm: String,
    pub queries: usize,
    pub top1: f64,
    pub top3: f64,
    pub top1_ci: (f64, f64),
}

/// Gold antecedent: the nearest earlier sentence sharing an entity. Mechanical,
/// so the answer key is not an opinion.
pub fn gold_antecedents(entities: &[Vec<String>]) -> Vec<Option<usize>> {
    (0..entities.len())
        .map(|p| {
            (0..p).rev().find(|&q| {
                entities[q].iter().any(|e| entities[p].contains(e))
            })
        })
        .collect()
}

/// Score one antecedent arm. `rank` returns, for a query position, the ranked
/// candidate positions that arm proposes.
pub fn score_antecedents(
    arm: &str,
    gold: &[Option<usize>],
    rank: impl Fn(usize) -> Vec<usize>,
    seed: u64,
) -> AntecedentScore {
    let mut h1 = Vec::new();
    let mut h3 = Vec::new();
    for (p, g) in gold.iter().enumerate() {
        let Some(g) = g else { continue };
        let r = rank(p);
        h1.push(f64::from(u8::from(r.first() == Some(g))));
        h3.push(f64::from(u8::from(r.iter().take(3).any(|c| c == g))));
    }
    AntecedentScore {
        arm: arm.to_string(),
        queries: h1.len(),
        top1: mean(&h1),
        top3: mean(&h3),
        top1_ci: bootstrap_ci(&h1, 2000, seed),
    }
}

/// The E56 artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionRun {
    pub corpus_sha256: String,
    pub n_items: usize,
    pub embedder: String,
    pub k: usize,
    pub transitions: TransitionScore,
    pub locality_observed: f64,
    pub locality_null: f64,
    pub locality_ratio: f64,
    pub antecedents: Vec<AntecedentScore>,
}

impl PositionRun {
    pub fn render(&self) -> String {
        let t = &self.transitions;
        let mut s = format!(
            "corpus {} ({} items)  embedder={}  k={}\n\nL1 transition derivation\n  accuracy          {:.3}\n  majority baseline {:.3}\n  shuffled null     {:.3} ± {:.3}   z = {:+.2}\n",
            &self.corpus_sha256[..12],
            self.n_items,
            self.embedder,
            self.k,
            t.accuracy,
            t.majority_baseline,
            t.shuffled_null_mean,
            t.shuffled_null_sd,
            t.z
        );
        for (label, count, recall) in &t.per_label {
            s.push_str(&format!("    {label:<11} n={count:<3} recall {recall:.3}\n"));
        }
        s.push_str(&format!(
            "\nL2 position locality\n  mean |Δpos| over kNN edges {:.2} vs null {:.2}  ratio {:.3}\n\nL3 antecedent retrieval\narm                 n    top1    top3    top1 95% CI\n",
            self.locality_observed, self.locality_null, self.locality_ratio
        ));
        for a in &self.antecedents {
            s.push_str(&format!(
                "{:<18} {:<4} {:.3}   {:.3}   [{:.3}, {:.3}]\n",
                a.arm, a.queries, a.top1, a.top3, a.top1_ci.0, a.top1_ci.1
            ));
        }
        s
    }
}

#[cfg(test)]
mod position_tests {
    use super::*;

    #[test]
    fn first_mention_is_emergence_and_repeat_is_persistence() {
        let ents = vec![vec!["a".to_string()], vec!["a".to_string()]];
        let facts = vec![
            Some(("a".into(), "Requires".into(), "b".into())),
            Some(("a".into(), "Requires".into(), "b".into())),
        ];
        let texts = vec!["a requires b", "a still requires b"];
        let d = derive_transitions(&ents, &facts, &texts);
        assert_eq!(d, vec!["emerge", "persist"]);
    }

    #[test]
    fn a_changed_predicate_over_the_same_pair_is_change() {
        let ents = vec![
            vec!["a".to_string()],
            vec!["a".to_string()],
            vec!["a".to_string()],
        ];
        let facts = vec![
            Some(("a".into(), "Requires".into(), "b".into())),
            Some(("a".into(), "Requires".into(), "b".into())),
            Some(("a".into(), "Contradicts".into(), "b".into())),
        ];
        let texts = vec!["x", "y", "z"];
        assert_eq!(derive_transitions(&ents, &facts, &texts)[2], "change");
    }

    #[test]
    fn gold_antecedent_is_the_nearest_earlier_shared_entity() {
        let ents = vec![
            vec!["a".to_string()],
            vec!["b".to_string()],
            vec!["a".to_string()],
        ];
        let g = gold_antecedents(&ents);
        assert_eq!(g, vec![None, None, Some(0)]);
    }

    #[test]
    fn locality_ratio_is_one_when_neighbours_are_arbitrary() {
        let n = 40;
        let adj: Vec<Vec<usize>> = (0..n).map(|i| vec![(i + n / 2) % n]).collect();
        let (_, _, ratio) = position_locality(&adj, 3, 4000);
        assert!(ratio > 0.8, "ratio = {ratio}");
    }
}

// ---------------------------------------------------------------------------
// E58 — extraction, because every number above was handed its entity links.
//
// E57's winning arm (entity + position + cosine, top-1 0.700) reads the
// corpus's gold entity annotation. A deployed system has no such field: it has
// text. So the same arm is re-run on links this module extracts, and the drop
// between the two is the cost of extraction, stated rather than assumed.
//
// The extractor is deterministic, model-free and unsupervised — three rules,
// all visible:
//
//   R1  an identifier-shaped token: letters, a hyphen, digits (`lathe-3`)
//   R2  a capitalised token that is not sentence-initial (`Rossi`, `Filter-A`)
//   R3  a repeated content token: length >= 4, not a stopword, occurring in at
//       least `min_df` sentences of the corpus
//
// R3 is the only rule that looks at the corpus rather than the sentence, and it
// is what makes the extractor unsupervised rather than a lexicon in disguise.

const STOPWORDS: [&str; 40] = [
    "the", "and", "that", "with", "from", "this", "then", "than", "into", "over",
    "under", "after", "before", "while", "when", "were", "was", "are", "its",
    "his", "her", "their", "they", "them", "have", "has", "had", "been", "being",
    "which", "what", "each", "both", "some", "more", "most", "less", "very",
    "also", "still",
];

/// Normalise a surface form to an entity key: lowercase, spaces to hyphens,
/// trailing punctuation dropped. `Filter-A` and `filter-a` become one key;
/// `coolant pump` and `coolant-pump` become one key.
pub fn entity_key(s: &str) -> String {
    s.trim_matches(|c: char| !c.is_alphanumeric() && c != '-')
        .to_lowercase()
        .replace(' ', "-")
}

/// Extract an entity set per sentence. `min_df` is R3's document-frequency
/// threshold; 3 is the value E58 runs at and it is not swept here.
pub fn extract_entities(texts: &[&str], min_df: usize) -> Vec<Vec<String>> {
    extract_entities_rules(texts, min_df, true, true, true)
}

/// As [`extract_entities`], with each rule switchable, because E58 found the
/// three together destroy the very thing they feed: R3 admits `pressure`,
/// `tolerance`, `line` — tokens every sentence shares — so an entity filter
/// built on it links everything to everything and collapses to no filter at
/// all. Which rule costs what is an ablation, not an opinion.
pub fn extract_entities_rules(
    texts: &[&str],
    min_df: usize,
    use_id: bool,
    use_cap: bool,
    use_df: bool,
) -> Vec<Vec<String>> {
    let toks: Vec<Vec<String>> = texts
        .iter()
        .map(|t| {
            t.split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-'))
                .filter(|w| !w.is_empty())
                .map(|w| w.to_string())
                .collect()
        })
        .collect();

    let mut df: HashMap<String, usize> = HashMap::new();
    for ts in &toks {
        let mut seen: Vec<String> = Vec::new();
        for w in ts {
            let k = entity_key(w);
            if k.len() >= 4 && !STOPWORDS.contains(&k.as_str()) && !seen.contains(&k) {
                seen.push(k.clone());
                *df.entry(k).or_insert(0) += 1;
            }
        }
    }

    toks.iter()
        .map(|ts| {
            let mut out: Vec<String> = Vec::new();
            for (i, w) in ts.iter().enumerate() {
                let k = entity_key(w);
                if k.is_empty() || out.contains(&k) {
                    continue;
                }
                let id_shaped = w
                    .split_once('-')
                    .map(|(a, b)| {
                        !a.is_empty()
                            && a.chars().all(|c| c.is_alphabetic())
                            && !b.is_empty()
                            && b.chars().all(|c| c.is_alphanumeric())
                    })
                    .unwrap_or(false);
                let capitalised_midsentence =
                    i > 0 && w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
                let repeated_content = k.len() >= 4
                    && !STOPWORDS.contains(&k.as_str())
                    && df.get(&k).copied().unwrap_or(0) >= min_df;
                if (use_id && id_shaped)
                    || (use_cap && capitalised_midsentence)
                    || (use_df && repeated_content)
                {
                    out.push(k);
                }
            }
            out
        })
        .collect()
}

/// Micro-averaged precision / recall / F1 of extracted entity sets against gold.
///
/// Gold keys are normalised the same way, so `filter-a` vs `Filter-A` is not
/// counted as an error. Multi-word gold entities (`coolant-pump`) also match a
/// single extracted head token (`coolant` or `pump`) — otherwise the score would
/// measure tokenisation rather than extraction. Both relaxations are declared,
/// and both inflate the number relative to exact matching.
pub fn extraction_prf(extracted: &[Vec<String>], gold: &[Vec<String>]) -> (f64, f64, f64) {
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut fneg = 0.0;
    let hits = |needle: &str, hay: &[String]| -> bool {
        hay.iter().any(|h| {
            h == needle
                || h.split('-').any(|part| part == needle)
                || needle.split('-').any(|part| part == h)
        })
    };
    for (e, g) in extracted.iter().zip(gold.iter()) {
        let gk: Vec<String> = g.iter().map(|x| entity_key(x)).collect();
        for x in e {
            if hits(x, &gk) {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
        }
        for x in &gk {
            if !hits(x, e) {
                fneg += 1.0;
            }
        }
    }
    let p = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let r = if tp + fneg > 0.0 { tp / (tp + fneg) } else { 0.0 };
    let f = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
    (p, r, f)
}

#[cfg(test)]
mod extraction_tests {
    use super::*;

    #[test]
    fn identifier_shaped_tokens_are_entities() {
        let texts = vec!["Lathe-3 stands at the wall.", "the shop was swept"];
        let e = extract_entities(&texts, 99);
        assert!(e[0].contains(&"lathe-3".to_string()));
        assert!(e[1].is_empty(), "got {:?}", e[1]);
    }

    #[test]
    fn repeated_content_tokens_are_entities_and_rare_ones_are_not() {
        let texts = vec![
            "coolant reaches the head",
            "coolant is filtered",
            "coolant returns to the tank",
            "a kestrel landed outside",
        ];
        let e = extract_entities(&texts, 3);
        assert!(e[0].contains(&"coolant".to_string()));
        assert!(!e[3].contains(&"kestrel".to_string()));
    }

    #[test]
    fn prf_is_perfect_when_extraction_equals_gold() {
        let g = vec![vec!["lathe-3".to_string()], vec!["press-2".to_string()]];
        let (p, r, f) = extraction_prf(&g, &g);
        assert!((p - 1.0).abs() < 1e-9 && (r - 1.0).abs() < 1e-9 && (f - 1.0).abs() < 1e-9);
    }
}

// ---------------------------------------------------------------------------
// E59 — noun phrases, because real text does not name things like a machine.
//
// E58's `id` rule is end-to-end free on the generated corpus (precision 0.928,
// recall 0.966) and recovers 30.4% of the hand corpus's entities, because that
// corpus says `the coolant pump`, `Filter-A`, `Rossi`. This rule chunks a noun
// phrase without a POS tagger: a determiner, then up to `max_len` tokens that
// are not stopwords and not verbs-by-suffix, keyed on the phrase and on its
// head.
//
// It is a heuristic and it is stated as one. What makes it testable is that the
// same downstream arm runs on its output, so a chunker that looks reasonable
// and helps nothing is visible as such.

const DETERMINERS: [&str; 6] = ["the", "a", "an", "this", "that", "its"];

/// Suffixes that mark a token as a verb form often enough to exclude it from a
/// noun phrase. Crude on purpose: the alternative is a tagger dependency, and
/// the point of the experiment is what a rule this cheap can carry.
const VERBISH: [&str; 2] = ["ing", "ed"];

fn verbish(w: &str) -> bool {
    w.len() >= 4 && VERBISH.iter().any(|suf| w.ends_with(suf))
}

/// A trailing `-s` is a plural noun at the head of a phrase (`the parts`) and a
/// present-tense verb after one (`the coolant pump feeds`). Position decides,
/// because nothing cheaper does: the first version of this file put `s` in
/// [`VERBISH`] and chunked `the coolant pump feeds` as one three-word entity.
fn verb_after_head(w: &str, phrase_len: usize) -> bool {
    phrase_len >= 1 && w.ends_with('s') && !w.ends_with("ss")
}

/// Chunk noun phrases and return one entity key per phrase plus its head.
///
/// `the coolant pump` yields `coolant-pump` **and** `pump`, so a later mention
/// of `the pump` still links. Head keys are what make the phrase useful for
/// coreference without a coreference model.
pub fn extract_noun_phrases(texts: &[&str], max_len: usize) -> Vec<Vec<String>> {
    texts
        .iter()
        .map(|t| {
            let toks: Vec<String> = t
                .split_whitespace()
                .map(|w| {
                    w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-')
                        .to_string()
                })
                .filter(|w| !w.is_empty())
                .collect();
            let mut out: Vec<String> = Vec::new();
            let mut i = 0;
            while i < toks.len() {
                let lower = toks[i].to_lowercase();
                let id_shaped = toks[i]
                    .split_once('-')
                    .map(|(a, b)| {
                        !a.is_empty()
                            && a.chars().all(|c| c.is_alphabetic())
                            && !b.is_empty()
                            && b.chars().all(|c| c.is_alphanumeric())
                    })
                    .unwrap_or(false);
                let capital_mid =
                    i > 0 && toks[i].chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
                if id_shaped || capital_mid {
                    let k = entity_key(&toks[i]);
                    if !k.is_empty() && !out.contains(&k) {
                        out.push(k);
                    }
                    i += 1;
                    continue;
                }
                if DETERMINERS.contains(&lower.as_str()) {
                    let mut phrase: Vec<String> = Vec::new();
                    let mut j = i + 1;
                    while j < toks.len() && phrase.len() < max_len {
                        let w = toks[j].to_lowercase();
                        if STOPWORDS.contains(&w.as_str())
                            || DETERMINERS.contains(&w.as_str())
                            || verbish(&w)
                            || verb_after_head(&w, phrase.len())
                            || w.len() < 3
                        {
                            break;
                        }
                        phrase.push(w);
                        j += 1;
                    }
                    if !phrase.is_empty() {
                        let full = entity_key(&phrase.join("-"));
                        let head = entity_key(phrase.last().unwrap());
                        for k in [full, head] {
                            if !k.is_empty() && !out.contains(&k) {
                                out.push(k);
                            }
                        }
                        i = j;
                        continue;
                    }
                }
                i += 1;
            }
            out
        })
        .collect()
}

/// Union of two extractors' outputs, per sentence.
pub fn merge_entities(a: &[Vec<String>], b: &[Vec<String>]) -> Vec<Vec<String>> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let mut out = x.clone();
            for k in y {
                if !out.contains(k) {
                    out.push(k.clone());
                }
            }
            out
        })
        .collect()
}

#[cfg(test)]
mod np_tests {
    use super::*;

    #[test]
    fn a_two_word_phrase_yields_phrase_and_head() {
        let texts = vec!["The coolant pump feeds the lathe."];
        let e = extract_noun_phrases(&texts, 3);
        assert!(e[0].contains(&"coolant-pump".to_string()), "got {:?}", e[0]);
        assert!(e[0].contains(&"pump".to_string()));
    }

    #[test]
    fn identifiers_and_midsentence_capitals_survive() {
        let texts = vec!["Rossi runs lathe-3 on shift B."];
        let e = extract_noun_phrases(&texts, 3);
        assert!(e[0].contains(&"lathe-3".to_string()), "got {:?}", e[0]);
    }

    #[test]
    fn merge_is_a_union_without_duplicates() {
        let a = vec![vec!["x".to_string()]];
        let b = vec![vec!["x".to_string(), "y".to_string()]];
        assert_eq!(merge_entities(&a, &b)[0], vec!["x".to_string(), "y".to_string()]);
    }
}

/// Code-shaped identifiers: `act::bearing_on`, `src/models.rs`, `RECUR_GAP`,
/// `fuse_rrf`. Added for E60 after the chunker scored precision 0.224 on this
/// repository's own prose while the author's backticks marked exactly these
/// forms. Declared as a fourth rule and scored on **both** corpora, because a
/// rule added after seeing one corpus's failure is tuning unless it is also
/// checked where it could hurt.
///
/// A token qualifies if it contains `::`, a `/` with a dot after it, a `.rs`
/// ending, an underscore, or is ALL_CAPS of length >= 4. Each contributes its
/// full key and its last segment.
pub fn extract_code_identifiers(texts: &[&str]) -> Vec<Vec<String>> {
    texts
        .iter()
        .map(|t| {
            let mut out: Vec<String> = Vec::new();
            for w in t.split_whitespace() {
                let w = w.trim_matches(|c: char| !c.is_alphanumeric() && !"-_./:".contains(c));
                if w.len() < 3 {
                    continue;
                }
                let code_shaped = w.contains("::")
                    || (w.contains('/') && w.contains('.'))
                    || w.ends_with(".rs")
                    || w.contains('_')
                    || (w.len() >= 4
                        && w.chars().all(|c| c.is_ascii_uppercase() || c == '_'));
                if !code_shaped {
                    continue;
                }
                let full = entity_key(w);
                let seg = full
                    .rsplit(['/', ':'])
                    .next()
                    .unwrap_or(&full)
                    .to_string();
                let stem = seg.strip_suffix(".rs").unwrap_or(&seg).to_string();
                for k in [full, seg, stem] {
                    if k.len() > 2 && !out.contains(&k) {
                        out.push(k);
                    }
                }
            }
            out
        })
        .collect()
}

#[cfg(test)]
mod code_tests {
    use super::*;

    #[test]
    fn code_forms_yield_full_key_and_last_segment() {
        let texts = vec!["see physis-core/src/models.rs and act::bearing_on for RECUR_GAP"];
        let e = extract_code_identifiers(&texts);
        assert!(e[0].contains(&"models".to_string()), "got {:?}", e[0]);
        assert!(e[0].contains(&"bearing_on".to_string()), "got {:?}", e[0]);
        assert!(e[0].contains(&"recur_gap".to_string()), "got {:?}", e[0]);
    }

    #[test]
    fn ordinary_prose_yields_nothing() {
        let texts = vec!["The coolant pump feeds the lathe at the north wall."];
        assert!(extract_code_identifiers(&texts)[0].is_empty());
    }
}

// ---------------------------------------------------------------------------
// E64 — a lookup table of words, instead of a model at query time.
//
// E63 found that a search function matches a 7B interpreter on real text. The
// obvious next question is how far down the stack that goes: if word-level
// evidence is what carries these tasks, does the *embedder* need to run at all,
// or can a table of word vectors — built once, looked up per word, averaged per
// sentence — stand in for it?
//
// Two tables, and the difference between them is the whole point:
//
//   table-model   each word's vector is the model's embedding OF THAT WORD.
//                 Built once per vocabulary, then no model at query time. Tests
//                 whether sentence meaning here is approximately the mean of its
//                 words.
//   table-count   each word's vector is built by counting: random indexing with
//                 PPMI weighting over a co-occurrence window, from the corpus
//                 itself. NO MODEL EVER, at build time or query time.
//
// `table-count` is the one that matters for the architecture: if it holds up,
// the semantic layer is arithmetic over counts and the machine needs no neural
// inference to navigate itself.

/// Word vectors by random indexing with PPMI weighting — pure counting.
///
/// Each context word gets a fixed random ±1 sparse signature (deterministic from
/// its hash, so the table is reproducible and needs no storage of the basis).
/// A word's vector is the PPMI-weighted sum of the signatures of the words that
/// co-occur with it inside `window`. This is the cheap, well-understood
/// alternative to SVD, and it needs no linear-algebra dependency.
pub fn word_table_count(texts: &[&str], dim: usize, window: usize) -> HashMap<String, Vec<f32>> {
    let toks: Vec<Vec<String>> = texts
        .iter()
        .map(|t| {
            t.split_whitespace()
                .map(|w| {
                    w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
                        .to_lowercase()
                })
                .filter(|w| w.len() >= 2)
                .collect()
        })
        .collect();

    let mut freq: HashMap<&str, f64> = HashMap::new();
    let mut pair: HashMap<(&str, &str), f64> = HashMap::new();
    let mut total_pairs = 0.0f64;
    for ts in &toks {
        for (i, w) in ts.iter().enumerate() {
            *freq.entry(w.as_str()).or_insert(0.0) += 1.0;
            let lo = i.saturating_sub(window);
            let hi = (i + window + 1).min(ts.len());
            for c in ts.iter().take(hi).skip(lo) {
                if c == w {
                    continue;
                }
                *pair.entry((w.as_str(), c.as_str())).or_insert(0.0) += 1.0;
                total_pairs += 1.0;
            }
        }
    }
    let total_tokens: f64 = freq.values().sum();

    let signature = |w: &str| -> Vec<f32> {
        // Deterministic sparse ±1 signature: eight positions set from the word's
        // hash. Storing no basis is what keeps the table regenerable anywhere.
        let mut v = vec![0.0f32; dim];
        let mut h = Sha256::new();
        h.update(w.as_bytes());
        let d = h.finalize();
        for chunk in d.chunks(4).take(8) {
            let idx = (u32::from_le_bytes(chunk.try_into().unwrap()) as usize) % dim;
            let sign = if chunk[3] & 1 == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        }
        v
    };
    let mut sig_cache: HashMap<&str, Vec<f32>> = HashMap::new();
    for w in freq.keys() {
        sig_cache.insert(w, signature(w));
    }

    let mut table: HashMap<String, Vec<f32>> = HashMap::new();
    for ((w, c), n) in &pair {
        let pw = freq[w] / total_tokens;
        let pc = freq[c] / total_tokens;
        let pwc = n / total_pairs.max(1.0);
        let ppmi = (pwc / (pw * pc).max(1e-12)).ln().max(0.0) as f32;
        if ppmi <= 0.0 {
            continue;
        }
        let sig = &sig_cache[c];
        let e = table.entry((*w).to_string()).or_insert_with(|| vec![0.0; dim]);
        for (slot, s) in e.iter_mut().zip(sig.iter()) {
            *slot += ppmi * s;
        }
    }
    for v in table.values_mut() {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        v.iter_mut().for_each(|x| *x /= n);
    }
    table
}

/// A sentence vector by lookup and mean: no model call, whatever built the table.
///
/// Words absent from the table contribute nothing, and a sentence with no known
/// word returns a zero vector — which `cosine` scores as 0 against everything,
/// rather than silently landing somewhere in the space.
pub fn sentence_from_table(table: &HashMap<String, Vec<f32>>, text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    let mut hits = 0.0f32;
    for w in text.split_whitespace() {
        let k = w
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .to_lowercase();
        if let Some(wv) = table.get(&k) {
            for (slot, x) in v.iter_mut().zip(wv.iter()) {
                *slot += x;
            }
            hits += 1.0;
        }
    }
    if hits > 0.0 {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        v.iter_mut().for_each(|x| *x /= n);
    }
    v
}

/// Vocabulary of a corpus, for building a model-backed table once.
pub fn vocabulary(texts: &[&str], min_len: usize) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for t in texts {
        for w in t.split_whitespace() {
            let k = w
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
                .to_lowercase();
            if k.len() >= min_len && !seen.contains(&k) {
                seen.push(k);
            }
        }
    }
    seen.sort();
    seen
}

#[cfg(test)]
mod table_tests {
    use super::*;

    #[test]
    fn counted_table_puts_co_occurring_words_together() {
        let texts = vec![
            "the lathe cuts steel with coolant",
            "the lathe cuts steel slowly",
            "the invoice was paid by the customer",
            "the invoice was paid late",
        ];
        let t = word_table_count(&texts, 128, 3);
        let lathe = &t["lathe"];
        let steel = &t["steel"];
        let invoice = &t["invoice"];
        assert!(
            cosine(lathe, steel) > cosine(lathe, invoice),
            "lathe~steel {:.3} should exceed lathe~invoice {:.3}",
            cosine(lathe, steel),
            cosine(lathe, invoice)
        );
    }

    #[test]
    fn a_sentence_with_no_known_word_is_zero_not_arbitrary() {
        let t = word_table_count(&["alpha beta gamma"], 64, 2);
        let v = sentence_from_table(&t, "zzz qqq", 64);
        assert!(v.iter().all(|x| *x == 0.0));
    }
}

// ---------------------------------------------------------------------------
// E65 — pick the filter from the corpus, not from a flag.
//
// E63 measured the same two arms swapping places between regimes: BM25 with a
// position restriction wins on lexically diverse text (docs 0.535 vs the entity
// arm's 0.209) and collapses under near-duplicate load (generated 0.013 vs
// 0.700). The deciding property is redundancy, so it should be measured rather
// than guessed — and a shell that measures it can choose its own arm and say
// why.
//
// The statistic is deliberately the crudest one that could work: for each
// document, the highest term-Jaccard against any other document, averaged. It
// asks "does almost every sentence have a near-twin?", which is exactly the
// condition that defeats a lexical filter.

/// Mean over documents of the maximum term-Jaccard to any other document.
///
/// 0.0 = every document's vocabulary is its own. 1.0 = every document has a
/// perfect twin. `sample` caps the pairwise work on large corpora by scoring a
/// deterministic stride rather than all pairs.
pub fn redundancy(texts: &[&str], sample: usize) -> f64 {
    let sets: Vec<Vec<String>> = texts
        .iter()
        .map(|t| {
            let mut v: Vec<String> = t
                .split_whitespace()
                .map(|w| {
                    w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
                        .to_lowercase()
                })
                .filter(|w| w.len() >= 3 && !STOPWORDS.contains(&w.as_str()))
                .collect();
            v.sort();
            v.dedup();
            v
        })
        .collect();
    let n = sets.len();
    if n < 2 {
        return 0.0;
    }
    let stride = if n > sample && sample > 0 {
        n / sample
    } else {
        1
    };
    let mut total = 0.0;
    let mut counted = 0.0;
    for i in (0..n).step_by(stride.max(1)) {
        if sets[i].is_empty() {
            continue;
        }
        let mut best = 0.0f64;
        for (j, other) in sets.iter().enumerate() {
            if i == j || other.is_empty() {
                continue;
            }
            let inter = sets[i].iter().filter(|w| other.contains(w)).count() as f64;
            let union = (sets[i].len() + other.len()) as f64 - inter;
            if union > 0.0 {
                best = best.max(inter / union);
            }
        }
        total += best;
        counted += 1.0;
    }
    if counted > 0.0 {
        total / counted
    } else {
        0.0
    }
}

/// Mean BM25 top-1 vs top-2 margin: how *decisively* the best lexical match wins.
///
/// Added after [`redundancy`] failed (E65): raw term overlap called the machine
/// log 0.786 — as redundant as the generated templates at 0.802 — because its
/// file paths share long common prefixes, while BM25's IDF ignores exactly those
/// shared tokens. Overlap is the wrong thing to measure; **ambiguity after
/// weighting** is the right one.
///
/// For each document, query the corpus with its own terms and take
/// `(score1 - score2) / score1` over the best two *other* documents. A corpus of
/// near-twins produces ties — a small margin — which is precisely the condition
/// that makes a lexical filter pick a distractor. Diverse text produces a clear
/// winner.
pub fn lexical_margin(texts: &[String], sample: usize) -> f64 {
    let index = crate::rag::Bm25Index::build(texts);
    let n = texts.len();
    if n < 3 {
        return 1.0;
    }
    let stride = if sample > 0 && n > sample { n / sample } else { 1 };
    let mut total = 0.0;
    let mut counted = 0.0;
    for i in (0..n).step_by(stride.max(1)) {
        let terms = crate::rag::bm25_terms(&texts[i]);
        if terms.is_empty() {
            continue;
        }
        let mut scores: Vec<f32> = (0..n)
            .filter(|&j| j != i)
            .map(|j| index.score(j, &terms))
            .filter(|s| *s > 0.0)
            .collect();
        if scores.len() < 2 {
            continue;
        }
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let (s1, s2) = (scores[0] as f64, scores[1] as f64);
        if s1 > 0.0 {
            total += (s1 - s2) / s1;
            counted += 1.0;
        }
    }
    if counted > 0.0 {
        total / counted
    } else {
        1.0
    }
}

/// Which filter the measured redundancy points at.
///
/// **Four corpora is four data points.** This is a reading of them, not a law,
/// and the boundary sits in the wide empty gap between the two groups rather
/// than at a fitted value — because with n = 4 anything more precise would be
/// false precision.
///
/// | corpus | redundancy | arm that won |
/// |---|---|---|
/// | docs | low | bm25 + position (0.535 vs entity 0.209) |
/// | machine log | low | bm25 + position (0.447 ≈ interpreter 0.465) |
/// | hand-written | low | either (n too small to separate) |
/// | generated templates | high | entity link (0.700 vs bm25 0.013) |
pub fn filter_advice(redundancy: f64) -> &'static str {
    if redundancy >= 0.50 {
        "entity-link (high redundancy: a lexical filter cannot partition near-twins)"
    } else {
        "bm25 + position (low redundancy: term overlap discriminates, and costs nothing)"
    }
}

/// Which filter the measured **margin** points at — the predictor that works.
///
/// Measured on four corpora (E65). It is right wherever the two arms actually
/// differ, and wrong on the one corpus where they are statistically tied:
///
/// | corpus | margin | bm25+pos | entity link | predictor |
/// |---|---|---|---|---|
/// | generated templates | 0.044 | 0.013 | **0.700** | correct |
/// | documentation | 0.235 | **0.535** | 0.209 | correct |
/// | machine log | 0.520 | **0.447** | 0.296 | correct |
/// | hand-written (n=28) | 0.265 | 0.286 | 0.357 | **wrong, inside the noise** |
///
/// The boundary sits at 0.15, in the wide empty gap between 0.044 and 0.235,
/// rather than at a fitted value: **four corpora is four points**, and anything
/// more precise would be false precision.
pub fn filter_from_margin(margin: f64) -> &'static str {
    if margin < 0.15 {
        "entity-link"
    } else {
        "bm25+position"
    }
}

#[cfg(test)]
mod redundancy_tests {
    use super::*;

    #[test]
    fn distinct_documents_score_low_and_twins_score_high() {
        let distinct = vec![
            "the lathe cuts steel with coolant",
            "invoices were reconciled against the ledger",
            "a heron stood in the shallow river",
        ];
        let twins = vec![
            "machine alpha holds pressure within tolerance",
            "machine beta holds pressure within tolerance",
            "machine gamma holds pressure within tolerance",
        ];
        let lo = redundancy(&distinct, 0);
        let hi = redundancy(&twins, 0);
        assert!(lo < 0.2, "distinct scored {lo}");
        assert!(hi > 0.5, "twins scored {hi}");
        assert!(filter_advice(hi).starts_with("entity-link"));
        assert!(filter_advice(lo).starts_with("bm25"));
    }
}
