// NOTE: the no-embed-onnx build is a stub; the analysis below is intentionally
// dead there (it serves the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Perspective invention — mission: can Physis discover a new perspective on
//! data rather than merely classify data within perspectives supplied to it?
//!
//! Protocol (pre-registered in research/perspective_invention/{hypothesis,
//! methodology}.md, written BEFORE this run):
//!
//!   S0  synthetic dataset, surface structure != latent organizing principle
//!       (difficulty calibrated so raw k-means recovers the surface but the
//!       latent structure only weakly — recoverable, not free)
//!   S1  baseline organization: k-means on raw embeddings, internal k
//!   S2  residual extraction: cross-cluster relations the baseline discarded,
//!       z-positioned against size-matched shuffled nulls
//!   S3  candidate perspectives generated FROM the residual (P_edge,
//!       P_soft, P_ref) + controls (random projection, no-residual ablation,
//!       random-references ablation) + evaluation-only oracle
//!   S4  re-representation + re-discovery; internal score = silhouette of the
//!       discovered partition measured in RAW space (no perspective grades
//!       its own space)
//!   S5  internal selection + recursive invention with lineage; stop on no
//!       internal gain
//!   S6  cross-perspective convergence (same embedder + across MiniLM/BGE,
//!       against the published cross-embedder partition null ARI ~ 0.10)
//!   S7  held-out prediction from frozen perspectives (1-NN accuracy +
//!       same-flow link AUC + compression + stability)
//!   S8  transfer to a second dataset with different surface domains
//!   S9  tier verdicts exactly as pre-registered
//!
//! Anti-leakage contract: latent labels exist ONLY in the sidecar and in
//! evaluation code paths. Selection, recursion, stopping and k-choice use
//! internal (label-free) measures only.
//!
//! Run (from workspace root, picks up models/model.onnx):
//!   cargo run -p physis-core --features embed-onnx --release --example perspective_invention_experiment -- --seed 20260911
//! Flags: --skip-bge, --out <dir>, --probe (calibration grid only)

use physis_core::embed::VectorEmbed;
use physis_core::models::cosine_sim;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::collections::HashMap;

// ───────────────────────── dataset vocabulary ─────────────────────────

struct DomVocab {
    name: &'static str,
    /// domain log-prefix — makes the surface signal lexically dominant,
    /// as the mission requires (surface easy, latent hidden)
    log: &'static str,
    actors: &'static [&'static str],
    objects: &'static [&'static str],
    settings: &'static [&'static str],
}

const DOMAINS_TRAIN: &[DomVocab] = &[
    DomVocab {
        name: "workshop",
        log: "Workshop",
        actors: &["the foreman", "the welder", "the apprentice", "the inspector", "the machinist", "the rigger", "the fitter", "the surveyor"],
        objects: &["a lathe", "a bench vise", "a bundle of steel rods", "an angle grinder", "a crate of bolts", "a pneumatic drill", "a roll of sheet metal", "a welding mask", "a paint drum", "a torque wrench", "an oil can", "a stack of planks"],
        settings: &["by the loading bay", "under the steel gantry", "beside the tool wall", "near the furnace", "along the assembly line", "in the yard"],
    },
    DomVocab {
        name: "garden",
        log: "Garden",
        actors: &["the gardener", "the planter", "the grower", "the pruner", "the allotment keeper", "the nursery hand", "the greenhouse keeper", "the groundsman"],
        objects: &["a rake", "a watering can", "a tray of seedlings", "a bag of compost", "a coil of garden hose", "a wheelbarrow", "a pair of shears", "a trellis", "a sack of mulch", "a lawnmower", "a flowerbed border", "a basket of cuttings"],
        settings: &["under the pear trees", "by the greenhouse wall", "along the south border", "near the rain barrel", "beside the potting shed", "in the cold frame"],
    },
    DomVocab {
        name: "harbor",
        log: "Harbor",
        actors: &["the harbormaster", "the dockworker", "the stevedore", "the crane operator", "the customs officer", "the boatman", "the tally clerk", "the tide watcher"],
        objects: &["a cargo net", "a shipping container", "a coil of mooring rope", "a stack of crates", "a winch drum", "a bollard", "a pallet of sacks", "a harbor buoy", "a gangway", "a fuel drum", "a manifest clipboard", "a derrick arm"],
        settings: &["at the east quay", "under the gantry crane", "by the tide gate", "along the breakwater", "near the fish market", "in the dry dock"],
    },
    DomVocab {
        name: "kitchen",
        log: "Kitchen",
        actors: &["the cook", "the baker", "the sous-chef", "the dishwasher", "the pantry keeper", "the griller", "the confectioner", "the steward"],
        objects: &["a mixing bowl", "a cast-iron pan", "a sack of flour", "a block of knives", "a cooling rack", "a stockpot", "a butter churn", "a spice rack", "a cutting board", "a set of measuring cups", "a proofing basket", "a shelf of jars"],
        settings: &["by the pantry door", "under the extraction hood", "along the prep counter", "near the ovens", "beside the sink", "in the cold room"],
    },
];

// Transfer dataset: DIFFERENT surface domains, SAME latent grammar.
const DOMAINS_TRANSFER: &[DomVocab] = &[
    DomVocab {
        name: "clinic",
        log: "Clinic",
        actors: &["the nurse", "the physician", "the intern", "the phlebotomist", "the radiographer", "the orderly", "the dietitian", "the custodian"],
        objects: &["a stethoscope", "a gurney", "a tray of vials", "an X-ray film", "a chart binder", "an IV stand", "a box of gloves", "a heart monitor", "a crutch", "a specimen jar", "a dosing cup", "a supply cart"],
        settings: &["by the reception desk", "under the skylight", "along the corridor", "near the lab bench", "beside the ward door", "in the supply room"],
    },
    DomVocab {
        name: "studio",
        log: "Studio",
        actors: &["the potter", "the painter", "the printmaker", "the sculptor", "the weaver", "the sketcher", "the muralist", "the framer"],
        objects: &["a canvas frame", "a kiln shelf", "a tray of pigments", "a bundle of brushes", "a slab of clay", "an easel", "a roll of linen", "a palette knife", "a jar of glaze", "a litho stone", "a spool of warp thread", "a chisel set"],
        settings: &["by the north window", "under the mezzanine", "along the drying wall", "near the sink", "beside the flat files", "in the spray booth"],
    },
    DomVocab {
        name: "barnyard",
        log: "Barnyard",
        actors: &["the farmer", "the herder", "the milker", "the farrier", "the farmhand", "the shepherd", "the poultry keeper", "the fence mender"],
        objects: &["a hay bale", "a milking pail", "a sack of feed", "a water trough", "a bundle of fence posts", "an egg crate", "a wheel of straw", "a leather harness", "a bucket of grain", "a pitchfork", "a storm lantern", "a cart of manure"],
        settings: &["by the barn door", "under the old oak", "along the fence line", "near the trough", "beside the silo", "in the paddock"],
    },
    DomVocab {
        name: "radio-room",
        log: "Radio room",
        actors: &["the operator", "the dispatcher", "the technician", "the signalman", "the coder", "the monitor keeper", "the maintainer", "the listener"],
        objects: &["a HF receiver", "a coil of antenna wire", "a signal lamp", "a logbook", "a bank of switches", "a morse key", "a headset", "a crate of vacuum tubes", "a battery rack", "an oscilloscope", "a patch panel", "a dial gauge"],
        settings: &["by the antenna mast", "under the noise filter", "along the patch bay", "near the transformer", "beside the blackout curtain", "in the relay closet"],
    },
];

// Latent flows — the TRUE organizing principle. Cue A: a pseudo-word clause
// (sub-lexical signal only; no pretrained semantics attaches to it). Cue B: a
// mild shared phase verb. Both are shared ACROSS surface domains, so the
// latent structure exists only as cross-cluster relations, never as a domain.
const FLOW_CLAUSE: [&str; 3] = [
    ", as the keltrin step requires",
    ", following the vozane order",
    ", under the halmex rule",
];
const FLOW_PSEUDOTOKEN: [&str; 3] = ["keltrin", "vozane", "halmex"];
const FLOW_VERB: [&[&str]; 3] = [
    &["starts up", "opens up", "kicks off"],
    &["reworks", "converts", "reshapes"],
    &["winds down", "closes off", "sets aside"],
];
const NEUTRAL_VERB: &[&str] = &["handles", "moves", "checks", "watches"];

const N_FLOWS: usize = 3;
const BRIDGE_PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

#[derive(Clone, Serialize)]
struct Item {
    text: String,
    surface: usize,
    flows: Vec<usize>,
    is_bridge: bool,
    split: &'static str,
}

fn gen_sentence(rng: &mut StdRng, dom: &DomVocab, flow: usize, p_cue: f64, q_cue: f64) -> String {
    let actor = dom.actors[rng.gen_range(0..dom.actors.len())];
    let object = dom.objects[rng.gen_range(0..dom.objects.len())];
    let setting = dom.settings[rng.gen_range(0..dom.settings.len())];
    let verb_pool: &[&str] = if rng.gen::<f64>() < q_cue { FLOW_VERB[flow] } else { NEUTRAL_VERB };
    let verb = verb_pool[rng.gen_range(0..verb_pool.len())];
    let mut s = format!("{} note: {} {} {}", dom.log, actor, verb, object);
    s.push_str(", ");
    s.push_str(setting);
    if rng.gen::<f64>() < p_cue {
        s.push_str(FLOW_CLAUSE[flow]);
    }
    s
}

fn generate_split(
    rng: &mut StdRng,
    doms: &[DomVocab],
    singles_per_flow: usize,
    p_cue: f64,
    q_cue: f64,
    split: &'static str,
) -> Vec<Item> {
    let mut items = Vec::new();
    for (di, dom) in doms.iter().enumerate() {
        for flow in 0..N_FLOWS {
            for _ in 0..singles_per_flow {
                let text = gen_sentence(rng, dom, flow, p_cue, q_cue);
                items.push(Item { text, surface: di, flows: vec![flow], is_bridge: false, split });
            }
        }
        for (fa, fb) in BRIDGE_PAIRS {
            let actor = dom.actors[rng.gen_range(0..dom.actors.len())];
            let object = dom.objects[rng.gen_range(0..dom.objects.len())];
            let setting = dom.settings[rng.gen_range(0..dom.settings.len())];
            let verb_pool: &[&str] = if rng.gen::<f64>() < q_cue { FLOW_VERB[fa] } else { NEUTRAL_VERB };
            let verb = verb_pool[rng.gen_range(0..verb_pool.len())];
            let mut text = format!("{} note: {} {} {}", dom.log, actor, verb, object);
            text.push_str(", ");
            text.push_str(setting);
            text.push_str(FLOW_CLAUSE[fa]);
            text.push_str(FLOW_CLAUSE[fb]); // bridges ALWAYS carry both clauses
            items.push(Item { text, surface: di, flows: vec![fa, fb], is_bridge: true, split });
        }
    }
    items
}

// ───────────────────────── metrics (self-contained) ─────────────────────────

/// Standard adjusted Rand index over two partitions.
fn ari(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len();
    assert_eq!(n, b.len());
    let mut map_a: HashMap<(usize, usize), usize> = HashMap::new();
    let mut ra: HashMap<usize, usize> = HashMap::new();
    let mut rb: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        *map_a.entry((a[i], b[i])).or_insert(0) += 1;
        *ra.entry(a[i]).or_insert(0) += 1;
        *rb.entry(b[i]).or_insert(0) += 1;
    }
    let c2 = |x: u64| x * (x - 1) / 2;
    let index: u64 = map_a.values().map(|&v| c2(v as u64)).sum();
    let sa: u64 = ra.values().map(|&v| c2(v as u64)).sum();
    let sb: u64 = rb.values().map(|&v| c2(v as u64)).sum();
    let total = c2(n as u64);
    if total == 0 {
        return 0.0;
    }
    let expected = sa as f64 * sb as f64 / total as f64;
    let max_index = (sa + sb) as f64 / 2.0;
    if (max_index - expected).abs() < 1e-12 {
        return 1.0; // degenerate: both partitions singletons
    }
    (index as f64 - expected) / (max_index - expected)
}

fn entropy_from_counts(counts: &[usize], n: usize) -> f64 {
    let n = n as f64;
    counts.iter().filter(|&&c| c > 0).map(|&c| {
        let p = c as f64 / n;
        -p * p.ln()
    }).sum()
}

/// Normalized mutual information (arithmetic-mean normalization), nats.
fn nmi(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len();
    let mut joint: HashMap<(usize, usize), usize> = HashMap::new();
    let mut ca: HashMap<usize, usize> = HashMap::new();
    let mut cb: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        *joint.entry((a[i], b[i])).or_insert(0) += 1;
        *ca.entry(a[i]).or_insert(0) += 1;
        *cb.entry(b[i]).or_insert(0) += 1;
    }
    let n_f = n as f64;
    let mi: f64 = joint.iter().map(|((kx, ky), &c)| {
        let (x, y) = (*kx, *ky);
        let pxy = c as f64 / n_f;
        pxy * ((pxy / ((ca[&x] as f64 / n_f) * (cb[&y] as f64 / n_f))).ln())
    }).sum();
    let ha: Vec<usize> = ca.values().copied().collect();
    let hb: Vec<usize> = cb.values().copied().collect();
    let denom = (entropy_from_counts(&ha, n) + entropy_from_counts(&hb, n)) / 2.0;
    if denom < 1e-12 { 0.0 } else { (mi / denom).max(0.0) }
}

/// Majority purity of a partition against labels.
fn purity(part: &[usize], labels: &[usize]) -> f64 {
    let n = part.len();
    let mut groups: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    for i in 0..n {
        *groups.entry(part[i]).or_default().entry(labels[i]).or_insert(0) += 1;
    }
    let hits: usize = groups.values().map(|g| *g.values().max().unwrap()).sum();
    hits as f64 / n as f64
}

/// Mean silhouette (cosine distance) of a partition in the given space.
fn silhouette(data: &[Vec<f32>], part: &[usize]) -> f64 {
    let n = data.len();
    // per-cluster pairwise distance sums
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n { clusters.entry(part[i]).or_default().push(i); }
    let ids: Vec<usize> = clusters.keys().copied().collect();
    let mut total = 0.0f64;
    for &ci in &ids {
        let members = &clusters[&ci];
        for (pos, &i) in members.iter().enumerate() {
            let a = if members.len() <= 1 {
                0.0
            } else {
                let mut s = 0.0f64;
                for (pos2, &j) in members.iter().enumerate() {
                    if pos2 == pos { continue; }
                    s += (1.0 - cosine_sim(&data[i], &data[j])) as f64;
                }
                s / (members.len() - 1) as f64
            };
            let mut b = f64::INFINITY;
            for &cj in &ids {
                if cj == ci { continue; }
                let other = &clusters[&cj];
                let s: f64 = other.iter().map(|&j| (1.0 - cosine_sim(&data[i], &data[j])) as f64).sum();
                b = b.min(s / other.len() as f64);
            }
            let m = a.max(b);
            total += if m < 1e-12 { 0.0 } else { (b - a) / m };
        }
    }
    total / n as f64
}

/// Seeded k-means with k-means++ init. Empty clusters re-seeded at the point
/// farthest from its center. Deterministic given the rng.
fn kmeans(data: &[Vec<f32>], k: usize, rng: &mut StdRng, iters: usize) -> Vec<usize> {
    let n = data.len();
    let dim = data[0].len();
    assert!(k >= 2 && k <= n);
    // k-means++
    let mut centers: Vec<Vec<f32>> = Vec::with_capacity(k);
    let first = rng.gen_range(0..n);
    centers.push(data[first].clone());
    let mut dist = |x: &Vec<f32>, cs: &[Vec<f32>]| -> f64 {
        cs.iter().map(|c| (1.0 - cosine_sim(x, c)) as f64).fold(f64::INFINITY, f64::min)
    };
    for _ in 1..k {
        let d: Vec<f64> = (0..n).map(|i| dist(&data[i], &centers)).collect();
        let sum: f64 = d.iter().sum();
        let pick = if sum <= 1e-12 {
            rng.gen_range(0..n)
        } else {
            let mut r = rng.gen::<f64>() * sum;
            let mut idx = n - 1;
            for (i, &dv) in d.iter().enumerate() {
                r -= dv;
                if r <= 0.0 { idx = i; break; }
            }
            idx
        };
        centers.push(data[pick].clone());
    }
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        let mut changed = false;
        for (i, x) in data.iter().enumerate() {
            let mut best = 0usize;
            let mut best_sim = f32::NEG_INFINITY;
            for (c, ctr) in centers.iter().enumerate() {
                let s = cosine_sim(x, ctr);
                if s > best_sim { best_sim = s; best = c; }
            }
            if assign[i] != best { assign[i] = best; changed = true; }
        }
        // update
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, x) in data.iter().enumerate() {
            counts[assign[i]] += 1;
            for d in 0..dim { sums[assign[i]][d] += x[d]; }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // re-seed at the point farthest from its assigned center
                let (far, _) = (0..n).map(|i| (i, (1.0 - cosine_sim(&data[i], &centers[assign[i]])) as f64))
                    .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap()).unwrap();
                centers[c] = data[far].clone();
            } else {
                let norm: f32 = sums[c].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                centers[c] = sums[c].iter().map(|x| x / norm).collect();
            }
        }
        if !changed { break; }
    }
    assign
}

/// Mann-Whitney AUC of scores for a binary flag (ties get 0.5 credit).
fn auc(scores: &[f64], pos: &[bool]) -> f64 {
    let n = scores.len();
    let n_pos = pos.iter().filter(|&&p| p).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 { return 0.5; }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && scores[order[j + 1]] == scores[order[i]] { j += 1; }
        let r = ((i + j + 2) as f64) / 2.0; // average rank, 1-based
        for t in i..=j { ranks[order[t]] = r; }
        i = j + 1;
    }
    let sum_pos: f64 = (0..n).filter(|&i| pos[i]).map(|i| ranks[i]).sum();
    (sum_pos - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos * n_neg) as f64
}

/// 2-D PCA via power iteration with deflation, for scatter plots only.
fn pca2(data: &[Vec<f32>]) -> Vec<(f32, f32)> {
    let n = data.len();
    let dim = data[0].len();
    let mut mean = vec![0.0f32; dim];
    for x in data { for d in 0..dim { mean[d] += x[d] / n as f32; } }
    let centered: Vec<Vec<f32>> = data.iter().map(|x| x.iter().zip(&mean).map(|(a, b)| a - b).collect()).collect();
    let mut component = |seed: &[f32]| -> Vec<f32> {
        let mut v: Vec<f32> = seed.to_vec();
        for _ in 0..100 {
            let mut nv = vec![0.0f32; dim];
            for x in &centered {
                let dot: f32 = x.iter().zip(&v).map(|(a, b)| a * b).sum();
                for d in 0..dim { nv[d] += dot * x[d]; }
            }
            let norm: f32 = nv.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v = nv.iter().map(|x| x / norm).collect();
        }
        v
    };
    let v1 = component(&vec![1.0 / (dim as f32).sqrt(); dim]);
    let v2_seed: Vec<f32> = centered.iter().map(|x| {
        let d: f32 = x.iter().zip(&v1).map(|(a, b)| a * b).sum();
        x.iter().zip(&v1).map(|(a, b)| a - d * b).sum::<f32>()
    }).collect();
    let v2 = component(&v2_seed);
    (0..n).map(|i| {
        let c1: f32 = centered[i].iter().zip(&v1).map(|(a, b)| a * b).sum();
        let c2: f32 = centered[i].iter().zip(&v2).map(|(a, b)| a * b).sum();
        (c1, c2)
    }).collect()
}

// ───────────────── residual extraction + perspective machinery ─────────────────

/// Cross-cluster relational residual: how much more related two items are
/// than their two baseline clusters normally are. Edges above a size-matched
/// shuffled null are kept — these are the cross-links a hard partition
/// treats as noise (mission Section 4).
struct Residual {
    edges: Vec<(usize, usize, f32)>,
    degree: Vec<f32>,
    baseline: Vec<usize>,
    k_baseline: usize,
    null_p95: f32,
}

fn cos_matrix(data: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = data.len();
    (0..n).map(|i| (0..n).map(|j| cosine_sim(&data[i], &data[j])).collect()).collect()
}

fn cluster_pair_means(m: &[Vec<f32>], part: &[usize], k: usize) -> Vec<f32> {
    let mut sums = vec![0.0f64; k * k];
    let mut cnts = vec![0usize; k * k];
    for i in 0..m.len() {
        for j in (i + 1)..m.len() {
            let (a, b) = (part[i], part[j]);
            sums[a * k + b] += m[i][j] as f64;
            sums[b * k + a] += m[i][j] as f64;
            cnts[a * k + b] += 1;
            cnts[b * k + a] += 1;
        }
    }
    (0..k * k).map(|ix| if cnts[ix] > 0 { (sums[ix] / cnts[ix] as f64) as f32 } else { 0.0 }).collect()
}

fn extract_residual(m: &[Vec<f32>], baseline: &[usize], k: usize, rng: &mut StdRng) -> Residual {
    let n = m.len();
    let means = cluster_pair_means(m, baseline, k);
    let means_r = &means;
    let cross_w: Vec<(usize, usize, f32)> = (0..n).flat_map(|i| {
        let bi = baseline[i];
        let row = &m[i];
        (i + 1..n).filter_map(move |j| {
            if bi == baseline[j] { return None; }
            let w = row[j] - means_r[bi * k + baseline[j]];
            Some((i, j, w))
        })
    }).collect();
    // Size-matched null: shuffle baseline labels, recompute the same
    // cross-pair weights, collect the distribution (construction-matched
    // control — same pairs, same cluster sizes, alignment destroyed).
    let mut null_vals: Vec<f32> = Vec::new();
    let mut perm = baseline.to_vec();
    for _ in 0..20 {
        perm.shuffle(rng);
        let null_means = cluster_pair_means(m, &perm, k);
        for &(i, j, _) in &cross_w {
            if perm[i] != perm[j] {
                null_vals.push(m[i][j] - null_means[perm[i] * k + perm[j]]);
            }
        }
    }
    null_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let null_p95 = if null_vals.is_empty() {
        0.0
    } else {
        null_vals[(0.95 * (null_vals.len() - 1) as f64).round() as usize]
    };
    let edges: Vec<(usize, usize, f32)> = cross_w
        .into_iter()
        .filter(|&(_, _, w)| w > 0.0 && w > null_p95)
        .collect();
    let mut degree = vec![0.0f32; n];
    for &(i, j, w) in &edges {
        degree[i] += w;
        degree[j] += w;
    }
    Residual { edges, degree, baseline: baseline.to_vec(), k_baseline: k, null_p95 }
}

/// Average-linkage greedy agglomeration over a weighted edge list.
/// Deterministic: max average weight, ties broken by lowest pair of
/// smallest-cluster ids. Falls back to merging the two smallest clusters
/// when no positive-weight pair remains. Returns labels 0..k_target.
fn agglomerate(n: usize, edges: &[(usize, usize, f32)], k_target: usize) -> Vec<usize> {
    assert!(k_target >= 2 && k_target <= n);
    let mut members: HashMap<usize, Vec<usize>> = (0..n).map(|i| (i, vec![i])).collect();
    let mut live: Vec<usize> = (0..n).collect();
    // incremental average-linkage state: (sum_w, count) per cluster pair
    let mut pw: HashMap<(usize, usize), (f64, usize)> = HashMap::new();
    for &(i, j, w) in edges {
        let key = if i < j { (i, j) } else { (j, i) };
        let e = pw.entry(key).or_insert((0.0, 0));
        e.0 += w as f64;
        e.1 += 1;
    }
    while live.len() > k_target {
        // best merge over live pairs
        let mut best: Option<(f64, usize, usize)> = None;
        for a in 0..live.len() {
            for b in (a + 1)..live.len() {
                let (ca, cb) = (live[a], live[b]);
                let key = if ca < cb { (ca, cb) } else { (cb, ca) };
                let (sw, c) = pw.get(&key).copied().unwrap_or((0.0, 0));
                let avg = if c > 0 { sw / c as f64 } else { 0.0 };
                let better = match best {
                    None => true,
                    Some((bw, _, _)) => avg > bw + 1e-12,
                };
                if better {
                    best = Some((avg, ca.min(cb), ca.max(cb)));
                }
            }
        }
        let (a, b) = match best {
            Some((w, a, b)) if w > 0.0 => (a, b),
            _ => {
                // no positive edge: merge the two smallest clusters (ties by id)
                let mut order = live.clone();
                order.sort_by_key(|&c| (members[&c].len(), c));
                (order[0], order[1])
            }
        };
        let merged = members.remove(&b).unwrap();
        members.get_mut(&a).unwrap().extend(merged);
        live.retain(|&c| c != b);
        // incremental pairwise sums; drop stale b entries
        let mut updates: Vec<((usize, usize), (f64, usize))> = Vec::new();
        for &c in live.iter() {
            if c == a { continue; }
            let kac = if a < c { (a, c) } else { (c, a) };
            let kbc = if b < c { (b, c) } else { (c, b) };
            let sac = pw.get(&kac).copied().unwrap_or((0.0, 0));
            let sbc = pw.get(&kbc).copied().unwrap_or((0.0, 0));
            updates.push((kac, (sac.0 + sbc.0, sac.1 + sbc.1)));
        }
        for (k, v) in updates { pw.insert(k, v); }
        pw.retain(|k, _| k.0 != b && k.1 != b);
    }
    // emit labels 0..k_target by scanning live clusters
    let mut out = vec![0usize; n];
    let mut ordered: Vec<usize> = live.clone();
    ordered.sort_unstable();
    for (label, &c) in ordered.iter().enumerate() {
        for &i in &members[&c] {
            out[i] = label;
        }
    }
    out
}

// ─────────────────────────── perspective object ───────────────────────────

/// A perspective is a first-class object: reference set + operator +
/// transformation + scoring function + discovered structure + lineage
/// (mission Sections 3 and 7). Serialized into results.json.
#[derive(Clone, Serialize)]
struct Perspective {
    id: String,
    parent: Option<String>,
    generator: String,
    level: usize,
    /// reference texts (empty for projection-type perspectives)
    reference_texts: Vec<String>,
    /// reference vectors in the base embedding space (transferable)
    reference_vecs: Vec<Vec<f32>>,
    operator: String,
    embedding_dim: usize,
    // Section-7 bookkeeping
    residual_edges: usize,
    coherence_before: f64,
    coherence_after: f64,
    silhouette_raw_before: f64,
    silhouette_raw_after: f64,
    internal_gain: f64,
    prediction_consistency_before: f64,
    prediction_consistency_after: f64,
    // the discovered structure
    partition: Vec<usize>,
    k_chosen: usize,
    /// silhouette of the discovered partition in the perspective's own space
    silhouette_own_space: f64,
    // evaluation-only fields (filled by eval, never used in selection)
    latent_ari: f64,
    latent_nmi: f64,
    latent_purity: f64,
}

/// Apply a frozen perspective transformation: x ↦ [cos(x, r) for r in refs].
fn transform_via_refs(x: &[f32], refs: &[Vec<f32>]) -> Vec<f32> {
    refs.iter().map(|r| cosine_sim(x, r)).collect()
}

/// Soft-membership profile: softmax over cosine to baseline centroids (tau).
fn soft_profile(x: &[f32], centroids: &[Vec<f32>], tau: f32) -> Vec<f32> {
    let sims: Vec<f32> = centroids.iter().map(|c| cosine_sim(x, c)).collect();
    let max = sims.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = sims.iter().map(|&s| ((s - max) / tau).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-9);
    exps.iter().map(|e| e / sum).collect()
}

fn centroid_of(vs: &[&Vec<f32>]) -> Vec<f32> {
    let dim = vs[0].len();
    let mut sum = vec![0.0f32; dim];
    for v in vs {
        for d in 0..dim { sum[d] += v[d]; }
    }
    let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    sum.iter().map(|x| x / norm).collect()
}

/// A candidate perspective generated from observed structure (never labels).
struct Candidate {
    name: String,
    generator: String,
    /// reference texts backing the transformation (for provenance/transfer)
    ref_texts: Vec<String>,
    /// reference vectors in the base space (frozen, transferable)
    ref_vecs: Vec<Vec<f32>>,
    operator: &'static str,
    /// build the transformation for arbitrary items
    build: Box<dyn Fn(&[Vec<f32>]) -> Vec<Vec<f32>>>,
}

// ─────────────────────── discovery protocol (uniform) ───────────────────────

/// Uniform protocol applied to EVERY candidate (mission Sections 5-6):
/// re-represent, re-discover with the same internal machinery (k swept 2..=6
/// by silhouette in the perspective's own space), then score the discovered
/// partition in RAW space — the primary internal measure. No labels enter.
fn discover_under(
    id: &str,
    parent: Option<&str>,
    generator: &str,
    level: usize,
    baseline_sil: f64,
    baseline_coh: f64,
    baseline_pred: f64,
    residual_edges: usize,
    reference_texts: Vec<String>,
    reference_vecs: Vec<Vec<f32>>,
    operator: &str,
    transformed: &[Vec<f32>],
    train_emb: &[Vec<f32>],
    rng: &mut StdRng,
) -> Perspective {
    let transformed_owned = transformed.to_vec();
    let transformed = &transformed_owned;
    // internal k-choice, in the perspective's own space
    let mut best: Option<(f64, usize, Vec<usize>)> = None;
    for k in 2..=6 {
        let part = kmeans(&transformed, k, rng, 50);
        let s = silhouette(&transformed, &part);
        let better = match &best { None => true, Some((bs, _, _)) => s > *bs + 1e-9 };
        if better { best = Some((s, k, part)); }
    }
    let (sil_p, k_chosen, partition) = best.unwrap();
    // primary internal score: the discovered partition's coherence in RAW space
    let sil_raw = silhouette(train_emb, &partition);
    // coherence: mean within-group cosine (in the perspective's own space)
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &g) in partition.iter().enumerate() { groups.entry(g).or_default().push(i); }
    let mut coh_sum = 0.0f64;
    let mut coh_n = 0usize;
    for members in groups.values() {
        if members.len() < 2 { continue; }
        for (a, &i) in members.iter().enumerate() {
            for &j in &members[a + 1..] {
                coh_sum += cosine_sim(&transformed[i], &transformed[j]) as f64;
                coh_n += 1;
            }
        }
    }
    let coh = if coh_n > 0 { coh_sum / coh_n as f64 } else { 0.0 };
    // label-free prediction consistency: fraction of items whose LOO-1NN
    // (in the perspective space) shares their discovered group
    let mut hits = 0usize;
    for i in 0..train_emb.len() {
        let (mut best_j, mut best_s) = (usize::MAX, f32::NEG_INFINITY);
        for j in 0..train_emb.len() {
            if i == j { continue; }
            let s = cosine_sim(&transformed[i], &transformed[j]);
            if s > best_s { best_s = s; best_j = j; }
        }
        if best_j != usize::MAX && partition[best_j] == partition[i] { hits += 1; }
    }
    let pred = hits as f64 / train_emb.len() as f64;
    Perspective {
        id: id.to_string(),
        parent: parent.map(|p| p.to_string()),
        generator: generator.to_string(),
        level,
        reference_texts,
        reference_vecs,
        operator: operator.to_string(),
        embedding_dim: train_emb[0].len(),
        residual_edges,
        coherence_before: baseline_coh,
        coherence_after: coh,
        silhouette_raw_before: baseline_sil,
        silhouette_raw_after: sil_raw,
        internal_gain: sil_raw - baseline_sil,
        prediction_consistency_before: baseline_pred,
        prediction_consistency_after: pred,
        partition,
        k_chosen,
        silhouette_own_space: sil_p,
        latent_ari: 0.0,
        latent_nmi: 0.0,
        latent_purity: 0.0,
    }
}

// ───────────────── candidate generation from residual (Section 5) ─────────────────

const EDGE_COMMUNITIES: usize = 6; // fixed cap for residual-graph communities
const N_REF_REFS: usize = 12;

/// Build all candidates from the residual structure + baseline. Every
/// reference vector/text comes from the data, never from labels.
fn build_candidates(
    residual: &Residual,
    m: &[Vec<f32>],
    train_emb: &[Vec<f32>],
    train_texts: &[String],
    baseline_centroids: &[Vec<f32>],
    rng: &mut StdRng,
) -> Vec<Candidate> {
    let n = train_emb.len();
    let dim = train_emb[0].len();
    let mut cands: Vec<Candidate> = Vec::new();

    // P_edge — partition the residual graph itself; references = community centroids
    {
        let communities = agglomerate(n, &residual.edges, EDGE_COMMUNITIES);
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &g) in communities.iter().enumerate() { groups.entry(g).or_default().push(i); }
        let mut keys: Vec<usize> = groups.keys().copied().collect();
        keys.sort_unstable();
        let vecs: Vec<Vec<f32>> = keys.iter().map(|&g| {
            let vs: Vec<&Vec<f32>> = groups[&g].iter().map(|&i| &train_emb[i]).collect();
            centroid_of(&vs)
        }).collect();
        let texts: Vec<String> = keys.iter().map(|&g| {
            let members = &groups[&g];
            train_texts[members[0]].clone()
        }).collect();
        cands.push(Candidate {
            name: "P_edge".into(),
            generator: "residual-edge".into(),
            ref_texts: texts,
            ref_vecs: vecs.clone(),
            operator: "cosine-to-references",
            build: Box::new(move |items: &[Vec<f32>]| {
                items.iter().map(|x| transform_via_refs(x, &vecs)).collect()
            }),
        });
    }

    // P_soft — soft-membership profiles over baseline centroids
    {
        let c0 = baseline_centroids.to_vec();
        cands.push(Candidate {
            name: "P_soft".into(),
            generator: "soft-membership".into(),
            ref_texts: Vec::new(),
            ref_vecs: c0.clone(),
            operator: "softmax-cos-tau0.1",
            build: Box::new(move |items: &[Vec<f32>]| {
                items.iter().map(|x| soft_profile(x, &c0, 0.1)).collect()
            }),
        });
    }

    // P_ref — residual-degree-selected references
    {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| residual.degree[b].partial_cmp(&residual.degree[a]).unwrap());
        let refs_idx: Vec<usize> = order.into_iter().take(N_REF_REFS).collect();
        let vecs: Vec<Vec<f32>> = refs_idx.iter().map(|&i| train_emb[i].clone()).collect();
        let texts: Vec<String> = refs_idx.iter().map(|&i| train_texts[i].clone()).collect();
        cands.push(Candidate {
            name: "P_ref".into(),
            generator: "residual-reference".into(),
            ref_texts: texts,
            ref_vecs: vecs.clone(),
            operator: "cosine-to-references",
            build: Box::new(move |items: &[Vec<f32>]| {
                items.iter().map(|x| transform_via_refs(x, &vecs)).collect()
            }),
        });
    }

    // P_edge-nores — ablation: same agglomeration on the FULL similarity graph
    {
        let full: Vec<(usize, usize, f32)> = (0..n).flat_map(|i| {
            (i + 1..n).map(move |j| (i, j, m[i][j]))
        }).collect();
        let communities = agglomerate(n, &full, EDGE_COMMUNITIES);
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &g) in communities.iter().enumerate() { groups.entry(g).or_default().push(i); }
        let mut keys: Vec<usize> = groups.keys().copied().collect();
        keys.sort_unstable();
        let vecs: Vec<Vec<f32>> = keys.iter().map(|&g| {
            let vs: Vec<&Vec<f32>> = groups[&g].iter().map(|&i| &train_emb[i]).collect();
            centroid_of(&vs)
        }).collect();
        cands.push(Candidate {
            name: "P_edge-nores".into(),
            generator: "full-edge-ablation".into(),
            ref_texts: Vec::new(),
            ref_vecs: vecs.clone(),
            operator: "cosine-to-references",
            build: Box::new(move |items: &[Vec<f32>]| {
                items.iter().map(|x| transform_via_refs(x, &vecs)).collect()
            }),
        });
    }

    // P_ref-randref — ablation: same count of references, chosen at random
    {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(rng);
        let vecs: Vec<Vec<f32>> = idx.iter().take(N_REF_REFS).map(|&i| train_emb[i].clone()).collect();
        cands.push(Candidate {
            name: "P_ref-randref".into(),
            generator: "random-reference-ablation".into(),
            ref_texts: idx.iter().take(N_REF_REFS).map(|&i| train_texts[i].clone()).collect(),
            ref_vecs: vecs.clone(),
            operator: "cosine-to-references",
            build: Box::new(move |items: &[Vec<f32>]| {
                items.iter().map(|x| transform_via_refs(x, &vecs)).collect()
            }),
        });
    }

    // P_rand — random projection control (mission Section 11: decisive)
    {
        let mut g: Vec<Vec<f32>> = (0..64).map(|_| {
            (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
        }).collect();
        for row in g.iter_mut() {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in row.iter_mut() { *x /= norm; }
        }
        cands.push(Candidate {
            name: "P_rand".into(),
            generator: "random-projection".into(),
            ref_texts: Vec::new(),
            ref_vecs: Vec::new(),
            operator: "random-projection-64",
            build: Box::new(move |items: &[Vec<f32>]| {
                items.iter().map(|x| {
                    g.iter().map(|r| {
                        let d: f32 = x.iter().zip(r).map(|(a, b)| a * b).sum();
                        d
                    }).collect::<Vec<f32>>()
                }).collect()
            }),
        });
    }

    cands
}

// ───────────────────── held-out evaluation (Section 10) ─────────────────────

#[derive(Serialize, Clone)]
struct HeldoutReport {
    method: String,
    /// 1-NN latent-role accuracy on test items (bridges excluded)
    acc_1nn: f64,
    acc_1nn_ci: (f64, f64),
    /// same-flow link AUC over cross-domain test pairs (bridges excluded)
    link_auc: f64,
    link_auc_ci: (f64, f64),
    /// number of test items / pairs the metrics were computed over
    n_test_items: usize,
    n_test_pairs: usize,
}

/// Frozen-transform held-out protocol: train and test re-represented by the
/// SAME frozen transformation; prediction = nearest train item in that space.
fn heldout_eval(
    method: &str,
    train_emb: &[Vec<f32>],
    test_emb: &[Vec<f32>],
    build: &dyn Fn(&[Vec<f32>]) -> Vec<Vec<f32>>,
    train_latent: &[usize],
    test_latent: &[usize],
    test_surface: &[usize],
    test_is_bridge: &[bool],
    rng: &mut StdRng,
) -> HeldoutReport {
    let mut all: Vec<Vec<f32>> = train_emb.to_vec();
    all.extend(test_emb.to_vec());
    let transformed = build(&all);
    let n_train = train_emb.len();
    let tr = &transformed[..n_train];
    let te = &transformed[n_train..];

    // 1-NN accuracy over non-bridge test items
    let idxs: Vec<usize> = (0..te.len()).filter(|&i| !test_is_bridge[i]).collect();
    let mut correct = 0usize;
    for &i in &idxs {
        let (mut bj, mut bs) = (usize::MAX, f32::NEG_INFINITY);
        for (j, t) in tr.iter().enumerate() {
            let s = cosine_sim(&te[i], t);
            if s > bs { bs = s; bj = j; }
        }
        if train_latent[bj] == test_latent[i] { correct += 1; }
    }
    let acc = correct as f64 / idxs.len().max(1) as f64;
    // bootstrap CI over test items
    let mut accs: Vec<f64> = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let mut ok = 0usize;
        let mut m = 0usize;
        for _ in 0..idxs.len() {
            let i = idxs[rng.gen_range(0..idxs.len())];
            m += 1;
            let (mut bj, mut bs) = (usize::MAX, f32::NEG_INFINITY);
            for (j, t) in tr.iter().enumerate() {
                let s = cosine_sim(&te[i], t);
                if s > bs { bs = s; bj = j; }
            }
            if train_latent[bj] == test_latent[i] { ok += 1; }
        }
        accs.push(ok as f64 / m.max(1) as f64);
    }
    accs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ci = (accs[24], accs[974]);

    // link AUC on cross-domain non-bridge test pairs
    let mut pairs: Vec<(f64, bool)> = Vec::new();
    for (a, &i) in idxs.iter().enumerate() {
        for &j in idxs[a + 1..].iter() {
            if test_surface[i] == test_surface[j] { continue; }
            let s = cosine_sim(&te[i], &te[j]) as f64;
            pairs.push((s, test_latent[i] == test_latent[j]));
        }
    }
    let scores: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let flags: Vec<bool> = pairs.iter().map(|p| p.1).collect();
    let a = auc(&scores, &flags);
    let mut aucs: Vec<f64> = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let mut sc = scores.clone();
        let mut fl = flags.clone();
        // bootstrap over pairs
        let np = pairs.len();
        let mut bs_s = Vec::with_capacity(np);
        let mut bs_f = Vec::with_capacity(np);
        for _ in 0..np {
            let k = rng.gen_range(0..np);
            bs_s.push(sc[k]);
            bs_f.push(fl[k]);
        }
        sc = bs_s; fl = bs_f;
        aucs.push(auc(&sc, &fl));
    }
    aucs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    HeldoutReport {
        method: method.to_string(),
        acc_1nn: acc,
        acc_1nn_ci: ci,
        link_auc: a,
        link_auc_ci: (aucs[24], aucs[974]),
        n_test_items: idxs.len(),
        n_test_pairs: pairs.len(),
    }
}

// ───────────────────────── visualizations (SVG) ─────────────────────────

const PALETTE: [&str; 8] = ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951", "#a463f2", "#ff8ab7", "#9498a0"];

fn svg_open(w: u32, h: u32) -> String {
    format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">\n<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n")
}

fn scatter_svg(pts: &[(f32, f32)], colors: &[usize], title: &str, sub: &str) -> String {
    let mut s = svg_open(560, 560);
    s.push_str(&format!("<text x=\"16\" y=\"24\" font-size=\"15\" font-family=\"sans-serif\" font-weight=\"bold\">{title}</text>\n"));
    s.push_str(&format!("<text x=\"16\" y=\"42\" font-size=\"11\" font-family=\"sans-serif\" fill=\"#555\">{sub}</text>\n"));
    let xs: Vec<f32> = pts.iter().map(|p| p.0).collect();
    let ys: Vec<f32> = pts.iter().map(|p| p.1).collect();
    let (xmin, xmax) = (xs.iter().cloned().fold(f32::INFINITY, f32::min), xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    let (ymin, ymax) = (ys.iter().cloned().fold(f32::INFINITY, f32::min), ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    let sx = |v: f32| 50.0 + (v - xmin) / (xmax - xmin + 1e-6) * 480.0;
    let sy = |v: f32| 520.0 - (v - ymin) / (ymax - ymin + 1e-6) * 460.0;
    for (p, &c) in pts.iter().zip(colors) {
        s.push_str(&format!("<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"3.2\" fill=\"{}\" fill-opacity=\"0.75\"/>\n", sx(p.0), sy(p.1), PALETTE[c % PALETTE.len()]));
    }
    s.push_str("</svg>\n");
    s
}

/// Adjacency heat matrix: items ordered by surface blocks; residual edges
/// as dots. Cross-block dots ARE the discarded relational structure.
fn residual_matrix_svg(edges: &[(usize, usize, f32)], order: &[usize], blocks: &[usize], title: &str) -> String {
    let n = order.len();
    let cell = 2.2f32;
    let pad = 60.0;
    let w = pad + n as f32 * cell + 30.0;
    let h = pad + n as f32 * cell + 30.0;
    let mut s = svg_open(w as u32, h as u32);
    s.push_str(&format!("<text x=\"10\" y=\"18\" font-size=\"12\" font-family=\"sans-serif\" font-weight=\"bold\">{title}</text>\n"));
    // block boundaries
    let mut pos_of = vec![0usize; n];
    for (pos, &i) in order.iter().enumerate() { pos_of[i] = pos; }
    let mut bstart = 0usize;
    for b in 1..=blocks.len() + 1 {
        let end = order.iter().take_while(|&&i| blocks[i] < b).count();
        if end > bstart {
            s.push_str(&format!("<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"none\" stroke=\"#999\" stroke-width=\"0.6\"/>\n",
                pad + bstart as f32 * cell, pad + bstart as f32 * cell, (end - bstart) as f32 * cell, (end - bstart) as f32 * cell));
            bstart = end;
        }
        if b > blocks.len() { break; }
    }
    for &(i, j, wgt) in edges {
        let (pi, pj) = (pos_of[i], pos_of[j]);
        let opacity = (0.25 + 0.75 * wgt).min(1.0);
        s.push_str(&format!("<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"#c0392b\" fill-opacity=\"{:.2}\"/>\n",
            pad + pi as f32 * cell, pad + pj as f32 * cell, cell, cell, opacity));
        s.push_str(&format!("<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"#c0392b\" fill-opacity=\"{:.2}\"/>\n",
            pad + pj as f32 * cell, pad + pi as f32 * cell, cell, cell, opacity));
    }
    s.push_str("</svg>\n");
    s
}

fn heatmap_svg(labels: &[String], vals: &[Vec<f64>], title: &str) -> String {
    let n = labels.len();
    let cell = 60.0;
    let pad = 190.0;
    let w = pad + n as f32 * cell + 20.0;
    let h = 40.0 + n as f32 * cell + 20.0;
    let mut s = svg_open(w as u32, h as u32);
    s.push_str(&format!("<text x=\"10\" y=\"20\" font-size=\"13\" font-family=\"sans-serif\" font-weight=\"bold\">{title}</text>\n"));
    for (r, row) in vals.iter().enumerate() {
        s.push_str(&format!("<text x=\"{}\" y=\"{:.1}\" text-anchor=\"end\" font-size=\"11\" font-family=\"sans-serif\">{}</text>\n", pad - 6.0, 40.0 + r as f32 * cell + cell / 2.0 + 4.0, labels[r]))
            ;
        for (c, &v) in row.iter().enumerate() {
            let t = v.clamp(0.0, 1.0);
            let rcol = (255.0 * t) as u32;
            let gcol = (255.0 * (1.0 - t)) as u32;
            s.push_str(&format!("<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"rgb({rcol},{gcol},60)\" stroke=\"#ddd\"/>\n",
                pad + c as f32 * cell, 40.0 + r as f32 * cell, cell - 2.0, cell - 2.0));
            s.push_str(&format!("<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"11\" font-family=\"monospace\">{:.3}</text>\n",
                pad + c as f32 * cell + (cell - 2.0) / 2.0, 40.0 + r as f32 * cell + cell / 2.0 + 4.0, v));
        }
    }
    s.push_str("</svg>\n");
    s
}

/// Perspective lineage tree: parent -> child with internal gains.
fn tree_svg(nodes: &[(String, Option<String>, f64)], title: &str) -> String {
    let mut s = svg_open(760, 300);
    s.push_str(&format!("<text x=\"10\" y=\"20\" font-size=\"13\" font-family=\"sans-serif\" font-weight=\"bold\">{title}</text>\n"));
    let mut y = 60.0f32;
    for (id, parent, gain) in nodes {
        let x = match parent { None => 60.0, Some(_) => 360.0 };
        s.push_str(&format!("<rect x=\"{x:.0}\" y=\"{y:.0}\" width=\"300\" height=\"26\" rx=\"5\" fill=\"#eef\" stroke=\"#889\"/>\n"));
        s.push_str(&format!("<text x=\"{:.0}\" y=\"{:.0}\" font-size=\"11\" font-family=\"sans-serif\">{} (raw-sil gain {:+.4})</text>\n", x + 8.0, y + 17.0, id, gain));
        if let Some(p) = parent {
            s.push_str(&format!("<line x1=\"360\" y1=\"{:.0}\" x2=\"{}\" y2=\"{:.0}\" stroke=\"#889\"/>\n", y + 13.0, x, y + 13.0));
            let _ = p;
        }
        y += 36.0;
    }
    s.push_str("</svg>\n");
    s
}

fn bars_svg(entries: &[(String, f64, f64, f64)], title: &str, ylabel: &str) -> String {
    let n = entries.len();
    let w = 40.0 + n as f32 * 90.0 + 40.0;
    let mut s = svg_open(w as u32, 340);
    s.push_str(&format!("<text x=\"10\" y=\"20\" font-size=\"13\" font-family=\"sans-serif\" font-weight=\"bold\">{title}</text>\n"));
    s.push_str(&format!("<text x=\"10\" y=\"36\" font-size=\"10\" font-family=\"sans-serif\" fill=\"#555\">{ylabel}</text>\n"));
    let mut max_v = 0.001f64;
    for e in entries { max_v = max_v.max(e.1).max(e.2); }
    let y = |v: f64| 300.0 - (v / max_v) as f32 * 240.0;
    for (i, (name, v, lo, hi)) in entries.iter().enumerate() {
        let x0 = 40.0 + i as f32 * 90.0;
        s.push_str(&format!("<rect x=\"{:.1}\" y=\"{:.1}\" width=\"46\" height=\"{:.1}\" fill=\"#4269d0\"/>\n", x0, y(*v), 300.0 - y(*v)));
        s.push_str(&format!("<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#c0392b\" stroke-width=\"2\"/>\n", x0 + 8.0, y(*lo), x0 + 38.0, y(*hi)));
        s.push_str(&format!("<text x=\"{:.1}\" y=\"315\" text-anchor=\"middle\" font-size=\"9\" font-family=\"sans-serif\" transform=\"rotate(-20 {:.1} 315)\">{}</text>\n", x0 + 23.0, x0 + 23.0, name));
        s.push_str(&format!("<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"9\" font-family=\"monospace\">{:.3}</text>\n", x0 + 23.0, y(*v) - 4.0, v));
    }
    s.push_str("</svg>\n");
    s
}

fn write_file(path: &std::path::Path, content: &str) {
    if let Some(p) = path.parent() { let _ = std::fs::create_dir_all(p); }
    std::fs::write(path, content).unwrap_or_else(|e| eprintln!("warning: could not write {}: {e}", path.display()));
}

// ─────────────────────────── pipeline (S1–S8) ───────────────────────────

#[derive(Serialize, Clone)]
struct ResidualInfo {
    n_edges: usize,
    null_p95: f32,
    top_residual_items: Vec<String>,
    /// edges for the residual-matrix visualization (items indexed as trained)
    edges: Vec<(usize, usize, f32)>,
}

#[derive(Serialize, Clone)]
struct RoutePair {
    a: String,
    b: String,
    ari: f64,
    nmi: f64,
}

#[derive(Serialize)]
struct ConvergenceReport {
    route_pairs: Vec<RoutePair>,
    permuted_null_ari_mean: f64,
    permuted_null_ari_p95: f64,
    consensus_pair_precision: f64,
    consensus_pair_recall: f64,
    n_consensus_pairs: usize,
    cross_embedder: Vec<RoutePair>,
}

#[derive(Serialize, Clone)]
struct CompressionPoint {
    method: String,
    n_refs: usize,
    acc_1nn: f64,
}

#[derive(Serialize)]
struct SelectionControl {
    rule: String,
    selected: String,
    heldout: HeldoutReport,
}

#[derive(Serialize)]
struct PipelineOutput {
    embedder: String,
    dim: usize,
    baseline_k: usize,
    baseline_surface_purity: f64,
    baseline_latent_ari: f64,
    baseline_silhouette: f64,
    residual: ResidualInfo,
    candidates: Vec<Perspective>,
    winner_id: String,
    convergence: ConvergenceReport,
    heldout: Vec<HeldoutReport>,
    transfer: Vec<HeldoutReport>,
    compression: Vec<CompressionPoint>,
    stability_winner: f64,
    stability_baseline: f64,
    selection_controls: Vec<SelectionControl>,
    oracle_train_ari: f64,
    oracle_heldout: HeldoutReport,
    recursion: Vec<Perspective>,
    recursion_accepted: bool,
}

struct Splits {
    train: Vec<Item>,
    test: Vec<Item>,
    transfer: Vec<Item>,
}

#[allow(clippy::too_many_arguments)]
fn run_pipeline(
    embedder_name: &str,
    train_emb: &[Vec<f32>],
    test_emb: &[Vec<f32>],
    transfer_emb: &[Vec<f32>],
    splits: &Splits,
    rng: &mut StdRng,
) -> PipelineOutput {
    let train = &splits.train;
    let test = &splits.test;
    let transfer = &splits.transfer;
    let train_texts: Vec<String> = train.iter().map(|i| i.text.clone()).collect();
    let train_surface: Vec<usize> = train.iter().map(|i| i.surface).collect();
    let train_latent: Vec<usize> = train.iter().map(|i| i.flows[0]).collect();
    let test_latent: Vec<usize> = test.iter().map(|i| i.flows[0]).collect();
    let test_surface: Vec<usize> = test.iter().map(|i| i.surface).collect();
    let test_bridge: Vec<bool> = test.iter().map(|i| i.is_bridge).collect();
    let transfer_latent: Vec<usize> = transfer.iter().map(|i| i.flows[0]).collect();
    let transfer_surface: Vec<usize> = transfer.iter().map(|i| i.surface).collect();
    let transfer_bridge: Vec<bool> = transfer.iter().map(|i| i.is_bridge).collect();
    let n = train_emb.len();
    let dim = train_emb[0].len();

    // S1 — baseline organization (internal k by raw-space silhouette)
    let m = cos_matrix(train_emb);
    let mut baseline: Option<(f64, usize, Vec<usize>)> = None;
    for k in 2..=6 {
        let part = kmeans(train_emb, k, rng, 50);
        let s = silhouette(train_emb, &part);
        let better = match &baseline { None => true, Some((bs, _, _)) => s > *bs + 1e-9 };
        if better { baseline = Some((s, k, part)); }
    }
    let (baseline_sil, baseline_k, baseline_part) = baseline.unwrap();
    let base_surf_pur = purity(&baseline_part, &train_surface);
    let base_lat_ari = ari(&baseline_part, &train_latent);
    // baseline coherence: mean within-cluster cosine (raw)
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &g) in baseline_part.iter().enumerate() { groups.entry(g).or_default().push(i); }
    let mut coh = 0.0f64;
    let mut coh_n = 0usize;
    for mem in groups.values() {
        for (a, &i) in mem.iter().enumerate() {
            for &j in &mem[a + 1..] {
                coh += m[i][j] as f64;
                coh_n += 1;
            }
        }
    }
    let baseline_coh = if coh_n > 0 { coh / coh_n as f64 } else { 0.0 };
    // baseline prediction consistency (LOO-1NN raw)
    let mut hits = 0usize;
    for i in 0..n {
        let (mut bj, mut bs) = (usize::MAX, f32::NEG_INFINITY);
        for j in 0..n {
            if i == j { continue; }
            if m[i][j] > bs { bs = m[i][j]; bj = j; }
        }
        if baseline_part[bj] == baseline_part[i] { hits += 1; }
    }
    let baseline_pred = hits as f64 / n as f64;

    // S2 — residual
    let residual = extract_residual(&m, &baseline_part, baseline_k, rng);
    let mut deg_order: Vec<usize> = (0..n).collect();
    deg_order.sort_by(|&a, &b| residual.degree[b].partial_cmp(&residual.degree[a]).unwrap());
    let top_items: Vec<String> = deg_order.iter().take(N_REF_REFS).map(|&i| train_texts[i].clone()).collect();
    let res_info = ResidualInfo { n_edges: residual.edges.len(), null_p95: residual.null_p95, top_residual_items: top_items.clone(), edges: residual.edges.clone() };

    // baseline centroids for P_soft + oracle
    let mut base_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &g) in baseline_part.iter().enumerate() { base_groups.entry(g).or_default().push(i); }
    let mut base_keys: Vec<usize> = base_groups.keys().copied().collect();
    base_keys.sort_unstable();
    let baseline_centroids: Vec<Vec<f32>> = base_keys.iter().map(|&g| {
        let vs: Vec<&Vec<f32>> = base_groups[&g].iter().map(|&i| &train_emb[i]).collect();
        centroid_of(&vs)
    }).collect();

    // S3/S4 — candidates + discovery
    let mut cands = build_candidates(&residual, &m, train_emb, &train_texts, &baseline_centroids, rng);
    let mut perspectives: Vec<Perspective> = Vec::new();
    for cand in cands.iter() {
        let transformed = (cand.build)(train_emb);
        let p = discover_under(
            &cand.name,
            None,
            &cand.generator,
            1,
            baseline_sil,
            baseline_coh,
            baseline_pred,
            residual.edges.len(),
            cand.ref_texts.clone(),
            cand.ref_vecs.clone(),
            cand.operator,
            &transformed,
            train_emb,
            rng,
        );
        let mut p = p;
        p.latent_ari = ari(&p.partition, &train_latent);
        p.latent_nmi = nmi(&p.partition, &train_latent);
        p.latent_purity = purity(&p.partition, &train_latent);
        perspectives.push(p);
    }

    // oracle (evaluation-only upper bound)
    let mut oracle_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &f) in train_latent.iter().enumerate() { oracle_groups.entry(f).or_default().push(i); }
    let oracle_vecs: Vec<Vec<f32>> = (0..N_FLOWS).map(|f| {
        let vs: Vec<&Vec<f32>> = oracle_groups[&f].iter().map(|&i| &train_emb[i]).collect();
        centroid_of(&vs)
    }).collect();
    let oracle_transformed: Vec<Vec<f32>> = train_emb.iter().map(|x| transform_via_refs(x, &oracle_vecs)).collect();
    let oracle_part = kmeans(&oracle_transformed, N_FLOWS, rng, 50);
    let oracle_ari = ari(&oracle_part, &train_latent);
    let oracle_build = |items: &[Vec<f32>]| -> Vec<Vec<f32>> {
        items.iter().map(|x| transform_via_refs(x, &oracle_vecs)).collect()
    };

    // internal selection: max raw-space silhouette (never labels)
    let winner_idx = perspectives.iter().enumerate()
        .max_by(|a, b| a.1.silhouette_raw_after.partial_cmp(&b.1.silhouette_raw_after).unwrap())
        .map(|(i, _)| i).unwrap();
    let winner_id = perspectives[winner_idx].id.clone();
    // selection controls: same candidates, different internal rules
    let coh_pick = perspectives.iter().enumerate()
        .max_by(|a, b| a.1.coherence_after.partial_cmp(&b.1.coherence_after).unwrap())
        .map(|(i, _)| i).unwrap();
    let rand_pick = rng.gen_range(0..perspectives.len());
    let winner_build = std::mem::replace(&mut cands[winner_idx].build, Box::new(|x: &[Vec<f32>]| x.to_vec()));
    let coh_build = std::mem::replace(&mut cands[coh_pick].build, Box::new(|x: &[Vec<f32>]| x.to_vec()));
    let rand_build = std::mem::replace(&mut cands[rand_pick].build, Box::new(|x: &[Vec<f32>]| x.to_vec()));
    let selection_controls = vec![
        SelectionControl { rule: "max-raw-silhouette".into(), selected: winner_id.clone(), heldout: heldout_eval(&winner_id, train_emb, test_emb, winner_build.as_ref(), &train_latent, &test_latent, &test_surface, &test_bridge, rng) },
        SelectionControl { rule: "max-own-space-coherence".into(), selected: perspectives[coh_pick].id.clone(), heldout: heldout_eval(&perspectives[coh_pick].id, train_emb, test_emb, coh_build.as_ref(), &train_latent, &test_latent, &test_surface, &test_bridge, rng) },
        SelectionControl { rule: "random".into(), selected: perspectives[rand_pick].id.clone(), heldout: heldout_eval(&perspectives[rand_pick].id, train_emb, test_emb, rand_build.as_ref(), &train_latent, &test_latent, &test_surface, &test_bridge, rng) },
    ];

    // S5 — recursive invention: residual under the winner's own partition.
    // Accept a child only on internal (raw-space silhouette) improvement.
    let mut recursion_chain: Vec<Perspective> = Vec::new();
    let mut accepted = false;
    {
        let mut parent_part = perspectives[winner_idx].partition.clone();
        let mut parent_sil = perspectives[winner_idx].silhouette_raw_after;
        let mut parent_id = winner_id.clone();
        for level in 2..=3 {
            let res2 = extract_residual(&m, &parent_part, perspectives[winner_idx].k_chosen, rng);
            let mut c2groups: HashMap<usize, Vec<usize>> = HashMap::new();
            for (i, &g) in parent_part.iter().enumerate() { c2groups.entry(g).or_default().push(i); }
            let mut c2keys: Vec<usize> = c2groups.keys().copied().collect();
            c2keys.sort_unstable();
            let parent_centroids: Vec<Vec<f32>> = c2keys.iter().map(|&g| {
                let vs: Vec<&Vec<f32>> = c2groups[&g].iter().map(|&i| &train_emb[i]).collect();
                centroid_of(&vs)
            }).collect();
            // children: P_edge child + P_ref child under the new residual
            let mut children: Vec<(String, Vec<String>, Vec<Vec<f32>>, &'static str, Vec<Vec<f32>>)> = Vec::new();
            {
                let comm = agglomerate(n, &res2.edges, EDGE_COMMUNITIES);
                let mut grp: HashMap<usize, Vec<usize>> = HashMap::new();
                for (i, &g) in comm.iter().enumerate() { grp.entry(g).or_default().push(i); }
                let mut keys: Vec<usize> = grp.keys().copied().collect();
                keys.sort_unstable();
                let vecs: Vec<Vec<f32>> = keys.iter().map(|&g| {
                    let vs: Vec<&Vec<f32>> = grp[&g].iter().map(|&i| &train_emb[i]).collect();
                    centroid_of(&vs)
                }).collect();
                children.push(("P_edge-child".into(), Vec::new(), vecs.clone(), "cosine-to-references",
                    train_emb.iter().map(|x| transform_via_refs(x, &vecs)).collect()));
            }
            {
                let mut ord: Vec<usize> = (0..n).collect();
                ord.sort_by(|&a, &b| res2.degree[b].partial_cmp(&res2.degree[a]).unwrap());
                let ridx: Vec<usize> = ord.into_iter().take(N_REF_REFS).collect();
                let vecs: Vec<Vec<f32>> = ridx.iter().map(|&i| train_emb[i].clone()).collect();
                let texts: Vec<String> = ridx.iter().map(|&i| train_texts[i].clone()).collect();
                children.push(("P_ref-child".into(), texts, vecs.clone(), "cosine-to-references",
                    train_emb.iter().map(|x| transform_via_refs(x, &vecs)).collect()));
            }
            let _ = &parent_centroids;
            let mut best_child: Option<(f64, Perspective)> = None;
            for (name, texts, vecs, op, transformed) in children {
                let mut p = discover_under(
                    &name, Some(&parent_id), "recursive", level,
                    parent_sil, baseline_coh, baseline_pred, res2.edges.len(),
                    texts, vecs, op, &transformed, train_emb, rng,
                );
                p.latent_ari = ari(&p.partition, &train_latent);
                p.latent_nmi = nmi(&p.partition, &train_latent);
                p.latent_purity = purity(&p.partition, &train_latent);
                let better = match &best_child {
                    None => true,
                    Some((bs, _)) => p.silhouette_raw_after > *bs + 1e-12,
                };
                if better { best_child = Some((p.silhouette_raw_after, p)); }
            }
            match best_child {
                Some((sil, child)) if sil > parent_sil + 0.005 => {
                    accepted = true;
                    recursion_chain.push(child.clone());
                    parent_part = child.partition.clone();
                    parent_sil = child.silhouette_raw_after;
                    parent_id = child.id.clone();
                }
                _ => break, // stop: no meaningful improvement (Section 7)
            }
        }
    }

    // S6 — cross-perspective convergence (same embedder)
    let route_names = ["P_edge", "P_soft", "P_ref"];
    let mut route_pairs: Vec<RoutePair> = Vec::new();
    for a in 0..route_names.len() {
        for b in (a + 1)..route_names.len() {
            let pa = &perspectives.iter().find(|p| p.id == route_names[a]).unwrap().partition;
            let pb = &perspectives.iter().find(|p| p.id == route_names[b]).unwrap().partition;
            route_pairs.push(RoutePair { a: route_names[a].into(), b: route_names[b].into(), ari: ari(pa, pb), nmi: nmi(pa, pb) });
        }
    }
    // size-matched permuted null for the strongest pair
    let pe = &perspectives.iter().find(|p| p.id == "P_edge").unwrap().partition;
    let ps = &perspectives.iter().find(|p| p.id == "P_soft").unwrap().partition;
    let mut nulls: Vec<f64> = Vec::new();
    let mut perm = pe.clone();
    for _ in 0..20 {
        perm.shuffle(rng);
        nulls.push(ari(&perm, ps));
    }
    nulls.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let null_mean = nulls.iter().sum::<f64>() / nulls.len() as f64;
    let null_p95 = nulls[(0.95 * (nulls.len() - 1) as f64).round() as usize];
    // consensus structure: pairs co-grouped by all three routes (singles only)
    let singles: Vec<usize> = (0..n).filter(|&i| !train[i].is_bridge).collect();
    let psoft = &perspectives.iter().find(|p| p.id == "P_soft").unwrap().partition;
    let pref = &perspectives.iter().find(|p| p.id == "P_ref").unwrap().partition;
    let mut n_cons = 0usize;
    let mut n_cons_lat = 0usize;
    let mut n_lat_total = 0usize;
    for (a, &i) in singles.iter().enumerate() {
        for &j in singles[a + 1..].iter() {
            let lat = train_latent[i] == train_latent[j];
            if lat { n_lat_total += 1; }
            let cons = pe[i] == pe[j] && psoft[i] == psoft[j] && pref[i] == pref[j];
            if cons { n_cons += 1; if lat { n_cons_lat += 1; } }
        }
    }
    let convergence = ConvergenceReport {
        route_pairs,
        permuted_null_ari_mean: null_mean,
        permuted_null_ari_p95: null_p95,
        consensus_pair_precision: if n_cons > 0 { n_cons_lat as f64 / n_cons as f64 } else { 0.0 },
        consensus_pair_recall: if n_lat_total > 0 { n_cons_lat as f64 / n_lat_total as f64 } else { 0.0 },
        n_consensus_pairs: n_cons,
        cross_embedder: Vec::new(),
    };

    // S7 — held-out prediction from frozen transformations
    let identity = |items: &[Vec<f32>]| -> Vec<Vec<f32>> { items.to_vec() };
    let mut heldout: Vec<HeldoutReport> = Vec::new();
    heldout.push(heldout_eval("raw", train_emb, test_emb, &identity, &train_latent, &test_latent, &test_surface, &test_bridge, rng));
    for (i, cand) in cands.iter().enumerate() {
        heldout.push(heldout_eval(&cand.name, train_emb, test_emb, cand.build.as_ref(), &train_latent, &test_latent, &test_surface, &test_bridge, rng));
        let _ = i;
    }
    if accepted {
        let last = recursion_chain.last().unwrap();
        let child_vecs = last.reference_vecs.clone();
        let child_build = move |items: &[Vec<f32>]| -> Vec<Vec<f32>> {
            items.iter().map(|x| transform_via_refs(x, &child_vecs)).collect()
        };
        heldout.push(heldout_eval("winner-child", train_emb, test_emb, &child_build, &train_latent, &test_latent, &test_surface, &test_bridge, rng));
    }

    // oracle held-out (upper bound, never used in selection)
    let oracle_heldout = heldout_eval("P_oracle", train_emb, test_emb, &oracle_build, &train_latent, &test_latent, &test_surface, &test_bridge, rng);

    // S8 — transfer to a second dataset with different surface domains
    let mut transfer_reports: Vec<HeldoutReport> = Vec::new();
    transfer_reports.push(heldout_eval("raw", train_emb, transfer_emb, &identity, &train_latent, &transfer_latent, &transfer_surface, &transfer_bridge, rng));
    for cand in cands.iter() {
        transfer_reports.push(heldout_eval(&cand.name, train_emb, transfer_emb, cand.build.as_ref(), &train_latent, &transfer_latent, &transfer_surface, &transfer_bridge, rng));
    }

    // compression: held-out accuracy with only the top-k residual references
    let mut compression: Vec<CompressionPoint> = Vec::new();
    for krefs in [3usize, 6, 12] {
        let vecs: Vec<Vec<f32>> = top_items.iter().take(krefs).zip(deg_order.iter().take(krefs))
            .map(|(_, &i)| train_emb[i].clone()).collect();
        let build = move |items: &[Vec<f32>]| -> Vec<Vec<f32>> {
            items.iter().map(|x| transform_via_refs(x, &vecs)).collect()
        };
        let rep = heldout_eval("P_ref-topk", train_emb, test_emb, &build, &train_latent, &test_latent, &test_surface, &test_bridge, rng);
        compression.push(CompressionPoint { method: "P_ref-topk".into(), n_refs: krefs, acc_1nn: rep.acc_1nn });
    }

    // stability: bootstrap re-discovery (80% subsamples), same internal rules
    let stability_winner = {
        let mut aris: Vec<f64> = Vec::new();
        for _ in 0..5 {
            let mut sub: Vec<usize> = (0..n).collect();
            sub.shuffle(rng);
            let take = (n * 8) / 10;
            sub.truncate(take);
            let sub_emb: Vec<Vec<f32>> = sub.iter().map(|&i| train_emb[i].clone()).collect();
            let ms = cos_matrix(&sub_emb);
            // baseline on the subsample
            let mut bpart = kmeans(&sub_emb, baseline_k, rng, 50);
            let rs = extract_residual(&ms, &bpart, baseline_k, rng);
            bpart = agglomerate(sub_emb.len(), &rs.edges, EDGE_COMMUNITIES);
            // winner partition restricted to the same items, relabeled coherently:
            // compare via ARI (ARI is label-permutation invariant)
            let w_restrict: Vec<usize> = sub.iter().map(|&i| perspectives[winner_idx].partition[i]).collect();
            aris.push(ari(&bpart, &w_restrict));
        }
        aris.iter().sum::<f64>() / aris.len() as f64
    };
    let stability_baseline = {
        let mut aris: Vec<f64> = Vec::new();
        for _ in 0..5 {
            let mut sub: Vec<usize> = (0..n).collect();
            sub.shuffle(rng);
            let take = (n * 8) / 10;
            sub.truncate(take);
            let sub_emb: Vec<Vec<f32>> = sub.iter().map(|&i| train_emb[i].clone()).collect();
            let bpart = kmeans(&sub_emb, baseline_k, rng, 50);
            let w_restrict: Vec<usize> = sub.iter().map(|&i| baseline_part[i]).collect();
            aris.push(ari(&bpart, &w_restrict));
        }
        aris.iter().sum::<f64>() / aris.len() as f64
    };

    PipelineOutput {
        embedder: embedder_name.to_string(),
        dim,
        baseline_k,
        baseline_surface_purity: base_surf_pur,
        baseline_latent_ari: base_lat_ari,
        baseline_silhouette: baseline_sil,
        residual: res_info,
        candidates: perspectives,
        winner_id,
        convergence,
        heldout,
        transfer: transfer_reports,
        compression,
        stability_winner,
        stability_baseline,
        selection_controls,
        oracle_train_ari: oracle_ari,
        oracle_heldout,
        recursion: recursion_chain,
        recursion_accepted: accepted,
    }
}

// ─────────────────────── tier resolution (pre-registered) ───────────────────────

fn heldout_of<'a>(reps: &'a [HeldoutReport], method: &str) -> Option<&'a HeldoutReport> {
    reps.iter().find(|r| r.method == method)
}

fn resolve_tiers(out: &PipelineOutput, cross_pairs: &[RoutePair]) -> (u8, Vec<(String, String)>) {
    let mut verdicts: Vec<(String, String)> = Vec::new();
    let winner = out.candidates.iter().find(|p| p.id == out.winner_id).unwrap();
    let rand_c = out.candidates.iter().find(|p| p.id == "P_rand").unwrap();
    let raw = heldout_of(&out.heldout, "raw").unwrap();
    let w_held = heldout_of(&out.heldout, &out.winner_id).unwrap_or(raw);
    let w_transfer = heldout_of(&out.transfer, &out.winner_id).unwrap_or_else(|| heldout_of(&out.transfer, "raw").unwrap());

    let margin_routes_ok = |rep: &HeldoutReport| {
        rep.acc_1nn >= raw.acc_1nn + 0.05 && rep.link_auc >= raw.link_auc + 0.05
    };

    let tier1 = winner.latent_ari >= out.baseline_latent_ari + 0.10
        && winner.silhouette_raw_after >= out.baseline_silhouette - 0.01
        && winner.latent_ari >= rand_c.latent_ari + 0.10;
    verdicts.push(("T1 train organization".into(), if tier1 { "PASS".into() } else { format!("FAIL (winner ARI {:.3} vs base {:.3} / rand {:.3}, raw-sil {:.4} vs {:.4})", winner.latent_ari, out.baseline_latent_ari, rand_c.latent_ari, winner.silhouette_raw_after, out.baseline_silhouette) }));

    let rand_held = heldout_of(&out.heldout, "P_rand").unwrap();
    let rand_control_ok = rand_held.acc_1nn < raw.acc_1nn + 0.05 && rand_held.acc_1nn < w_held.acc_1nn - 0.02;
    let t2 = tier1 && w_held.acc_1nn >= raw.acc_1nn + 0.05 && w_held.link_auc >= raw.link_auc + 0.05;
    verdicts.push(("T2 held-out".into(), if t2 { "PASS".into() } else { format!("FAIL (held acc {:+.3} vs raw {:.3}, AUC {:+.3} vs {:.3})", w_held.acc_1nn - raw.acc_1nn, raw.acc_1nn, w_held.link_auc - raw.link_auc, raw.link_auc) }));
    verdicts.push(("T2 P_rand control".into(), if rand_control_ok { "PASS (random projection stays below the margins)".into() } else { format!("FAIL (P_rand acc {:.3} — random perspective reaches the margins: hypothesis FALSIFIED)", rand_held.acc_1nn) }));

    let route_names = ["P_edge", "P_soft", "P_ref"];
    let routes_ok: Vec<&str> = route_names.iter().copied().filter(|rn| {
        heldout_of(&out.heldout, rn).map(margin_routes_ok).unwrap_or(false)
    }).collect();
    let max_route_ari = out.convergence.route_pairs.iter().map(|p| p.ari).fold(0.0f64, f64::max);
    let max_cross = cross_pairs.iter().map(|p| p.ari).fold(0.0f64, f64::max);
    let convergence_ok = max_route_ari >= 0.5
        && out.convergence.permuted_null_ari_p95 < 0.2
        && out.convergence.permuted_null_ari_p95 + 0.3 < max_route_ari;
    let t3 = t2 && ((routes_ok.len() >= 2 && convergence_ok) || max_cross >= 0.5);
    verdicts.push(("T3 convergence".into(), if t3 { "PASS".into() } else { format!("FAIL (routes_ok={:?}, max route ARI {:.3} null p95 {:.3}, max cross-embedder ARI {:.3})", routes_ok, max_route_ari, out.convergence.permuted_null_ari_p95, max_cross) }));

    let raw_transfer = heldout_of(&out.transfer, "raw").unwrap();
    let t4 = t3 && w_transfer.acc_1nn >= raw_transfer.acc_1nn + 0.05;
    verdicts.push(("T4 transfer".into(), if t4 { "PASS".into() } else { format!("FAIL (transfer acc {:+.3} vs raw {:.3})", w_transfer.acc_1nn - raw_transfer.acc_1nn, raw_transfer.acc_1nn) }));

    let t5 = t4 && out.recursion_accepted;
    if out.recursion_accepted {
        let last = out.recursion.last().unwrap();
        let child_held = heldout_of(&out.heldout, "winner-child");
        let keeps = child_held.map(|c| c.acc_1nn >= w_held.acc_1nn - 0.02).unwrap_or(false);
        verdicts.push(("T5 recursion keeps gain".into(), if keeps { "PASS".into() } else { "FAIL (child drops held-out gain)".into() }));
    } else {
        verdicts.push(("T5 recursion keeps gain".into(), "FAIL (recursion stopped: no internal gain at level 2)".into()));
    }

    let tier = if t5 { 5 } else if t4 { 4 } else if t3 { 3 } else if t2 { 2 } else if tier1 { 1 } else { 0 };
    (tier, verdicts)
}

// ─────────────────────────── embedder + calibration ───────────────────────────

fn embed_texts(e: &dyn VectorEmbed, texts: &[String]) -> Vec<Vec<f32>> {
    texts.iter().map(|t| e.embed(t)).collect()
}

#[derive(Serialize, Clone)]
struct CalibPoint {
    p_cue: f64,
    q_cue: f64,
    surface_purity: f64,
    latent_ari: f64,
    cross_flow_margin: f64,
}

fn calibration(e: &dyn VectorEmbed, seed: u64) -> (CalibPoint, Vec<CalibPoint>) {
    let ps = [0.15f64, 0.3, 0.5, 0.7, 0.9];
    let qs = [0.0f64, 0.3, 0.6];
    let mut pts: Vec<CalibPoint> = Vec::new();
    for p in ps {
        for q in qs {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add((p * 1000.0) as u64).wrapping_add((q * 77.0) as u64));
            let probe = generate_split(&mut rng, DOMAINS_TRAIN, 4, p, q, "probe");
            let texts: Vec<String> = probe.iter().map(|i| i.text.clone()).collect();
            let emb = embed_texts(e, &texts);
            let singles: Vec<usize> = (0..emb.len()).filter(|&i| !probe[i].is_bridge).collect();
            let single_emb: Vec<Vec<f32>> = singles.iter().map(|&i| emb[i].clone()).collect();
            let part4 = kmeans(&single_emb, 4, &mut rng, 50);
            let surf: Vec<usize> = singles.iter().map(|&i| probe[i].surface).collect();
            let part3 = kmeans(&single_emb, 3, &mut rng, 50);
            let lat: Vec<usize> = singles.iter().map(|&i| probe[i].flows[0]).collect();
            // cross-domain same-flow minus different-flow margin
            let mut same = 0.0f64;
            let mut diff = 0.0f64;
            let mut ns = 0usize;
            let mut nd = 0usize;
            for a in 0..singles.len() {
                for b in (a + 1)..singles.len() {
                    if surf[a] == surf[b] { continue; }
                    let c = cosine_sim(&single_emb[a], &single_emb[b]) as f64;
                    if lat[a] == lat[b] { same += c; ns += 1; } else { diff += c; nd += 1; }
                }
            }
            pts.push(CalibPoint {
                p_cue: p,
                q_cue: q,
                surface_purity: purity(&part4, &surf),
                latent_ari: ari(&part3, &lat),
                cross_flow_margin: if ns > 0 && nd > 0 { same / ns as f64 - diff / nd as f64 } else { 0.0 },
            });
        }
    }
    // frozen rule: purity>=0.7, ARI>=0.05, margin>=0.005; minimize |ARI-0.2|
    let mut eligible: Vec<&CalibPoint> = pts.iter()
        .filter(|c| c.surface_purity >= 0.7 && c.latent_ari >= 0.05 && c.cross_flow_margin >= 0.005)
        .collect();
    if eligible.is_empty() {
        eprintln!("warning: no calibration point satisfied the difficulty window; relaxing (recorded)");
        eligible = pts.iter().collect();
    }
    let chosen = eligible.iter().copied()
        .min_by(|a, b| {
            let da = (a.latent_ari - 0.2).abs();
            let db = (b.latent_ari - 0.2).abs();
            da.partial_cmp(&db).unwrap()
        }).unwrap().clone();
    (chosen, pts)
}

// ───────────────────────────────── main ─────────────────────────────────

#[derive(Serialize)]
struct Sidecar {
    seed: u64,
    calibration: Vec<CalibPoint>,
    chosen: CalibPoint,
    items: Vec<Item>,
}

#[derive(Serialize)]
struct Results {
    seed: u64,
    primary: PipelineOutput,
    bge_route: Option<PipelineOutput>,
    tier: u8,
    tier_verdicts: Vec<(String, String)>,
}

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    let ix = args.iter().position(|a| a == flag)?;
    args.get(ix + 1).cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    #[cfg(feature = "embed-onnx")]
    {
        if args.iter().any(|a| a == "--help") {
            println!("flags: --seed <u64> | --out <dir> | --skip-bge | --probe");
            return;
        }
        run(&args);
    }
    #[cfg(not(feature = "embed-onnx"))]
    {
        eprintln!("requires --features embed-onnx (MiniLM/BGE embeddings)");
        std::process::exit(1);
    }
}

#[cfg(feature = "embed-onnx")]
fn run(args: &[String]) {
    use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

    let seed = parse_flag(args, "--seed").and_then(|s| s.parse::<u64>().ok()).unwrap_or(20260911);
    let out_dir = parse_flag(args, "--out").unwrap_or_else(|| "research/perspective_invention".into());
    let skip_bge = args.iter().any(|a| a == "--skip-bge");
    let probe_only = args.iter().any(|a| a == "--probe");
    println!("perspective invention experiment — seed {seed}");

    // MiniLM (the pinned primary embedder)
    let minilm = OnnxEmbedder::with_config(&OnnxConfig::default());
    let minilm_ok = minilm.is_available();
    if minilm_ok {
        println!("embedder: MiniLM-L6-v2 384-d (models/model.onnx) [OK]");
    } else {
        eprintln!("FATAL: models/model.onnx not found from cwd. Run from the workspace root. Results under the random-projection fallback are NOT evidence (see README). Re-run with --features embed-onnx from /home/gio/dev/physis-pro.");
        std::process::exit(2);
    }

    println!("S0 — difficulty calibration (probe grid, train-side labels only, declared)…");
    let (chosen, calib) = calibration(&minilm, seed);
    println!("chosen: p_cue={:.2} q_cue={:.2} → probe surface purity {:.3}, latent ARI {:.3}, cross-flow margin {:+.4}",
        chosen.p_cue, chosen.q_cue, chosen.surface_purity, chosen.latent_ari, chosen.cross_flow_margin);

    if probe_only {
        write_file(&std::path::Path::new(&out_dir).join("datasets/calibration.json"),
            &serde_json::to_string_pretty(&serde_json::json!({ "seed": seed, "chosen": chosen, "grid": calib })).unwrap());
        println!("probe run complete — calibration grid written");
        return;
    }

    // dataset generation with the calibrated difficulty
    let mut rng_gen = StdRng::seed_from_u64(seed.wrapping_add(1));
    let splits = Splits {
        train: generate_split(&mut rng_gen, DOMAINS_TRAIN, 19, chosen.p_cue, chosen.q_cue, "train"),
        test: generate_split(&mut rng_gen, DOMAINS_TRAIN, 9, chosen.p_cue, chosen.q_cue, "test"),
        transfer: generate_split(&mut rng_gen, DOMAINS_TRANSFER, 9, chosen.p_cue, chosen.q_cue, "transfer"),
    };
    println!("dataset: train {} / test {} / transfer {} (transfer domains differ)", splits.train.len(), splits.test.len(), splits.transfer.len());

    write_file(&std::path::Path::new(&out_dir).join("datasets/dataset_sidecar.json"),
        &serde_json::to_string_pretty(&Sidecar { seed, calibration: calib, chosen: chosen.clone(), items: splits.train.iter().chain(splits.test.iter()).chain(splits.transfer.iter()).cloned().collect() }).unwrap());

    // embeddings
    println!("embedding (MiniLM)…");
    let all_texts: Vec<String> = splits.train.iter().chain(splits.test.iter()).chain(splits.transfer.iter()).map(|i| i.text.clone()).collect();
    let (n_train, n_test) = (splits.train.len(), splits.test.len());
    let emb_mini_all = embed_texts(&minilm, &all_texts);
    let train_mini = &emb_mini_all[..n_train];
    let test_mini = &emb_mini_all[n_train..n_train + n_test];
    let transfer_mini = &emb_mini_all[n_train + n_test..];

    println!("S1–S8 pipeline (MiniLM route)…");
    let mut rng_a = StdRng::seed_from_u64(seed.wrapping_add(100));
    let mut out_a = run_pipeline("minilm-l6-v2", train_mini, test_mini, transfer_mini, &splits, &mut rng_a);

    // BGE route (independent embedding, same data)
    let mut out_b: Option<PipelineOutput> = None;
    if !skip_bge {
        let bge = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 768,
            model_dir: Some("models/bge-base-en-v1.5".into()),
            pooling: PoolingStrategy::Mean,
            ..OnnxConfig::default()
        });
        if bge.is_available() {
            println!("embedder: BGE-base 768-d mean-pool [OK] — independent route");
            let emb_bge_all = embed_texts(&bge, &all_texts);
            let train_bge = &emb_bge_all[..n_train];
            let test_bge = &emb_bge_all[n_train..n_train + n_test];
            let transfer_bge = &emb_bge_all[n_train + n_test..];
            println!("S1–S8 pipeline (BGE route)…");
            let mut rng_b = StdRng::seed_from_u64(seed.wrapping_add(200));
            let mut ob = run_pipeline("bge-base-en-v1.5", train_bge, test_bge, transfer_bge, &splits, &mut rng_b);
            // cross-embedder route convergence: same generator, independent geometry
            let mut xpairs: Vec<RoutePair> = Vec::new();
            for rn in ["P_edge", "P_soft", "P_ref"] {
                let pa = &out_a.candidates.iter().find(|p| p.id == rn).unwrap().partition;
                let pb = &ob.candidates.iter().find(|p| p.id == rn).unwrap().partition;
                xpairs.push(RoutePair { a: format!("minilm/{rn}"), b: format!("bge/{rn}"), ari: ari(pa, pb), nmi: nmi(pa, pb) });
            }
            out_a.convergence.cross_embedder = xpairs.clone();
            ob.convergence.cross_embedder = xpairs.clone();
            out_b = Some(ob);
        } else {
            eprintln!("warning: models/bge-base-en-v1.5 unavailable — cross-embedder convergence NOT tested (recorded)");
        }
    }

    // tiers + outputs
    let (tier, verdicts) = resolve_tiers(&out_a, &out_a.convergence.cross_embedder);
    print_summary(&out_a, &out_b, tier, &verdicts);

    emit_svgs(&out_dir, &out_a, &splits, &emb_mini_all, n_train, n_test);

    let results = Results { seed, primary: out_a, bge_route: out_b, tier, tier_verdicts: verdicts };
    write_file(&std::path::Path::new(&out_dir).join("experiments/results.json"),
        &serde_json::to_string_pretty(&results).unwrap());
    println!("results written to {out_dir}/experiments/results.json");
}

#[cfg(feature = "embed-onnx")]
fn print_summary(out: &PipelineOutput, out_b: &Option<PipelineOutput>, tier: u8, verdicts: &[(String, String)]) {
    println!("\n════════════ SUMMARY ({}) ════════════", out.embedder);
    println!("baseline: k={} surface purity {:.3} latent ARI {:.3} raw-sil {:.4}",
        out.baseline_k, out.baseline_surface_purity, out.baseline_latent_ari, out.baseline_silhouette);
    println!("residual: {} cross-cluster edges (null p95 {:+.4})", out.residual.n_edges, out.residual.null_p95);
    println!("\ncandidates (internal selection = max raw-space silhouette):\n");
    println!("  {:<16} {:<26} {:>10} {:>9} {:>9} {:>9} {:>9}", "id", "generator", "raw-sil", "own-sil", "ARI", "NMI", "purity");
    for p in &out.candidates {
        println!("  {:<16} {:<26} {:>10.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4}{}",
            p.id, p.generator, p.silhouette_raw_after, p.silhouette_own_space, p.latent_ari, p.latent_nmi, p.latent_purity,
            if p.id == out.winner_id { "  <— winner" } else { "" });
    }
    if let Some(ob) = out_b {
        println!("\nBGE route (independent embedder):\n");
        println!("  {:<16} {:<26} {:>10} {:>9} {:>9} {:>9} {:>9}", "id", "generator", "raw-sil", "own-sil", "ARI", "NMI", "purity");
        for p in &ob.candidates {
            println!("  {:<16} {:<26} {:>10.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4}{}",
                p.id, p.generator, p.silhouette_raw_after, p.silhouette_own_space, p.latent_ari, p.latent_nmi, p.latent_purity,
                if p.id == ob.winner_id { "  <— winner" } else { "" });
        }
    }
    println!("\nconvergence:");
    for rp in &out.convergence.route_pairs {
        println!("  {} vs {}: ARI {:.3} NMI {:.3}", rp.a, rp.b, rp.ari, rp.nmi);
    }
    println!("  permuted null ARI: mean {:.3} p95 {:.3}", out.convergence.permuted_null_ari_mean, out.convergence.permuted_null_ari_p95);
    println!("  consensus pairs: {} precision {:.3} recall {:.3}", out.convergence.n_consensus_pairs, out.convergence.consensus_pair_precision, out.convergence.consensus_pair_recall);
    for xp in &out.convergence.cross_embedder {
        println!("  cross-embedder {} vs {}: ARI {:.3} (known null ≈ 0.10)", xp.a, xp.b, xp.ari);
    }
    println!("\nheld-out (frozen transforms, n_items/n_pairs in JSON):");
    for h in &out.heldout {
        println!("  {:<16} acc {:.3} CI [{:.3},{:.3}]  AUC {:.3} CI [{:.3},{:.3}]",
            h.method, h.acc_1nn, h.acc_1nn_ci.0, h.acc_1nn_ci.1, h.link_auc, h.link_auc_ci.0, h.link_auc_ci.1);
    }
    println!("\ntransfer (different surface domains):" );
    for h in &out.transfer {
        println!("  {:<16} acc {:.3} CI [{:.3},{:.3}]  AUC {:.3}", h.method, h.acc_1nn, h.acc_1nn_ci.0, h.acc_1nn_ci.1, h.link_auc);
    }
    println!("\ncompression (top-k residual references):" );
    for c in &out.compression {
        println!("  {} k={}: held acc {:.3}", c.method, c.n_refs, c.acc_1nn);
    }
    println!("\nstability: winner {:.3} baseline {:.3}", out.stability_winner, out.stability_baseline);
    println!("selection controls:");
    for s in &out.selection_controls {
        println!("  {} → {} held acc {:.3} AUC {:.3}", s.rule, s.selected, s.heldout.acc_1nn, s.heldout.link_auc);
    }
    println!("\nrecursion: {}", if out.recursion_accepted { "level-2 accepted" } else { "stopped (no internal gain)" });
    for p in &out.recursion {
        println!("  {} (parent {:?}) raw-sil {:.4} ARI {:.3}", p.id, p.parent, p.silhouette_raw_after, p.latent_ari);
    }
    println!("\n════════════ TIER {} ════════════", tier);
    for (claim, v) in verdicts {
        println!("  {claim:<28} {v}");
    }
    let tier_name = match tier {
        0 => "FAILED", 1 => "INTERESTING BUT INSUFFICIENT", 2 => "SUPPORTED",
        3 => "STRONG STRUCTURAL EVIDENCE", 4 => "REPRESENTATION DISCOVERY", _ => "HIGH-ORDER GENERATIVE REPRESENTATION",
    };
    println!("  → {tier_name}");
}

#[cfg(feature = "embed-onnx")]
fn emit_svgs(out_dir: &str, out: &PipelineOutput, splits: &Splits, emb: &[Vec<f32>], n_train: usize, n_test: usize) {
    let viz = std::path::Path::new(out_dir).join("visualizations");
    let train_emb: Vec<Vec<f32>> = emb[..n_train].to_vec();
    let pts = pca2(&train_emb);
    let train_surface: Vec<usize> = splits.train.iter().map(|i| i.surface).collect();
    let train_latent: Vec<usize> = splits.train.iter().map(|i| i.flows[0]).collect();
    let winner = out.candidates.iter().find(|p| p.id == out.winner_id).unwrap();
    write_file(&viz.join("train_by_surface.svg"), &scatter_svg(&pts, &train_surface, "train — colored by SURFACE domain (the dominant signal)", "PCA-2D of MiniLM embeddings; this is the organization ordinary classification finds"));
    write_file(&viz.join("train_by_latent.svg"), &scatter_svg(&pts, &train_latent, "train — colored by LATENT flow (the true organizing principle)", "evaluation-only coloring; discovery never saw these labels"));
    write_file(&viz.join("train_discovered.svg"), &scatter_svg(&pts, &winner.partition, &format!("train — discovered partition D({})", out.winner_id), &format!("raw-space silhouette {:.4} vs baseline {:.4}", winner.silhouette_raw_after, out.baseline_silhouette)));
    // residual matrix, items ordered by surface blocks
    let mut order: Vec<usize> = (0..n_train).collect();
    order.sort_by_key(|&i| (train_surface[i], train_latent[i]));
    let surface_blocks: Vec<usize> = train_surface.clone();
    write_file(&viz.join("residual_graph.svg"), &residual_matrix_svg(&out.residual.edges, &order, &surface_blocks, "residual graph — cross-cluster relations the baseline partition discarded"));
    // agreement heatmap
    let mut labels: Vec<String> = Vec::new();
    let mut parts: Vec<Vec<usize>> = Vec::new();
    labels.push("P_edge".into()); parts.push(out.candidates.iter().find(|p| p.id == "P_edge").unwrap().partition.clone());
    labels.push("P_soft".into()); parts.push(out.candidates.iter().find(|p| p.id == "P_soft").unwrap().partition.clone());
    labels.push("P_ref".into()); parts.push(out.candidates.iter().find(|p| p.id == "P_ref").unwrap().partition.clone());
    labels.push("winner".into()); parts.push(winner.partition.clone());
    let mut vals: Vec<Vec<f64>> = Vec::new();
    for a in &parts {
        vals.push(parts.iter().map(|b| ari(a, b)).collect());
    }
    write_file(&viz.join("cross_perspective_agreement.svg"), &heatmap_svg(&labels, &vals, "partition agreement (ARI) across independent routes"));
    // lineage tree
    let mut tree: Vec<(String, Option<String>, f64)> = vec![("baseline".into(), None, 0.0), (out.winner_id.clone(), Some("baseline".into()), winner.internal_gain)];
    for p in &out.recursion {
        tree.push((p.id.clone(), p.parent.clone(), p.internal_gain));
    }
    write_file(&viz.join("perspective_tree.svg"), &tree_svg(&tree, "perspective lineage — each child accepted only on internal raw-silhouette gain"));
    // held-out bars
    let mut acc_entries: Vec<(String, f64, f64, f64)> = Vec::new();
    for h in &out.heldout {
        acc_entries.push((h.method.clone(), h.acc_1nn, h.acc_1nn_ci.0, h.acc_1nn_ci.1));
    }
    write_file(&viz.join("heldout_accuracy.svg"), &bars_svg(&acc_entries, "held-out latent-role 1-NN accuracy (95% bootstrap CI)", "frozen transforms applied to test embeddings; red line = CI"));
    let mut auc_entries: Vec<(String, f64, f64, f64)> = Vec::new();
    for h in &out.heldout {
        auc_entries.push((h.method.clone(), h.link_auc, h.link_auc_ci.0, h.link_auc_ci.1));
    }
    write_file(&viz.join("heldout_link_auc.svg"), &bars_svg(&auc_entries, "held-out same-flow link AUC (cross-domain test pairs)", "does the perspective expose cross-cutting relations raw cosine cannot?"));
    println!("SVGs written to {}/visualizations/", out_dir);
    let _ = n_test;
}
