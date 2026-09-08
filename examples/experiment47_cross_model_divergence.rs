// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 47 — E1: does cross-model DISAGREEMENT carry epistemic signal?
//!
//! Three prior experiments failed to make cross-embedder AGREEMENT useful
//! (Iteration 18: ARI 0.10; Iteration 19: FPR gain 0, recall 5/5 -> 4/5;
//! Iteration 24: split agreement fails backwards). This inverts that failed
//! assumption: not "the models should agree", but "where they disagree, is the
//! filing suspect?"
//!
//! ## What is reused, unchanged, from experiment38 (Stage 12)
//!
//! Corpus, ground truth, split, evaluation and metric are lifted verbatim:
//!   - `OntologyLoader::load_all()` classification domains, mode anchors excluded
//!   - the WordNet-anchored gate (only entries anchored into the noun DAG are
//!     scored) — this defines n, so it must stay even though E1 uses no
//!     structural scorer
//!   - the deterministic stride/mult injection, character for character
//!   - the held-out half from `cell_vocab_*.json` `test_entries`
//!   - `auc()` (rank-sum, ties averaged) and the cosine-strata procedure
//!   - the same 22 configurations (11 stride/mult pairs x {test, all})
//!
//! ## What is new, and why
//!
//! 1. A SECOND embedder (BGE-base-en-v1.5, 768-d) scores the same entry against
//!    the same peer set in its own space. Same pooling as MiniLM (mean), so the
//!    only variable is model identity.
//! 2. The sweep runs INSIDE one process. experiment38 was re-invoked per
//!    configuration; embedding 731 entries twice per invocation would cost ~20x
//!    more for identical vectors. The corpus does not change across
//!    configurations — only the injection does — so the embeddings are computed
//!    once and the injection loop became a function. No scored quantity changes.
//! 3. A logistic combiner, fit on the TRAIN half and scored on TEST only, for
//!    the joint arms. Single signals are monotone so their held-out AUC needs no
//!    fit; a joint arm does.
//!
//! ## The primitives, kept separate on purpose
//!
//!   support_1 = cos(e1_i, centroid_1(cellmates of i, excluding i))
//!   support_2 = cos(e2_i, centroid_2(cellmates of i, excluding i))
//!   delta     = support_1 - support_2
//!   D         = |delta|
//!
//! Nothing here is called certainty. The baseline scorer of Stage 12 is
//! `1 - support_1`, so arm B is the baseline restated with the sign flipped.
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment47_cross_model_divergence

use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use std::collections::{HashMap, HashSet};

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter().map(|x| x / n).collect()
}

fn words(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|w| w.len() > 2)
        .collect()
}

/// WordNet noun DAG: lemma -> synset offsets, and offset -> hypernym offsets.
/// Verbatim from experiment38 — E1 uses it only for the anchored gate, which
/// decides which entries are scored and therefore must stay identical.
fn load_wordnet(path: &std::path::Path) -> (HashMap<String, Vec<u32>>, HashMap<u32, Vec<u32>>) {
    let mut lemma: HashMap<String, Vec<u32>> = HashMap::new();
    let mut hyper: HashMap<u32, Vec<u32>> = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return (lemma, hyper);
    };
    for line in text.lines() {
        if line.starts_with("  ") || line.trim().is_empty() {
            continue;
        }
        let head = line.split(" | ").next().unwrap_or("");
        let f: Vec<&str> = head.split_whitespace().collect();
        if f.len() < 5 {
            continue;
        }
        let Ok(off) = f[0].parse::<u32>() else {
            continue;
        };
        let Ok(wcnt) = usize::from_str_radix(f[3], 16) else {
            continue;
        };
        for k in 0..wcnt {
            if let Some(w) = f.get(4 + k * 2) {
                lemma
                    .entry(w.replace('_', " ").to_lowercase())
                    .or_default()
                    .push(off);
            }
        }
        let pstart = 4 + wcnt * 2;
        if let Some(pc) = f.get(pstart).and_then(|s| s.parse::<usize>().ok()) {
            for p in 0..pc {
                let b = pstart + 1 + p * 4;
                if let (Some(sym), Some(o), Some(pos)) = (f.get(b), f.get(b + 1), f.get(b + 2)) {
                    if (*sym == "@" || *sym == "@i") && *pos == "n" {
                        if let Ok(t) = o.parse::<u32>() {
                            hyper.entry(off).or_default().push(t);
                        }
                    }
                }
            }
        }
    }
    (lemma, hyper)
}

fn ancestors(seed: &[u32], hyper: &HashMap<u32, Vec<u32>>, max_depth: usize) -> HashSet<u32> {
    let mut seen: HashSet<u32> = seed.iter().copied().collect();
    let mut frontier: Vec<u32> = seed.to_vec();
    for _ in 0..max_depth {
        let mut next = Vec::new();
        for n in frontier.drain(..) {
            if let Some(ps) = hyper.get(&n) {
                for &p in ps {
                    if seen.insert(p) {
                        next.push(p);
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    seen
}

/// Area under ROC for `score` separating flagged (true) from clean (false).
/// Verbatim from experiment38.
fn auc(scored: &[(f64, bool)]) -> f64 {
    let mut v: Vec<&(f64, bool)> = scored.iter().collect();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (pos, neg) = (
        v.iter().filter(|x| x.1).count(),
        v.iter().filter(|x| !x.1).count(),
    );
    if pos == 0 || neg == 0 {
        return 0.5;
    }
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1].0 == v[i].0 {
            j += 1;
        }
        let avg = ((i + 1 + j + 1) as f64) / 2.0;
        for x in &v[i..=j] {
            if x.1 {
                rank_sum += avg;
            }
        }
        i = j + 1;
    }
    (rank_sum - (pos * (pos + 1)) as f64 / 2.0) / (pos * neg) as f64
}

/// Mean, sample sd, and the paired t statistic of a set of differences.
fn paired_t(d: &[f64]) -> (f64, f64, f64) {
    let n = d.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0, 0.0);
    }
    let m = d.iter().sum::<f64>() / n;
    let var = d.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0);
    let sd = var.sqrt();
    let t = if sd > 0.0 { m / (sd / n.sqrt()) } else { 0.0 };
    (m, sd, t)
}

/// Deterministic LCG, so the bootstrap and the pairing control are reproducible.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// Percentile bootstrap CI for the difference in AUC between two scores on the
/// same rows (paired: the same resampled row set scores both).
fn boot_auc_diff(a: &[f64], b: &[f64], y: &[bool], iters: usize, seed: u64) -> (f64, f64, f64) {
    let n = y.len();
    let point = auc(&a.iter().copied().zip(y.iter().copied()).collect::<Vec<_>>())
        - auc(&b.iter().copied().zip(y.iter().copied()).collect::<Vec<_>>());
    let mut rng = Lcg(seed);
    let mut diffs: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let idx: Vec<usize> = (0..n).map(|_| rng.below(n)).collect();
        let ra: Vec<(f64, bool)> = idx.iter().map(|&i| (a[i], y[i])).collect();
        let rb: Vec<(f64, bool)> = idx.iter().map(|&i| (b[i], y[i])).collect();
        if ra.iter().filter(|x| x.1).count() == 0 || ra.iter().filter(|x| !x.1).count() == 0 {
            continue;
        }
        diffs.push(auc(&ra) - auc(&rb));
    }
    diffs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    if diffs.is_empty() {
        return (point, 0.0, 0.0);
    }
    let lo = diffs[(0.025 * diffs.len() as f64) as usize];
    let hi = diffs[((0.975 * diffs.len() as f64) as usize).min(diffs.len() - 1)];
    (point, lo, hi)
}

/// Logistic regression, standardized features, deterministic full-batch descent.
/// Only ever fit on the train half.
fn fit_logistic(
    x: &[Vec<f64>],
    y: &[bool],
    iters: usize,
    lr: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let p = x[0].len();
    let n = x.len() as f64;
    let mut mu = vec![0.0; p];
    let mut sd = vec![1.0; p];
    for j in 0..p {
        mu[j] = x.iter().map(|r| r[j]).sum::<f64>() / n;
        let v = x.iter().map(|r| (r[j] - mu[j]).powi(2)).sum::<f64>() / n;
        sd[j] = v.sqrt().max(1e-9);
    }
    let z: Vec<Vec<f64>> = x
        .iter()
        .map(|r| (0..p).map(|j| (r[j] - mu[j]) / sd[j]).collect())
        .collect();
    let mut w = vec![0.0; p + 1];
    for _ in 0..iters {
        let mut g = vec![0.0; p + 1];
        for (i, row) in z.iter().enumerate() {
            let mut s = w[p];
            for j in 0..p {
                s += w[j] * row[j];
            }
            let pr = 1.0 / (1.0 + (-s).exp());
            let e = pr - if y[i] { 1.0 } else { 0.0 };
            for j in 0..p {
                g[j] += e * row[j];
            }
            g[p] += e;
        }
        for j in 0..=p {
            w[j] -= lr * g[j] / n;
        }
    }
    (w, mu, sd)
}

fn apply_logistic(w: &[f64], mu: &[f64], sd: &[f64], row: &[f64]) -> f64 {
    let p = row.len();
    let mut s = w[p];
    for j in 0..p {
        s += w[j] * (row[j] - mu[j]) / sd[j];
    }
    s
}

/// One scored entry: everything E1 needs, with the primitives kept separate.
#[derive(Clone)]
struct Row {
    in_test: bool,
    injected: bool,
    s1: f64,  // support_1: cos to cell centroid in model 1's space
    s2: f64,  // support_2: same, model 2
    s2r: f64, // support_2 under the random-pairing control
}

fn main() {
    println!("Experiment 47 — E1: does cross-model divergence carry epistemic signal?\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed::VectorEmbed;
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

        let Some(wn) = ["models/wordnet", "../models/wordnet"]
            .iter()
            .map(std::path::Path::new)
            .find(|d| d.join("data.noun").exists())
        else {
            println!("WARNING: WordNet not found — aborting.");
            return;
        };
        let (lemma, hyper) = load_wordnet(&wn.join("data.noun"));

        let Some(m1dir) = ["models", "../models"]
            .iter()
            .find(|d| std::path::Path::new(d).join("model.onnx").exists())
        else {
            println!("WARNING: MiniLM not available — aborting.");
            return;
        };
        let Some(m2dir) = ["models/bge-base-en-v1.5", "../models/bge-base-en-v1.5"]
            .iter()
            .find(|d| std::path::Path::new(d).join("onnx/model.onnx").exists())
        else {
            println!("WARNING: BGE not available — aborting.");
            return;
        };
        // Same pooling for both, so the only variable is model identity.
        let e1 = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 384,
            model_dir: Some(m1dir.to_string()),
            pooling: PoolingStrategy::Mean,
            ..OnnxConfig::default()
        });
        let e2 = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 768,
            model_dir: Some(m2dir.to_string()),
            pooling: PoolingStrategy::Mean,
            ..OnnxConfig::default()
        });
        if !e1.is_available() || !e2.is_available() {
            println!("WARNING: an embedder is unavailable — aborting.");
            return;
        }
        println!("model 1 = MiniLM 384-d (mean)   [Stage 12's baseline embedder]");
        println!("model 2 = BGE-base 768-d (mean) [Iteration 18's stronger embedder]\n");

        // ── mode anchors: excluded from the tested entries, as in experiment38 ──
        let anchor_path = ["physis-core/config", "config", "../config"]
            .iter()
            .map(|d| std::path::Path::new(d).join("mode_anchors_ontology.json"))
            .find(|p| p.exists());
        let mut anchor_names: HashSet<String> = HashSet::new();
        if let Some(ap) = &anchor_path {
            if let Ok(txt) = std::fs::read_to_string(ap) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                    for e in v["domains"].as_array().into_iter().flatten() {
                        if let Some(nm) = e["name"].as_str() {
                            anchor_names.insert(nm.to_string());
                        }
                    }
                }
            }
        }

        // ── the held-out half, from the same file experiment38 reads ──
        let mut test_names: HashSet<String> = HashSet::new();
        let vocab_paths: Vec<String> = match std::env::var("PHYSIS_VOCAB") {
            Ok(v) => vec![v],
            Err(_) => [
                "research/perspective-discovery/cell_vocab_AB.json",
                "../research/perspective-discovery/cell_vocab_AB.json",
                "cell_vocab_AB.json",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        };
        for vp in vocab_paths {
            let Ok(txt) = std::fs::read_to_string(&vp) else {
                continue;
            };
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else {
                continue;
            };
            for (_, arr) in v["test_entries"].as_object().into_iter().flatten() {
                for x in arr.as_array().into_iter().flatten() {
                    if let Some(n) = x.as_str() {
                        test_names.insert(n.to_string());
                    }
                }
            }
            println!("split: {} held-out entries ({vp})", test_names.len());
            break;
        }

        // ── corpus, identical to experiment38 ──
        let ontology = OntologyLoader::load_all();
        let (mut texts, mut true_cell, mut names, mut entry_names) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for def in ontology.classification_domains() {
            let (Some(d), Some(m)) = (&def.domain, &def.mode) else {
                continue;
            };
            if anchor_names.contains(&def.name) {
                continue;
            }
            let mut t = def.name.clone();
            for h in &def.hints {
                t.push(' ');
                t.push_str(h);
            }
            entry_names.push(def.name.clone());
            names.push(words(&def.name));
            texts.push(t);
            true_cell.push((d.clone(), m.clone()));
        }
        let n = texts.len();
        println!("{n} entries, embedding with both models (once — the corpus does not vary across configurations)...");
        let emb1: Vec<Vec<f32>> = texts.iter().map(|t| normalize(&e1.embed(t))).collect();
        let emb2: Vec<Vec<f32>> = texts.iter().map(|t| normalize(&e2.embed(t))).collect();

        // ── the anchored gate: decides WHICH entries are scored ──
        let anc: Vec<HashSet<u32>> = (0..n)
            .map(|i| {
                let mut seed: Vec<u32> = Vec::new();
                for w in &names[i] {
                    if let Some(offs) = lemma.get(w) {
                        seed.extend(offs.iter().take(2));
                    }
                }
                if seed.is_empty() {
                    HashSet::new()
                } else {
                    ancestors(&seed, &hyper, 6)
                }
            })
            .collect();
        let anchored = anc.iter().filter(|a| !a.is_empty()).count();
        println!("entries anchored into the DAG: {anchored}/{n}\n");

        let cell_names: Vec<(String, String)> = {
            let mut v: Vec<(String, String)> = true_cell.clone();
            v.sort();
            v.dedup();
            v
        };

        // RANDOM PAIRING CONTROL: a fixed derangement of the model-2 vectors.
        // Entry i is scored in model 2's space using ANOTHER entry's vector
        // against i's own cell centroid, so the marginal distribution of
        // support_2 is preserved but the per-entry pairing is destroyed.
        let perm: Vec<usize> = {
            let mut rng = Lcg(0x5EED_1234);
            let mut p: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.below(i + 1);
                p.swap(i, j);
            }
            // force a derangement
            for i in 0..n {
                if p[i] == i {
                    p.swap(i, (i + 1) % n);
                }
            }
            p
        };

        // ── the scoring pass, parameterised by injection configuration ──
        // The body is experiment38's loop with the structural scorers removed
        // and the second embedder added.
        let score = |stride: usize, mult: usize| -> Vec<Row> {
            let mut cell = true_cell.clone();
            let mut injected: Vec<bool> = vec![false; n];
            let mut k = 0usize;
            for i in 0..n {
                if anc[i].is_empty() {
                    continue;
                }
                k += 1;
                // Verbatim from experiment38, clippy's is_multiple_of hint included:
                // this line defines the ground truth, so it stays character for character.
                #[allow(clippy::manual_is_multiple_of)]
                if stride != 0 && k % stride == 0 {
                    let mut c = cell_names[(i * mult + 5) % cell_names.len()].clone();
                    if c == true_cell[i] {
                        c = cell_names[(i * mult + 6) % cell_names.len()].clone();
                    }
                    if c != true_cell[i] {
                        cell[i] = c;
                        injected[i] = true;
                    }
                }
            }
            let mut members: HashMap<(String, String), Vec<usize>> = HashMap::new();
            for (i, c) in cell.iter().enumerate() {
                members.entry(c.clone()).or_default().push(i);
            }
            let mut rows = Vec::new();
            for i in 0..n {
                if anc[i].is_empty() {
                    continue;
                }
                let peers: Vec<usize> = members[&cell[i]]
                    .iter()
                    .copied()
                    .filter(|&j| j != i && !anc[j].is_empty())
                    .collect();
                if peers.len() < 2 {
                    continue;
                }
                let centroid = |e: &Vec<Vec<f32>>, dim: usize| -> Vec<f32> {
                    let mut acc = vec![0.0f32; dim];
                    for &j in &peers {
                        for d in 0..dim {
                            acc[d] += e[j][d];
                        }
                    }
                    normalize(&acc)
                };
                let c1 = centroid(&emb1, 384);
                let c2 = centroid(&emb2, 768);
                rows.push(Row {
                    in_test: test_names.contains(&entry_names[i]),
                    injected: injected[i],
                    s1: cosine_sim(&emb1[i], &c1) as f64,
                    s2: cosine_sim(&emb2[i], &c2) as f64,
                    s2r: cosine_sim(&emb2[perm[i]], &c2) as f64,
                });
            }
            rows
        };

        // The 22 Stage 12 configurations, in the order the sweep ran them.
        let mut configs: Vec<(usize, usize)> = Vec::new();
        for s in [5, 6, 7, 8, 9, 11, 13] {
            configs.push((s, 13));
        }
        for m in [11, 17, 19, 23] {
            configs.push((7, m));
        }

        // Arms. Every score is oriented so HIGHER = more suspect, which is the
        // orientation Stage 12's cosine baseline already used (it scored
        // distance, not similarity). `sgn` is the exception and is reported raw.
        type Arm = (&'static str, fn(&Row) -> f64);
        let arms: Vec<Arm> = vec![
            ("A base  ", |r: &Row| 1.0 - r.s1),
            ("B supp1 ", |r: &Row| -r.s1),
            ("C supp2 ", |r: &Row| -r.s2),
            ("D absD  ", |r: &Row| (r.s1 - r.s2).abs()),
            ("E sgnD  ", |r: &Row| r.s1 - r.s2),
            ("F sum   ", |r: &Row| -(r.s1 + r.s2)),
            ("Dr absDr", |r: &Row| (r.s1 - r.s2r).abs()),
        ];

        println!("=== Per-configuration AUC (higher score = more suspect; 0.500 = chance) ===\n");
        println!("stride mult half     n  inj | A base  B supp1  C supp2  D absD   E sgnD   F sum   | Dr rand");
        let mut per_arm: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        let mut cfg_labels: Vec<String> = Vec::new();
        for &(stride, mult) in &configs {
            let rows = score(stride, mult);
            for half in ["test", "all"] {
                let sel: Vec<&Row> = rows
                    .iter()
                    .filter(|r| half == "all" || test_names.is_empty() || r.in_test)
                    .collect();
                let y: Vec<bool> = sel.iter().map(|r| r.injected).collect();
                let mut vals = Vec::new();
                for (k, (_, f)) in arms.iter().enumerate() {
                    let a = auc(&sel.iter().map(|r| (f(r), r.injected)).collect::<Vec<_>>());
                    per_arm[k].push(a);
                    vals.push(a);
                }
                cfg_labels.push(format!("s{stride}m{mult}{half}"));
                println!(
                    "{stride:>6} {mult:>4} {half:>4} {:>5} {:>4} | {:.3}    {:.3}    {:.3}    {:.3}    {:.3}    {:.3}  |  {:.3}",
                    sel.len(), y.iter().filter(|x| **x).count(),
                    vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6]
                );
            }
        }

        let nconf = per_arm[0].len();
        println!("\n=== Mean over {nconf} configurations, and the paired test against the baseline ===\n");
        println!("arm         mean AUC   mean delta vs A   sd      paired t   wins/{nconf}");
        for (k, (name, _)) in arms.iter().enumerate() {
            let mean = per_arm[k].iter().sum::<f64>() / nconf as f64;
            let d: Vec<f64> = (0..nconf).map(|c| per_arm[k][c] - per_arm[0][c]).collect();
            let (m, sd, t) = paired_t(&d);
            let wins = d.iter().filter(|x| **x > 0.0).count();
            println!("{name}      {mean:.3}         {m:+.3}        {sd:.3}   {t:+.2}      {wins}");
        }

        // ── the published configuration, in detail ──
        println!("\n=== Published configuration (stride 7, mult 13), both halves ===\n");
        let rows = score(7, 13);
        for half in ["test", "all"] {
            let sel: Vec<&Row> = rows.iter().filter(|r| half == "all" || r.in_test).collect();
            let y: Vec<bool> = sel.iter().map(|r| r.injected).collect();
            let base: Vec<f64> = sel.iter().map(|r| 1.0 - r.s1).collect();
            for (name, f) in arms.iter().skip(1) {
                let s: Vec<f64> = sel.iter().map(|r| f(r)).collect();
                let (p, lo, hi) = boot_auc_diff(&s, &base, &y, 2000, 0xC0FFEE);
                println!(
                    "  half={half:<4} {name} minus baseline: {p:+.3}  95% CI [{lo:+.3}, {hi:+.3}]"
                );
            }
            println!();
        }

        // ── joint arms: fit on TRAIN, score on TEST ──
        println!(
            "=== Joint arms — logistic combiner fit on the TRAIN half, scored on TEST only ===\n"
        );
        println!(
            "(a single monotone signal needs no fit, so its held-out AUC is unchanged by this;"
        );
        println!(
            " the joint arms are the only ones that could overfit, and they are the ones fit.)\n"
        );
        let feats: Vec<(&str, Vec<usize>)> = vec![
            ("s1 alone            ", vec![0]),
            ("s1 + D              ", vec![0, 2]),
            ("s1 + s2             ", vec![0, 1]),
            ("s1 + s2 + D  (joint)", vec![0, 1, 2]),
            ("D alone             ", vec![2]),
        ];
        println!(
            "cfg          {}",
            feats
                .iter()
                .map(|f| f.0.trim())
                .collect::<Vec<_>>()
                .join("  ")
        );
        let mut joint_auc: Vec<Vec<f64>> = vec![Vec::new(); feats.len()];
        for &(stride, mult) in &configs {
            let rows = score(stride, mult);
            let tr: Vec<&Row> = rows.iter().filter(|r| !r.in_test).collect();
            let te: Vec<&Row> = rows.iter().filter(|r| r.in_test).collect();
            let mk = |r: &Row| vec![r.s1, r.s2, (r.s1 - r.s2).abs()];
            let mut line = format!("s{stride}m{mult}      ");
            for (k, (_, cols)) in feats.iter().enumerate() {
                let xtr: Vec<Vec<f64>> = tr
                    .iter()
                    .map(|r| cols.iter().map(|&c| mk(r)[c]).collect())
                    .collect();
                let ytr: Vec<bool> = tr.iter().map(|r| r.injected).collect();
                let (w, mu, sd) = fit_logistic(&xtr, &ytr, 4000, 0.5);
                let scored: Vec<(f64, bool)> = te
                    .iter()
                    .map(|r| {
                        let row: Vec<f64> = cols.iter().map(|&c| mk(r)[c]).collect();
                        (apply_logistic(&w, &mu, &sd, &row), r.injected)
                    })
                    .collect();
                let a = auc(&scored);
                joint_auc[k].push(a);
                line.push_str(&format!("{a:.3}                 "));
            }
            println!("{line}");
        }
        println!();
        let nj = joint_auc[0].len();
        for (k, (name, _)) in feats.iter().enumerate() {
            let mean = joint_auc[k].iter().sum::<f64>() / nj as f64;
            let d: Vec<f64> = (0..nj).map(|c| joint_auc[k][c] - joint_auc[0][c]).collect();
            let (m, _, t) = paired_t(&d);
            println!(
                "  {name}  mean held-out AUC {mean:.3}   vs s1 alone {m:+.3}  paired t {t:+.2}"
            );
        }

        // ── CONTROL 1: model swap ──
        println!("\n=== Control: model swap (MiniLM <-> BGE as reference) ===\n");
        let rows = score(7, 13);
        let sel: Vec<&Row> = rows.iter().collect();
        let a_of = |f: &dyn Fn(&Row) -> f64| {
            auc(&sel.iter().map(|r| (f(r), r.injected)).collect::<Vec<_>>())
        };
        println!(
            "  baseline    1-s1  = {:.3}   |  swapped 1-s2  = {:.3}",
            a_of(&|r: &Row| 1.0 - r.s1),
            a_of(&|r: &Row| 1.0 - r.s2)
        );
        println!(
            "  absolute D  |s1-s2| = {:.3} |  swapped |s2-s1| = {:.3}",
            a_of(&|r: &Row| (r.s1 - r.s2).abs()),
            a_of(&|r: &Row| (r.s2 - r.s1).abs())
        );
        println!(
            "  signed  D   s1-s2 = {:.3}   |  swapped s2-s1 = {:.3}",
            a_of(&|r: &Row| r.s1 - r.s2),
            a_of(&|r: &Row| r.s2 - r.s1)
        );
        println!("  -> |D| is invariant by construction; signed D is directional (AUC -> 1-AUC).");
        println!("     A signed arm therefore requires naming a reference and a challenger.");

        // ── CONTROL 2: random pairing ──
        println!("\n=== Control: random pairing (model-2 vectors deranged across entries) ===\n");
        let dr: Vec<f64> = sel.iter().map(|r| (r.s1 - r.s2).abs()).collect();
        let dp: Vec<f64> = sel.iter().map(|r| (r.s1 - r.s2r).abs()).collect();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        println!("  mean |D| real pairs   = {:.4}", mean(&dr));
        println!("  mean |D| random pairs = {:.4}", mean(&dp));
        println!(
            "  AUC  |D| real pairs   = {:.3}",
            a_of(&|r: &Row| (r.s1 - r.s2).abs())
        );
        println!(
            "  AUC  |D| random pairs = {:.3}",
            a_of(&|r: &Row| (r.s1 - r.s2r).abs())
        );

        // ── CONTROL 3: does D add anything WITHIN cosine strata? ──
        // experiment38's own procedure, pointed at D instead of at the
        // structural scorers.
        println!("\n=== Does D add anything BEYOND the baseline? (experiment38's strata procedure) ===\n");
        let mut order: Vec<usize> = (0..sel.len()).collect();
        order.sort_by(|&a, &b| (1.0 - sel[a].s1).partial_cmp(&(1.0 - sel[b].s1)).unwrap());
        const STRATA: usize = 8;
        let per = order.len() / STRATA;
        for (label, f) in [
            (
                "absolute D ",
                Box::new(|r: &Row| (r.s1 - r.s2).abs()) as Box<dyn Fn(&Row) -> f64>,
            ),
            ("support_2  ", Box::new(|r: &Row| -r.s2)),
            ("signed   D ", Box::new(|r: &Row| r.s1 - r.s2)),
        ] {
            let mut within: Vec<(f64, bool)> = Vec::new();
            for s in 0..STRATA {
                let lo = s * per;
                let hi = if s == STRATA - 1 {
                    order.len()
                } else {
                    (s + 1) * per
                };
                let mut idx: Vec<usize> = order[lo..hi].to_vec();
                idx.sort_by(|&a, &b| f(sel[a]).partial_cmp(&f(sel[b])).unwrap());
                for (r, &k) in idx.iter().enumerate() {
                    within.push((r as f64 / idx.len().max(1) as f64, sel[k].injected));
                }
            }
            println!(
                "  {label} within baseline strata:  AUC = {:.3}",
                auc(&within)
            );
        }

        // ── the decisive conditioning test ──
        // If divergence carries information of its own, it must survive being
        // conditioned on the SECOND model's support too, not only on the
        // baseline's. Otherwise "disagreement" is just a lossy read-out of the
        // support it disagrees with.
        println!("\n=== Conditioned on support_2 (the strongest single signal) instead ===\n");
        let strata_auc = |key: &dyn Fn(&Row) -> f64, f: &dyn Fn(&Row) -> f64| -> f64 {
            let mut order: Vec<usize> = (0..sel.len()).collect();
            order.sort_by(|&a, &b| key(sel[a]).partial_cmp(&key(sel[b])).unwrap());
            let per = order.len() / STRATA;
            let mut within: Vec<(f64, bool)> = Vec::new();
            for s in 0..STRATA {
                let lo = s * per;
                let hi = if s == STRATA - 1 {
                    order.len()
                } else {
                    (s + 1) * per
                };
                let mut idx: Vec<usize> = order[lo..hi].to_vec();
                idx.sort_by(|&a, &b| f(sel[a]).partial_cmp(&f(sel[b])).unwrap());
                for (r, &k) in idx.iter().enumerate() {
                    within.push((r as f64 / idx.len().max(1) as f64, sel[k].injected));
                }
            }
            auc(&within)
        };
        let k2 = |r: &Row| 1.0 - r.s2;
        let k1 = |r: &Row| 1.0 - r.s1;
        println!(
            "  absolute D  within support_2 strata: AUC = {:.3}",
            strata_auc(&k2, &|r: &Row| (r.s1 - r.s2).abs())
        );
        println!(
            "  signed   D  within support_2 strata: AUC = {:.3}",
            strata_auc(&k2, &|r: &Row| r.s1 - r.s2)
        );
        println!(
            "  support_1   within support_2 strata: AUC = {:.3}",
            strata_auc(&k2, &|r: &Row| -r.s1)
        );
        println!(
            "  (for scale: support_2 within support_1 strata = {:.3})",
            strata_auc(&k1, &|r: &Row| -r.s2)
        );

        println!("\n  Same procedure on the RANDOM-PAIRED divergence, as the null:");
        println!(
            "  absolute Dr within support_1 strata: AUC = {:.3}",
            strata_auc(&k1, &|r: &Row| (r.s1 - r.s2r).abs())
        );
        println!(
            "  absolute Dr within support_2 strata: AUC = {:.3}",
            strata_auc(&k2, &|r: &Row| (r.s1 - r.s2r).abs())
        );

        // ── one more natural combination the linear combiner cannot express ──
        println!(
            "\n=== Non-linear combinations of the two supports (published config, half=all) ===\n"
        );
        for (name, f) in [
            (
                "min(s1,s2)  ",
                Box::new(|r: &Row| -(r.s1.min(r.s2))) as Box<dyn Fn(&Row) -> f64>,
            ),
            ("max(s1,s2)  ", Box::new(|r: &Row| -(r.s1.max(r.s2)))),
            ("mean(s1,s2) ", Box::new(|r: &Row| -(r.s1 + r.s2) / 2.0)),
            ("s2 only     ", Box::new(|r: &Row| -r.s2)),
        ] {
            println!("  {name} AUC = {:.3}", a_of(&f));
        }

        // ── E1b: the sign/magnitude decomposition (user-proposed) ──────────
        // E1 tested |D| and signed D, and a LINEAR combiner over (s1, s2, D).
        // A linear model already spans signed D (it is s1 - s2), but it cannot
        // express "partition by which model is more supportive, then rank by
        // how far apart they are" -- and the strata procedure, which is
        // rank-based and conditional, is exactly where |D| looked strongest.
        // So the decomposition is a real gap in E1's coverage, not a restatement.
        //
        //   coherence = +1 when support_1 >= support_2, -1 otherwise
        //   magnitude = |D|
        //   lexicographic rank: coherence block first, |D| within it
        println!("\n=== E1b: coherence sign as a separate axis from the modulus ===\n");
        let coh = |r: &Row| if r.s1 >= r.s2 { 1.0 } else { -1.0 };
        let mag = |r: &Row| (r.s1 - r.s2).abs();
        let n_pos = sel.iter().filter(|r| coh(r) > 0.0).count();
        println!(
            "  coherence split: {} rows at +1 (MiniLM more supportive), {} at -1",
            n_pos,
            sel.len() - n_pos
        );
        println!();
        type Arm2 = (&'static str, Box<dyn Fn(&Row) -> f64>);
        let arms2: Vec<Arm2> = vec![
            (
                "coherence sign alone      ",
                Box::new(move |r: &Row| coh(r)),
            ),
            (
                "coherence sign, flipped   ",
                Box::new(move |r: &Row| -coh(r)),
            ),
            (
                "modulus |D| alone         ",
                Box::new(move |r: &Row| mag(r)),
            ),
            (
                "lex: +1 block, then |D|   ",
                Box::new(move |r: &Row| (if coh(r) > 0.0 { 1000.0 } else { 0.0 }) + mag(r)),
            ),
            (
                "lex: -1 block, then |D|   ",
                Box::new(move |r: &Row| (if coh(r) < 0.0 { 1000.0 } else { 0.0 }) + mag(r)),
            ),
            (
                "signed D = sign x modulus ",
                Box::new(move |r: &Row| r.s1 - r.s2),
            ),
        ];
        println!("  arm                            AUC     vs baseline (0.739)");
        for (name, f) in &arms2 {
            let a = a_of(f.as_ref());
            println!("  {name}  {a:.3}    {:+.3}", a - 0.739);
        }

        // Does the modulus carry anything ONCE THE SIGN IS HELD CONSTANT? This
        // is the decomposition's own question, asked with the harness's own
        // strata procedure.
        println!("\n  Conditioned on the coherence sign (the decomposition's own claim):");
        println!(
            "    modulus |D| within sign strata      : AUC = {:.3}",
            strata_auc(&|r: &Row| coh(r), &|r: &Row| mag(r))
        );
        println!(
            "    support_2   within sign strata      : AUC = {:.3}",
            strata_auc(&|r: &Row| coh(r), &|r: &Row| -r.s2)
        );

        // And the control that decided E1, applied to the decomposition. If the
        // sign axis is a real second channel it must survive conditioning on the
        // stronger support; if it is another read-out of s2, it will not.
        println!("\n  The E1 control, applied to the decomposition:");
        println!(
            "    coherence sign within support_1 strata : AUC = {:.3}",
            strata_auc(&k1, &|r: &Row| coh(r))
        );
        println!(
            "    coherence sign within support_2 strata : AUC = {:.3}",
            strata_auc(&k2, &|r: &Row| coh(r))
        );
        println!(
            "    lex(+1, |D|)  within support_2 strata  : AUC = {:.3}",
            strata_auc(&k2, &|r: &Row| (if coh(r) > 0.0 { 1000.0 } else { 0.0 })
                + mag(r))
        );
        println!(
            "    RANDOM-PAIRED sign within support_2    : AUC = {:.3}",
            strata_auc(&k2, &|r: &Row| if r.s1 >= r.s2r { 1.0 } else { -1.0 })
        );

        // Held out, per configuration: does the sign add to the joint model?
        println!("\n  Held out (fit on TRAIN, scored on TEST), mean over the 22 configurations:");
        let feats2: Vec<(&str, Vec<usize>)> = vec![
            ("s1 + s2                    ", vec![0, 1]),
            ("s1 + s2 + |D|              ", vec![0, 1, 2]),
            ("s1 + s2 + |D| + sign       ", vec![0, 1, 2, 3]),
            ("|D| + sign  (decomposition)", vec![2, 3]),
            ("sign alone                 ", vec![3]),
        ];
        let mut acc2: Vec<Vec<f64>> = vec![Vec::new(); feats2.len()];
        for &(stride, mult) in &configs {
            let rows = score(stride, mult);
            let tr: Vec<&Row> = rows.iter().filter(|r| !r.in_test).collect();
            let te: Vec<&Row> = rows.iter().filter(|r| r.in_test).collect();
            let mk = |r: &Row| {
                vec![
                    r.s1,
                    r.s2,
                    (r.s1 - r.s2).abs(),
                    if r.s1 >= r.s2 { 1.0 } else { -1.0 },
                ]
            };
            for (k, (_, cols)) in feats2.iter().enumerate() {
                let xtr: Vec<Vec<f64>> = tr
                    .iter()
                    .map(|r| cols.iter().map(|&c| mk(r)[c]).collect())
                    .collect();
                let ytr: Vec<bool> = tr.iter().map(|r| r.injected).collect();
                let (w, mu, sd) = fit_logistic(&xtr, &ytr, 4000, 0.5);
                let sc: Vec<(f64, bool)> = te
                    .iter()
                    .map(|r| {
                        let row: Vec<f64> = cols.iter().map(|&c| mk(r)[c]).collect();
                        (apply_logistic(&w, &mu, &sd, &row), r.injected)
                    })
                    .collect();
                acc2[k].push(auc(&sc));
            }
        }
        for (k, (name, _)) in feats2.iter().enumerate() {
            let m = acc2[k].iter().sum::<f64>() / acc2[k].len() as f64;
            let d: Vec<f64> = (0..acc2[k].len())
                .map(|c| acc2[k][c] - acc2[0][c])
                .collect();
            let (dm, _, t) = paired_t(&d);
            let wins = d.iter().filter(|x| **x > 0.0).count();
            println!("    {name}  {m:.3}   vs s1+s2 {dm:+.3}  paired t {t:+.2}  wins {wins}/22");
        }

        // ── E1c: the decomposition again, with the model scales removed ────
        // E1b's sign is +1 on 86% of rows, because BGE's cosines live in a
        // narrower band than MiniLM's. A near-constant sign cannot discriminate
        // whatever it means, so E1b tested a scale artifact rather than the
        // proposal. Standardising each model's supports within its OWN
        // distribution (no labels involved) gives a sign that actually reports
        // per-item disagreement, which is the version the idea deserves.
        println!("\n=== E1c: the same decomposition, with each model's scale removed ===\n");
        let z1v: Vec<f64> = sel.iter().map(|r| r.s1).collect();
        let z2v: Vec<f64> = sel.iter().map(|r| r.s2).collect();
        let (mu1, mu2) = (mean(&z1v), mean(&z2v));
        let sd1 = (z1v.iter().map(|x| (x - mu1).powi(2)).sum::<f64>() / z1v.len() as f64).sqrt();
        let sd2 = (z2v.iter().map(|x| (x - mu2).powi(2)).sum::<f64>() / z2v.len() as f64).sqrt();
        println!(
            "  support_1  mean {mu1:.3}  sd {sd1:.3}      support_2  mean {mu2:.3}  sd {sd2:.3}"
        );
        let z1 = move |r: &Row| (r.s1 - mu1) / sd1;
        let z2 = move |r: &Row| (r.s2 - mu2) / sd2;
        let zcoh = move |r: &Row| if z1(r) >= z2(r) { 1.0 } else { -1.0 };
        let zmag = move |r: &Row| (z1(r) - z2(r)).abs();
        let zpos = sel.iter().filter(|r| zcoh(r) > 0.0).count();
        println!(
            "  coherence split now: {} at +1, {} at -1  (was {} / {})\n",
            zpos,
            sel.len() - zpos,
            n_pos,
            sel.len() - n_pos
        );
        let arms3: Vec<Arm2> = vec![
            (
                "z coherence sign alone    ",
                Box::new(move |r: &Row| zcoh(r)),
            ),
            (
                "z modulus |dz| alone      ",
                Box::new(move |r: &Row| zmag(r)),
            ),
            (
                "z lex: +1 block, then |dz|",
                Box::new(move |r: &Row| (if zcoh(r) > 0.0 { 1000.0 } else { 0.0 }) + zmag(r)),
            ),
            (
                "z lex: -1 block, then |dz|",
                Box::new(move |r: &Row| (if zcoh(r) < 0.0 { 1000.0 } else { 0.0 }) + zmag(r)),
            ),
            (
                "z signed dz               ",
                Box::new(move |r: &Row| z1(r) - z2(r)),
            ),
        ];
        println!("  arm                            AUC     vs baseline (0.739)");
        for (name, f) in &arms3 {
            let a = a_of(f.as_ref());
            println!("  {name}  {a:.3}    {:+.3}", a - 0.739);
        }
        println!("\n  The control that decided E1, on the scale-free version:");
        println!(
            "    z sign      within support_2 strata : AUC = {:.3}",
            strata_auc(&k2, &|r: &Row| zcoh(r))
        );
        println!(
            "    z modulus   within support_2 strata : AUC = {:.3}",
            strata_auc(&k2, &|r: &Row| zmag(r))
        );
        println!(
            "    z modulus   within z-sign strata    : AUC = {:.3}",
            strata_auc(&|r: &Row| zcoh(r), &|r: &Row| zmag(r))
        );
        {
            // the derangement null, standardised the same way
            let s2rv: Vec<f64> = sel.iter().map(|r| r.s2r).collect();
            let mu2r = mean(&s2rv);
            let sd2r =
                (s2rv.iter().map(|x| (x - mu2r).powi(2)).sum::<f64>() / s2rv.len() as f64).sqrt();
            let zr = move |r: &Row| (r.s2r - mu2r) / sd2r;
            println!(
                "    RANDOM-PAIRED z sign within support_2 : AUC = {:.3}",
                strata_auc(&k2, &|r: &Row| if z1(r) >= zr(r) { 1.0 } else { -1.0 })
            );
            println!(
                "    RANDOM-PAIRED z modulus within supp_2 : AUC = {:.3}",
                strata_auc(&k2, &|r: &Row| (z1(r) - zr(r)).abs())
            );
        }
        println!("\n  Held out (fit on TRAIN, scored on TEST), mean over the 22 configurations:");
        let feats3: Vec<(&str, Vec<usize>)> = vec![
            ("s1 + s2                     ", vec![0, 1]),
            ("s1 + s2 + |dz| + z sign     ", vec![0, 1, 2, 3]),
            ("|dz| + z sign (decomposition)", vec![2, 3]),
        ];
        let mut acc3: Vec<Vec<f64>> = vec![Vec::new(); feats3.len()];
        for &(stride, mult) in &configs {
            let rows = score(stride, mult);
            let tr: Vec<&Row> = rows.iter().filter(|r| !r.in_test).collect();
            let te: Vec<&Row> = rows.iter().filter(|r| r.in_test).collect();
            let mk = |r: &Row| {
                let (a, b) = ((r.s1 - mu1) / sd1, (r.s2 - mu2) / sd2);
                vec![r.s1, r.s2, (a - b).abs(), if a >= b { 1.0 } else { -1.0 }]
            };
            for (k, (_, cols)) in feats3.iter().enumerate() {
                let xtr: Vec<Vec<f64>> = tr
                    .iter()
                    .map(|r| cols.iter().map(|&c| mk(r)[c]).collect())
                    .collect();
                let ytr: Vec<bool> = tr.iter().map(|r| r.injected).collect();
                let (w, mu, sd) = fit_logistic(&xtr, &ytr, 4000, 0.5);
                let sc: Vec<(f64, bool)> = te
                    .iter()
                    .map(|r| {
                        let row: Vec<f64> = cols.iter().map(|&c| mk(r)[c]).collect();
                        (apply_logistic(&w, &mu, &sd, &row), r.injected)
                    })
                    .collect();
                acc3[k].push(auc(&sc));
            }
        }
        for (k, (name, _)) in feats3.iter().enumerate() {
            let m = acc3[k].iter().sum::<f64>() / acc3[k].len() as f64;
            let d: Vec<f64> = (0..acc3[k].len())
                .map(|c| acc3[k][c] - acc3[0][c])
                .collect();
            let (dm, _, t) = paired_t(&d);
            let wins = d.iter().filter(|x| **x > 0.0).count();
            println!("    {name} {m:.3}   vs s1+s2 {dm:+.3}  paired t {t:+.2}  wins {wins}/22");
        }

        // ── PHYSIS_E1_POINTS: every scored row at the published configuration,
        // for the graphical inspector. Scores only — no mechanism depends on it.
        {
            let out = std::env::var("PHYSIS_E1_POINTS")
                .unwrap_or_else(|_| "research/perspective-discovery/e1_points.json".to_string());
            let mut j = String::from(
                "{\n  \"config\": \"stride 7, mult 13, half=all\",\n  \"points\": [\n",
            );
            for (k, r) in sel.iter().enumerate() {
                j.push_str(&format!(
                    "    {{\"s1\": {:.4}, \"s2\": {:.4}, \"inj\": {}, \"test\": {}}}{}\n",
                    r.s1,
                    r.s2,
                    r.injected,
                    r.in_test,
                    if k + 1 == sel.len() { "" } else { "," }
                ));
            }
            j.push_str("  ]\n}\n");
            if std::fs::write(&out, j).is_ok() {
                println!("\n  wrote {} scored points to {out}", sel.len());
            }
        }

        // ── correlation between the two supports ──
        let s1v: Vec<f64> = sel.iter().map(|r| r.s1).collect();
        let s2v: Vec<f64> = sel.iter().map(|r| r.s2).collect();
        let (m1, m2v) = (mean(&s1v), mean(&s2v));
        let cov: f64 = s1v
            .iter()
            .zip(&s2v)
            .map(|(a, b)| (a - m1) * (b - m2v))
            .sum::<f64>();
        let v1: f64 = s1v.iter().map(|a| (a - m1).powi(2)).sum::<f64>().sqrt();
        let v2: f64 = s2v.iter().map(|b| (b - m2v).powi(2)).sum::<f64>().sqrt();
        println!(
            "\n  Pearson r(support_1, support_2) = {:.3}",
            cov / (v1 * v2).max(1e-9)
        );
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
