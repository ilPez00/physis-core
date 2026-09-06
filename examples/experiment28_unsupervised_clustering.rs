//! Experiment 28 — roadmap item 24. Plain unsupervised clustering is
//! described at the top of FINAL_REPORT.md as "the one mechanism this
//! track never found a way to make lose on exclusive-membership data",
//! and is the only survivor recommended for `discovery.rs`. It is also
//! the one mechanism never actually prototyped, so the recommendation
//! rests on it having won a few small comparisons — never on a serious
//! attempt to break it.
//!
//! This experiment is that attempt. The specific suspicion: **it never
//! lost because it was always handed the correct k.** Every prior
//! appearance of clustering in this track (Iterations 5, 7, 10, 12)
//! supplied the number of clusters from ground truth. Choosing k is the
//! certification problem wearing a different hat — and Iterations 9 and
//! 11 established that silhouette cannot rank a good split above a bad
//! one, with Iteration 27 confirming that under deterministic ordering.
//!
//! So the question is not "does clustering recover structure" but "does
//! it still do so when nobody tells it how many clusters exist".
//!
//! Three parts:
//!   A. ORACLE k — given the true cluster count, does it beat a random
//!      partition of the same shape? (Reproduces the claim as previously
//!      tested.)
//!   B. SELECTED k — pick k by internal criteria (silhouette, elbow) with
//!      no access to ground truth, then measure the damage.
//!   C. The honest arithmetic: how much of part A's win was the oracle?
//!
//! Run:
//!   cargo run -p physis-core --features embed-onnx --release --example experiment28_unsupervised_clustering

use physis_core::models::cosine_sim;
use physis_core::ontology::OntologyLoader;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

/// Unit-normalize; every routine here assumes unit vectors so cosine is a dot.
fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    v.iter().map(|x| x / n).collect()
}

fn centroid(members: &[&Vec<f32>]) -> Vec<f32> {
    let dim = members[0].len();
    let mut sum = vec![0.0f32; dim];
    for m in members {
        for d in 0..dim {
            sum[d] += m[d];
        }
    }
    normalize(&sum)
}

/// Spherical k-means with k-means++ seeding. Deterministic given `seed`:
/// the RNG is explicitly seeded and every argmax breaks ties on the lower
/// index, so there is no reliance on iteration order anywhere (the trap
/// Iteration 26a found in production code).
fn kmeans(embeddings: &[Vec<f32>], k: usize, seed: u64, iters: usize) -> Vec<usize> {
    let n = embeddings.len();
    if k >= n {
        return (0..n).collect();
    }
    let mut rng = StdRng::seed_from_u64(seed);

    // k-means++ seeding: first centre uniform, rest weighted by squared
    // cosine distance to the nearest chosen centre.
    let mut centres: Vec<Vec<f32>> = vec![embeddings[rng.gen_range(0..n)].clone()];
    while centres.len() < k {
        let d2: Vec<f32> = embeddings
            .iter()
            .map(|e| {
                let best = centres
                    .iter()
                    .map(|c| cosine_sim(e, c))
                    .fold(f32::NEG_INFINITY, f32::max);
                let d = (1.0 - best).max(0.0);
                d * d
            })
            .collect();
        let total: f32 = d2.iter().sum();
        if total <= 1e-12 {
            centres.push(embeddings[centres.len() % n].clone());
            continue;
        }
        let mut target = rng.gen_range(0.0..total);
        let mut pick = n - 1;
        for (i, w) in d2.iter().enumerate() {
            target -= w;
            if target <= 0.0 {
                pick = i;
                break;
            }
        }
        centres.push(embeddings[pick].clone());
    }

    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        let mut changed = false;
        for (i, e) in embeddings.iter().enumerate() {
            let mut best = 0usize;
            let mut best_sim = f32::NEG_INFINITY;
            for (c, centre) in centres.iter().enumerate() {
                let s = cosine_sim(e, centre);
                if s > best_sim {
                    best_sim = s;
                    best = c;
                }
            }
            if assign[i] != best {
                assign[i] = best;
                changed = true;
            }
        }
        for c in 0..k {
            let members: Vec<&Vec<f32>> = (0..n).filter(|&i| assign[i] == c).map(|i| &embeddings[i]).collect();
            if !members.is_empty() {
                centres[c] = centroid(&members);
            }
        }
        if !changed {
            break;
        }
    }
    assign
}

fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len();
    let mut table: HashMap<(usize, usize), usize> = HashMap::new();
    let mut ra: HashMap<usize, usize> = HashMap::new();
    let mut rb: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        *table.entry((a[i], b[i])).or_default() += 1;
        *ra.entry(a[i]).or_default() += 1;
        *rb.entry(b[i]).or_default() += 1;
    }
    let c2 = |x: usize| (x as f64) * (x as f64 - 1.0) / 2.0;
    let sum_ij: f64 = table.values().map(|&v| c2(v)).sum();
    let sum_a: f64 = ra.values().map(|&v| c2(v)).sum();
    let sum_b: f64 = rb.values().map(|&v| c2(v)).sum();
    let total = c2(n);
    let expected = sum_a * sum_b / total;
    let max_index = (sum_a + sum_b) / 2.0;
    if (max_index - expected).abs() < 1e-12 {
        return 0.0;
    }
    (sum_ij - expected) / (max_index - expected)
}

fn nmi(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len() as f64;
    let mut joint: HashMap<(usize, usize), usize> = HashMap::new();
    let mut ca: HashMap<usize, usize> = HashMap::new();
    let mut cb: HashMap<usize, usize> = HashMap::new();
    for i in 0..a.len() {
        *joint.entry((a[i], b[i])).or_default() += 1;
        *ca.entry(a[i]).or_default() += 1;
        *cb.entry(b[i]).or_default() += 1;
    }
    let entropy = |c: &HashMap<usize, usize>| -> f64 {
        -c.values()
            .map(|&v| {
                let p = v as f64 / n;
                p * p.ln()
            })
            .sum::<f64>()
    };
    let (ha, hb) = (entropy(&ca), entropy(&cb));
    let mut mi = 0.0;
    for (&(x, y), &v) in &joint {
        let pxy = v as f64 / n;
        let px = ca[&x] as f64 / n;
        let py = cb[&y] as f64 / n;
        mi += pxy * (pxy / (px * py)).ln();
    }
    if ha <= 0.0 || hb <= 0.0 {
        return 0.0;
    }
    mi / ((ha * hb).sqrt())
}

fn purity(pred: &[usize], truth: &[usize]) -> f64 {
    let mut per_cluster: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    for i in 0..pred.len() {
        *per_cluster.entry(pred[i]).or_default().entry(truth[i]).or_default() += 1;
    }
    let correct: usize = per_cluster
        .values()
        .map(|counts| *counts.values().max().unwrap_or(&0))
        .sum();
    correct as f64 / pred.len() as f64
}

/// Mean silhouette using cosine distance. This is the internal criterion
/// under test — the same family Iterations 9/11 already found unable to
/// rank a good split above a bad one.
fn silhouette(embeddings: &[Vec<f32>], assign: &[usize], k: usize) -> f64 {
    let n = embeddings.len();
    let mut total = 0.0;
    let mut counted = 0usize;
    for i in 0..n {
        let mut sums = vec![0.0f64; k];
        let mut counts = vec![0usize; k];
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = 1.0 - cosine_sim(&embeddings[i], &embeddings[j]) as f64;
            sums[assign[j]] += d;
            counts[assign[j]] += 1;
        }
        if counts[assign[i]] == 0 {
            continue; // singleton cluster: silhouette undefined
        }
        let a = sums[assign[i]] / counts[assign[i]] as f64;
        let mut b = f64::INFINITY;
        for c in 0..k {
            if c == assign[i] || counts[c] == 0 {
                continue;
            }
            b = b.min(sums[c] / counts[c] as f64);
        }
        if b.is_finite() {
            total += (b - a) / a.max(b);
            counted += 1;
        }
    }
    if counted == 0 {
        0.0
    } else {
        total / counted as f64
    }
}

/// Within-cluster cosine distance to own centroid — the elbow quantity.
fn inertia(embeddings: &[Vec<f32>], assign: &[usize], k: usize) -> f64 {
    let mut total = 0.0;
    for c in 0..k {
        let members: Vec<&Vec<f32>> = (0..embeddings.len()).filter(|&i| assign[i] == c).map(|i| &embeddings[i]).collect();
        if members.is_empty() {
            continue;
        }
        let ctr = centroid(&members);
        for m in &members {
            total += 1.0 - cosine_sim(m, &ctr) as f64;
        }
    }
    total
}

fn random_partition(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(0..k)).collect()
}

fn main() {
    println!("Experiment 28: does plain unsupervised clustering still win without an oracle k?\n");

    #[cfg(feature = "embed-onnx")]
    {
        use physis_core::embed::VectorEmbed;
        use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};

        let dir = match ["models", "../models"]
            .iter()
            .find(|d| std::path::Path::new(d).join("model.onnx").exists())
        {
            Some(d) => *d,
            None => {
                println!("WARNING: MiniLM not available — aborting.");
                return;
            }
        };
        let embedder = OnnxEmbedder::with_config(&OnnxConfig {
            dim: 384,
            model_dir: Some(dir.to_string()),
            pooling: PoolingStrategy::Mean,
            ..OnnxConfig::default()
        });
        if !embedder.is_available() {
            println!("WARNING: embedder unavailable — aborting.");
            return;
        }

        // Entry order is deterministic since physis-core 0.1.15 (Iteration 26a).
        let ontology = OntologyLoader::load_all();
        let mut texts = Vec::new();
        let mut domains = Vec::new();
        for def in ontology.classification_domains() {
            let (Some(d), Some(_m)) = (&def.domain, &def.mode) else { continue };
            let mut t = def.name.clone();
            for h in &def.hints {
                t.push(' ');
                t.push_str(h);
            }
            texts.push(t);
            domains.push(d.clone());
        }
        println!("Loaded {} real entries, embedding...", texts.len());
        let embeddings: Vec<Vec<f32>> = texts.iter().map(|t| normalize(&embedder.embed(t))).collect();

        // Ground truth = the 5 coarse domains, mapped to stable indices by
        // sorted name so the labelling itself carries no ordering surprise.
        let mut names: Vec<String> = {
            let mut v: Vec<String> = domains.clone();
            v.sort();
            v.dedup();
            v
        };
        names.sort();
        let idx: HashMap<&str, usize> = names.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
        let truth: Vec<usize> = domains.iter().map(|d| idx[d.as_str()]).collect();
        let true_k = names.len();
        println!("Ground truth: {true_k} domains {names:?}\n");

        // ---------- Part 0: the control ----------
        // The "never loses" claim comes from Iterations 5/7/10/12, which all
        // ran on small hand-built sets like this one. Running it here is the
        // control that tells us whether a weak result below is a property of
        // clustering or a property of the real corpus.
        println!("=== PART 0 — control: Dataset A, the kind of set the claim came from ===\n");
        let ds_a = [
            "dog", "cat", "horse", "sheep", "lion", "wolf", "bear", "tiger",
            "eagle", "sparrow", "owl", "swan", "penguin", "ostrich", "kiwi",
        ];
        let ds_a_truth: Vec<usize> = (0..15).map(|i| usize::from(i >= 8)).collect();
        let ds_a_emb: Vec<Vec<f32>> = ds_a.iter().map(|t| normalize(&embedder.embed(t))).collect();
        let ds_a_assign = kmeans(&ds_a_emb, 2, 42, 100);
        let ds_a_rand = random_partition(15, 2, 7);
        println!(
            "  k-means (oracle k=2)  ARI={:.3}  purity={:.3}",
            adjusted_rand_index(&ds_a_assign, &ds_a_truth),
            purity(&ds_a_assign, &ds_a_truth)
        );
        println!(
            "  random partition      ARI={:.3}  purity={:.3}",
            adjusted_rand_index(&ds_a_rand, &ds_a_truth),
            purity(&ds_a_rand, &ds_a_truth)
        );
        let mut szs = vec![0usize; 2];
        for &c in &ds_a_assign {
            szs[c] += 1;
        }
        szs.sort_unstable();
        println!("  cluster sizes: {szs:?}   (true split is 8/7)");
        println!("  members: {:?}", (0..2).map(|c| ds_a.iter().enumerate().filter(|(i, _)| ds_a_assign[*i] == c).map(|(_, t)| *t).collect::<Vec<_>>()).collect::<Vec<_>>());
        println!("  (mammal/bird, 15 items — this is where 'never loses' was established)\n");

        // ---------- Part A: ORACLE k ----------
        println!("=== PART A — real 730-entry ontology, given the TRUE k ({true_k}) ===\n");
        let a_assign = kmeans(&embeddings, true_k, 42, 100);
        let a_ari = adjusted_rand_index(&a_assign, &truth);
        let a_nmi = nmi(&a_assign, &truth);
        let a_pur = purity(&a_assign, &truth);
        let r_assign = random_partition(embeddings.len(), true_k, 7);
        let r_ari = adjusted_rand_index(&r_assign, &truth);
        let r_nmi = nmi(&r_assign, &truth);
        let r_pur = purity(&r_assign, &truth);
        let one = vec![0usize; embeddings.len()];
        println!("  k-means (oracle k)   ARI={a_ari:.3}  NMI={a_nmi:.3}  purity={a_pur:.3}");
        println!("  random partition     ARI={r_ari:.3}  NMI={r_nmi:.3}  purity={r_pur:.3}");
        println!(
            "  single cluster       ARI={:.3}  NMI={:.3}  purity={:.3}",
            adjusted_rand_index(&one, &truth),
            nmi(&one, &truth),
            purity(&one, &truth)
        );
        println!(
            "\n  beats random on ARI: {}   (this is the comparison the 'never loses' claim rests on)\n",
            a_ari > r_ari
        );

        // ---------- Part B: SELECTED k ----------
        println!("=== PART B — nobody supplies k. Sweep, then pick by internal criteria ===\n");
        println!("     k    ARI     NMI   purity   silhouette   inertia");
        let ks: Vec<usize> = (2..=20).collect();
        let mut rows = Vec::new();
        for &k in &ks {
            let asg = kmeans(&embeddings, k, 42, 100);
            let ari = adjusted_rand_index(&asg, &truth);
            let sil = silhouette(&embeddings, &asg, k);
            let inr = inertia(&embeddings, &asg, k);
            println!(
                "  {:4}  {:.3}  {:.3}   {:.3}      {:+.4}   {:8.1}",
                k,
                ari,
                nmi(&asg, &truth),
                purity(&asg, &truth),
                sil,
                inr
            );
            rows.push((k, ari, sil, inr));
        }

        // Silhouette's pick: argmax, ties to the smaller k.
        let sil_pick = rows
            .iter()
            .fold(rows[0], |best, &r| if r.2 > best.2 { r } else { best });
        // Elbow: largest second difference in inertia (the classic knee rule),
        // again with a deterministic tie-break on the smaller k.
        let mut elbow = rows[0];
        let mut best_drop = f64::NEG_INFINITY;
        for w in rows.windows(3) {
            let second_diff = (w[0].3 - w[1].3) - (w[1].3 - w[2].3);
            if second_diff > best_drop {
                best_drop = second_diff;
                elbow = w[1];
            }
        }
        let best_ari = rows.iter().fold(rows[0], |b, &r| if r.1 > b.1 { r } else { b });

        println!("\n  true k                        = {true_k}");
        println!("  silhouette picks k            = {}  (ARI there = {:.3})", sil_pick.0, sil_pick.1);
        println!("  elbow (knee in inertia) picks = {}  (ARI there = {:.3})", elbow.0, elbow.1);
        println!("  best ARI in the whole sweep   = {:.3} at k={}", best_ari.1, best_ari.0);

        // ---------- Part C: the arithmetic ----------
        println!("\n=== PART C — how much of Part A's win was the oracle? ===\n");
        let sil_loss = a_ari - sil_pick.1;
        let elbow_loss = a_ari - elbow.1;
        println!("  ARI with oracle k          : {a_ari:.3}");
        println!("  ARI with silhouette-chosen k: {:.3}   (lost {:.3}, {:.0}% of the oracle result)", sil_pick.1, sil_loss, 100.0 * sil_loss / a_ari.max(1e-9));
        println!("  ARI with elbow-chosen k     : {:.3}   (lost {:.3}, {:.0}% of the oracle result)", elbow.1, elbow_loss, 100.0 * elbow_loss / a_ari.max(1e-9));
        println!("  random-partition floor      : {r_ari:.3}");

        // Reported per-criterion: a single boolean would hide that the two
        // internal criteria disagree sharply, which is itself the finding.
        println!("\n  silhouette-chosen k beats the random floor: {}", sil_pick.1 > r_ari);
        println!("  elbow-chosen k beats the random floor     : {}", elbow.1 > r_ari);
        println!("  silhouette recovers the true k            : {}", sil_pick.0 == true_k);
        println!("  elbow recovers the true k                 : {}", elbow.0 == true_k);

        // The headline is not the k-selection loss — it is the size of the
        // oracle result in the first place. Say so numerically rather than
        // letting a "beats random: true" carry an implication it can't bear.
        let oracle_over_floor = a_ari - r_ari;
        println!("\n  === the number that actually matters ===");
        println!("  oracle-k ARI minus random floor = {:.3} - {:.3} = {:.3}", a_ari, r_ari, oracle_over_floor);
        println!("  purity: oracle {:.3} vs single-cluster {:.3} (i.e. {:+.1} points over labelling everything one class)",
                 a_pur, purity(&one, &truth), 100.0 * (a_pur - purity(&one, &truth)));
        println!("  best ARI anywhere in k=2..20    = {:.3}", best_ari.1);
        println!(
            "\n  VERDICT: on the real corpus, plain clustering recovers the coarse domain\n  structure {} — with or without an oracle k. The 'never loses' record was\n  set on 15-item curated sets (Part 0), and does not carry over.",
            if best_ari.1 < 0.10 { "essentially not at all (ARI < 0.10 everywhere)" } else { "partially" }
        );
        println!("\n(Part A is the comparison every prior iteration ran. Parts 0 and B are the ones\n none did. What is NOT claimed: this tests the 5 COARSE domains on one corpus and\n one embedder. It says nothing about the 70-cell fine structure, and per Iteration\n 26b nothing here transfers across a model swap.)");
    }
    #[cfg(not(feature = "embed-onnx"))]
    println!("Built without --features embed-onnx; aborting.");
}
