// NOTE: the no-embed-onnx build is a stub; the analysis helpers below are
// intentionally dead there (they serve the embed-onnx analysis path only).
#![cfg_attr(not(feature = "embed-onnx"), allow(dead_code, unused_imports))]
//! Experiment 44 — the worklist ranking: **does the disposer rank, not just filter?**
//!
//! Two mechanisms in this entire track survived their controls:
//!
//! 1. Iteration 30's proposer (z = +4.87): near-neighbour midpoints point at
//!    concepts the ontology is missing, at ~13% precision.
//! 2. `coverage` (t = +12.11): a candidate that re-adds what the ontology
//!    already has rescues 0/2000 records; interpolations rescue real ones.
//!
//! Iteration 33 proved the disposer FILTERS (2000 -> 141) but never tested
//! whether it RANKS: whether sorting the proposer's output by coverage gain
//! concentrates the true gaps (the held-out concepts) at the top, where a
//! human would read first. That composition is the PDCA discovery loop the
//! user asked for: Plan proposes, Check ranks, the human only Act(s) on the
//! head of the list.
//!
//! Records are the OPERATIONAL corpus (6,234 events: telemetry, document
//! paragraphs, git workflow) — external to the proposal mechanism, which is
//! what makes this check operational rather than geometric. The proposer is
//! built and validated on the ontology alone; it never sees these records.
//!
//! Arms (same proposals, same hit definition, only the ranking signal
//! differs — the construction-matched discipline):
//!   COVERAGE   : rank by candidate_gain over the operational records
//!   GEOMETRY   : rank by the proposal's max cosine to known entries
//!                (how deeply embedded in known territory it is)
//!   PAIR-SIM   : rank by the parents' cosine with each other
//!   ISOLATION  : rank by ASCENDING cosine to the nearest NON-PARENT known
//!                entry — "how empty is this region of the ontology alone".
//!                The construction-matched control for LIFT: if the ontology
//!                already knows where its own holes are, the operational
//!                corpus adds nothing. Deliberately label-coupled — this
//!                quantity is one side of the HIT comparison — so it is a
//!                HARD control, not a fair rival.
//!   RANDOM     : seeded shuffle, the floor for precision@k
//!
//! A proposal is a HIT if its nearest held-out concept is closer than any
//! non-parent known entry (Iterations 30/31's ground truth, parent-adjacency
//! excluded — the artifact that voided two earlier metrics).
//!
//!   cargo run -p physis-core --features embed-onnx --release --example experiment44_worklist_ranking

fn main() {
    println!("Experiment 44: does the coverage disposer RANK the proposer's output?\n");
    #[cfg(feature = "embed-onnx")]
    run();
}

#[cfg(feature = "embed-onnx")]
fn run() {
    use physis_core::embed::VectorEmbed;
    use physis_core::embed_onnx::{OnnxConfig, OnnxEmbedder, PoolingStrategy};
    use physis_core::ontology::OntologyLoader;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use std::collections::{HashMap, HashSet};

    let dirs = ["physis-core", ".", "..", "../physis-core", "/home/gio/dev/physis-pro/physis-core"];
    let model_dir = dirs
        .iter()
        .find_map(|d| {
            let m = std::path::Path::new(d).join("models");
            m.join("model.onnx").exists().then(|| m.to_string_lossy().into_owned())
        })
        .expect("models/model.onnx not found");
    let embedder = OnnxEmbedder::with_config(&OnnxConfig {
        model_dir: Some(model_dir.to_string()),
        pooling: PoolingStrategy::Mean,
        ..OnnxConfig::default()
    });
    if !embedder.is_available() {
        println!("WARNING: embedder unavailable — aborting.");
        return;
    }

    // ---------- the ontology and the hold-out ----------
    let ontology = OntologyLoader::load_all();
    let mut texts = Vec::new();
    let mut names = Vec::new();
    let mut cells = Vec::new();
    for def in ontology.classification_domains() {
        let (Some(d), Some(m)) = (&def.domain, &def.mode) else { continue };
        let mut t = def.name.clone();
        for h in &def.hints {
            t.push(' ');
            t.push_str(h);
        }
        texts.push(t);
        names.push(def.name.clone());
        cells.push((d.clone(), m.clone()));
    }
    println!("physis: {} entries, embedding...", texts.len());
    let emb: Vec<Vec<f32>> = texts.iter().map(|t| normalize(&embedder.embed(t))).collect();

    // Stratified hold-out, identical construction to Iterations 30/31.
    let mut by_cell: HashMap<(String, String), Vec<usize>> = HashMap::new();
    for (i, c) in cells.iter().enumerate() {
        by_cell.entry(c.clone()).or_default().push(i);
    }
    let mut cell_keys: Vec<&(String, String)> = by_cell.keys().collect();
    cell_keys.sort();
    let mut held: Vec<usize> = Vec::new();
    for k in &cell_keys {
        let m = &by_cell[*k];
        let n_hold = ((m.len() as f64 * 0.10).round() as usize).min(m.len().saturating_sub(2));
        for j in 0..n_hold {
            held.push(m[(j * 7 + 3) % m.len()]);
        }
    }
    held.sort_unstable();
    held.dedup();
    let held_set: HashSet<usize> = held.iter().copied().collect();
    let known: Vec<usize> = (0..texts.len()).filter(|i| !held_set.contains(i)).collect();
    println!("hold-out: {} known / {} held out", known.len(), held.len());

    // ---------- propose (Iteration 30's validated construction) ----------
    const NN: usize = 5;
    let mut proposals: Vec<(Vec<f32>, usize, usize)> = Vec::new();
    for (pos, &i) in known.iter().enumerate() {
        let mut sims: Vec<(f32, usize)> = known
            .iter()
            .enumerate()
            .filter(|(p, _)| *p != pos)
            .map(|(_, &j)| (cosine_sim(&emb[i], &emb[j]), j))
            .collect();
        sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.cmp(&b.1)));
        for &(_, j) in sims.iter().take(NN) {
            if i < j {
                proposals.push((midpoint(&emb[i], &emb[j]), i, j));
            }
        }
    }
    println!("proposals: {} near-neighbour midpoints (ranks 1-{NN})\n", proposals.len());

    // ---------- the operational records ----------
    #[derive(serde::Deserialize)]
    struct Event {
        subject: String,
        evidence: Vec<String>,
        observed_at: String,
        event_key: String,
    }
    let raw = std::fs::read("operational_corpus.json")
        .or_else(|_| std::fs::read("../operational_corpus.json"))
        .expect("operational_corpus.json not found (run from the physis-pro root)");
    let mut events: Vec<Event> = serde_json::from_slice(&raw).expect("corpus parse");
    // Chronological order, tie-broken on the key — same as Experiment 39.
    events.sort_by(|a, b| a.observed_at.cmp(&b.observed_at).then(a.event_key.cmp(&b.event_key)));
    let rec_texts: Vec<String> = events
        .iter()
        .map(|e| format!("{} {}", e.subject, e.evidence.join(" ")).to_lowercase())
        .collect();
    println!("records: {} operational events, embedding...", rec_texts.len());
    let rec_emb: Vec<Vec<f32>> = rec_texts.iter().map(|t| normalize(&embedder.embed(t))).collect();

    // ---------- live coverage over records ----------
    // Raw best-entry cosine against the live (90%) ontology, following
    // discovery.rs and Experiment 33 ("undiluted by the blended classifier").
    let live: Vec<f32> = rec_emb
        .iter()
        .map(|r| known.iter().map(|&k| cosine_sim(r, &emb[k])).fold(f32::NEG_INFINITY, f32::max))
        .collect();
    let mut sorted = live.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold = quantile(&sorted, 0.50);
    let uncovered: Vec<usize> = (0..rec_emb.len()).filter(|&i| live[i] < threshold).collect();
    println!(
        "coverage: threshold {:.3} (median) — {}/{} records uncovered ({:.0}%)\n",
        threshold,
        uncovered.len(),
        rec_emb.len(),
        100.0 * uncovered.len() as f64 / rec_emb.len() as f64
    );

    // ---------- signals per proposal ----------
    let n = proposals.len();
    // The planner's binarized rule (newly covered at threshold) is structurally
    // dead on this record population: dev-corpus records match the ontology
    // strongly (median live 0.741), and a midpoint — sandwiched between two
    // entries — cannot beat all 657 known entries for any record. Iteration
    // 33's records were held-out ENTRIES (median ~0.5) where the bar is
    // reachable; operational TEXT raises it out of reach. So the ranking
    // signal is the continuous form of the same rule — how much the candidate
    // RAISES each record's best score — summed over records, plus the
    // binarized rule at a sweep of thresholds reported for transparency.
    let mut lift = vec![0.0f64; n];
    let mut gain_p25 = vec![0usize; n];
    let mut gain_p50 = vec![0usize; n];
    let mut geo = vec![f32::NEG_INFINITY; n]; // max cosine to any known entry
    let mut pair = vec![0.0f32; n]; // parents' mutual cosine
    let mut traffic = vec![f32::NEG_INFINITY; n]; // max cosine to ANY record — popularity control
    let mut isolation = vec![0.0f32; n]; // cosine to nearest NON-PARENT known entry
    let mut hit = vec![false; n];
    let t25 = quantile(&sorted, 0.25);
    for (pi, (p, i, j)) in proposals.iter().enumerate() {
        for r in 0..rec_emb.len() {
            let s = cosine_sim(p, &rec_emb[r]);
            if s > traffic[pi] {
                traffic[pi] = s;
            }
            if s > live[r] {
                lift[pi] += (s - live[r]) as f64;
            }
            if s >= t25 && live[r] < t25 {
                gain_p25[pi] += 1;
            }
            if s >= threshold && live[r] < threshold {
                gain_p50[pi] += 1;
            }
        }
        geo[pi] = known.iter().map(|&k| cosine_sim(p, &emb[k])).fold(f32::NEG_INFINITY, f32::max);
        pair[pi] = cosine_sim(&emb[*i], &emb[*j]);

        // Hit ground truth (Iterations 30/31): nearest held-out concept beats
        // the nearest NON-PARENT known entry. Parents are adjacent to their
        // own midpoint by construction and must be excluded.
        let best_held = held.iter().map(|&h| cosine_sim(p, &emb[h])).fold(f32::NEG_INFINITY, f32::max);
        let rival = known
            .iter()
            .filter(|&&k| k != *i && k != *j)
            .map(|&k| cosine_sim(p, &emb[k]))
            .fold(f32::NEG_INFINITY, f32::max);
        hit[pi] = best_held > rival;
        isolation[pi] = rival;
    }
    let p25_rescuers = gain_p25.iter().filter(|&&g| g > 0).count();
    let p50_rescuers = gain_p50.iter().filter(|&&g| g > 0).count();
    println!(
        "planner-rule check: candidates rescuing >=1 record — p25 threshold {:.3}: {p25_rescuers}/{n}, p50 (median) {:.3}: {p50_rescuers}/{n}",
        t25, threshold
    );
    let hits = hit.iter().filter(|x| **x).count();
    let base = hits as f64 / n as f64;
    println!("ground truth: {hits}/{n} proposals are hits (base rate {base:.3})\n");

    // ---------- DIAG: why is the rescue count structurally zero? ----------
    // For sampled records: their top-2 known entries, the parents' mutual
    // cosine, whether that midpoint exists in the proposal set, and the best
    // any proposal scores against the record. If a midpoint can never beat
    // the record's best entry, the planner rule is inert HERE by geometry,
    // not by signal — and the composition claim needs the reachability
    // analysis, not a ranking table over a constant.
    println!("=== DIAG: rescue reachability on sampled records ===");
    let proposal_of_pair: HashMap<(usize, usize), usize> = proposals
        .iter()
        .enumerate()
        .map(|(pi, (_, i, j))| ((*i, *j), pi))
        .collect();
    for &r in [0, rec_emb.len() / 4, rec_emb.len() / 2, rec_emb.len() - 1].iter() {
        let mut top: Vec<(f32, usize)> = known.iter().map(|&k| (cosine_sim(&rec_emb[r], &emb[k]), k)).collect();
        top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let (l1, e1) = top[0];
        let (l2, e2) = top[1];
        let key = (e1.min(e2), e1.max(e2));
        let mp = proposal_of_pair.get(&key).copied();
        let best_p = proposals.iter().map(|(p, _, _)| cosine_sim(p, &rec_emb[r])).fold(f32::NEG_INFINITY, f32::max);
        let pmid = mp.map(|_| cosine_sim(&proposals[proposal_of_pair[&key]].0, &rec_emb[r]));
        println!(
            "  record {r}: live {l1:.3} (e{}) / {:.3} (e{}), mutual {:.3}, midpoint-in-set {:?} -> its score {:?}; best proposal {:.3}",
            e1, l2, e2, cosine_sim(&emb[e1], &emb[e2]), mp.is_some(), pmid, best_p
        );
    }
    println!();

    // ---------- ranking arms ----------
    let order_by = |key: &[f64]| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| key[b].partial_cmp(&key[a]).unwrap().then(a.cmp(&b)));
        idx
    };
    let lift_key: Vec<f64> = lift.clone();
    let p25_key: Vec<f64> = gain_p25.iter().map(|&g| g as f64).collect();
    let geo_key: Vec<f64> = geo.iter().map(|&g| g as f64).collect();
    let pair_key: Vec<f64> = pair.iter().map(|&g| g as f64).collect();
    let traffic_key: Vec<f64> = traffic.iter().map(|&g| g as f64).collect();
    // Ascending nearest-non-parent cosine: high key = deep in an empty region.
    let isol_key: Vec<f64> = isolation.iter().map(|&g| -(g as f64)).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(20260907);
    let mut rand_idx: Vec<usize> = (0..n).collect();
    rand_idx.shuffle(&mut rng);

    let arms: [(&str, Vec<usize>, &str); 7] = [
        ("LIFT     ", order_by(&lift_key), "operational: total score lift over the record pile"),
        ("ISOLATION", order_by(&isol_key), "ontology-only emptiness — the construction-matched control"),
        ("TRAFFIC  ", order_by(&traffic_key), "max cosine to any record — popularity control"),
        ("P25-GAIN ", order_by(&p25_key), "planner rule at the p25 threshold"),
        ("GEOMETRY ", order_by(&geo_key), "max cosine to known — the proposer's own preference"),
        ("PAIR-SIM ", order_by(&pair_key), "parents' mutual cosine"),
        ("RANDOM   ", rand_idx, "seeded shuffle — the floor"),
    ];

    println!("=== precision@k: share of hits in the top k, per ranking arm ===\n");
    println!(
        "  arm        {:>7} {:>7} {:>7} {:>7} {:>7} {:>10}   (base {:.3})",
        "k=25", "k=50", "k=100", "k=200", "k=400", "p(k=25)", base
    );
    for (name, idx, note) in &arms {
        let at = |k: usize| -> f64 {
            let h = idx.iter().take(k).filter(|&&i| hit[i]).count();
            h as f64 / k as f64
        };
        // Exact hypergeometric tail: P(>= observed hits in a random 25 drawn
        // from n proposals containing `hits`). The null is "this ordering
        // carries no information", which is what RANDOM samples once.
        let top25 = idx.iter().take(25).filter(|&&i| hit[i]).count();
        let p25 = hypergeom_sf(top25, 25, hits, n);
        println!(
            "  {name}   {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>10.2e}   {note}",
            at(25), at(50), at(100), at(200), at(400), p25
        );
    }

    // AUC: each signal as a classifier of `hit`, on the same population.
    println!("\n=== AUC of each signal as a hit classifier (0.5 = chance) ===");
    for (name, key, _) in [
        ("LIFT", &lift_key, ""),
        ("TRAFFIC", &traffic_key, ""),
        ("P25-GAIN", &p25_key, ""),
        ("GEOMETRY", &geo_key, ""),
        ("PAIR-SIM", &pair_key, ""),
        ("ISOLATION", &isol_key, ""),
    ] {
        println!("  {name}: {:.3}", auc(key, &hit));
    }

    // The honest scope: a proposal can only lift records the operation
    // actually wrote. How much of the space does the corpus even visit?
    let worklist: Vec<usize> = (0..n).filter(|&pi| lift[pi] > 1e-9).collect();
    println!(
        "\n=== operational scope ===\n  proposals with any lift: {}/{} ({:.0}%)\n",
        worklist.len(),
        n,
        100.0 * worklist.len() as f64 / n as f64
    );

    // Conditional enrichment: among the worklist a human would actually read
    // (lift > 0), does the LIFT ordering still enrich?
    if worklist.len() > 30 {
        let hits_in = worklist.iter().filter(|&&i| hit[i]).count();
        let cond_base = hits_in as f64 / worklist.len() as f64;
        println!("  within that worklist: {hits_in} hits, conditional base rate {cond_base:.3}");
        let lift_order: Vec<usize> = {
                let mut t: Vec<usize> = worklist.clone();
                t.sort_by(|&a, &b| lift[b].partial_cmp(&lift[a]).unwrap().then(a.cmp(&b)));
                t
            };
        let k = 50.min(lift_order.len());
        let top = lift_order.iter().take(k).filter(|&&i| hit[i]).count();
        println!("  top {k} of the worklist by lift: {top} hits ({:.3} vs {cond_base:.3} conditional, {base:.3} overall)", top as f64 / k as f64);
    }

    // Where do the hits live? Corpus traffic per held-out concept.
    let held_traffic: Vec<f32> = held
        .iter()
        .map(|&h| rec_emb.iter().map(|r| cosine_sim(r, &emb[h])).fold(f32::NEG_INFINITY, f32::max))
        .collect();
    let mut ht = held_traffic.clone();
    ht.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "\n  corpus traffic per held-out concept (max record cosine): median {:.3}, p90 {:.3}",
        quantile(&ht, 0.5),
        quantile(&ht, 0.9)
    );
    println!(
        "\n(Read: LIFT > ISOLATION at equal k means the operational corpus adds ranking\n information the ontology's own geometry does not already carry — that is the\n composition working. LIFT <= ISOLATION means the ranking is available for free\n from the entries alone and the records are decoration. Equal to RANDOM means the\n disposer filters but does not rank, and the worklist is worth reading only as a\n set, not in order.)"
    );
}

#[cfg(feature = "embed-onnx")]
fn midpoint(a: &[f32], b: &[f32]) -> Vec<f32> {
    // Normalized bisector — Experiment 33's convention. The unnormalized
    // average (Iteration 31's) makes the planner's rescue condition
    // structurally impossible: cos(midpoint, r) <= max parent cosine, so no
    // candidate can ever beat a record's best known entry. Normalization
    // inflates the bisector just enough that a record sitting between the
    // parents can be rescued. The HIT metric is invariant to this choice —
    // both sides of its comparison divide by the same ||a+b|| — so the
    // ground truth stays comparable with Iterations 30/31 (base rate 0.132
    // reproduced above).
    normalize(&a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<f32>>())
}

#[cfg(feature = "embed-onnx")]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut d = 0.0f32;
    for i in 0..a.len() {
        d += a[i] * b[i];
    }
    d
}

#[cfg(feature = "embed-onnx")]
fn normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

#[cfg(feature = "embed-onnx")]
fn quantile(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// P(X >= k) for X ~ Hypergeometric(N population, K successes, n draws).
/// Log-factorials summed directly — N here is ~1600, so exactness costs nothing.
#[cfg(feature = "embed-onnx")]
fn hypergeom_sf(k: usize, n: usize, big_k: usize, big_n: usize) -> f64 {
    let mut lf = vec![0.0f64; big_n + 1];
    for i in 1..=big_n {
        lf[i] = lf[i - 1] + (i as f64).ln();
    }
    let lchoose = |a: usize, b: usize| -> f64 {
        if b > a {
            f64::NEG_INFINITY
        } else {
            lf[a] - lf[b] - lf[a - b]
        }
    };
    let denom = lchoose(big_n, n);
    let hi = n.min(big_k);
    let mut p = 0.0f64;
    for x in k..=hi {
        if n - x > big_n - big_k {
            continue;
        }
        p += (lchoose(big_k, x) + lchoose(big_n - big_k, n - x) - denom).exp();
    }
    p.clamp(0.0, 1.0)
}

#[cfg(feature = "embed-onnx")]
fn auc(scores: &[f64], labels: &[bool]) -> f64 {
    // Mann-Whitney: P(score_pos > score_neg) + 0.5 P(=). Each positive/negative
    // pair contributes once.
    let mut pos_neg = 0u64;
    let mut concordant = 0.0f64;
    for i in 0..scores.len() {
        if !labels[i] {
            continue;
        }
        for j in 0..scores.len() {
            if labels[j] {
                continue;
            }
            pos_neg += 1;
            match scores[i].partial_cmp(&scores[j]).unwrap() {
                std::cmp::Ordering::Greater => concordant += 1.0,
                std::cmp::Ordering::Equal => concordant += 0.5,
                _ => {}
            }
        }
    }
    if pos_neg == 0 {
        return 0.5;
    }
    concordant / pos_neg as f64
}
