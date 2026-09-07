//! Experiment 41 — Stage 0 + Stage 1 of the n-grams plan: an external gold
//! standard, and whether suffix backoff lifts signature recurrence above fixed-n.
//!
//! ## Why this exists
//!
//! Iterations 39-40 failed at the same step three times: producing a sense
//! partition. `becoming`'s runs test is sound but cannot induce one, and every
//! geometric attempt to induce one measured something else (genre, breadth of
//! use, scattered cell labels). Item 31 proposed the symbolic alternative — an
//! n-gram is an *exact* symbol, so "same context" is decided rather than
//! measured — and measured its blocker: at fixed n = 3, only 21% of `coverage`
//! occurrences share a trigram with any other occurrence. Signatures that never
//! recur cannot form families.
//!
//! infini-gram (arXiv 2401.17377) supplies the fix: do not fix n. Take the
//! LONGEST window that still has support, and back off when it runs out.
//!
//! ## Stage 0 — the evaluation, fixed before anything is built
//!
//! The prior corpus was one project's own documentation, eight terms chosen by
//! me, no real polysemy, and ground truth I authored. Every result on it was
//! unfalsifiable, which is precisely how Iterations 39-40 went wrong.
//!
//! This uses SemEval-2020 Task 1 English (Schlechtweg, Dubossarsky, Hengchen,
//! McGillivray, Tahmasebi): 37 lemmas annotated for lexical semantic change
//! between CCOHA1 (1810-1860) and CCOHA2 (1960-2010), with binary and graded
//! gold scores. Nobody in this project authored any of it, and the task's
//! published state of the art (66.5% accuracy, 51.8 Spearman) is the yardstick.
//!
//! ## Stage 1 — what is measured here
//!
//! A *signature* is the exact token window around one occurrence of a target,
//! written with the target's offset so that "the attack" (target last) and
//! "attack the" (target first) are different signatures.
//!
//! *Support* is counted within the term's OWN occurrence set, not the corpus.
//! That is the property the sense-family argument needs: a signature earns its
//! place by recurring across that term's uses, not by being a common English
//! phrase.
//!
//! *Recurrence* is the fraction of a term's occurrences that receive any
//! signature at all — i.e. that share some window of >= 1 context token with
//! at least k-1 other occurrences. Occurrences whose every window is unique
//! get no signature and are counted against the term.
//!
//! Backoff picks, per occurrence, the longest window with support >= k; ties on
//! length break on higher support, then lexicographically, so the result does
//! not depend on iteration order (standing constraint: determinism was a real
//! bug in this pipeline, fixed twice).
//!
//! **Gate**: median recurrence across the 37 terms >= 50%.
//! **Kill**: if backoff cannot beat the fixed-n baseline this file also
//! computes, the sparsity is structural and the whole n-gram line ends here.
//!
//! No embedder is involved at any point. That is the entire attraction: nothing
//! here can be undone by a model swap, which killed cross-embedder transfer at
//! ARI ~0.10.
//!
//! Run:
//!   cargo run -p physis-core --release --example experiment41_semeval_backoff
//!   SEMEVAL_DIR=/path/to/semeval2020_ulscd_eng cargo run --release --example experiment41_semeval_backoff

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// How many tokens of context the search may take on each side. Reported
/// against the observed length distribution so it is visible whether the cap
/// ever binds.
const MAX_SIDE: usize = 4;
/// Minimum number of occurrences sharing a window for it to count as support.
const MIN_SUPPORT: usize = 2;

/// One occurrence of a target: which period it came from, and its context.
struct Occ {
    period: u8,
    tokens: Vec<String>,
    pos: usize,
}

/// The window [pos-a, pos+b] rendered as a signature key.
///
/// The offset is part of the key: the same tokens with the target in a
/// different slot are a different context, and keying on the string alone
/// would merge them whenever the target word repeats in its own window.
fn window_key(o: &Occ, a: usize, b: usize) -> Option<(usize, String)> {
    let lo = o.pos.checked_sub(a)?;
    let hi = o.pos.checked_add(b)?;
    if hi >= o.tokens.len() {
        return None;
    }
    Some((a, o.tokens[lo..=hi].join(" ")))
}

/// Read a corpus file, keeping only sentences that contain a target.
fn load_corpus(path: &Path, period: u8, targets: &[String], out: &mut BTreeMap<String, Vec<Occ>>) {
    let text =
        fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    for line in text.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        for (i, tok) in tokens.iter().enumerate() {
            if targets.iter().any(|t| t == tok) {
                out.entry((*tok).to_string()).or_default().push(Occ {
                    period,
                    tokens: tokens.iter().map(|s| (*s).to_string()).collect(),
                    pos: i,
                });
            }
        }
    }
}

/// Per-occurrence signature chosen by longest-with-support backoff.
///
/// Returns one entry per occurrence: `Some((length, key))` or `None` when even
/// the best 2-gram is unique to this occurrence.
fn backoff_signatures(occs: &[Occ]) -> Vec<Option<(usize, (usize, String))>> {
    // Count every candidate window across this term's occurrences.
    let mut counts: BTreeMap<(usize, String), usize> = BTreeMap::new();
    for o in occs {
        for a in 0..=MAX_SIDE {
            for b in 0..=MAX_SIDE {
                if a + b == 0 {
                    continue; // length 1 is the bare target: no context, no signal
                }
                if let Some(k) = window_key(o, a, b) {
                    *counts.entry(k).or_insert(0) += 1;
                }
            }
        }
    }

    occs.iter()
        .map(|o| {
            let mut best: Option<(usize, usize, (usize, String))> = None; // (len, support, key)
            for a in 0..=MAX_SIDE {
                for b in 0..=MAX_SIDE {
                    if a + b == 0 {
                        continue;
                    }
                    let Some(key) = window_key(o, a, b) else {
                        continue;
                    };
                    let support = counts.get(&key).copied().unwrap_or(0);
                    if support < MIN_SUPPORT {
                        continue;
                    }
                    let len = a + b + 1;
                    // Longest wins; then most-supported; then lexicographic —
                    // a total order, so the output does not depend on the
                    // iteration order of anything.
                    let cand = (len, support, key);
                    best = match best {
                        None => Some(cand),
                        Some(cur) => {
                            let better = (cand.0, cand.1, &cand.2) > (cur.0, cur.1, &cur.2);
                            Some(if better { cand } else { cur })
                        }
                    };
                }
            }
            best.map(|(len, _, key)| (len, key))
        })
        .collect()
}

/// The construction-matched control: same occurrences, same window geometry,
/// same context-token frequencies — collocation structure destroyed.
///
/// Three positives in this track died to a control that differed from the
/// treatment in more than one way, so this one differs in exactly one: each
/// context token is resampled independently from the term's own pooled context
/// distribution. Sentence lengths and target positions are untouched.
///
/// If recurrence survives this, it is measuring token frequency and not
/// context structure, and no family can be built on it.
fn shuffled_control(occs: &[Occ], seed: u64) -> Vec<Occ> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // Pool every context token this term actually appears near, at any offset.
    let mut pool: Vec<&str> = Vec::new();
    for o in occs {
        for (i, t) in o.tokens.iter().enumerate() {
            if i != o.pos {
                pool.push(t.as_str());
            }
        }
    }
    if pool.is_empty() {
        return Vec::new();
    }
    let mut rng = StdRng::seed_from_u64(seed);
    occs.iter()
        .map(|o| Occ {
            period: o.period,
            pos: o.pos,
            tokens: o
                .tokens
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    if i == o.pos {
                        t.clone()
                    } else {
                        pool[rng.gen_range(0..pool.len())].to_string()
                    }
                })
                .collect(),
        })
        .collect()
}

/// The control the gate names: fixed n = 3, one token each side, no backoff.
fn fixed_trigram_recurrence(occs: &[Occ]) -> f64 {
    let mut counts: BTreeMap<(usize, String), usize> = BTreeMap::new();
    for o in occs {
        if let Some(k) = window_key(o, 1, 1) {
            *counts.entry(k).or_insert(0) += 1;
        }
    }
    let hit = occs
        .iter()
        .filter(|o| {
            window_key(o, 1, 1)
                .and_then(|k| counts.get(&k).copied())
                .is_some_and(|c| c >= MIN_SUPPORT)
        })
        .count();
    if occs.is_empty() {
        0.0
    } else {
        hit as f64 / occs.len() as f64
    }
}

fn median(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

fn main() {
    let dir: PathBuf = std::env::var("SEMEVAL_DIR")
        .unwrap_or_else(|_| {
            format!(
                "{}/datasets/semeval2020_ulscd_eng",
                std::env::var("HOME").unwrap_or_else(|_| ".".into())
            )
        })
        .into();

    println!("Experiment 41 — Stage 0 (external gold standard) + Stage 1 (backoff)\n");
    println!("Dataset: {}", dir.display());

    let targets: Vec<String> = fs::read_to_string(dir.join("targets.txt"))
        .expect("targets.txt — set SEMEVAL_DIR to the unpacked semeval2020_ulscd_eng")
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    let binary: BTreeMap<String, u8> = fs::read_to_string(dir.join("truth/binary.txt"))
        .expect("truth/binary.txt")
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?.to_string(), it.next()?.parse().ok()?))
        })
        .collect();

    println!("\n=== Stage 0 — the evaluation ===\n");
    println!("  targets with gold labels : {}", targets.len());
    println!(
        "  changed / stable         : {} / {}",
        binary.values().filter(|v| **v == 1).count(),
        binary.values().filter(|v| **v == 0).count()
    );
    println!("  authored by              : SemEval-2020 Task 1 organisers, not this project");
    if targets.len() < 30 {
        println!("\n  GATE FAILED: fewer than 30 terms with external ground truth. Stop.");
        return;
    }
    println!("  GATE PASSED (>= 30 terms with ground truth nobody here authored)");

    let mut occs: BTreeMap<String, Vec<Occ>> = BTreeMap::new();
    load_corpus(
        &dir.join("corpus1/lemma/ccoha1.txt"),
        1,
        &targets,
        &mut occs,
    );
    load_corpus(
        &dir.join("corpus2/lemma/ccoha2.txt"),
        2,
        &targets,
        &mut occs,
    );

    println!("\n=== Stage 1 — signature recurrence, backoff vs fixed n = 3 ===\n");
    println!(
        "  {:<18} {:>6} {:>6} {:>8} {:>9} {:>9} {:>7} {:>7}",
        "term", "occ", "c1/c2", "fixed-3", "backoff", "shuffled", "medLen", "distinct"
    );
    println!("  {}", "-".repeat(80));

    let mut backoff_rates = Vec::new();
    let mut fixed_rates = Vec::new();
    let mut shuffled_rates = Vec::new();
    let mut ctrl_lens: Vec<f64> = Vec::new();
    let mut real_lens: Vec<f64> = Vec::new();
    let mut len_hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut ctrl_len_hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut rows = Vec::new();

    for t in &targets {
        let Some(os) = occs.get(t) else {
            println!("  {t:<18}    (no occurrences found)");
            continue;
        };
        let sigs = backoff_signatures(os);
        let hit = sigs.iter().filter(|s| s.is_some()).count();
        let rate = hit as f64 / os.len() as f64;
        let fixed = fixed_trigram_recurrence(os);
        let mut lens: Vec<f64> = sigs
            .iter()
            .filter_map(|s| s.as_ref().map(|(l, _)| *l as f64))
            .collect();
        for l in &lens {
            *len_hist.entry(*l as usize).or_insert(0) += 1;
        }
        let med_len = median(std::mem::take(&mut lens));
        real_lens.push(med_len);
        let distinct: std::collections::BTreeSet<_> = sigs
            .iter()
            .filter_map(|s| s.as_ref().map(|(_, k)| k.clone()))
            .collect();
        let c1 = os.iter().filter(|o| o.period == 1).count();

        // Same term, same geometry, same token frequencies, no collocations.
        // Seeded per term so the whole run is byte-reproducible.
        let ctrl = shuffled_control(
            os,
            t.bytes()
                .map(u64::from)
                .sum::<u64>()
                .wrapping_mul(2654435761),
        );
        let ctrl_sigs = backoff_signatures(&ctrl);
        let ctrl_rate = if ctrl.is_empty() {
            0.0
        } else {
            ctrl_sigs.iter().filter(|s| s.is_some()).count() as f64 / ctrl.len() as f64
        };
        // Length is the sharper discriminator than rate: independent resampling
        // can still hit a shared 2-gram by chance, but a shared 5-gram is a
        // collocation. Track how far the control's signatures actually extend.
        let ctrl_med_len = median(
            ctrl_sigs
                .iter()
                .filter_map(|s| s.as_ref().map(|(l, _)| *l as f64))
                .collect(),
        );
        ctrl_lens.push(ctrl_med_len);
        for l in ctrl_sigs.iter().flatten() {
            *ctrl_len_hist.entry(l.0).or_insert(0) += 1;
        }

        println!(
            "  {:<18} {:>6} {:>6} {:>7.1}% {:>8.1}% {:>8.1}% {:>7.1} {:>7}",
            t,
            os.len(),
            format!("{}/{}", c1, os.len() - c1),
            fixed * 100.0,
            rate * 100.0,
            ctrl_rate * 100.0,
            med_len,
            distinct.len()
        );

        backoff_rates.push(rate);
        fixed_rates.push(fixed);
        shuffled_rates.push(ctrl_rate);
        rows.push((t.clone(), os.len(), fixed, rate, ctrl_rate, distinct.len()));
    }

    let med_backoff = median(backoff_rates.clone());
    let med_fixed = median(fixed_rates.clone());
    let med_shuffled = median(shuffled_rates.clone());

    println!("\n=== Verdict ===\n");
    println!(
        "  median recurrence, fixed n = 3 : {:.1}%",
        med_fixed * 100.0
    );
    println!(
        "  median recurrence, backoff     : {:.1}%",
        med_backoff * 100.0
    );
    println!(
        "  median recurrence, SHUFFLED    : {:.1}%  (construction-matched control)",
        med_shuffled * 100.0
    );
    let med_real_len = median(real_lens.clone());
    let med_ctrl_len = median(ctrl_lens.clone());
    println!(
        "  median signature length        : {med_real_len:.1} real vs {med_ctrl_len:.1} shuffled"
    );
    println!("  signature length distribution  :");
    let total_sigs: usize = len_hist.values().sum();
    for (len, n) in &len_hist {
        println!(
            "      n = {len:<2} {:>7} ({:>4.1}%){}",
            n,
            100.0 * *n as f64 / total_sigs.max(1) as f64,
            if *len == 2 * MAX_SIDE + 1 {
                "   <- cap; if this is large, raise MAX_SIDE"
            } else {
                ""
            }
        );
    }

    let ctrl_total: usize = ctrl_len_hist.values().sum();
    let long_real: usize = len_hist
        .iter()
        .filter(|(l, _)| **l >= 4)
        .map(|(_, n)| *n)
        .sum();
    let long_ctrl: usize = ctrl_len_hist
        .iter()
        .filter(|(l, _)| **l >= 4)
        .map(|(_, n)| *n)
        .sum();
    let long_real_pct = 100.0 * long_real as f64 / total_sigs.max(1) as f64;
    let long_ctrl_pct = 100.0 * long_ctrl as f64 / ctrl_total.max(1) as f64;
    println!(
        "  signatures of length >= 4      : {long_real_pct:.1}% real vs {long_ctrl_pct:.1}% shuffled"
    );

    println!();
    println!("  What the control says, stated against the headline rather than under it:");
    println!("  recurrence RATE is largely a frequency artifact — independent resampling of");
    println!(
        "  the same tokens still reaches {:.1}%, so {:.0}% of the {:.1}% is explained without",
        med_shuffled * 100.0,
        100.0 * med_shuffled / med_backoff,
        med_backoff * 100.0
    );
    println!("  any collocation structure at all. What does NOT survive the control is");
    println!("  signature LENGTH: {med_real_len:.1} vs {med_ctrl_len:.1} median, {long_real_pct:.1}% vs {long_ctrl_pct:.1}% at length >= 4.");
    println!("  Stage 2 must therefore build families from the long signatures, not from");
    println!("  the fact that a signature exists.");

    println!();
    if med_backoff >= 0.50 {
        println!(
            "  GATE PASSED: median recurrence {:.1}% >= 50%.",
            med_backoff * 100.0
        );
    } else {
        println!(
            "  GATE FAILED: median recurrence {:.1}% < 50%.",
            med_backoff * 100.0
        );
    }
    if med_backoff <= med_fixed {
        println!("  KILL CRITERION MET: backoff does not beat fixed-n. The sparsity is");
        println!("  structural and the n-gram line ends here.");
    } else {
        println!(
            "  Backoff beats fixed-n by {:.1} points, so the sparsity was the fixed n,",
            (med_backoff - med_fixed) * 100.0
        );
        println!("  not the method.");
    }

    // Machine-readable, for the next stage to consume rather than recompute.
    let json = serde_json::json!({
        "dataset": "semeval2020_ulscd_eng",
        "terms": targets.len(),
        "min_support": MIN_SUPPORT,
        "max_side": MAX_SIDE,
        "median_recurrence_fixed3": med_fixed,
        "median_recurrence_backoff": med_backoff,
        "median_recurrence_shuffled": med_shuffled,
        "median_signature_len_real": med_real_len,
        "median_signature_len_shuffled": med_ctrl_len,
        "per_term": rows.iter().map(|(t, n, f, b, s, d)| serde_json::json!({
            "term": t, "occurrences": n, "fixed3": f, "backoff": b, "shuffled": s, "distinct_signatures": d
        })).collect::<Vec<_>>(),
    });
    let out = "experiment41_results.json";
    fs::write(out, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    println!("\n  wrote {out}");
}
