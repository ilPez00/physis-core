//! Experiment 42 — Stage 2 (signature families) and Stage 3 (`becoming` over
//! family labels) of the refined n-gram plan, on the SemEval-2020 gold standard.
//!
//! ## Stage 2 — families by SUBSTITUTABILITY, not similarity
//!
//! Stage 1 established that exact signatures recur, and that only the LONG ones
//! do so for a reason (41.4% of real signatures reach length >= 4, against 2.6%
//! when the same tokens are resampled independently). But exact n-grams are
//! still a vocabulary, not a sense inventory: "the attack of" and "an attack of"
//! are two symbols for one context.
//!
//! The synonym layer is the classic distributional test, done symbolically
//! rather than geometrically: two signatures are substitutable when they occupy
//! **the same slot** — identical length, identical target offset, differing in
//! exactly one context position. Families are the connected components of that
//! relation. No cosine, no embedder, no threshold.
//!
//! The obvious failure mode is transitive collapse: substitutability is
//! chained, so a single promiscuous frame ("the _ of") can merge everything
//! into one component. That is not a bug to be tuned away — it is exactly
//! Stage 2's kill criterion, so it is measured and reported rather than
//! prevented.
//!
//! **Gate**: families must form — >= 2 families each covering >= 10% of a
//! term's signed occurrences, for a majority of the 37 terms, with the largest
//! family under 90%.
//! **Kill**: if families do not form (or one swallows everything), n-grams are
//! a vocabulary and not a sense inventory. Report and stop.
//!
//! ## Stage 3 — `becoming` over family labels
//!
//! `classify_labeled` takes discrete labels and asks whether the two leading
//! ones are time-separated (Becoming) or interleaved (Split). Stage 2 supplies
//! the labels. The corpus supplies the order — and only coarsely, which is
//! worth stating plainly: CCOHA sentences are randomly shuffled WITHIN each
//! period, so the only real order is the C1 (1810-1860) -> C2 (1960-2010)
//! block. That is enough for the runs test to ask the one question the gold
//! labels answer, and nothing finer should be read into `change_at`.
//!
//! Mapping to the gold binary task: Becoming -> changed (a sense displaced
//! another across the periods); Stable and Split -> unchanged (one dominant
//! sense, or two that coexist throughout).
//!
//! **Gate (restated for the external set)**: the original plan said ">= 5/8 on
//! the known terms", written when the terms were eight I had chosen myself.
//! The honest translation to 37 terms nobody here authored is: beat the
//! majority-class baseline (21/37 = 56.8%), with SemEval's best published
//! accuracy of 66.5% as the yardstick.
//! **Kill**: at or below the majority baseline, the partition is still not the
//! problem and the trajectory question should be abandoned rather than tuned.
//!
//! Run:
//!   cargo run -p physis-core --release --example experiment42_signature_families

use physis_core::becoming::{classify_labeled, Trajectory};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const MAX_SIDE: usize = 4;
const MIN_SUPPORT: usize = 2;
/// A family must reach this share of a term's signed occurrences to count as
/// one of the "real" families the gate asks for.
const FAMILY_SHARE: f64 = 0.10;

struct Occ {
    period: u8,
    tokens: Vec<String>,
    pos: usize,
}

/// A signature: target offset within the window, and the window's tokens.
type Sig = (usize, Vec<String>);

fn window(o: &Occ, a: usize, b: usize) -> Option<Sig> {
    let lo = o.pos.checked_sub(a)?;
    let hi = o.pos.checked_add(b)?;
    if hi >= o.tokens.len() {
        return None;
    }
    Some((a, o.tokens[lo..=hi].to_vec()))
}

fn load_corpus(path: &Path, period: u8, targets: &[String], out: &mut BTreeMap<String, Vec<Occ>>) {
    let text =
        fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    for line in text.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
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

/// Longest-with-support backoff, as established in Stage 1.
fn backoff_signatures(occs: &[Occ], min_len: usize) -> Vec<Option<Sig>> {
    let mut counts: BTreeMap<Sig, usize> = BTreeMap::new();
    for o in occs {
        for a in 0..=MAX_SIDE {
            for b in 0..=MAX_SIDE {
                if a + b == 0 {
                    continue;
                }
                if let Some(k) = window(o, a, b) {
                    *counts.entry(k).or_insert(0) += 1;
                }
            }
        }
    }
    occs.iter()
        .map(|o| {
            let mut best: Option<(usize, usize, Sig)> = None;
            for a in 0..=MAX_SIDE {
                for b in 0..=MAX_SIDE {
                    if a + b == 0 {
                        continue;
                    }
                    let Some(key) = window(o, a, b) else { continue };
                    let support = counts.get(&key).copied().unwrap_or(0);
                    if support < MIN_SUPPORT {
                        continue;
                    }
                    if a + b + 1 < min_len {
                        continue;
                    }
                    let cand = (a + b + 1, support, key);
                    best = match best {
                        None => Some(cand),
                        Some(cur) => {
                            let better = (cand.0, cand.1, &cand.2) > (cur.0, cur.1, &cur.2);
                            Some(if better { cand } else { cur })
                        }
                    };
                }
            }
            best.map(|(_, _, k)| k)
        })
        .collect()
}

/// Disjoint-set over signature indices.
struct Dsu(Vec<usize>);
impl Dsu {
    fn new(n: usize) -> Self {
        Dsu((0..n).collect())
    }
    fn find(&mut self, x: usize) -> usize {
        if self.0[x] != x {
            let r = self.find(self.0[x]);
            self.0[x] = r;
        }
        self.0[x]
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // Always attach the larger root to the smaller: deterministic,
            // independent of the order unions arrive in.
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.0[hi] = lo;
        }
    }
}

/// Group signatures into families by substitutability.
///
/// Two signatures share a frame when they have the same length, the same target
/// offset, and differ in exactly one context position. Building a
/// frame -> signatures index and unioning within each frame gives the connected
/// components in one pass.
fn families(sigs: &BTreeSet<Sig>) -> BTreeMap<Sig, usize> {
    let list: Vec<&Sig> = sigs.iter().collect();
    let mut dsu = Dsu::new(list.len());
    // frame = (offset, wildcard position, tokens with that position blanked)
    let mut frames: BTreeMap<(usize, usize, Vec<String>), Vec<usize>> = BTreeMap::new();
    for (idx, (a, toks)) in list.iter().enumerate() {
        for i in 0..toks.len() {
            if i == *a {
                continue; // the target slot itself is not a variable
            }
            let mut blanked = (*toks).clone();
            blanked[i] = "\u{0}".into();
            frames.entry((*a, i, blanked)).or_default().push(idx);
        }
    }
    for members in frames.values() {
        for w in members.windows(2) {
            dsu.union(w[0], w[1]);
        }
    }
    // Relabel roots to dense ids in a deterministic order.
    let mut root_of: BTreeMap<usize, usize> = BTreeMap::new();
    let mut out = BTreeMap::new();
    for (idx, sig) in list.iter().enumerate() {
        let r = dsu.find(idx);
        let next = root_of.len();
        let fam = *root_of.entry(r).or_insert(next);
        out.insert((*sig).clone(), fam);
    }
    out
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

/// Spearman rank correlation, average ranks on ties.
fn spearman(xs: &[f64], ys: &[f64]) -> f64 {
    fn ranks(v: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|a, b| v[*a].partial_cmp(&v[*b]).unwrap());
        let mut r = vec![0.0; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let avg = (i + j) as f64 / 2.0 + 1.0;
            for k in i..=j {
                r[idx[k]] = avg;
            }
            i = j + 1;
        }
        r
    }
    let (rx, ry) = (ranks(xs), ranks(ys));
    let n = xs.len() as f64;
    let mx = rx.iter().sum::<f64>() / n;
    let my = ry.iter().sum::<f64>() / n;
    let cov: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sx: f64 = rx.iter().map(|a| (a - mx).powi(2)).sum::<f64>().sqrt();
    let sy: f64 = ry.iter().map(|b| (b - my).powi(2)).sum::<f64>().sqrt();
    if sx == 0.0 || sy == 0.0 {
        0.0
    } else {
        cov / (sx * sy)
    }
}

/// Jensen-Shannon divergence between the family distributions of the two
/// periods — the graded change score, and the natural SemEval subtask-2 read.
fn jsd(c1: &BTreeMap<usize, f64>, c2: &BTreeMap<usize, f64>) -> f64 {
    let keys: BTreeSet<usize> = c1.keys().chain(c2.keys()).copied().collect();
    let n1: f64 = c1.values().sum();
    let n2: f64 = c2.values().sum();
    if n1 == 0.0 || n2 == 0.0 {
        return 0.0;
    }
    let mut d = 0.0;
    for k in keys {
        let p = c1.get(&k).copied().unwrap_or(0.0) / n1;
        let q = c2.get(&k).copied().unwrap_or(0.0) / n2;
        let m = (p + q) / 2.0;
        if p > 0.0 {
            d += 0.5 * p * (p / m).log2();
        }
        if q > 0.0 {
            d += 0.5 * q * (q / m).log2();
        }
    }
    d.max(0.0)
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

    println!("Experiment 42 — Stage 2 (families) + Stage 3 (becoming over families)\n");

    let targets: Vec<String> = fs::read_to_string(dir.join("targets.txt"))
        .expect("targets.txt — set SEMEVAL_DIR")
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
    let graded: BTreeMap<String, f64> = fs::read_to_string(dir.join("truth/graded.txt"))
        .expect("truth/graded.txt")
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?.to_string(), it.next()?.parse().ok()?))
        })
        .collect();

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

    for min_len in [2usize, 4usize] {
        println!("\n########## ARM: signatures of length >= {min_len} ##########\n");
        println!("=== Stage 2 — do families form? ===\n");
        println!(
            "  {:<18} {:>7} {:>7} {:>8} {:>8} {:>8} {:>7}",
            "term", "signed", "sigs", "families", "largest", ">=10%", "gold"
        );
        println!("  {}", "-".repeat(72));

        let mut largest_shares = Vec::new();
        let mut real_family_counts = Vec::new();
        let mut stage3: Vec<(String, Trajectory, f64, u8, f64)> = Vec::new();
        let mut terms_forming = 0usize;
        let mut top2_cover: Vec<f64> = Vec::new();
        let mut label_cards: Vec<f64> = Vec::new();

        for t in &targets {
            let Some(os) = occs.get(t) else { continue };
            let sigs = backoff_signatures(os, min_len);
            let distinct: BTreeSet<Sig> = sigs.iter().flatten().cloned().collect();
            let fam_of = families(&distinct);

            // Per-occurrence family label, in corpus order (C1 block then C2 block).
            let mut labels: Vec<String> = Vec::new();
            let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
            let mut c1: BTreeMap<usize, f64> = BTreeMap::new();
            let mut c2: BTreeMap<usize, f64> = BTreeMap::new();
            for (o, s) in os.iter().zip(&sigs) {
                let Some(sig) = s else { continue };
                let f = fam_of[sig];
                *counts.entry(f).or_insert(0) += 1;
                if o.period == 1 {
                    *c1.entry(f).or_insert(0.0) += 1.0;
                } else {
                    *c2.entry(f).or_insert(0.0) += 1.0;
                }
                labels.push(format!("f{f}"));
            }
            let signed = labels.len();
            if signed == 0 {
                continue;
            }
            let largest = *counts.values().max().unwrap_or(&0) as f64 / signed as f64;
            let real = counts
                .values()
                .filter(|c| **c as f64 / signed as f64 >= FAMILY_SHARE)
                .count();
            let gold = binary.get(t).copied().unwrap_or(0);

            println!(
                "  {:<18} {:>7} {:>7} {:>8} {:>7.1}% {:>8} {:>7}",
                t,
                signed,
                distinct.len(),
                counts.len(),
                largest * 100.0,
                real,
                if gold == 1 { "CHANGED" } else { "stable" }
            );

            largest_shares.push(largest);
            real_family_counts.push(real as f64);
            if real >= 2 && largest < 0.90 {
                terms_forming += 1;
            }

            // Stage 3 inputs. Record how much of the term the two leading labels
            // actually cover: `classify_labeled` reads only those two, so this is
            // the share of the evidence the statistic is allowed to see.
            let mut by_size: Vec<usize> = counts.values().copied().collect();
            by_size.sort_unstable_by(|a, b| b.cmp(a));
            let top2 = by_size.iter().take(2).sum::<usize>() as f64 / signed as f64;
            top2_cover.push(top2);
            label_cards.push(counts.len() as f64);
            let refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            let v = classify_labeled(&refs);
            stage3.push((
                t.clone(),
                v.trajectory,
                jsd(&c1, &c2),
                gold,
                graded.get(t).copied().unwrap_or(0.0),
            ));
        }

        let med_largest = median(largest_shares.clone());
        let med_real = median(real_family_counts.clone());
        println!(
            "\n  median largest-family share : {:.1}%",
            med_largest * 100.0
        );
        println!("  median families >= 10%      : {med_real:.1}");
        println!(
            "  terms with >= 2 real families and no >90% giant : {terms_forming}/{}",
            stage3.len()
        );

        let stage2_pass = terms_forming * 2 > stage3.len() && med_largest < 0.90;
        if stage2_pass {
            println!("\n  STAGE 2 GATE PASSED: families form for a majority of terms.");
        } else {
            println!("\n  STAGE 2 GATE FAILED / KILL CRITERION MET.");
            if med_largest >= 0.90 {
                println!("  Transitive collapse: one family swallows the term. Substitutability");
                println!("  chains through promiscuous frames, so n-grams are a vocabulary and");
                println!("  not a sense inventory. This is the documented stopping point.");
            } else {
                println!("  Families are too fragmented to act as a sense inventory.");
            }
        }

        println!("\n=== Stage 3 — becoming over family labels, against gold ===\n");
        println!(
            "  {:<18} {:>12} {:>10} {:>9} {:>8}",
            "term", "trajectory", "predicted", "gold", "JSD"
        );
        println!("  {}", "-".repeat(62));

        let mut correct = 0usize;
        let (mut preds, mut golds) = (Vec::new(), Vec::new());
        for (t, traj, d, gold, _) in &stage3 {
            // Becoming = one sense displaced another across the periods = changed.
            let pred = u8::from(*traj == Trajectory::Becoming);
            if pred == *gold {
                correct += 1;
            }
            preds.push(*d);
            golds.push(*gold as f64);
            println!(
                "  {:<18} {:>12} {:>10} {:>9} {:>8.4}",
                t,
                format!("{traj:?}"),
                if pred == 1 { "CHANGED" } else { "stable" },
                if *gold == 1 { "CHANGED" } else { "stable" },
                d
            );
        }

        let n = stage3.len();
        let acc = correct as f64 / n as f64;
        let majority = golds
            .iter()
            .filter(|g| **g == 0.0)
            .count()
            .max(golds.iter().filter(|g| **g == 1.0).count()) as f64
            / n as f64;
        let graded_vec: Vec<f64> = stage3.iter().map(|(_, _, _, _, g)| *g).collect();
        let rho_graded = spearman(&preds, &graded_vec);

        let mut traj_hist: BTreeMap<String, usize> = BTreeMap::new();
        for (_, tr, _, _, _) in &stage3 {
            *traj_hist.entry(format!("{tr:?}")).or_insert(0) += 1;
        }
        println!("\n=== Verdict ===\n");
        println!("  trajectory distribution              : {traj_hist:?}");
        println!(
            "  median families per term             : {:.0}",
            median(label_cards.clone())
        );
        println!(
            "  median share seen by the top 2 labels: {:.1}%",
            median(top2_cover.clone()) * 100.0
        );
        println!(
            "  binary accuracy (Becoming = changed) : {correct}/{n} = {:.1}%",
            acc * 100.0
        );
        println!(
            "  majority-class baseline              : {:.1}%",
            majority * 100.0
        );
        println!("  SemEval best published accuracy      : 66.5%");
        println!("  Spearman(JSD, gold graded)           : {rho_graded:.3}");
        println!("  SemEval best published Spearman      : 0.518");
        println!();
        if acc > majority {
            println!(
                "  STAGE 3 GATE PASSED: beats the majority baseline by {:.1} points.",
                (acc - majority) * 100.0
            );
        } else {
            println!(
                "  STAGE 3 KILL CRITERION MET: {:.1}% is at or below the {:.1}% majority",
                acc * 100.0,
                majority * 100.0
            );
            println!("  baseline. The partition is still not the problem. Per the plan, the");
            println!("  trajectory question is abandoned rather than tuned.");
        }

        let json = serde_json::json!({
            "dataset": "semeval2020_ulscd_eng",
            "min_signature_len": min_len,
            "stage2": {
                "median_largest_family_share": med_largest,
                "median_real_families": med_real,
                "terms_forming": terms_forming,
                "passed": stage2_pass,
            },
            "stage3": {
                "accuracy": acc,
                "correct": correct,
                "n": n,
                "majority_baseline": majority,
                "spearman_jsd_vs_graded": rho_graded,
            },
            "per_term": stage3.iter().map(|(t, tr, d, g, gr)| serde_json::json!({
                "term": t, "trajectory": format!("{tr:?}"), "jsd": d, "gold_binary": g, "gold_graded": gr
            })).collect::<Vec<_>>(),
        });
        fs::write(
            format!("experiment42_results_minlen{min_len}.json"),
            serde_json::to_string_pretty(&json).unwrap(),
        )
        .unwrap();
        println!("\n  wrote experiment42_results_minlen{min_len}.json");
    }
}
