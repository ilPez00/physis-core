//! Experiment 43 — Stage 5: does substitutability recover hand-authored
//! oppositions it was never given?
//!
//! ## Why this survives Stage 3's kill
//!
//! Stage 3 died on label cardinality: substitutability produces ~57 families
//! per term where `classify_labeled` wants labels on the order of the sense
//! count. That killed the *trajectory* question. It said nothing about whether
//! the substitutability relation itself carries semantic structure — Stage 2
//! showed the families form and do not collapse, which is a separate result.
//!
//! Stage 5 tests the relation directly, against a target that is genuinely
//! held out: `physis_pro::models::SemioticGrid::dual()` hand-authors seven mode
//! oppositions on the Greimas square —
//!
//!   lift <-> rest, walk <-> work, create <-> destroy, learn <-> guide,
//!   sense <-> brainstorm, play <-> plan, maintain <-> move
//!
//! Those pairs were written by a person, are stored as a `match` arm, and are
//! independent of any corpus statistic. If a purely symbolic distributional
//! relation ranks a word's true dual above eleven distractors, it has recovered
//! structure it was not given. If it cannot recover KNOWN oppositions, it will
//! not find unknown ones, and the relation-typing line ends with the
//! trajectory line.
//!
//! ## The corpus, and why it is not the project's own
//!
//! CCOHA (SemEval-2020 Task 1, both periods pooled) is ~12M tokens of ordinary
//! English containing every one of these words as ordinary vocabulary. Stage 0
//! rejected the project's own documentation as unfalsifiable; this test does
//! not need it, so it does not use it. The grid's authorship and the corpus are
//! fully independent of each other.
//!
//! ## Method — substitutability, symbolically
//!
//! A *frame* is a context window with the target slot blanked: "want to _ the
//! house". Two words are substitutable to the extent that they fill the same
//! frames. Similarity is Jaccard over exact frame sets — a set operation, not a
//! cosine, so nothing here moves when an embedding model does.
//!
//! Frames carry >= 2 context tokens, per Stage 1's finding that short windows
//! are reachable by chance (41.4% of real signatures reach length >= 4 against
//! 2.6% under independent resampling).
//!
//! **Gate**: the true dual ranked first substantially above the 1-in-12 chance
//! rate.
//! **Kill**: at chance, relation typing by substitutability is dead and Stage 5
//! closes with Stage 3.
//!
//! Run:
//!   cargo run -p physis-core --release --example experiment43_substitutable_opposites

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Context tokens taken on each side when building frames.
const SIDE: usize = 2;
/// A frame needs at least this many real context tokens to count.
const MIN_CONTEXT: usize = 2;
/// Below this many occurrences a word cannot be scored — reported as excluded
/// rather than counted as a failure.
const MIN_OCCURRENCES: usize = 100;

/// `dual()`'s mode oppositions, transcribed from physis-pro's
/// `SemioticGrid::dual` (src/models.rs). Hand-authored, held out.
const DUALS: &[(&str, &str)] = &[
    ("lift", "rest"),
    ("walk", "work"),
    ("create", "destroy"),
    ("learn", "guide"),
    ("sense", "brainstorm"),
    ("play", "plan"),
    ("maintain", "move"),
];

/// A frame: target offset within the window, and the window's tokens with the
/// target slot blanked.
type Frame = (usize, Vec<String>);

fn frames_at(tokens: &[String], pos: usize) -> Vec<Frame> {
    let mut out = Vec::new();
    for a in 0..=SIDE {
        for b in 0..=SIDE {
            if a + b < MIN_CONTEXT {
                continue;
            }
            let Some(lo) = pos.checked_sub(a) else {
                continue;
            };
            let hi = pos + b;
            if hi >= tokens.len() {
                continue;
            }
            let mut w: Vec<String> = tokens[lo..=hi].to_vec();
            w[a] = "\u{0}".into(); // blank the target slot
            out.push((a, w));
        }
    }
    out
}

fn jaccard(a: &BTreeSet<Frame>, b: &BTreeSet<Frame>) -> f64 {
    let inter = a.intersection(b).count();
    let union = a.len() + b.len() - inter;
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
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

    println!("Experiment 43 — Stage 5: substitutability vs hand-authored oppositions\n");

    let words: Vec<String> = DUALS
        .iter()
        .flat_map(|(a, b)| [a.to_string(), b.to_string()])
        .collect();
    let wordset: BTreeSet<&str> = words.iter().map(|s| s.as_str()).collect();

    let mut frames: BTreeMap<String, BTreeSet<Frame>> = BTreeMap::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();

    for rel in ["corpus1/lemma/ccoha1.txt", "corpus2/lemma/ccoha2.txt"] {
        let path: &Path = &dir.join(rel);
        let text = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e} — set SEMEVAL_DIR", path.display()));
        for line in text.lines() {
            let toks: Vec<String> = line.split_whitespace().map(str::to_string).collect();
            for (i, t) in toks.iter().enumerate() {
                if wordset.contains(t.as_str()) {
                    *counts.entry(t.clone()).or_insert(0) += 1;
                    frames
                        .entry(t.clone())
                        .or_default()
                        .extend(frames_at(&toks, i));
                }
            }
        }
    }

    println!("=== Corpus support ===\n");
    let mut usable: Vec<String> = Vec::new();
    for w in &words {
        let n = counts.get(w).copied().unwrap_or(0);
        let f = frames.get(w).map(|s| s.len()).unwrap_or(0);
        let ok = n >= MIN_OCCURRENCES;
        println!(
            "  {:<12} {:>7} occurrences {:>8} distinct frames{}",
            w,
            n,
            f,
            if ok {
                ""
            } else {
                "   EXCLUDED (too rare to score)"
            }
        );
        if ok {
            usable.push(w.clone());
        }
    }

    // A pair is scorable only when both members cleared the support floor.
    let scorable: Vec<(&str, &str)> = DUALS
        .iter()
        .filter(|(a, b)| usable.iter().any(|u| u == a) && usable.iter().any(|u| u == b))
        .copied()
        .collect();
    let excluded: Vec<(&str, &str)> = DUALS
        .iter()
        .filter(|p| !scorable.contains(p))
        .copied()
        .collect();

    println!("\n  scorable pairs : {}/{}", scorable.len(), DUALS.len());
    for (a, b) in &excluded {
        println!("  excluded       : {a} <-> {b}");
    }

    println!("\n=== Stage 5 — is the true dual the nearest word? ===\n");
    println!(
        "  {:<12} {:>10} {:>7} {:>9}   top-3 by substitutability",
        "word", "dual", "rank", "jaccard"
    );
    println!("  {}", "-".repeat(78));

    let mut rank1 = 0usize;
    let mut top3 = 0usize;
    let mut scored = 0usize;
    let mut rows = Vec::new();

    for w in &usable {
        let Some((_, dual)) = scorable.iter().find_map(|(a, b)| {
            if a == w {
                Some((*a, *b))
            } else if b == w {
                Some((*b, *a))
            } else {
                None
            }
        }) else {
            continue;
        };

        let fw = &frames[w];
        // Rank every other usable word by shared-frame Jaccard. Ties break
        // lexicographically so the ranking is a total order.
        let mut sims: Vec<(f64, &str)> = usable
            .iter()
            .filter(|o| *o != w)
            .map(|o| (jaccard(fw, &frames[o]), o.as_str()))
            .collect();
        sims.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap().then_with(|| x.1.cmp(y.1)));

        let rank = sims
            .iter()
            .position(|(_, o)| *o == dual)
            .map(|i| i + 1)
            .unwrap_or(0);
        let jac = sims
            .iter()
            .find(|(_, o)| *o == dual)
            .map(|(s, _)| *s)
            .unwrap_or(0.0);
        let top: Vec<String> = sims
            .iter()
            .take(3)
            .map(|(s, o)| format!("{o} {s:.4}"))
            .collect();

        println!(
            "  {:<12} {:>10} {:>7} {:>9.4}   {}",
            w,
            dual,
            rank,
            jac,
            top.join(", ")
        );

        scored += 1;
        if rank == 1 {
            rank1 += 1;
        }
        if rank <= 3 {
            top3 += 1;
        }
        rows.push((w.clone(), dual.to_string(), rank, jac));
    }

    let n_alt = usable.len().saturating_sub(1) as f64;
    let chance_r1 = if n_alt > 0.0 { 1.0 / n_alt } else { 0.0 };
    let chance_t3 = if n_alt > 0.0 {
        (3.0 / n_alt).min(1.0)
    } else {
        0.0
    };

    println!("\n=== Verdict ===\n");
    println!("  words scored              : {scored}");
    println!(
        "  dual ranked 1st           : {rank1}/{scored} = {:.1}%   (chance {:.1}%)",
        100.0 * rank1 as f64 / scored.max(1) as f64,
        100.0 * chance_r1
    );
    println!(
        "  dual in top 3             : {top3}/{scored} = {:.1}%   (chance {:.1}%)",
        100.0 * top3 as f64 / scored.max(1) as f64,
        100.0 * chance_t3
    );
    let expected_r1 = chance_r1 * scored as f64;
    println!("  expected 1st by chance    : {expected_r1:.1}");

    println!();
    if rank1 as f64 > 2.0 * expected_r1 && rank1 >= 3 {
        println!("  STAGE 5 GATE PASSED: substitutability recovers hand-authored oppositions");
        println!("  it was never given.");
    } else {
        println!("  STAGE 5 KILL CRITERION MET: recovery is at or near chance. If it cannot");
        println!("  recover KNOWN oppositions it will not find unknown ones. Relation typing");
        println!("  by substitutability closes here, with the trajectory line.");
    }

    let json = serde_json::json!({
        "corpus": "ccoha1+ccoha2 (semeval2020_ulscd_eng), pooled",
        "side": SIDE,
        "min_context": MIN_CONTEXT,
        "scored": scored,
        "rank1": rank1,
        "top3": top3,
        "chance_rank1": chance_r1,
        "excluded_pairs": excluded.iter().map(|(a, b)| format!("{a}<->{b}")).collect::<Vec<_>>(),
        "per_word": rows.iter().map(|(w, d, r, j)| serde_json::json!({
            "word": w, "dual": d, "rank": r, "jaccard": j
        })).collect::<Vec<_>>(),
    });
    fs::write(
        "experiment43_results.json",
        serde_json::to_string_pretty(&json).unwrap(),
    )
    .unwrap();
    println!("\n  wrote experiment43_results.json");
}
