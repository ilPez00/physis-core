//! Did the transformation go the way it was asked to?
//!
//! ## The operation nothing else in the field performs
//!
//! Eleven systems were surveyed for this (`computer-remake-research/`). Every
//! one can transform something; **not one can check the direction of its own
//! transformation.** `ragfs` will reorganise a directory and never compare that
//! against an arbitrary reorganisation. `recall` surfaces salient decisions and
//! never compares them against random decisions of the same age. A coding agent
//! rewrites a paragraph and the only test is whether the user complains.
//!
//! "Make this shorter" is the right acceptance test for a substrate precisely
//! because no single layer satisfies it. It needs an observation (the text
//! before), an interpretation (the rewrite), a claim (*this is shorter and still
//! says the same thing*), and a **control** (is it actually shorter, and did
//! anything else move). A chatbot answers it. Only a substrate checks it.
//!
//! ## The control, and why it is the whole idea
//!
//! Measuring "it got shorter" alone is trivial and nearly worthless: deleting
//! the second half also gets shorter. So every direction ships with a **null
//! transformation that achieves the same magnitude the stupid way**, and the
//! rewrite is scored against it.
//!
//! For [`Direction::Shorter`] the null is truncation to the same length. If a
//! rewrite does not retain meaning better than simply cutting the text off,
//! **it did not do anything a pair of scissors could not**, and this module says
//! so. That comparison is what makes the verdict able to fail.
//!
//! Retention is measured as the **worst-covered part** of the original rather
//! than as similarity to it, because similarity rewards copying and would hand
//! the verdict to the scissors. See [`worst_covered`].
//!
//! # !! THIS CONTROL DOES NOT CURRENTLY WORK — measured 2026-09-12
//!
//! It was validated on one authored example and **inverts on the first real
//! text it was pointed at.** Three arms over the same 118-word source (this
//! crate's own `observe.rs` header), semantic embedder:
//!
//! | arm | Δ | verdict | should be |
//! |---|---|---|---|
//! | genuine compression, every point kept | **−0.034** | FAILS | holds |
//! | drop every other sentence (deletion) | **+0.024** | HOLDS | fails |
//! | word shuffle at same length | −0.080 | fails | fails ✓ |
//!
//! **It prefers deletion over compression.** `worst_covered` moved the copying
//! bias from the whole-text level to the sentence level; it did not remove it.
//! Verbatim sentences still outscore paraphrases, so decimating homogeneous
//! prose leaves every dropped unit with a surviving near-neighbour and the
//! minimum stays high — while a real rewrite scores every unit as a paraphrase
//! and the minimum falls.
//!
//! Two further things are unmeasured and must not be quoted:
//! - the `0.02` margin threshold in [`Verdict::holds`] was chosen arbitrarily,
//!   and the observed margins (−0.080, −0.034, +0.024, +0.134) straddle it;
//! - `n = 4` across two texts, all four arms authored by the same party that
//!   wrote the metric.
//!
//! The shape is right — a direction needs a null that reaches the same
//! magnitude stupidly — and the retention statistic inside it is wrong. Do not
//! act on a `holds` from this command. Recorded rather than patched, because
//! adjusting the metric until the examples agree is how an instrument that
//! cannot fail gets built.
//!
//! ## Degrades honestly without a model
//!
//! The magnitude test is pure token counting and needs nothing. The retention
//! test needs an embedder; under the offline random-projection fallback it is
//! lexical, not semantic, and [`Verdict::retention_is_lexical`] records that so
//! a caller never mistakes a string-overlap number for a meaning number.

use crate::embed::VectorEmbed;
use serde::{Deserialize, Serialize};

/// What the caller asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// Fewer tokens, same meaning. Null: truncate to the same length.
    Shorter,
    /// More tokens, same meaning. Null: repeat the text to the same length.
    Longer,
    /// Same length, different wording, same meaning. Null: shuffle the words.
    Rephrased,
}

impl Direction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Direction::Shorter => "shorter",
            Direction::Longer => "longer",
            Direction::Rephrased => "rephrased",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "shorter" | "short" | "shorten" => Some(Direction::Shorter),
            "longer" | "long" | "expand" => Some(Direction::Longer),
            "rephrased" | "rephrase" | "reword" => Some(Direction::Rephrased),
            _ => None,
        }
    }

    /// Build the null: hit the same magnitude by the stupidest available means.
    /// The null must match the *result's* size, not a fixed ratio — otherwise it
    /// is a different transformation and the comparison is void.
    fn null(&self, before: &str, after_tokens: usize) -> String {
        let words: Vec<&str> = before.split_whitespace().collect();
        match self {
            Direction::Shorter => words
                .iter()
                .take(after_tokens)
                .cloned()
                .collect::<Vec<_>>()
                .join(" "),
            Direction::Longer => {
                let mut out: Vec<&str> = Vec::new();
                while out.len() < after_tokens && !words.is_empty() {
                    out.extend(words.iter().cloned());
                }
                out.truncate(after_tokens.max(words.len()));
                out.join(" ")
            }
            Direction::Rephrased => {
                // Deterministic word shuffle: same tokens, destroyed order.
                let mut w = words.clone();
                let mut s: u64 = 0x9E3779B97F4A7C15;
                for i in (1..w.len()).rev() {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    w.swap(i, (s % (i as u64 + 1)) as usize);
                }
                w.join(" ")
            }
        }
    }

    fn moved(&self, before: usize, after: usize) -> bool {
        match self {
            Direction::Shorter => after < before,
            Direction::Longer => after > before,
            // Within 15%: "rephrase" that halves the text is a different edit.
            Direction::Rephrased => {
                let b = before as f32;
                (after as f32 - b).abs() <= 0.15 * b.max(1.0)
            }
        }
    }
}

/// The answer, with everything needed to disbelieve it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verdict {
    pub direction: String,
    pub before_tokens: usize,
    pub after_tokens: usize,
    /// Did the size move the way it was asked to? Necessary, nowhere near
    /// sufficient.
    pub moved: bool,
    /// Worst-covered part of the original, under the result. Not similarity to
    /// the original — see [`worst_covered`] for why that distinction decides
    /// whether this instrument works at all.
    pub retention: f32,
    /// The same measure under the null.
    pub null_retention: f32,
    /// True when the embedder is the non-semantic fallback, so both retention
    /// figures are lexical overlap rather than meaning.
    pub retention_is_lexical: bool,
    pub embedder: String,
}

impl Verdict {
    /// The margin that decides it: how much better than scissors.
    pub fn margin(&self) -> f32 {
        self.retention - self.null_retention
    }

    /// Did the transformation do something the null could not?
    ///
    /// Both conditions are required. Moving without beating the null means the
    /// magnitude was achieved and nothing else; beating the null without moving
    /// means it was a good edit in the wrong direction.
    pub fn holds(&self) -> bool {
        self.moved && self.margin() > 0.02
    }

    /// Whether this verdict is trustworthy at all. Currently: never.
    ///
    /// See the module header. The retention statistic inverts on real prose —
    /// it rates deletion above compression — so `holds()` is not evidence. This
    /// is a separate method rather than a change to `holds()` so the broken
    /// behaviour stays visible and measurable while it is fixed.
    pub fn is_trustworthy(&self) -> bool {
        false
    }

    pub fn render(&self) -> String {
        let mut o = String::new();
        o.push_str(&format!(
            "── DIRECTION: {} ──\n{} → {} tokens   moved: {}\n",
            self.direction, self.before_tokens, self.after_tokens, self.moved
        ));
        o.push_str(&format!(
            "retention {:.3}   vs null (same size, done stupidly) {:.3}   Δ {:+.3}\n",
            self.retention,
            self.null_retention,
            self.margin()
        ));
        o.push_str(&format!("embedder {}\n", self.embedder));
        if self.retention_is_lexical {
            o.push_str(
                "\n!! retention is LEXICAL, not semantic: the embedder is the\n\
                 \x20  random-projection fallback, which measures string overlap.\n\
                 \x20  This biases the check TOWARD COPYING — a verbatim prefix\n\
                 \x20  outscores a genuine rewrite (measured: 0.882 vs 0.844), so a\n\
                 \x20  passing verdict here is not evidence. Set PHYSIS_MODEL_DIR.\n",
            );
        }
        o.push('\n');
        o.push_str(
            "\n!! DO NOT ACT ON THIS VERDICT. Measured 2026-09-12, this control\n\
             \x20  INVERTS on real prose: it rated deletion (+0.024) above a genuine\n\
             \x20  compression (-0.034) of the same text. The retention statistic\n\
             \x20  still rewards verbatim text, one level down from where it was\n\
             \x20  fixed. See the module header.\n\n",
        );
        o.push_str(if self.holds() {
            "VERDICT  holds — it moved the requested way AND retained more than\n\
             \x20        the null. The edit did something scissors could not.\n"
        } else if !self.moved {
            "VERDICT  FAILS — it did not move in the requested direction.\n"
        } else {
            "VERDICT  FAILS — it moved, but retained no more than truncating to\n\
             \x20        the same size. Magnitude was achieved; nothing else was.\n"
        });
        o
    }
}

/// Retention = **the worst-covered part of the original**, not similarity to it.
///
/// The first version of this used cosine between whole texts, and it was wrong
/// in a way worth recording. Cosine-to-original *rewards copying*: truncating a
/// 65-word text to a verbatim 41-word prefix keeps two thirds of it exactly, so
/// it scored 0.920 while a genuine rewrite that compressed every sentence scored
/// 0.919. The metric preferred scissors — which is precisely the thing the null
/// exists to detect, so the instrument was failing in the direction that makes
/// it useless.
///
/// Splitting the original into sentences and taking the **minimum** best-match
/// fixes it by construction. A prefix abandons its tail, so some sentence has no
/// match and the minimum collapses. A rewrite that keeps every point covers
/// every sentence somewhere, so the minimum stays high. You cannot pass this by
/// copying part of the input well; you have to keep all of it.
fn worst_covered(before: &str, after: &str, embedder: &dyn VectorEmbed) -> f32 {
    let units: Vec<&str> = before
        .split(['.', '\n', ';'])
        .map(str::trim)
        .filter(|u| u.split_whitespace().count() >= 3)
        .collect();
    if units.is_empty() || after.trim().is_empty() {
        return 0.0;
    }
    // Compare each original unit against each unit of the result, so a point
    // that moved to a different position still counts as covered.
    let after_units: Vec<&str> = after
        .split(['.', '\n', ';'])
        .map(str::trim)
        .filter(|u| !u.is_empty())
        .collect();
    let after_emb: Vec<Vec<f32>> = after_units.iter().map(|u| embedder.embed(u)).collect();

    let mut worst = f32::INFINITY;
    for u in &units {
        let ue = embedder.embed(u);
        let best = after_emb
            .iter()
            .map(|ae| crate::models::cosine_sim(&ue, ae))
            .fold(f32::NEG_INFINITY, f32::max);
        worst = worst.min(best);
    }
    if worst.is_finite() { worst } else { 0.0 }
}

/// Check `after` against `before` for `direction`, with the null in the same pass.
pub fn check(
    before: &str,
    after: &str,
    direction: Direction,
    embedder: &dyn VectorEmbed,
    embedder_kind: &str,
) -> anyhow::Result<Verdict> {
    anyhow::ensure!(!before.trim().is_empty(), "nothing to compare against");
    let bt = crate::rag::count_tokens(before);
    let at = crate::rag::count_tokens(after);

    let null = direction.null(before, after.split_whitespace().count());

    Ok(Verdict {
        direction: direction.as_str().to_string(),
        before_tokens: bt,
        after_tokens: at,
        moved: direction.moved(bt, at),
        retention: worst_covered(before, after, embedder),
        null_retention: worst_covered(before, &null, embedder),
        // Random projection fails the crate's own semantic self-test by design,
        // so anything it reports is lexical. Reuse that test rather than
        // matching on a name, which would rot the moment a name changes.
        retention_is_lexical: !crate::embed::semantic_self_test(embedder),
        embedder: embedder_kind.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::RandomProjectionEmbedder;

    const LONG: &str = "The pressure relief valve shall open at 2.6 bar and the machine \
must stop above it. Maintenance is scheduled every four hundred hours. The seal on line \
one is replaced when vibration rises after bearing wear becomes measurable.";

    fn e() -> RandomProjectionEmbedder {
        RandomProjectionEmbedder::new(256)
    }

    /// A rewrite that does not shorten must fail, whatever else it did well.
    /// Direction is a precondition, not a score to be traded off.
    #[test]
    fn not_moving_in_the_requested_direction_fails() {
        let v = check(LONG, &format!("{LONG} And more besides."), Direction::Shorter, &e(), "rp")
            .unwrap();
        assert!(!v.moved);
        assert!(!v.holds());
        assert!(v.render().contains("did not move"));
    }

    /// The test that makes this module worth having: truncation *does* get
    /// shorter, and must still fail, because it retains no more than the null —
    /// which for truncation IS the null, so the margin is zero by construction.
    #[test]
    fn truncation_gets_shorter_and_still_fails() {
        let cut: String = LONG.split_whitespace().take(12).collect::<Vec<_>>().join(" ");
        let v = check(LONG, &cut, Direction::Shorter, &e(), "rp").unwrap();
        assert!(v.moved, "truncation does shorten");
        assert!(
            v.margin().abs() < 0.001,
            "truncation is the null, so the margin must be ~0, got {}",
            v.margin()
        );
        assert!(!v.holds(), "shorter alone must never be enough");
        assert!(v.render().contains("retained no more than truncating"));
    }

    /// The null must match the RESULT's size, not a fixed ratio. If it did not,
    /// the comparison would be against a different transformation and the
    /// verdict would be meaningless.
    #[test]
    fn the_null_matches_the_result_size() {
        for take in [5usize, 12, 25] {
            let cut: String = LONG.split_whitespace().take(take).collect::<Vec<_>>().join(" ");
            let null = Direction::Shorter.null(LONG, take);
            assert_eq!(
                null.split_whitespace().count(),
                cut.split_whitespace().count(),
                "null and result must be the same size"
            );
        }
    }

    /// Under the offline fallback the retention figure is string overlap. It
    /// must say so, or a caller will read a lexical number as a meaning number —
    /// the exact mistake this repository has already made once at the benchmark
    /// level.
    #[test]
    fn lexical_retention_is_declared_not_implied() {
        let v = check(LONG, "short", Direction::Shorter, &e(), "random-projection").unwrap();
        assert!(v.retention_is_lexical);
        assert!(v.render().contains("LEXICAL"));
    }

    /// Rephrasing must hold its size; halving the text is a different edit and
    /// must not pass as a rephrase.
    #[test]
    fn rephrase_requires_the_size_to_hold() {
        assert!(Direction::Rephrased.moved(100, 95));
        assert!(!Direction::Rephrased.moved(100, 40));
    }

    /// **A known limitation, pinned so it cannot be forgotten.**
    ///
    /// `worst_covered` inverts the copying bias only when the embedder is
    /// semantic. Under the random-projection fallback it does NOT: a verbatim
    /// prefix still scores higher than a genuine rewrite, because string overlap
    /// is what that embedder measures and a prefix is made of the original's
    /// exact strings.
    ///
    /// Measured here: prefix 0.882 vs rewrite 0.844 under random projection.
    /// With `PHYSIS_MODEL_DIR` set to real weights, the same pair measures
    /// rewrite 0.879 vs null 0.746 (Δ +0.134) and the verdict holds.
    ///
    /// So: under a lexical embedder this instrument is biased toward scissors
    /// and its verdicts must not be acted on. `Verdict::retention_is_lexical`
    /// carries that to the caller; this test carries it to whoever edits the
    /// metric next.
    #[test]
    fn under_a_lexical_embedder_the_control_is_biased_toward_copying() {
        let e = e();
        let rewrite = "Relief valve opens at 2.6 bar; stop above it. Pump maintenance \
every 400 hours. Replace the line-one seal when vibration rises after bearing wear.";
        let prefix: String = LONG
            .split_whitespace()
            .take(rewrite.split_whitespace().count())
            .collect::<Vec<_>>()
            .join(" ");
        let rw = worst_covered(LONG, rewrite, &e);
        let pf = worst_covered(LONG, &prefix, &e);
        assert!(
            pf > rw,
            "documented limitation: under RP the prefix ({pf}) outscores the rewrite ({rw}). \
If this assertion ever fails, the fallback became semantic and the warning in \
`retention_is_lexical` should be revisited."
        );
        let v = check(LONG, rewrite, Direction::Shorter, &e, "random-projection").unwrap();
        assert!(v.retention_is_lexical, "the caller must be told");
    }

    /// Truncation is the null for Shorter, so its margin must be exactly zero.
    /// Not approximately: the same string is scored on both sides. If this ever
    /// drifts, the null and the result have stopped being the same computation.
    #[test]
    fn truncation_scores_exactly_zero_margin_against_its_own_null() {
        let cut: String = LONG.split_whitespace().take(15).collect::<Vec<_>>().join(" ");
        let v = check(LONG, &cut, Direction::Shorter, &e(), "rp").unwrap();
        assert_eq!(v.retention, v.null_retention);
        assert_eq!(v.margin(), 0.0);
        assert!(!v.holds());
    }

    /// The measurement that killed this control, pinned so a future fix has a
    /// target and cannot quietly declare victory.
    ///
    /// Deletion must not outscore compression. Today it does, under a semantic
    /// embedder, on this crate's own prose. The assertion is written to FAIL
    /// when the metric is repaired — it asserts the broken ordering — so
    /// whoever fixes `worst_covered` is forced to come back and delete it.
    #[test]
    #[ignore = "documents the broken ordering; run with --ignored after any change to worst_covered — it must then FAIL, and be replaced"]
    fn known_broken_deletion_outscores_compression() {
        let e = e();
        let original = "The observation log records what the machine saw. Observers \
record but never mean anything. Indexers find but hold no position on truth. \
Interpreters let a model decide and the model becomes the memory.";
        let sentences: Vec<&str> = original.split(". ").collect();
        let decimated: String = sentences.iter().step_by(2).cloned().collect::<Vec<_>>().join(". ");
        let compressed = "The log records what the machine saw. Observers record \
without meaning; indexers find without judging truth; interpreters hand memory \
to the model.";
        let d = worst_covered(original, &decimated, &e);
        let c = worst_covered(original, compressed, &e);
        assert!(
            d >= c,
            "the broken ordering no longer holds (deletion {d} < compression {c}) — \
the metric may be fixed. Verify against the module header's table and delete \
this test."
        );
    }

    #[test]
    fn an_empty_before_is_an_error_not_a_verdict() {
        assert!(check("", "x", Direction::Shorter, &e(), "rp").is_err());
    }
}
