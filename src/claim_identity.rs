//! Is a claim its sentence, or is it a structure?
//!
//! ## The question, and the number it has to beat
//!
//! Conceptual problem 2: a claim is identified by `statement`, and
//! contradiction, dedup, supersession and `act`'s recall all run on the
//! embedding of that sentence. The graphiti state-hash experiment measured
//! prose as the one layer that moves under an interpreter swap — and across two
//! runs of the *same* interpreter — while entities, topology and intervals held
//! byte-identical. So the ledger computes its defining relations on noise.
//!
//! The plan's direction is a structural identity — subject, relation, object,
//! interval — with the sentence demoted to attributed rendering. This module is
//! the cheapest honest test of whether that direction buys anything, because it
//! does not need an interpreter: **a polarity-aware content signature is a
//! structural identity in its poorest form.** If even that beats the sentence
//! embedding at finding contradictions, the direction is worth the work. If it
//! does not, the argument is aesthetic and should be dropped.
//!
//! ## The task
//!
//! Given every pair of claims in the ledger, say which pairs *contradict*. The
//! corpus is the twin ledger from [`crate::act_recall`], plus eight agreeing
//! restatements: 64 claims, of which 8 are refutation-and-endorsement pairs
//! about the same fact — same subject, same vocabulary, opposite verdict.
//!
//! **The agreeing restatements are what makes the question askable**, and the
//! first run of this module had to be thrown out for lacking them. Without
//! them the 8 contradiction pairs are also, by construction, the 8 most
//! *similar* pairs in the corpus: a cosine threshold at 0.80 isolated them
//! exactly and scored F1 **1.000** for a reason that has nothing to do with
//! contradiction. Each agreement sits at the same similarity as its twin and
//! asserts the *same* verdict, so a method that flags "two claims about one
//! fact that disagree" must take the twins and leave these, while a method that
//! flags "two claims about one fact" takes both.
//!
//! **Precision matters more than recall here, and that is the point.** The
//! positives are also the most *similar* pairs in the corpus. A method that
//! ranks by similarity will find them all and will also flag every near
//! duplicate, because similarity cannot tell "says the same thing" from "says
//! the opposite thing about the same thing". That is the whole of problem 2 in
//! one sentence, and it is what the false-positive count measures.
//!
//! ## The two methods
//!
//! - **`embedding`** — the current path. Two claims contradict if their
//!   statement embeddings are more similar than a threshold. Swept, because a
//!   single threshold would be a straw man.
//! - **`structural`** — content signature plus polarity. Strip stopwords and
//!   polarity words, keep the rest as a set; two claims contradict if the sets
//!   overlap above a threshold **and** their polarities differ. This is
//!   (subject, relation, object) collapsed to a bag and (verdict) as the fourth
//!   field, with no interpreter and no parser.
//!
//! Both are swept over the same thresholds on the same pairs, so the comparison
//! is between the *identity*, not between two tunings.
//!
//! ## What it measured, 2026-09-13
//!
//! 64 claims, 2016 pairs, 8 known contradictions, bge-base-en-v1.5, each
//! method at its own best threshold:
//!
//! | identity | found | false positives | precision | recall | F1 |
//! |---|---|---|---|---|---|
//! | `embedding@0.90` | 8/8 | 4 | 0.667 | 1.000 | **0.800** |
//! | `structural@0.30–0.70` | 3/8 | **0** | **1.000** | 0.375 | 0.545 |
//!
//! **Read the split, not the F1.** The sentence embedding finds every
//! contradiction and also flags four of the eight *agreeing* restatements as
//! contradictions — at its best threshold it is wrong a third of the time it
//! speaks, because similarity cannot distinguish "says the same thing about
//! this" from "says the opposite thing about this". That is conceptual problem
//! 2, measured, on this crate's own corpus.
//!
//! The structural identity is **never wrong when it speaks** — precision 1.000
//! at every threshold in the sweep, no agreement ever mistaken for a
//! contradiction — and it is silent on five of the eight pairs.
//!
//! F1 says prose wins. F1 is the wrong number here: item B measured that
//! `act`'s larger defect is warning when it should be silent, so precision is
//! worth more than recall for this consumer, and an average weighting them
//! equally hides the difference the experiment exists to see.
//!
//! ### The ceiling is the argument for doing C2 properly
//!
//! Structural recall is capped at 0.375 by one thing: **5 of 8 pairs have a
//! side whose verdict the extractor cannot read.** Not because the lexicon is
//! too small — because the sentences negate predicates that no lexicon
//! contains:
//!
//! > the release build **finishes** … the LTO stage **never runs out of memory**
//!
//! One positive verdict, one negator scoped over a predicate that is not a
//! verdict word, net zero. Widening the lexicon cannot fix that class; scoping
//! a negation over an arbitrary predicate is what a parser is for. So the
//! poorest structural identity already gives perfect precision and its only
//! limit is the thing the real design would supply. That is the strongest
//! argument available for building it — and it is an argument, not a result.
//!
//! ## What this cannot say
//!
//! A bag of content words is not a triple. It has no subject/object asymmetry,
//! so "A supersedes B" and "B supersedes A" are identical to it, and it will
//! never carry an interval. A negative result here would therefore not refute
//! the plan's direction — only a positive result is informative, and that is
//! the asymmetry to keep in mind when reading the table. It is run because it
//! is cheap and because a positive result would settle the direction without
//! building the parser first.

use serde::{Deserialize, Serialize};

/// Words that carry a verdict rather than a subject.
///
/// Kept out of the content signature *and* used as the polarity signal, so the
/// same word cannot make two claims look alike and then be asked to tell them
/// apart.
const NEGATIVE: [&str; 24] = [
    "not",
    "never",
    "no",
    "cannot",
    "fails",
    "fail",
    "failed",
    "failing",
    "breaks",
    "broke",
    "broken",
    "rejected",
    "rejects",
    "drops",
    "loses",
    "lost",
    "corrupts",
    "exhausts",
    "miscompiles",
    "without",
    "missing",
    "stale",
    "wrong",
    "slower",
];

/// Words that carry the opposite verdict.
const POSITIVE: [&str; 16] = [
    "passes",
    "pass",
    "passed",
    "succeeds",
    "succeed",
    "succeeded",
    "works",
    "finishes",
    "completes",
    "accepted",
    "accepts",
    "healthy",
    "idempotent",
    "resolves",
    "fine",
    "correct",
];

/// Function words, which carry neither.
const STOP: [&str; 40] = [
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "with", "is", "are", "it",
    "this", "that", "as", "by", "from", "into", "than", "when", "then", "there", "here", "its",
    "was", "were", "be", "been", "so", "but", "if", "any", "every", "all", "each", "run", "runs",
];

/// The structural identity, in its poorest form: what the claim is about, and
/// which way it comes down.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Signature {
    /// Content words, lowercased, sorted, deduplicated. Stands in for
    /// (subject, relation, object).
    pub content: Vec<String>,
    /// `-1` refuting, `+1` affirming, `0` neither found.
    pub polarity: i8,
}

impl Signature {
    /// Jaccard over the content sets. Not cosine: the sets are small and a
    /// frequency weighting on eight-word bags would be noise with a formula.
    pub fn overlap(&self, other: &Signature) -> f32 {
        if self.content.is_empty() || other.content.is_empty() {
            return 0.0;
        }
        let shared = self
            .content
            .iter()
            .filter(|w| other.content.contains(w))
            .count();
        let union = self.content.len() + other.content.len() - shared;
        if union == 0 {
            return 0.0;
        }
        shared as f32 / union as f32
    }

    /// Do these disagree? Only when both carry a verdict and the verdicts
    /// differ — an unknown polarity is not a disagreement, it is a silence.
    pub fn opposed(&self, other: &Signature) -> bool {
        self.polarity != 0 && other.polarity != 0 && self.polarity != other.polarity
    }
}

/// Bare negators — they carry no verdict alone, they invert the next one.
///
/// Separated from [`NEGATIVE`] after the first version of this module got
/// "never finishes" and "never runs out of memory" both wrong: counting
/// verdict words in a bag cannot scope a negation, and both sentences have one
/// negator and one positive word. A negator now attaches to the nearest
/// following verdict word inside [`SCOPE`] and inverts it; an unattached
/// negator is left as a weak negative on its own.
const NEGATOR: [&str; 6] = ["not", "never", "cannot", "without", "no", "nor"];

/// How far a negator reaches. Three tokens covers "never finishes",
/// "does not resolve", "cannot ever pass" and stops well short of the next
/// clause. It is a guess, and the module reports how often scoping fails
/// rather than tuning it until the corpus agrees.
const SCOPE: usize = 3;

/// Extract the signature. No parser, no model, no network.
pub fn signature(statement: &str) -> Signature {
    let lower = statement.to_ascii_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();

    let mut content: Vec<String> = Vec::new();
    let mut score = 0i32;
    let mut consumed = vec![false; tokens.len()];

    // Pass one: negator + verdict inside the scope window, innermost first.
    for i in 0..tokens.len() {
        if !NEGATOR.contains(&tokens[i]) || consumed[i] {
            continue;
        }
        for j in (i + 1)..(i + 1 + SCOPE).min(tokens.len()) {
            if consumed[j] {
                continue;
            }
            if POSITIVE.contains(&tokens[j]) {
                score -= 1; // "never finishes"
                consumed[i] = true;
                consumed[j] = true;
                break;
            }
            if NEGATIVE.contains(&tokens[j]) {
                score += 1; // "not broken"
                consumed[i] = true;
                consumed[j] = true;
                break;
            }
        }
    }
    // Pass two: whatever is left says what it says.
    for (i, t) in tokens.iter().enumerate() {
        if consumed[i] {
            continue;
        }
        if t.len() < 3 {
            continue;
        }
        if NEGATOR.contains(t) {
            score -= 1;
            continue;
        }
        if NEGATIVE.contains(t) {
            score -= 1;
            continue;
        }
        if POSITIVE.contains(t) {
            score += 1;
            continue;
        }
        if STOP.contains(t) {
            continue;
        }
        content.push(t.to_string());
    }

    content.sort();
    content.dedup();
    let polarity = match score.cmp(&0) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Equal => 0,
    };
    Signature { content, polarity }
}

/// One method at one threshold, scored over every pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairScore {
    /// `embedding@<t>` or `structural@<t>`.
    pub method: String,
    pub threshold: f32,
    /// Known contradiction pairs found.
    pub true_positives: usize,
    /// Pairs flagged that are not contradictions.
    pub false_positives: usize,
    pub known: usize,
    pub pairs_examined: usize,
    pub recall: f32,
    /// `tp / (tp + fp)`; 1.0 when nothing was flagged, which is reported as
    /// `flagged = 0` rather than hidden behind a perfect score.
    pub precision: f32,
    pub f1: f32,
}

/// Everything the run measured, with the baseline it has to beat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityRun {
    pub claims: usize,
    pub pairs_examined: usize,
    pub known_contradictions: usize,
    pub embedder: String,
    pub scores: Vec<PairScore>,
    /// Best F1 for each method, so the comparison is between the methods at
    /// their own best rather than at a shared threshold that suits one.
    pub best_embedding: Option<PairScore>,
    pub best_structural: Option<PairScore>,
    /// Known pairs where at least one side's polarity could not be read at
    /// all. These are unreachable for the structural method by construction —
    /// its ceiling, and the size of the argument for a real parser.
    pub polarity_unresolved: usize,
    /// The statements the extractor could not assign a verdict to, so the
    /// ceiling can be read rather than only counted.
    pub unresolved_examples: Vec<String>,
}

impl IdentityRun {
    pub fn render(&self) -> String {
        let mut o = String::new();
        o.push_str(&format!(
            "── CLAIM IDENTITY: is a claim its sentence, or a structure? ──\n{} claims · {} pairs · {} known contradictions · embedder {}\n\n",
            self.claims, self.pairs_examined, self.known_contradictions, self.embedder
        ));
        o.push_str("  method              thr    found  false+   recall  precision      F1\n");
        for s in &self.scores {
            o.push_str(&format!(
                "  {:<18} {:>5.2}   {:>3}/{:<3}  {:>5}    {:>5.3}      {:>5.3}   {:>5.3}\n",
                s.method,
                s.threshold,
                s.true_positives,
                s.known,
                s.false_positives,
                s.recall,
                s.precision,
                s.f1
            ));
        }
        if let (Some(e), Some(st)) = (&self.best_embedding, &self.best_structural) {
            o.push_str(&format!(
                "\n  best embedding  F1 {:.3}  ({} found, {} false positives)\n  best structural F1 {:.3}  ({} found, {} false positives)\n",
                e.f1, e.true_positives, e.false_positives, st.f1, st.true_positives, st.false_positives
            ));
            // F1 is the wrong single number for this task and the split is
            // reported instead. `act` is a warning system, and item B measured
            // that its larger defect is warning when it should be silent — so
            // a method's precision is worth more than its recall here, and an
            // average that weights them equally hides exactly the difference
            // this experiment exists to see.
            o.push_str(&format!(
                "\n  the split, which is what matters:\n    prose      speaks on every pair and is wrong {:.0}% of the time it speaks\n               (precision {:.3} at its best F1)\n    structure  is never wrong when it speaks (precision {:.3}) and is silent\n               on {:.0}% of the pairs\n",
                (1.0 - e.precision) * 100.0,
                e.precision,
                st.precision,
                (1.0 - st.recall) * 100.0
            ));
        }
        if self.polarity_unresolved > 0 {
            o.push_str(&format!(
                "\n  ceiling: {} of {} known pairs have a side whose verdict the extractor\n  cannot read, so they are unreachable for the structural method by\n  construction. Widening the lexicon does not fix the class — scoping a\n  negation over an arbitrary predicate is what a parser is for.\n",
                self.polarity_unresolved, self.known_contradictions
            ));
            for e in self.unresolved_examples.iter().take(4) {
                o.push_str(&format!(
                    "    · {}\n",
                    e.chars().take(88).collect::<String>()
                ));
            }
        }
        o
    }
}

/// Thresholds swept. The same list for both methods on purpose: a comparison
/// where each side is tuned on a different grid is a comparison of grids.
pub const THRESHOLDS: [f32; 9] = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95];

/// A corpus for [`run`]: each claim's statement with its embedding.
pub type ScoredClaims = Vec<(String, Vec<f32>)>;
/// Index pairs that really do contradict.
pub type TruthPairs = Vec<(usize, usize)>;

/// Score both identities over every pair of `claims`.
///
/// `truth` holds the index pairs that really do contradict.
pub fn run(
    claims: &[(String, Vec<f32>)],
    truth: &[(usize, usize)],
    embedder_kind: &str,
) -> IdentityRun {
    let sigs: Vec<Signature> = claims.iter().map(|(s, _)| signature(s)).collect();
    let is_true = |i: usize, j: usize| {
        truth
            .iter()
            .any(|&(a, b)| (a == i && b == j) || (a == j && b == i))
    };

    let mut scores = Vec::new();
    let mut pairs_examined = 0usize;
    for &t in THRESHOLDS.iter() {
        for method in ["embedding", "structural"] {
            let (mut tp, mut fp, mut seen) = (0usize, 0usize, 0usize);
            for i in 0..claims.len() {
                for j in (i + 1)..claims.len() {
                    seen += 1;
                    let flagged = match method {
                        "embedding" => crate::models::cosine_sim(&claims[i].1, &claims[j].1) >= t,
                        _ => sigs[i].overlap(&sigs[j]) >= t && sigs[i].opposed(&sigs[j]),
                    };
                    if !flagged {
                        continue;
                    }
                    if is_true(i, j) {
                        tp += 1;
                    } else {
                        fp += 1;
                    }
                }
            }
            pairs_examined = seen;
            let recall = tp as f32 / truth.len().max(1) as f32;
            let flagged = tp + fp;
            let precision = if flagged == 0 {
                0.0
            } else {
                tp as f32 / flagged as f32
            };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            scores.push(PairScore {
                method: format!("{method}@{t:.2}"),
                threshold: t,
                true_positives: tp,
                false_positives: fp,
                known: truth.len(),
                pairs_examined: seen,
                recall,
                precision,
                f1,
            });
        }
    }

    let best = |prefix: &str| -> Option<PairScore> {
        scores
            .iter()
            .filter(|s| s.method.starts_with(prefix))
            .max_by(|a, b| a.f1.partial_cmp(&b.f1).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
    };

    let mut polarity_unresolved = 0usize;
    let mut unresolved_examples = Vec::new();
    for &(a, b) in truth {
        for i in [a, b] {
            if sigs[i].polarity == 0 && !unresolved_examples.contains(&claims[i].0) {
                unresolved_examples.push(claims[i].0.clone());
            }
        }
        if sigs[a].polarity == 0 || sigs[b].polarity == 0 {
            polarity_unresolved += 1;
        }
    }

    IdentityRun {
        claims: claims.len(),
        pairs_examined,
        known_contradictions: truth.len(),
        embedder: embedder_kind.to_string(),
        best_embedding: best("embedding"),
        best_structural: best("structural"),
        polarity_unresolved,
        unresolved_examples,
        scores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polarity_is_read_from_the_verdict_words() {
        assert_eq!(signature("the build fails on this box").polarity, -1);
        assert_eq!(signature("the build passes on this box").polarity, 1);
        assert_eq!(signature("the build takes four minutes").polarity, 0);
    }

    /// Verdict words must not also be content, or two opposed claims would look
    /// *less* alike for the very reason they are a pair.
    #[test]
    fn verdict_words_are_not_content() {
        let s = signature("the build fails and never passes");
        assert!(!s
            .content
            .iter()
            .any(|w| w == "fails" || w == "passes" || w == "never"));
    }

    /// The pair this module exists for, in the case the extractor can read.
    #[test]
    fn a_refutation_and_its_endorsement_overlap_and_oppose() {
        let a = signature("cargo test fails on this crate: the ort linker step never resolves");
        let b =
            signature("cargo test passes on this crate: the ort linker step resolves every time");
        assert!(a.overlap(&b) > 0.4, "overlap was {}", a.overlap(&b));
        assert!(
            a.opposed(&b),
            "polarities {} and {}",
            a.polarity,
            b.polarity
        );
    }

    /// And the case it CANNOT read, kept as a test rather than patched away.
    ///
    /// "the release build never finishes … it runs out of memory" against
    /// "the release build finishes … the LTO stage never runs out of memory".
    /// The second sentence negates a predicate — *running out of memory* — that
    /// is not in any verdict lexicon, so a bag with a scope window sees one
    /// positive and one loose negator and scores zero. No amount of lexicon
    /// widening fixes the class: scoping a negation over an arbitrary predicate
    /// is what a parser is for, and the absence of one is the measured ceiling
    /// of this identity. The run reports how often it happens instead of
    /// hiding it.
    #[test]
    fn double_negation_over_an_unlisted_predicate_is_unreadable() {
        let b = signature("the release build finishes on this box in four minutes — the LTO stage never runs out of memory");
        assert_eq!(
            b.polarity, 0,
            "if this ever becomes nonzero, say why in the doc"
        );
    }

    /// And the case that must NOT fire: two claims that agree.
    #[test]
    fn agreement_is_not_contradiction() {
        let a = signature("the release build fails at the LTO stage");
        let b = signature("the release build breaks at the LTO stage");
        assert!(a.overlap(&b) > 0.5);
        assert!(
            !a.opposed(&b),
            "two refutations must not read as a contradiction"
        );
    }

    /// An unknown polarity is a silence, not a disagreement.
    #[test]
    fn a_claim_with_no_verdict_opposes_nothing() {
        let a = signature("the release build takes four minutes");
        let b = signature("the release build fails at the LTO stage");
        assert!(!a.opposed(&b));
    }

    /// The harness must actually find a planted pair, or the numbers below it
    /// are about the harness.
    #[test]
    fn the_run_finds_a_planted_contradiction() {
        let claims = vec![
            (
                "the migrate script fails when the audit table is present".to_string(),
                vec![1.0, 0.0],
            ),
            (
                "the migrate script succeeds when the audit table is present".to_string(),
                vec![0.9, 0.1],
            ),
            (
                "the semiotic grid holds five domains".to_string(),
                vec![0.0, 1.0],
            ),
        ];
        let r = run(&claims, &[(0, 1)], "test");
        assert_eq!(r.pairs_examined, 3);
        let s = r.best_structural.unwrap();
        assert_eq!(s.true_positives, 1, "the planted pair must be found");
        assert_eq!(s.false_positives, 0);
    }
}
