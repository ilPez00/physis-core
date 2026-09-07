//! Coverage impact — which candidate cells would let the ontology place
//! records it currently cannot.
//!
//! ## Why this exists
//!
//! [`crate::discovery`] proposes new ontology entries but has no way to say
//! which proposals are worth keeping. This is that missing half, and it is
//! deliberately the *narrow* half: it does not judge whether a candidate is a
//! good concept, only whether adding it changes what the ontology can place.
//!
//! ## Why the check is shaped this way
//!
//! A long research track (`research/perspective-discovery/` in the
//! superproject) tried nine mechanisms for certifying discovered structure and
//! all nine failed. Every one asked a REPRESENTATIONAL question — is this
//! split correct, does a lexicon recognise it, do many pairs agree on it — and
//! answered it with a statistic computed over the same embedding space that
//! produced the candidate.
//!
//! This asks an OPERATIONAL question instead: does adding the candidate make
//! records classifiable that were not? The corpus of records is external to
//! whatever produced the candidate, so the check cannot be satisfied by the
//! geometry agreeing with itself.
//!
//! Measured over 5-fold cross-validation on 691 held-out records: re-adding an
//! entry the ontology already contains scores **exactly zero** on 2000 out of
//! 2000 trials — correctly, since records near it were already covered — while
//! genuine interpolations score above zero (Welch t = +12.11). The check is not
//! fooled by a candidate that adds nothing, and it needs neither a human nor a
//! lexicon to say so. In that run it reduced 2000 candidates to 141 worth
//! reading.
//!
//! ## What it does NOT tell you
//!
//! **Coverage is not correctness.** A candidate that swallows records into a
//! wrong cell scores exactly like one that captures a real gap — both make the
//! records classifiable. Deciding which is which needs labels, and is the
//! question those nine mechanisms failed to answer. Use this to *shrink the
//! pile*, then have a person read what survives; do not use it to auto-accept.
//!
//! ```no_run
//! use physis_core::coverage::rank_candidates;
//! # fn demo(classifier: &physis_core::classify::CellClassifier,
//! #         candidates: &[physis_core::classify::Cell],
//! #         records: &[(String, Vec<f32>)]) {
//! for (idx, gain) in rank_candidates(classifier, candidates, records, 0.85) {
//!     println!("candidate {idx} would newly cover {} records", gain.count());
//! }
//! # }
//! ```

use crate::classify::{Cell, CellClassifier};
use crate::models::cosine_sim;

/// What one candidate cell would add.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CandidateGain {
    /// Ids of records that are not confidently placed today and would be
    /// placed by this candidate, in the order the records were supplied.
    pub newly_covered: Vec<String>,
}

impl CandidateGain {
    /// How many records this candidate would rescue.
    pub fn count(&self) -> usize {
        self.newly_covered.len()
    }

    /// Whether the candidate changes anything at all. A candidate that fails
    /// this is provably not worth a reader's attention: it leaves every record
    /// exactly where it was.
    pub fn is_useful(&self) -> bool {
        !self.newly_covered.is_empty()
    }
}

/// Best raw single-entry cosine for each record under the live ontology.
///
/// Raw best-entry similarity rather than the blended classifier score, matching
/// [`crate::discovery`]: the blend's population weighting is a ranking aid, and
/// mixing it into a coverage threshold would make the threshold mean different
/// things in dense and sparse cells.
fn live_scores(classifier: &CellClassifier, records: &[(String, Vec<f32>)]) -> Vec<f32> {
    records
        .iter()
        .map(|(_, e)| classifier.best_entry_sim(e).map(|(s, _, _)| s).unwrap_or(f32::NEG_INFINITY))
        .collect()
}

/// Highest cosine from `embedding` to any member entry of `cell`.
fn cell_score(cell: &Cell, embedding: &[f32]) -> f32 {
    cell.embeddings
        .iter()
        .map(|e| cosine_sim(embedding, e))
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Ids of records the live ontology cannot place at or above `threshold`.
///
/// This is the pile a candidate has to make a dent in. If it is empty, every
/// candidate will score zero and the ranking carries no information — pick a
/// threshold against the observed score distribution rather than an absolute
/// guess.
pub fn uncovered(
    classifier: &CellClassifier,
    records: &[(String, Vec<f32>)],
    threshold: f32,
) -> Vec<String> {
    let live = live_scores(classifier, records);
    records
        .iter()
        .zip(&live)
        .filter(|(_, &s)| s < threshold)
        .map(|((id, _), _)| id.clone())
        .collect()
}

/// What a single candidate cell would newly cover.
pub fn candidate_gain(
    classifier: &CellClassifier,
    candidate: &Cell,
    records: &[(String, Vec<f32>)],
    threshold: f32,
) -> CandidateGain {
    let live = live_scores(classifier, records);
    gain_against(&live, candidate, records, threshold)
}

/// Score every candidate and return them strongest-first.
///
/// The live scores are computed once and shared, so this is far cheaper than
/// calling [`candidate_gain`] in a loop. Ties break on candidate index, never
/// on iteration order, so the ranking is reproducible.
pub fn rank_candidates(
    classifier: &CellClassifier,
    candidates: &[Cell],
    records: &[(String, Vec<f32>)],
    threshold: f32,
) -> Vec<(usize, CandidateGain)> {
    let live = live_scores(classifier, records);
    let mut scored: Vec<(usize, CandidateGain)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (i, gain_against(&live, c, records, threshold)))
        .collect();
    scored.sort_by(|a, b| b.1.count().cmp(&a.1.count()).then_with(|| a.0.cmp(&b.0)));
    scored
}

fn gain_against(
    live: &[f32],
    candidate: &Cell,
    records: &[(String, Vec<f32>)],
    threshold: f32,
) -> CandidateGain {
    // Adding a cell can only raise a record's best score, so a record can never
    // move from covered to uncovered. "Newly covered" is therefore exactly
    // "was below the threshold, and this candidate is at or above it".
    let newly_covered = records
        .iter()
        .zip(live)
        .filter(|((_, e), &s)| s < threshold && cell_score(candidate, e) >= threshold)
        .map(|((id, _), _)| id.clone())
        .collect();
    CandidateGain { newly_covered }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Facets;

    fn basis(dim: usize, i: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        v
    }

    fn cell(domain: &str, mode: &str, embeddings: Vec<Vec<f32>>) -> Cell {
        Cell {
            domain: domain.to_string(),
            mode: mode.to_string(),
            entries: embeddings.iter().map(|_| "e".to_string()).collect(),
            facets: embeddings.iter().map(|_| Facets::default()).collect(),
            embeddings,
        }
    }

    fn fixture() -> CellClassifier {
        CellClassifier::from_cells(vec![
            cell("HEAL", "REST", vec![basis(4, 0)]),
            cell("BOND", "REST", vec![basis(4, 1)]),
        ])
    }

    #[test]
    fn uncovered_lists_only_records_below_threshold() {
        let clf = fixture();
        let records = vec![
            ("on_top".to_string(), basis(4, 0)),   // cosine 1.0 to a known entry
            ("far_away".to_string(), basis(4, 3)), // orthogonal to everything
        ];
        let u = uncovered(&clf, &records, 0.5);
        assert_eq!(u, vec!["far_away".to_string()]);
    }

    #[test]
    fn a_candidate_covering_the_gap_scores_it() {
        let clf = fixture();
        let records = vec![("far_away".to_string(), basis(4, 3))];
        let candidate = cell("NEW", "CELL", vec![basis(4, 3)]);
        let g = candidate_gain(&clf, &candidate, &records, 0.5);
        assert_eq!(g.count(), 1);
        assert!(g.is_useful());
        assert_eq!(g.newly_covered, vec!["far_away".to_string()]);
    }

    /// The behaviour the cross-validation actually established: re-adding
    /// something the ontology already has rescues nothing, because the records
    /// near it were already covered. This is what makes the check a usable
    /// filter rather than a rubber stamp — it scored 0 on 2000/2000 trials.
    #[test]
    fn duplicating_an_existing_entry_gains_nothing() {
        let clf = fixture();
        let records = vec![
            ("on_top".to_string(), basis(4, 0)),
            ("far_away".to_string(), basis(4, 3)),
        ];
        let duplicate = cell("HEAL", "REST", vec![basis(4, 0)]);
        let g = candidate_gain(&clf, &duplicate, &records, 0.5);
        assert_eq!(g.count(), 0, "a duplicate must not be credited with coverage");
        assert!(!g.is_useful());
    }

    #[test]
    fn ranking_is_strongest_first_and_reproducible() {
        let clf = fixture();
        let records = vec![
            ("a".to_string(), basis(4, 2)),
            ("b".to_string(), basis(4, 3)),
        ];
        let candidates = vec![
            cell("X", "1", vec![basis(4, 2)]),                 // covers one
            cell("Y", "2", vec![basis(4, 2), basis(4, 3)]),    // covers both
            cell("Z", "3", vec![basis(4, 0)]),                 // covers none
        ];
        let ranked = rank_candidates(&clf, &candidates, &records, 0.5);
        assert_eq!(ranked[0].0, 1, "the candidate covering both must rank first");
        assert_eq!(ranked[0].1.count(), 2);
        assert_eq!(ranked[2].1.count(), 0);
        for _ in 0..8 {
            assert_eq!(rank_candidates(&clf, &candidates, &records, 0.5), ranked);
        }
    }

    #[test]
    fn nothing_uncovered_means_no_candidate_can_gain() {
        // Guards the documented degenerate case: with a threshold every record
        // already clears, the ranking is uniformly zero and carries no signal.
        let clf = fixture();
        let records = vec![("on_top".to_string(), basis(4, 0))];
        assert!(uncovered(&clf, &records, -1.0).is_empty());
        let candidate = cell("NEW", "CELL", vec![basis(4, 0)]);
        assert_eq!(candidate_gain(&clf, &candidate, &records, -1.0).count(), 0);
    }
}
