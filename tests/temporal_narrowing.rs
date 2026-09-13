//! PH-019 — a validity window can be narrowed by better evidence, and the
//! narrowing is recorded.
//!
//! The failure this guards was measured in graphiti
//! (`computer-remake-research/experiments/graphiti/RESULTS.md`): it stamps its
//! world-time end once, from the first contradiction it happens to see, and
//! never narrows it when better evidence arrives — leaving one person employed
//! at two companies simultaneously in the stored record. It also mutates that
//! leg with no revision entry, so the weak boundary is indistinguishable
//! afterwards from one the evidence always supported.

use physis_core::hypothesis::Hypothesis;
use physis_core::temporal::TemporalValidity;

fn at(days: i64) -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::<chrono::Utc>::from_timestamp(days * 86_400, 0).unwrap()
}

fn hypothesis_valid_from(days: i64) -> Hypothesis {
    let mut h = Hypothesis::new("Mira works at Aurelia Systems", vec![0.0; 8]);
    h.temporal = TemporalValidity::from(at(days));
    h
}

#[test]
fn a_narrowing_records_exactly_one_revision() {
    let mut h = hypothesis_valid_from(0);
    let before = h.revision_history.len();

    assert!(h.narrow_validity(at(400), Some("contradicted by E2".into())));

    assert_eq!(
        h.revision_history.len(),
        before + 1,
        "a window that moved must leave a revision behind"
    );
    let last = h.revision_history.last().unwrap();
    assert!(
        last.description.contains("Validity closed"),
        "revision should say what happened, got: {}",
        last.description
    );
    assert_eq!(last.trigger.as_deref(), Some("contradicted by E2"));
}

#[test]
fn a_refused_narrowing_records_nothing() {
    let mut h = hypothesis_valid_from(0);
    assert!(h.narrow_validity(at(400), None));
    let after_real_change = h.revision_history.len();

    // Widening: refused, and must not appear in the ledger at all.
    assert!(!h.narrow_validity(at(800), Some("later, weaker evidence".into())));
    assert!(!h.narrow_validity(at(-10), None));

    assert_eq!(
        h.revision_history.len(),
        after_real_change,
        "a no-op must not report a change that never happened"
    );
    assert_eq!(h.temporal.valid_until, Some(at(400)));
}

/// The graphiti case end to end: a boundary set from a weak first
/// contradiction, then corrected when earlier evidence arrives.
#[test]
fn better_evidence_corrects_a_boundary_set_by_weaker_evidence() {
    let mut h = hypothesis_valid_from(0);
    let mid = at(600);

    assert!(h.narrow_validity(at(800), Some("first contradiction".into())));
    assert!(
        h.is_valid_at(mid),
        "with only the weak boundary the claim still reads true at T"
    );

    assert!(h.narrow_validity(at(400), Some("earlier contradiction".into())));
    assert!(
        !h.is_valid_at(mid),
        "after the correction the claim must be false at T"
    );

    // Both steps are in the record: the correction is replayable, which is the
    // part graphiti does not keep.
    let narrowings = h
        .revision_history
        .iter()
        .filter(|r| r.description.contains("Validity"))
        .count();
    assert_eq!(narrowings, 2, "both boundary changes must be replayable");
}
