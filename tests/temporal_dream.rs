//! P2 + 23.6 tests — the temporal dream over the time machine.
//!
//! - 23.6: `dream_proposes_reactivation_from_retained_branches`,
//!   `dream_proposes_restoring_severed_connections`,
//!   `dream_proposes_retiring_repeatedly_contradicted_connections`,
//!   `dream_never_writes` — proposals cite real historical ids; the trail
//!   is byte-identical after a dream.

use chrono::TimeZone;
use physis_core::delta_engine::{MutationOp, OntologyMutation};
use physis_core::dream::{dream_over_history, ProposalKind};
use physis_core::epistemic::{EpistemicAuditTrail, EpistemicEvent, EpistemicEventType};
use physis_core::hypothesis::HypothesisStatus;

fn at(secs: i64) -> chrono::DateTime<chrono::Utc> {
    chrono::Utc.timestamp_opt(secs, 0).single().unwrap()
}

fn event(kind: EpistemicEventType, subject: &str, asserted: i64) -> EpistemicEvent {
    let mut ev = EpistemicEvent::new(kind, subject, "dream fixture");
    ev.asserted_at = Some(at(asserted));
    ev.timestamp = at(asserted + 5); // arrival after assertion
    ev
}

fn transition(subject: &str, asserted: i64, posterior: &str) -> EpistemicEvent {
    event(EpistemicEventType::StatusTransition, subject, asserted)
        .with_transition("prior", posterior)
}

#[test]
fn dream_proposes_reactivation_from_retained_branches() {
    // h_old was superseded at t200; at t800 an observation re-presents the
    // pattern. G7 kept the branch, so the dream may re-propose it.
    let mut trail = EpistemicAuditTrail::default();
    trail.record(event(EpistemicEventType::HypothesisGenerated, "h_old", 0));
    trail.record(transition("h_old", 200, "superseded"));
    trail.record(event(
        EpistemicEventType::ObservationIngested,
        "h_old",
        800,
    ));

    let proposals = dream_over_history(&trail, &[], chrono::Duration::seconds(1000));
    assert_eq!(proposals.len(), 1);
    let p = &proposals[0];
    assert_eq!(p.subject_id, "h_old");
    assert_eq!(p.kind, ProposalKind::ReactivateHypothesis);
    assert_eq!(p.anchor_time, at(200));
    assert!(p.rationale.contains("retained"), "the rationale cites G7");

    // The citation resolves to the real re-presenting event.
    assert_eq!(p.citing.len(), 1);
    assert!(trail.events.iter().any(|e| e.id == p.citing[0]));
}

#[test]
fn dream_proposes_restoring_severed_connections() {
    let mut mutation = OntologyMutation::new(
        "n0",
        MutationOp::RelationSevered {
            target_id: "n1".to_string(),
        },
    );
    mutation.timestamp = at(300);

    let mut trail = EpistemicAuditTrail::default();
    // n1 re-confirms AFTER the sever: the edge may belong back.
    trail.record(transition("n1", 500, "supported"));

    let proposals = dream_over_history(&trail, &[mutation], chrono::Duration::seconds(1000));
    assert_eq!(proposals.len(), 1);
    let p = &proposals[0];
    assert_eq!(p.kind, ProposalKind::RestoreConnection { to: "n1".to_string() });
    assert!(p.rationale.contains("re-confirmed"));
}

#[test]
fn dream_stays_silent_without_reconfirmation() {
    // Severed, but the target never re-confirmed: no proposal — the dream
    // does not invent reasons to reconnect.
    let mut mutation = OntologyMutation::new(
        "n0",
        MutationOp::RelationSevered {
            target_id: "n1".to_string(),
        },
    );
    mutation.timestamp = at(300);

    let mut trail = EpistemicAuditTrail::default();
    trail.record(transition("n1", 500, "contradicted"));

    let proposals = dream_over_history(&trail, &[mutation], chrono::Duration::seconds(1000));
    assert!(proposals.is_empty());
}

#[test]
fn dream_proposes_retiring_repeatedly_contradicted_connections() {
    let mut mutation = OntologyMutation::new(
        "n0",
        MutationOp::RelationCreated {
            target_id: "n1".to_string(),
            weight: 1.0,
        },
    );
    mutation.timestamp = at(100);

    let mut trail = EpistemicAuditTrail::default();
    trail.record(event(
        EpistemicEventType::ContradictionDetected,
        "n1",
        200,
    ));
    trail.record(event(
        EpistemicEventType::ContradictionDetected,
        "n1",
        400,
    ));

    let proposals = dream_over_history(&trail, &[mutation], chrono::Duration::seconds(1000));
    assert_eq!(proposals.len(), 1);
    let p = &proposals[0];
    assert_eq!(p.kind, ProposalKind::RetireConnection { to: "n1".to_string() });
    assert!(p.rationale.contains("2 contradiction"));
}

#[test]
fn dream_never_writes() {
    // The dream is a read over history: the trail and the mutation log are
    // byte-identical afterwards (enforced at compile time by &; asserted
    // here at runtime), and every citation resolves to a real record.
    let mut trail = EpistemicAuditTrail::default();
    trail.record(event(EpistemicEventType::HypothesisGenerated, "h_old", 0));
    trail.record(transition("h_old", 200, "superseded"));
    trail.record(event(
        EpistemicEventType::ObservationIngested,
        "h_old",
        800,
    ));

    let mut sever = OntologyMutation::new(
        "n0",
        MutationOp::RelationSevered {
            target_id: "n1".to_string(),
        },
    );
    sever.timestamp = at(300);
    trail.record(transition("n1", 500, "supported"));
    let mutations = vec![sever];

    let events_before = serde_json::to_string(&trail.events).unwrap();
    let mutations_before = serde_json::to_string(&mutations).unwrap();

    let proposals =
        dream_over_history(&trail, &mutations, chrono::Duration::seconds(10_000));

    assert_eq!(
        serde_json::to_string(&trail.events).unwrap(),
        events_before,
        "the trail is never mutated by a dream"
    );
    assert_eq!(
        serde_json::to_string(&mutations).unwrap(),
        mutations_before,
        "the mutation log is never mutated by a dream"
    );
    assert_eq!(trail.watermarks.len(), 0);

    // Every citation resolves to a real event or mutation id.
    let event_ids: Vec<&str> = trail.events.iter().map(|e| e.id.as_str()).collect();
    let mutation_ids: Vec<&str> = mutations.iter().map(|m| m.id.as_str()).collect();
    for p in &proposals {
        assert!(!p.citing.is_empty(), "proposals must cite their history");
        for id in &p.citing {
            assert!(
                event_ids.contains(&id.as_str()) || mutation_ids.contains(&id.as_str()),
                "citation {id} must resolve to a real record"
            );
        }
    }
    assert!(!proposals.is_empty());
    assert!(proposals
        .iter()
        .any(|p| p.kind == ProposalKind::ReactivateHypothesis));
    assert!(proposals
        .iter()
        .any(|p| matches!(p.kind, ProposalKind::RestoreConnection { .. })));
    // And nothing was applied: h_old is still retired in the trail's own
    // replay — the dream proposes, the caller decides.
    assert_eq!(
        trail.reconstruct_status_at("h_old", at(1000)),
        Some(HypothesisStatus::Superseded)
    );
}
