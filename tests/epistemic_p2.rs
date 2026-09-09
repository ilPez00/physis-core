//! P2 tests (ATLAS_GRAPHITI_INTEGRATION_PLAN.md §4 P2) — starting with G3.
//!
//! - G3: `late_evidence_replays_to_same_state` — an out-of-order intake
//!   replays to the in-order state; per-stream watermarks flag late
//!   episodes and carry both clocks (the T18 two-clock take); legacy
//!   trails without `asserted_at` replay unchanged.

use chrono::TimeZone;
use physis_core::epistemic::{
    EpistemicAuditTrail, EpistemicEvent, EpistemicEventType, IntakeReceipt,
};
use physis_core::hypothesis::HypothesisStatus;

fn at(secs: i64) -> chrono::DateTime<chrono::Utc> {
    chrono::Utc.timestamp_opt(secs, 0).single().unwrap()
}

fn formed(subject: &str, asserted: i64, arrival: i64) -> EpistemicEvent {
    let mut ev = EpistemicEvent::new(
        EpistemicEventType::HypothesisGenerated,
        subject,
        "belief formed",
    );
    ev.asserted_at = Some(at(asserted));
    ev.timestamp = at(arrival);
    ev
}

fn transition(subject: &str, asserted: i64, arrival: i64, posterior: &str) -> EpistemicEvent {
    let mut ev = EpistemicEvent::new(
        EpistemicEventType::StatusTransition,
        subject,
        format!("transition to {posterior}"),
    );
    ev.asserted_at = Some(at(asserted));
    ev.timestamp = at(arrival);
    ev = ev.with_transition("prior", posterior);
    ev
}

// ── G3: assertion-ordered replay ─────────────────────────────────────────

#[test]
fn late_evidence_replays_to_same_state() {
    // The same three assertions. The in-order stream arrives in assertion
    // order; the late stream has the *newest* episode arrive first and the
    // older two backfilled afterwards.
    let subj = "h1";

    let in_order = EpistemicAuditTrail {
        events: vec![
            formed(subj, 0, 10),
            transition(subj, 200, 210, "contradicted"),
            transition(subj, 400, 410, "supported"),
        ],
        watermarks: vec![],
    };

    let mut late = EpistemicAuditTrail::default();
    late.record(transition(subj, 400, 900, "supported")); // arrives first
    late.record(formed(subj, 0, 910)); // backfilled
    late.record(transition(subj, 200, 920, "contradicted")); // backfilled late

    // Every replay time agrees, whatever the arrival order was.
    for t in [0i64, 100, 200, 300, 400, 500, 900] {
        let a = in_order.reconstruct_status_at(subj, at(t));
        let b = late.reconstruct_status_at(subj, at(t));
        assert_eq!(a, b, "replay at t={t} must not depend on arrival order");
    }

    // And the states themselves are the expected assertion-time sequence.
    assert_eq!(
        late.reconstruct_status_at(subj, at(100)),
        Some(HypothesisStatus::Candidate)
    );
    assert_eq!(
        late.reconstruct_status_at(subj, at(300)),
        Some(HypothesisStatus::Contradicted)
    );
    assert_eq!(
        late.reconstruct_status_at(subj, at(500)),
        Some(HypothesisStatus::Supported)
    );

    // Beyond the cutoff nothing is visible.
    assert_eq!(late.reconstruct_status_at(subj, at(-1)), None);
}

#[test]
fn watermark_advances_and_flags_late_episodes() {
    let mut trail = EpistemicAuditTrail::default();

    let r1: IntakeReceipt = trail.note_intake("stream-a", at(1000), Some(at(400)));
    assert!(!r1.late);
    assert_eq!(r1.mark.last_arrival, at(1000));
    assert_eq!(r1.mark.last_episode_valid_at, Some(at(400)));

    // A late episode: asserted t200 arrives after t400 was seen. The mark
    // does not move backwards on either clock.
    let r2 = trail.note_intake("stream-a", at(1100), Some(at(200)));
    assert!(r2.late);
    assert_eq!(r2.mark.last_episode_valid_at, Some(at(400)));
    assert_eq!(r2.mark.last_arrival, at(1100));

    // A newer episode advances the valid clock; not late.
    let r3 = trail.note_intake("stream-a", at(1200), Some(at(600)));
    assert!(!r3.late);
    assert_eq!(r3.mark.last_episode_valid_at, Some(at(600)));

    // An intake without an episode time advances arrival only.
    let r4 = trail.note_intake("stream-a", at(1300), None);
    assert!(!r4.late);
    assert_eq!(r4.mark.last_arrival, at(1300));
    assert_eq!(r4.mark.last_episode_valid_at, Some(at(600)));

    // Streams are independent; a fresh stream starts its own mark.
    let r5 = trail.note_intake("stream-b", at(50), Some(at(10)));
    assert!(!r5.late);
    assert_eq!(r5.mark.stream_id, "stream-b");
    assert_eq!(r5.mark.last_episode_valid_at, Some(at(10)));
}

#[test]
fn legacy_trails_replay_by_arrival_unchanged() {
    // Pre-G3 events carry no asserted_at: assertion time == arrival, so
    // replay is byte-for-byte the old behavior.
    let mut trail = EpistemicAuditTrail::default();
    trail.record(formed("h", 100, 100));
    trail.record(transition("h", 300, 300, "contradicted"));

    assert_eq!(
        trail.reconstruct_status_at("h", at(200)),
        Some(HypothesisStatus::Candidate)
    );
    assert_eq!(
        trail.reconstruct_status_at("h", at(400)),
        Some(HypothesisStatus::Contradicted)
    );
}

#[test]
fn trail_serialises_with_g3_fields_and_reads_old_json() {
    // New JSON round-trips both G3 additions...
    let mut trail = EpistemicAuditTrail::default();
    trail.record(formed("h", 100, 110));
    trail.note_intake("stream-a", at(110), Some(at(100)));
    let json = serde_json::to_string(&trail).expect("serialise");
    assert!(json.contains("asserted_at"));
    assert!(json.contains("watermarks"));
    let back: EpistemicAuditTrail = serde_json::from_str(&json).expect("deserialise");
    assert_eq!(back.watermarks.len(), 1);
    assert_eq!(back.events[0].assertion_time(), at(100));

    // ...and old JSON without the fields deserialises with defaults.
    let old = r#"{"events":[{"id":"e1","timestamp":"2026-09-09T00:00:00Z","event_type":"HypothesisGenerated","subject_id":"h","description":"formed"}]}"#;
    let legacy: EpistemicAuditTrail = serde_json::from_str(old).expect("legacy deserialise");
    assert!(legacy.events[0].asserted_at.is_none());
    assert!(legacy.watermarks.is_empty());
}
