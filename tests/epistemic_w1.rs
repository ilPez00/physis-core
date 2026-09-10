//! W1 resource-integration tests (PLAN.md Phase 24, wave 1).
//!
//! - G4: `citations_resolve_to_intake_ids` — per-link intake ids stamp every
//!   evidence/provenance link and every audit event; an explanation's
//!   citations resolve to the concrete ingest episodes that produced them.
//! - G6: `point_in_time_matches_audit_trail` — the point-in-time surface
//!   honours the validity window including the `expired_at` system leg;
//!   history keeps what the surface no longer exposes.
//! - T3: `evidence_retraction_replays_to_state_without_it` — retracting a
//!   source re-derives the hypothesis without that evidence and then
//!   re-evaluates DependsOn dependents in the shadow frame (the JTMS
//!   Tell/Retract cascade).
//! - T4: `nixon_diamond_retains_both_sides` — the ATMS Nixon Diamond seed
//!   (`rules_nixon`): p and ¬p coexist with their evidence; a later verdict
//!   disprefers, never destroys, the losing side.
//!
//! Sources read, never depended on: `~/dev/ATMS-in-Python/atms_algo.py`
//! and the JTMS/ATMS lineage — see physis-pro/docs/resources.md.

use chrono::TimeZone;
use physis_core::coherence_dimensions::CoherenceProfile;
use physis_core::contradiction::{Contradiction, ContradictionParty, ResolutionStatus};
use physis_core::delta_engine::{retract_evidence_with_cascade, EvaluationContext};
use physis_core::epistemic::{EpistemicAuditTrail, EpistemicEvent, EpistemicEventType};
use physis_core::explanation::ExplanationReport;
use physis_core::hypothesis::{Evidence, Hypothesis, HypothesisStatus};
use physis_core::provenance::{ProvenanceChain, ProvenanceLink};
use physis_core::relation::{RelationType, TypedEdge};
use physis_core::temporal::TemporalValidity;
use physis_core::CoherenceNode;

fn at(secs: i64) -> chrono::DateTime<chrono::Utc> {
    chrono::Utc.timestamp_opt(secs, 0).single().unwrap()
}

// ── G4: per-link intake ids ───────────────────────────────────────────────

#[test]
fn citations_resolve_to_intake_ids() {
    let intake = "intk-7".to_string();
    let mut trail = EpistemicAuditTrail::default();
    trail.record(
        EpistemicEvent::new(
            EpistemicEventType::ObservationIngested,
            "obs-7",
            "intake of the maintenance log",
        )
        .with_intake_id(intake.clone()),
    );

    // Both evidence links were produced by the same intake episode.
    let ev_support = Evidence::supports("maintenance_log.txt", "bearing temperature rising")
        .with_intake_id(intake.clone());
    let ev_contra = Evidence::contradicts("shift_report.txt", "vibration within band")
        .with_intake_id(intake.clone());

    let mut h = Hypothesis::new("bearing is degrading", vec![0.25; 8]);
    h.add_supporting_evidence(ev_support);
    h.add_contradicting_evidence(ev_contra);

    let report = ExplanationReport {
        subject_id: h.id.clone(),
        statement: h.statement.clone(),
        status: h.status,
        supporting_evidence: h.supporting_evidence.clone(),
        contradicting_evidence: h.contradicting_evidence.clone(),
        historical_precedents: Vec::new(),
        expected_consequences: Vec::new(),
        observed_consequences: Vec::new(),
        coherence_profile: CoherenceProfile::new(),
        coherence_score: h.coherence,
        fitness_breakdown: h.fitness_breakdown.clone(),
        fitness_score: h.fitness,
        confidence: h.confidence,
        provenance_chain: ProvenanceChain::new(),
        human_readable_summary: "".to_string(),
    };

    // The explanation cites the one intake episode, deduped, and every
    // citation resolves to a trail event carrying the same intake id.
    let cited = report.cited_intake_ids();
    assert_eq!(cited.len(), 1, "both links share one intake episode");
    for id in &cited {
        assert!(
            trail.events.iter().any(|e| e.intake_id == Option::Some(id.clone())),
            "citation {id:?} must resolve to a trail event carrying that intake id"
        );
    }

    // Provenance links carry the per-link intake id on their own.
    let mut chain = ProvenanceChain::new();
    chain.add_link(
        ProvenanceLink::new("bearing is degrading", "maintenance_log.txt").with_intake_id(intake.clone()),
    );
    assert_eq!(chain.cited_intake_ids(), vec![intake.clone()]);
}

// ── G6: point-in-time queries ─────────────────────────────────────────────

#[test]
fn point_in_time_matches_audit_trail() {
    let subj = "h1";
    let mut trail = EpistemicAuditTrail::default();
    trail.record(
        EpistemicEvent::new(EpistemicEventType::HypothesisGenerated, subj, "formed")
            .with_asserted_at(at(0)),
    );
    trail.record(
        EpistemicEvent::new(EpistemicEventType::StatusTransition, subj, "to supported")
            .with_transition("candidate", "supported")
            .with_asserted_at(at(100)),
    );
    trail.record(
        EpistemicEvent::new(EpistemicEventType::StatusTransition, subj, "to contradicted")
            .with_transition("supported", "contradicted")
            .with_asserted_at(at(200)),
    );
    trail.record(
        EpistemicEvent::new(EpistemicEventType::StatusTransition, subj, "to supported")
            .with_transition("contradicted", "supported")
            .with_asserted_at(at(300)),
    );

    let mut validity = TemporalValidity::during(at(100), at(400));
    validity.expired_at = Some(at(500));

    // Inside the validity window the point-in-time query *is* the audit
    // trail — it must never disagree with the replay.
    assert_eq!(
        trail.point_in_time_status_at(subj, at(150), &validity),
        Some(HypothesisStatus::Supported)
    );
    assert_eq!(
        trail.point_in_time_status_at(subj, at(250), &validity),
        Some(HypothesisStatus::Contradicted)
    );

    // Outside the window the surface says None while history keeps the
    // state — invalidate-don't-delete.
    assert_eq!(
        trail.point_in_time_status_at(subj, at(50), &validity),
        None,
        "not yet valid at t=50"
    );
    assert_eq!(
        trail.point_in_time_status_at(subj, at(450), &validity),
        None,
        "past valid_until at t=450"
    );
    assert_eq!(
        trail.reconstruct_status_at(subj, at(450)),
        Some(HypothesisStatus::Supported),
        "history keeps the record"
    );

    // The expired_at system leg closes the surface at t=550 even though the
    // claim's own interval said it was still valid.
    assert_eq!(
        trail.point_in_time_status_at(subj, at(550), &validity),
        None,
        "superseded at t=550"
    );
    assert_eq!(
        trail.reconstruct_status_at(subj, at(550)),
        Some(HypothesisStatus::Supported),
        "the audit trail still replays it"
    );

    // The hypothesis-level helper agrees with the temporal legs.
    let mut h = Hypothesis::new("h", vec![0.5; 4]);
    h.temporal = validity;
    assert!(h.is_valid_at(at(150)));
    assert!(h.is_valid_at(at(350)));
    assert!(!h.is_valid_at(at(550)), "expired at the system leg");
}

// ── T3: JTMS retraction cascade ───────────────────────────────────────────

#[test]
fn evidence_retraction_replays_to_state_without_it() {
    // Level 1 — the hypothesis re-derives, evidence by evidence.
    let mut h = Hypothesis::new("machine degradation", vec![0.5; 4]);
    h.add_supporting_evidence(
        Evidence::supports("s1", "vibration harmonics").with_weight(0.9),
    );
    h.add_supporting_evidence(
        Evidence::supports("s2", "temperature rise").with_weight(0.7),
    );
    h.add_contradicting_evidence(
        Evidence::contradicts("s3", "sensor wiring fault explains the signal").with_weight(0.8),
    );
    assert_eq!(h.status, HypothesisStatus::Contradicted);

    assert_eq!(h.retract_evidence("s3"), 1);
    assert_eq!(h.status, HypothesisStatus::Supported, "retracting the contradiction re-derives Supported");
    assert_eq!(h.retract_evidence("s1"), 1);
    assert_eq!(h.status, HypothesisStatus::Supported, "one supporting source still stands");
    assert_eq!(h.retract_evidence("s2"), 1);
    assert_eq!(h.status, HypothesisStatus::Candidate, "no evidence left");
    assert!(h.supporting_evidence.is_empty());
    assert!(h.contradicting_evidence.is_empty());
    assert_eq!(h.retract_evidence("s1"), 0, "nothing left to retract");

    // Retraction is a revision: it shows in the history, it is not a delete.
    assert!(
        h.revision_history.iter().any(|r| r.description.contains("Retracted")),
        "the retraction must be audited"
    );

    // A Certified status survives an evidence removal (the authority leg
    // forbids telemetry from overturning a human/process verdict; the
    // authority registry itself is T11).
    let mut certified = Hypothesis::new("certified", vec![0.5; 4]);
    certified.add_supporting_evidence(Evidence::supports("s1", "verified by authority"));
    certified.add_contradicting_evidence(Evidence::contradicts("s3", "contradicts"));
    certified.status = HypothesisStatus::Certified;
    assert_eq!(certified.retract_evidence("s3"), 1);
    assert_eq!(certified.status, HypothesisStatus::Certified);

    // Level 2 — the JTMS cascade: hB loses its justification, so hA (the
    // declared DependsOn dependent) is re-evaluated in the shadow frame.
    let mut node_a = CoherenceNode::new(vec![1.0, 0.0, 0.0, 0.0]);
    node_a.id = "nA".into();
    let mut node_b = CoherenceNode::new(vec![0.0, 1.0, 0.0, 0.0]);
    node_b.id = "nB".into();
    let nodes = vec![node_a, node_b];

    let mut h_a = Hypothesis::new("A depends on B", vec![1.0, 0.0, 0.0, 0.0]);
    h_a.id = "hA".into();
    h_a.ontology_refs = vec!["nA".into()];
    h_a.status = HypothesisStatus::Supported;
    h_a.add_supporting_evidence(Evidence::supports("derived", "inferred from B"));

    let mut h_b = Hypothesis::new("B observed", vec![0.0, 1.0, 0.0, 0.0]);
    h_b.id = "hB".into();
    h_b.ontology_refs = vec!["nB".into()];
    h_b.status = HypothesisStatus::Supported;
    h_b.add_supporting_evidence(
        Evidence::supports("src-x", "direct observation").with_weight(0.9),
    );

    let hyps = vec![h_a, h_b];
    let edges = vec![TypedEdge::new(RelationType::DependsOn, "nA", "nB")];

    let mut ctx = EvaluationContext::from_base(&nodes, &hyps, &edges);
    let report = retract_evidence_with_cascade(&mut ctx, "hB", "src-x").expect("hB exists");
    assert_eq!(report.removed, 1);
    assert_eq!(
        report.dependents_revised,
        vec!["hA".to_string()],
        "declared DependsOn dependents only"
    );

    // The shadow of hB no longer carries the retracted source and its
    // status was re-derived; hA was shadowed and re-evaluated.
    let hb_shadow = ctx.shadow_hypotheses.get("hB").expect("hB is shadowed");
    assert!(
        hb_shadow.supporting_evidence.iter().all(|e| e.source != "src-x"),
        "the retracted evidence is gone from the shadow"
    );
    assert_eq!(hb_shadow.supporting_evidence.len(), 0);
    assert_eq!(hb_shadow.status, HypothesisStatus::Candidate);
    assert!(ctx.shadow_hypotheses.contains_key("hA"), "hA must be re-evaluated in shadow");

    // Base state is untouched — the retraction is a proposal until committed.
    assert_eq!(hyps[1].supporting_evidence.len(), 1);
    assert_eq!(hyps[1].supporting_evidence[0].source, "src-x");
}

// ── T4: Nixon Diamond benchmark seed ──────────────────────────────────────

#[test]
fn nixon_diamond_retains_both_sides() {
    // ATMS-in-Python `rules_nixon` scaled down: Quaker implies pacifist (p),
    // Republican implies ¬pacifist (np); `{p, np} → ⊥`. The ATMS answers ⊥
    // with environment labels; Physis's answer is a retained, open
    // contradiction with both sides audited and queryable.
    let mut h_p = Hypothesis::new("Nixon is a pacifist", vec![1.0, 0.0, 0.0, 0.0]);
    h_p.id = "h_p".into();
    h_p.add_supporting_evidence(
        Evidence::supports("quaker_census.txt", "Nixon is a Quaker; Quakers are pacifists")
            .with_weight(0.8),
    );
    h_p.add_assumption("Quakers are pacifists");

    let mut h_np = Hypothesis::new("Nixon is not a pacifist", vec![0.0, 1.0, 0.0, 0.0]);
    h_np.id = "h_np".into();
    h_np.add_supporting_evidence(
        Evidence::supports(
            "republican_roster.txt",
            "Nixon is a Republican; Republicans are not pacifists",
        )
        .with_weight(0.8),
    );
    h_np.add_assumption("Republicans are not pacifists");

    // Both sides are supported: the diamond did not collapse one into the
    // other, and neither side's evidence was discarded.
    assert_eq!(h_p.status, HypothesisStatus::Supported);
    assert_eq!(h_np.status, HypothesisStatus::Supported);
    assert_eq!(h_p.supporting_evidence.len(), 1);
    assert_eq!(h_np.supporting_evidence.len(), 1);

    // The contradiction is a first-class, retained object, resolution open.
    let contradiction = Contradiction::new(
        ContradictionParty::new("Nixon is a pacifist", "quaker_census.txt"),
        ContradictionParty::new("Nixon is not a pacifist", "republican_roster.txt"),
    );
    assert_eq!(contradiction.resolution, ResolutionStatus::Open);
    assert_eq!(contradiction.claim_a.source, "quaker_census.txt");
    assert_eq!(contradiction.claim_b.source, "republican_roster.txt");

    // Both sides leave a simultaneous replayable audit footprint.
    let mut trail = EpistemicAuditTrail::default();
    trail.record(
        EpistemicEvent::new(EpistemicEventType::HypothesisGenerated, "h_p", "formed")
            .with_asserted_at(at(0)),
    );
    trail.record(
        EpistemicEvent::new(EpistemicEventType::HypothesisGenerated, "h_np", "formed")
            .with_asserted_at(at(0)),
    );
    assert_eq!(
        trail.reconstruct_status_at("h_p", at(50)),
        Some(HypothesisStatus::Candidate)
    );
    assert_eq!(
        trail.reconstruct_status_at("h_np", at(50)),
        Some(HypothesisStatus::Candidate)
    );

    // A later human verdict (the rater decides; standing rules 6/7)
    // disprefers np. The record is not destroyed: the losing hypothesis
    // drops to Candidate with the retraction audited, the winner stands,
    // and the contradiction object still names both parties.
    let removed = h_np.retract_evidence("republican_roster.txt");
    assert_eq!(removed, 1);
    assert_eq!(h_np.status, HypothesisStatus::Candidate);
    assert_eq!(h_p.status, HypothesisStatus::Supported);
    assert!(
        h_np.revision_history.iter().any(|r| r.description.contains("Retracted")),
        "the retraction is an audited revision, not a silent deletion"
    );
    assert_eq!(contradiction.claim_a.claim, "Nixon is a pacifist");
    assert_eq!(contradiction.claim_b.claim, "Nixon is not a pacifist");
    assert_eq!(contradiction.resolution, ResolutionStatus::Open);
}