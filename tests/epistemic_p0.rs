//! P0 invariant + history tests (ATLAS_GRAPHITI_INTEGRATION_PLAN.md §4).
//!
//! - A7: zero-shift wave leaves fitness bit-identical, zero transitions.
//! - G7: superseded items stay history-readable, excluded from current queries.
//! - G1: temporal triple (incl. `expired_at`) serialises, back-compat.

use physis_core::delta_engine::{
    evaluate_mutation, EvaluationContext, MutationOp, OntologyMutation,
};
use physis_core::hypothesis::{current_hypotheses, hypotheses_including_history};
use physis_core::{CoherenceNode, Hypothesis, HypothesisStatus, TemporalValidity};

fn unit(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; 4];
    v[axis] = 1.0;
    v
}

// ── A7: No-perturbation invariant ─────────────────────────────────────────

#[test]
fn zero_shift_wave_leaves_fitness_untouched() {
    let mut n = CoherenceNode::new(unit(0));
    n.id = "n0".to_string();

    let mut h = Hypothesis::new("zero shift keeps fitness", unit(0));
    h.ontology_refs = vec!["n0".to_string()];
    h.status = HypothesisStatus::Supported;

    let fitness_before = h.fitness;
    let coherence_before = h.coherence;

    let nodes = vec![n];
    let hyps = vec![h];
    let edges = vec![];

    let mut ctx = EvaluationContext::from_base(&nodes, &hyps, &edges);
    let emb = unit(0);
    let mutation = OntologyMutation::new(
        "n0",
        MutationOp::EmbeddingShift {
            old_embedding: emb.clone(),
            new_embedding: emb,
        },
    );
    let report = evaluate_mutation(&mut ctx, mutation);

    // Zero HypothesisTransition entries, hence zero StatusTransition events
    // derivable on commit (transitions are built 1:1 from this list).
    assert!(
        report.hypothesis_status_shifts.is_empty(),
        "zero-shift wave must emit no transitions, got {:?}",
        report.hypothesis_status_shifts
    );

    // Base hypotheses are never touched (shadow-frame isolation) …
    assert_eq!(
        hyps[0].fitness.to_bits(),
        fitness_before.to_bits(),
        "base fitness must be bit-identical"
    );
    // … and any shadow copy must carry bit-identical fitness too.
    for (id, shadow) in &ctx.shadow_hypotheses {
        let base = hyps.iter().find(|b| &b.id == id).unwrap();
        assert_eq!(
            shadow.fitness.to_bits(),
            base.fitness.to_bits(),
            "shadow fitness for {id} moved on a zero shift"
        );
        assert_eq!(
            shadow.coherence.to_bits(),
            coherence_before.to_bits(),
            "shadow coherence for {id} moved on a zero shift"
        );
    }
}

// ── G7: Invalidate-don't-delete ──────────────────────────────────────────

#[test]
fn superseded_items_stay_queryable_for_history() {
    let mut map = std::collections::HashMap::new();
    let mut h1 = Hypothesis::new("live belief", unit(0));
    h1.id = "h1".to_string();
    h1.status = HypothesisStatus::Supported;
    let mut h2 = Hypothesis::new("superseded belief", unit(0));
    h2.id = "h2".to_string();
    h2.status = HypothesisStatus::Superseded;
    map.insert("h1".to_string(), h1);
    map.insert("h2".to_string(), h2);

    let live = current_hypotheses(&map);
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].id, "h1");

    let all = hypotheses_including_history(&map);
    assert_eq!(all.len(), 2);
}

// ── G1: Temporal triple serialisation ────────────────────────────────────

#[test]
fn temporal_triple_serialises() {
    let mut tv = TemporalValidity::permanent();
    tv.expired_at = Some(chrono::Utc::now());
    let json = serde_json::to_string(&tv).expect("serialise");
    assert!(json.contains("expired_at"));
    let back: TemporalValidity = serde_json::from_str(&json).expect("deserialise");
    assert_eq!(back.expired_at, tv.expired_at);
}
