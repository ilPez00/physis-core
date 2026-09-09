//! P1 revision-correctness tests (ATLAS_GRAPHITI_INTEGRATION_PLAN.md §4).
//!
//! - A2: `midchain_revision_revises_exact_dependents` — hypothesis revision
//!   is selected by the declared DependsOn closure (BFS order, exact
//!   dependents), not by the breadth wave; cycles terminate the walk and
//!   are recorded; a same-cell bystander is never revised.
//! - A6: `fitness_recompute_reports_term_breakdown` — every recompute names
//!   its terms, contributions sum to the composite where the clamp does not
//!   bite, and the published weights stay frozen.

use physis_core::delta_engine::{
    evaluate_mutation, EvaluationContext, MutationOp, OntologyMutation,
};
use physis_core::hypothesis::{
    Evidence, Hypothesis, HypothesisStatus, CONTRADICTION_PENALTY_PER_ITEM,
    FAILED_PREDICTION_PENALTY_PER_ITEM, FITNESS_WEIGHT_EMPIRICAL_SUPPORT,
    FITNESS_WEIGHT_LOGICAL_CONSISTENCY, FITNESS_WEIGHT_ONTOLOGICAL_FIT,
    FITNESS_WEIGHT_PREDICTIVE_SUCCESS, FITNESS_WEIGHT_SEMANTIC_FIT,
};
use physis_core::relation::{RelationType, TypedEdge};
use physis_core::{CoherenceNode, FitnessBreakdown};

fn unit(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; 4];
    v[axis] = 1.0;
    v
}

fn node(id: &str, embedding: Vec<f32>) -> CoherenceNode {
    let mut n = CoherenceNode::new(embedding);
    n.id = id.to_string();
    n
}

fn depends_on(source: &str, target: &str) -> TypedEdge {
    TypedEdge::new(RelationType::DependsOn, source, target)
}

fn hypothesis(id: &str, statement: &str, node_id: &str, embedding: Vec<f32>) -> Hypothesis {
    let mut h = Hypothesis::new(statement, embedding);
    h.id = id.to_string();
    h.ontology_refs = vec![node_id.to_string()];
    h.status = HypothesisStatus::Supported;
    h
}

// ── A2: DependsOn-only revision walk ─────────────────────────────────────

#[test]
fn midchain_revision_revises_exact_dependents() {
    // Chain: nA DependsOn nB, nB DependsOn nC. nX shares nC's cell pin and is
    // cosine-identical to it — under the old breadth wave a mere same-cell
    // arrival pulled hX into the revision set; under A2 only declared
    // dependencies justify a revision.
    let mut nodes = vec![
        node("nA", unit(0)),
        node("nB", unit(1)),
        node("nC", unit(2)),
        node("nX", unit(2)),
    ];
    let pin = Some(("test".to_string(), "cell".to_string()));
    for n in nodes.iter_mut() {
        if n.id == "nC" || n.id == "nX" {
            n.cell_pin = pin.clone();
        }
    }

    let h_x = hypothesis("hX", "X is a bystander", "nX", unit(3));
    let fitness_x_before = h_x.fitness;

    let hyps = vec![
        hypothesis("hA", "A depends on B", "nA", unit(0)),
        hypothesis("hB", "B depends on C", "nB", unit(1)),
        hypothesis("hC", "C is observed", "nC", unit(2)),
        h_x,
    ];
    let edges = vec![depends_on("nA", "nB"), depends_on("nB", "nC")];

    let mut ctx = EvaluationContext::from_base(&nodes, &hyps, &edges);
    let mutation = OntologyMutation::new(
        "nC",
        MutationOp::EmbeddingShift {
            old_embedding: unit(2),
            new_embedding: vec![0.6, 0.8, 0.0, 0.0], // a real, non-zero shift
        },
    );
    let report = evaluate_mutation(&mut ctx, mutation);

    let walk = report.revision_walk.expect("the walk must be attached");
    assert!(
        !walk.fallback_breadth_used,
        "declared dependents exist; the fallback must not fire"
    );
    assert!(walk.cycles.is_empty(), "a chain has no cycles");
    assert!(!walk.truncated);

    // BFS order: the mutated node's hypotheses (hops 0), then hop 1 (nB),
    // then hop 2 (nA). Exactly the dependents, exactly once each.
    assert_eq!(
        walk.revised_hypotheses,
        vec!["hC".to_string(), "hB".to_string(), "hA".to_string()]
    );

    // The bystander never enters the shadow frame — not revised, not moved.
    assert!(
        !ctx.shadow_hypotheses.contains_key("hX"),
        "bystander hX must not be revised by the wave"
    );
    assert_eq!(ctx.shadow_hypotheses.len(), 3);

    // Base state is untouched (shadow-frame isolation).
    assert_eq!(
        hyps[3].fitness.to_bits(),
        fitness_x_before.to_bits(),
        "base hX fitness must be bit-identical"
    );
}

#[test]
fn depends_on_walk_records_cycles_and_terminates() {
    // 2-cycle: nX DependsOn nY, nY DependsOn nX.
    let nodes = vec![node("nX", unit(0)), node("nY", unit(1))];
    let hyps = vec![
        hypothesis("hX", "X", "nX", unit(0)),
        hypothesis("hY", "Y", "nY", unit(1)),
    ];
    let edges = vec![depends_on("nX", "nY"), depends_on("nY", "nX")];

    let mut ctx = EvaluationContext::from_base(&nodes, &hyps, &edges);
    let mutation = OntologyMutation::new(
        "nX",
        MutationOp::EmbeddingShift {
            old_embedding: unit(0),
            new_embedding: vec![0.0, 0.0, 1.0, 0.0],
        },
    );
    let report = evaluate_mutation(&mut ctx, mutation);

    let walk = report.revision_walk.expect("the walk must be attached");
    // The cycle is recorded (never re-enqueued), the walk terminates, and
    // each node is visited exactly once.
    assert_eq!(walk.cycles.len(), 1);
    assert_eq!(walk.cycles[0], ["nY".to_string(), "nX".to_string()]);
    assert_eq!(walk.steps.len(), 2);
    assert!(!walk.truncated);
    // BFS order: the mutated node first, then its dependent.
    assert_eq!(
        walk.revised_hypotheses,
        vec!["hX".to_string(), "hY".to_string()]
    );
}


// ── A6: named weights + per-term breakdown ───────────────────────────────

#[test]
fn fitness_recompute_reports_term_breakdown() {
    let mut h = hypothesis("h1", "well supported", "n1", unit(0));
    h.fitness_breakdown = FitnessBreakdown {
        semantic_fit: 0.8,
        ontological_fit: 0.7,
        logical_consistency: 1.0,
        empirical_support: 0.5,
        predictive_success: 0.5,
        outcome_success: 0.5,
        contradiction_penalty: 0.0,
        failed_prediction_penalty: 0.0,
        composite_fitness: 0.0,
    };
    h.add_supporting_evidence(Evidence::supports("obs-1", "it worked"));
    h.recompute_fitness();

    let terms = h.fitness_term_breakdown();
    let names: Vec<&str> = terms.iter().map(|(n, _)| *n).collect();
    assert_eq!(
        names,
        vec![
            "semantic_fit",
            "ontological_fit",
            "logical_consistency",
            "empirical_support",
            "predictive_success",
            "contradiction_penalty",
            "failed_prediction_penalty",
        ]
    );

    // The contributions sum to the composite: the clamp is the only
    // nonlinearity and it does not bite for a supported hypothesis.
    let sum: f32 = terms.iter().map(|(_, v)| v).sum();
    assert!(
        (sum - h.fitness).abs() < 1e-6,
        "term contributions sum {sum} != composite fitness {}",
        h.fitness
    );

    // No penalties on this hypothesis: the penalty terms are present, zero.
    assert!(
        terms
            .iter()
            .all(|(n, v)| !n.contains("penalty") || *v == 0.0),
        "penalty terms must be named and zero here"
    );

    // The published weights are frozen (A6 gate: tuning is a separate
    // scored config, never an in-place edit).
    assert_eq!(FITNESS_WEIGHT_SEMANTIC_FIT, 0.20);
    assert_eq!(FITNESS_WEIGHT_ONTOLOGICAL_FIT, 0.15);
    assert_eq!(FITNESS_WEIGHT_LOGICAL_CONSISTENCY, 0.15);
    assert_eq!(FITNESS_WEIGHT_EMPIRICAL_SUPPORT, 0.25);
    assert_eq!(FITNESS_WEIGHT_PREDICTIVE_SUCCESS, 0.25);
    assert_eq!(CONTRADICTION_PENALTY_PER_ITEM, 0.10);
    assert_eq!(FAILED_PREDICTION_PENALTY_PER_ITEM, 0.15);
}
