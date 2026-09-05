//! Integration tests for the Dynamic Ontology Delta & Propagation Engine.

use physis_core::delta_engine::{
    evaluate_mutation, EvaluationContext, MutationOp, OntologyDeltaReport, OntologyMutation,
    MAX_PROPAGATION_DEPTH,
};
use physis_core::{
    CoherenceNode, Evidence, EvidencePolarity, Hypothesis, HypothesisStatus, RelationType,
    TypedEdge,
};

/// Build a 4-element unit vector pointing in the `axis` direction.
fn unit_vector(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; 4];
    if axis < v.len() {
        v[axis] = 1.0;
    }
    v
}

/// Convenience: create a labeled node with optional cell pin.
fn make_node(id: &str, embedding: Vec<f32>, coherence: f32, cell_pin: Option<(&str, &str)>) -> CoherenceNode {
    let mut n = CoherenceNode::new(embedding);
    n.id = id.to_string();
    n.coherence_score = coherence;
    n.label = Some(id.to_string());
    n.cell_pin = cell_pin.map(|(d, m)| (d.to_string(), m.to_string()));
    n
}

/// Convenience: build a simple chain graph of `n` nodes connected by edges.
fn chain_graph(count: usize) -> (Vec<CoherenceNode>, Vec<TypedEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    for i in 0..count {
        let emb = unit_vector(i % 4);
        nodes.push(make_node(&format!("n{i}"), emb, 0.5, None));
    }

    for i in 0..count - 1 {
        edges.push(TypedEdge::new(
            RelationType::Influences,
            format!("n{i}"),
            format!("n{}", i + 1),
        ));
    }

    (nodes, edges)
}

// ── Test 1: Single Node Mutation ─────────────────────────────────────────

#[test]
fn single_node_mutation_affects_radius() {
    // Build a chain of 6 nodes: n0 - n1 - n2 - n3 - n4 - n5
    let (nodes, edges) = chain_graph(6);

    let ctx = EvaluationContext::from_base(&nodes, &[], &edges);

    // Mutate n0's embedding
    let mutation = OntologyMutation::new(
        "n0",
        MutationOp::EmbeddingShift {
            old_embedding: unit_vector(0),
            new_embedding: vec![
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
                0.0,
                0.0,
            ],
        },
    );

    let mut ctx = ctx;
    let report = evaluate_mutation(&mut ctx, mutation);

    // The target node must be in the report
    assert!(
        report.affected_nodes.iter().any(|d| d.node_id == "n0"),
        "target node must be affected"
    );

    // Only nodes within MAX_PROPAGATION_DEPTH hops should appear
    let max_expected_depth = MAX_PROPAGATION_DEPTH;
    let max_affected = (max_expected_depth + 1).min(6); // +1 for the target itself
    assert!(
        report.affected_nodes.len() <= max_affected,
        "expected at most {} affected nodes, got {}",
        max_affected,
        report.affected_nodes.len()
    );

    // Nodes at depth 0 (the target) should have impact == source_sim * γ^0 = source_sim
    // Nodes at depth 1 should have impact == source_sim * γ^1, etc.
    // The target should always be first
    assert_eq!(report.affected_nodes[0].node_id, "n0");

    println!(
        "Single-node mutation: {} nodes affected out of 6 (depth limit: {})",
        report.affected_nodes.len(),
        max_expected_depth
    );
}

// ── Test 2: Hypothesis Cascade ───────────────────────────────────────────

#[test]
fn hypothesis_cascade_embedding_shift() {
    // Node A: embedding pointing in axis 0
    let node_a = make_node(
        "nodeA",
        unit_vector(0),
        0.9,
        Some(("HEAL", "WORK")),
    );

    // Node B: unrelated node
    let node_b = make_node(
        "nodeB",
        unit_vector(1),
        0.8,
        Some(("HEAL", "WORK")),
    );

    let nodes = vec![node_a.clone(), node_b.clone()];
    let edges: Vec<TypedEdge> = vec![];

    // Hypothesis that references nodeA and is Supported with high fitness
    let hyp_embedding = unit_vector(0); // perfectly aligned with nodeA
    let mut hypothesis = Hypothesis::new(
        "Node A represents a valid operational state",
        hyp_embedding.clone(),
    );
    hypothesis.ontology_refs = vec!["nodeA".to_string()];
    hypothesis.status = HypothesisStatus::Supported;
    hypothesis.coherence = 1.0; // perfect alignment
    hypothesis.fitness = 0.9; // high fitness
    hypothesis.fitness_breakdown.semantic_fit = 1.0;
    hypothesis.fitness_breakdown.logical_consistency = 1.0;
    hypothesis
        .supporting_evidence
        .push(Evidence::supports("sensor_log", "Reading confirms operational state"));

    let hypotheses = vec![hypothesis.clone()];
    let hyp_id = hypothesis.id.clone();
    let pre_status = hypothesis.status;

    let ctx = EvaluationContext::from_base(&nodes, &hypotheses, &edges);

    // Shift nodeA's embedding to the opposite direction
    let mutation = OntologyMutation::new(
        "nodeA",
        MutationOp::EmbeddingShift {
            old_embedding: unit_vector(0),
            new_embedding: vec![-1.0, 0.0, 0.0, 0.0],
        },
    );

    let mut ctx = ctx;
    let report = evaluate_mutation(&mut ctx, mutation);

    // Verify the hypothesis transitioned to Contradicted
    let transition = report
        .hypothesis_status_shifts
        .iter()
        .find(|t| t.hypothesis_id == hyp_id)
        .expect("hypothesis must appear in status shifts");

    assert_eq!(
        transition.previous_status, pre_status,
        "previous status must be Supported"
    );
    assert_eq!(
        transition.new_status,
        HypothesisStatus::Contradicted,
        "hypothesis must transition to Contradicted"
    );

    // Verify contradicting evidence was added to the shadow hypothesis
    let shadow_hyp = ctx
        .shadow_hypotheses
        .get(&hyp_id)
        .expect("hypothesis must be in shadow frame");

    assert!(
        !shadow_hyp.contradicting_evidence.is_empty(),
        "contradicting evidence must be added"
    );

    let contra_ev = &shadow_hyp.contradicting_evidence[0];
    assert_eq!(contra_ev.polarity, EvidencePolarity::Contradicts);
    assert!(
        !contra_ev.claim.is_empty(),
        "contradicting evidence must have a claim"
    );

    // Verify the hypothesis coherence dropped significantly
    assert!(
        shadow_hyp.coherence < hypothesis.coherence,
        "hypothesis coherence must drop after embedding shift"
    );

    println!(
        "Hypothesis cascade: {} → {:?} (coherence {:.4} → {:.4})",
        pre_status.as_str(),
        transition.new_status,
        hypothesis.coherence,
        shadow_hyp.coherence
    );
}

// ── Test 3: Shadow Frame Isolation ──────────────────────────────────────

#[test]
fn shadow_frame_does_not_modify_base() {
    let node_a = make_node(
        "nodeA",
        unit_vector(0),
        0.9,
        Some(("HEAL", "WORK")),
    );
    let node_b = make_node(
        "nodeB",
        unit_vector(1),
        0.8,
        Some(("HEAL", "WORK")),
    );

    let nodes = vec![node_a.clone(), node_b.clone()];
    let hypotheses = vec![Hypothesis::new(
        "test hypothesis",
        unit_vector(0),
    )];
    let hyp_id = hypotheses[0].id.clone();

    // Add an edge for adjacency
    let edges = vec![TypedEdge::new(
        RelationType::Influences,
        "nodeA",
        "nodeB",
    )];

    let base_nodes_snapshot = nodes.clone();
    let base_embeddings: Vec<Vec<f32>> = nodes.iter().map(|n| n.embedding.clone()).collect();
    let base_coherences: Vec<f32> = nodes.iter().map(|n| n.coherence_score).collect();

    let ctx = EvaluationContext::from_base(&nodes, &hypotheses, &edges);

    // Apply a mutation that shifts the embedding and coherence
    let mutation = OntologyMutation::new(
        "nodeA",
        MutationOp::EmbeddingShift {
            old_embedding: unit_vector(0),
            new_embedding: vec![0.0, 1.0, 0.0, 0.0],
        },
    );

    let mut ctx = ctx;
    let _report = evaluate_mutation(&mut ctx, mutation);

    // Verify base_nodes slice is unchanged (same lengths, values)
    assert_eq!(nodes.len(), base_nodes_snapshot.len());
    for (i, n) in nodes.iter().enumerate() {
        assert_eq!(n.id, base_nodes_snapshot[i].id, "node id must not change");
        assert_eq!(n.embedding, base_embeddings[i], "base embedding must be unchanged");
        assert_eq!(n.coherence_score, base_coherences[i], "base coherence must be unchanged");
    }

    // Verify base hypotheses are unchanged
    let base_hyp = &hypotheses[0];
    assert_eq!(base_hyp.id, hyp_id);
    assert_eq!(
        base_hyp.status,
        HypothesisStatus::Candidate,
        "base hypothesis status must not change"
    );
    assert!(
        base_hyp.supporting_evidence.is_empty(),
        "base hypothesis must not gain evidence"
    );
    assert_eq!(
        base_hyp.coherence, 0.5,
        "base hypothesis coherence must not change"
    );

    // Verify shadow_nodes has the mutated node
    let shadow_node = ctx
        .shadow_nodes
        .get("nodeA")
        .expect("nodeA must be in shadow frame");
    assert_ne!(
        shadow_node.embedding, unit_vector(0),
        "shadow node embedding must be mutated"
    );

    // Verify the base node reference is not the shadow node
    let base_ref = &nodes[0];
    assert_ne!(
        base_ref.embedding, shadow_node.embedding,
        "base and shadow embeddings must differ after mutation"
    );

    println!(
        "Shadow isolation: {} base nodes unchanged, {} shadow nodes mutated",
        nodes.len(),
        ctx.shadow_nodes.len()
    );
}

// ── Test 4: Report JSON Serialization ───────────────────────────────────

#[test]
fn report_serializes_to_json() {
    let node_a = make_node("nA", unit_vector(0), 0.8, None);
    let node_b = make_node("nB", unit_vector(1), 0.6, None);
    let nodes = vec![node_a.clone(), node_b.clone()];

    let edges = vec![TypedEdge::new(
        RelationType::Influences,
        "nA",
        "nB",
    )];

    let ctx = EvaluationContext::from_base(&nodes, &[], &edges);

    let mutation = OntologyMutation::new(
        "nA",
        MutationOp::EmbeddingShift {
            old_embedding: unit_vector(0),
            new_embedding: vec![0.0, 1.0, 0.0, 0.0],
        },
    );

    let mut ctx = ctx;
    let report = evaluate_mutation(&mut ctx, mutation);

    // Serialize to JSON
    let json = serde_json::to_string(&report).expect("report must serialize to JSON");
    assert!(!json.is_empty(), "JSON output must not be empty");

    // Deserialize back
    let deserialized: OntologyDeltaReport =
        serde_json::from_str(&json).expect("report must deserialize from JSON");

    // Verify key fields survive round-trip
    assert_eq!(
        deserialized.mutation_source.target_node_id, "nA",
        "target node ID must survive round-trip"
    );
    assert_eq!(
        deserialized.affected_nodes.len(),
        report.affected_nodes.len(),
        "affected nodes count must survive round-trip"
    );
    assert!(
        (deserialized.net_coherence_delta - report.net_coherence_delta).abs() < 1e-6,
        "net coherence delta must survive round-trip ({} vs {})",
        deserialized.net_coherence_delta,
        report.net_coherence_delta
    );

    // Verify net_coherence_delta is a valid f32
    assert!(
        deserialized.net_coherence_delta.is_finite(),
        "net coherence delta must be a finite number"
    );

    // Verify each NodeDelta round-trips
    for (orig, de) in report.affected_nodes.iter().zip(deserialized.affected_nodes.iter()) {
        assert_eq!(orig.node_id, de.node_id);
        assert!((orig.previous_coherence - de.previous_coherence).abs() < 1e-6);
        assert!((orig.new_coherence - de.new_coherence).abs() < 1e-6);
        assert_eq!(orig.delta_type, de.delta_type);
    }

    println!(
        "Report JSON round-trip: {} nodes, {} hypothesis shifts, net_coherence_delta={:.4}",
        deserialized.affected_nodes.len(),
        deserialized.hypothesis_status_shifts.len(),
        deserialized.net_coherence_delta
    );
}

// ── Additional: PropertySet mutation ───────────────────────────────────

#[test]
fn property_set_mutation_propagates() {
    let node_a = make_node("nA", unit_vector(0), 0.7, Some(("HEAL", "WORK")));
    let node_b = make_node("nB", unit_vector(0), 0.7, Some(("HEAL", "WORK")));
    let nodes = vec![node_a.clone(), node_b.clone()];

    let edges = vec![TypedEdge::new(
        RelationType::Influences,
        "nA",
        "nB",
    )];

    let ctx = EvaluationContext::from_base(&nodes, &[], &edges);

    let mutation = OntologyMutation::new(
        "nA",
        MutationOp::PropertySet {
            key: "label".to_string(),
            old_val: serde_json::Value::String("old_label".to_string()),
            new_val: serde_json::Value::String("new_label".to_string()),
        },
    );

    let mut ctx = ctx;
    let report = evaluate_mutation(&mut ctx, mutation);

    // Target must be in the report
    assert!(
        report.affected_nodes.iter().any(|d| d.node_id == "nA"),
        "target node must be in report"
    );

    // Mutation log must contain the mutation
    assert_eq!(ctx.mutation_log.len(), 1);
    assert_eq!(ctx.mutation_log[0].target_node_id, "nA");

    println!(
        "PropertySet mutation: {} nodes affected, {} mutations logged",
        report.affected_nodes.len(),
        ctx.mutation_log.len()
    );
}

// ── Additional: Cell-pin adjacency propagation ─────────────────────────

#[test]
fn cell_pin_propagates_to_shared_cell() {
    // Three nodes share the same cell pin
    let node_a = make_node("a", unit_vector(0), 0.5, Some(("HEAL", "WORK")));
    let node_b = make_node("b", unit_vector(1), 0.5, Some(("HEAL", "WORK")));
    let node_c = make_node("c", unit_vector(2), 0.5, Some(("HEAL", "WORK")));

    let nodes = vec![node_a.clone(), node_b.clone(), node_c.clone()];
    let edges: Vec<TypedEdge> = vec![];

    let ctx = EvaluationContext::from_base(&nodes, &[], &edges);

    let mutation = OntologyMutation::new(
        "a",
        MutationOp::EmbeddingShift {
            old_embedding: unit_vector(0),
            new_embedding: vec![-1.0, 0.0, 0.0, 0.0],
        },
    );

    let mut ctx = ctx;
    let report = evaluate_mutation(&mut ctx, mutation);

    // All three nodes share the cell pin, so they should all be affected
    let affected_ids: Vec<&str> = report.affected_nodes.iter().map(|d| d.node_id.as_str()).collect();
    assert!(
        affected_ids.contains(&"a"),
        "node a (mutated) must be affected"
    );
    assert!(
        affected_ids.contains(&"b"),
        "node b (shared cell pin) must be affected"
    );

    println!(
        "Cell-pin propagation: {} nodes affected via shared cell pin",
        report.affected_nodes.len()
    );
}

// ── Additional: Multiple mutations logged ──────────────────────────────

#[test]
fn multiple_mutations_accumulate_log() {
    let nodes = vec![make_node("n0", unit_vector(0), 0.5, None)];
    let ctx = EvaluationContext::from_base(&nodes, &[], &[]);

    let mut ctx = ctx;

    let m1 = OntologyMutation::new(
        "n0",
        MutationOp::EmbeddingShift {
            old_embedding: unit_vector(0),
            new_embedding: vec![0.0, 1.0, 0.0, 0.0],
        },
    );
    let _ = evaluate_mutation(&mut ctx, m1);

    let m2 = OntologyMutation::new(
        "n0",
        MutationOp::PropertySet {
            key: "label".to_string(),
            old_val: serde_json::Value::String("a".to_string()),
            new_val: serde_json::Value::String("b".to_string()),
        },
    );
    let _ = evaluate_mutation(&mut ctx, m2);

    assert_eq!(ctx.mutation_log.len(), 2, "both mutations must be logged");
}
