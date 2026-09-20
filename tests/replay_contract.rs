//! MS-002-harness — the runnable replay contract MS-002 deferred.
//!
//! `evaluate_mutation` has no production caller, so "incremental correctness"
//! cannot be checked against a live baseline. This file supplies the second
//! path instead: a full recompute built directly from mutated base nodes, and
//! an independent BFS that re-derives the wave from the documented neighbour
//! rule. Neither calls `find_neighbors`, `build_adjacency` or `dfs_collect`.
//!
//! ## Oracle semantics (decided before the oracle was written)
//!
//! The wave is **not** "the set of nodes whose coherence changed". The DFS
//! stops at `MAX_PROPAGATION_DEPTH` (5) and `MIN_IMPACT` (0.01), while
//! `compute_coherence` scans the whole effective graph — so a mutation can
//! change the coherence of a node the wave never visits. MS-002 called that
//! "hidden global work". An equality oracle over the *changed* set would
//! therefore fail on healthy input and prove nothing.
//!
//! What is checked instead:
//!
//! | id | property |
//! |----|----------|
//! | P1 | `previous_coherence` is the target's stored score, not an invented one |
//! | P2 | value agreement: every `new_coherence` equals the full recompute, epsilon 1e-6 |
//! | P3 | wave shape: `affected_nodes` equals the independent BFS over the documented rule |
//! | P7 | `net_coherence_delta` equals the sum of the reported deltas, epsilon 1e-6 |
//! | P8 | the known gap is pinned: at least one node changes under the full recompute without being visited |
//!
//! **P1 is not the "no phantom change" check the packet proposed.** The wave's
//! `previous_coherence` is the node's *stored* `coherence_score`, while a full
//! recompute produces a *recomputed* one — two different quantities. Comparing
//! them is what made the fanout arm fail on healthy input: `n2` carries a
//! fixture score of 0.5, recomputes to 0.0 both before and after the mutation,
//! and therefore reports a −0.5 delta that no recomputed-vs-recomputed check
//! can see. The honest property is that `previous_coherence` equals the base
//! node's stored score; the delta's meaning (stored → recomputed) is recorded
//! in the report, not asserted away.
//!
//! Amendment A1 discipline: `affected_nodes` is compared as a set sorted by
//! `node_id`, floats to epsilon 1e-6, never positionally and never bit-exact.

use std::collections::{HashMap, HashSet};

use physis_core::delta_engine::{
    evaluate_mutation, EvaluationContext, MutationOp, OntologyDeltaReport, OntologyMutation, GAMMA,
    MAX_PROPAGATION_DEPTH, MIN_IMPACT,
};
use physis_core::models::cosine_sim;
use physis_core::{CoherenceNode, RelationType, TypedEdge};

const EPS: f32 = 1e-6;

// ── Fixtures ─────────────────────────────────────────────────────────────

/// Unit vector along `axis` in 4 dimensions.
fn unit_vector(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; 4];
    if axis < v.len() {
        v[axis] = 1.0;
    }
    v
}

/// The shift used by every non-degenerate arm: cos(old, new) = 0.6.
fn shifted() -> Vec<f32> {
    vec![0.6, 0.8, 0.0, 0.0]
}

fn make_node(
    id: &str,
    embedding: Vec<f32>,
    coherence: f32,
    cell_pin: Option<(&str, &str)>,
) -> CoherenceNode {
    let mut n = CoherenceNode::new(embedding);
    n.id = id.to_string();
    n.coherence_score = coherence;
    n.label = Some(id.to_string());
    n.cell_pin = cell_pin.map(|(d, m)| (d.to_string(), m.to_string()));
    n
}

fn edge(relation: RelationType, from: &str, to: &str) -> TypedEdge {
    TypedEdge::new(relation, from, to)
}

fn shift(target: &str, old: Vec<f32>, new: Vec<f32>) -> OntologyMutation {
    OntologyMutation::new(
        target,
        MutationOp::EmbeddingShift {
            old_embedding: old,
            new_embedding: new,
        },
    )
}

/// Two isolated nodes, no edges: the wave must stay on the target.
fn local_graph() -> (Vec<CoherenceNode>, Vec<TypedEdge>) {
    (
        vec![
            make_node("n0", unit_vector(0), 0.5, None),
            make_node("n1", unit_vector(1), 0.5, None),
        ],
        vec![],
    )
}

/// n0–n1 edge, n0 and n2 sharing a cell pin, and n3 identical to n0 but
/// connected to nothing — the node the wave cannot reach.
fn fanout_graph() -> (Vec<CoherenceNode>, Vec<TypedEdge>) {
    (
        vec![
            make_node("n0", unit_vector(0), 0.5, Some(("d", "m"))),
            make_node("n1", unit_vector(1), 0.5, None),
            make_node("n2", unit_vector(2), 0.5, Some(("d", "m"))),
            make_node("n3", unit_vector(0), 0.5, None),
        ],
        vec![edge(RelationType::Influences, "n0", "n1")],
    )
}

/// A DependsOn loop: the revision walk must record the closing edge.
fn cycle_graph() -> (Vec<CoherenceNode>, Vec<TypedEdge>) {
    (
        vec![
            make_node("n0", unit_vector(0), 0.5, None),
            make_node("n1", unit_vector(1), 0.5, None),
        ],
        vec![
            edge(RelationType::DependsOn, "n0", "n1"),
            edge(RelationType::DependsOn, "n1", "n0"),
        ],
    )
}

// ── The independent oracle ───────────────────────────────────────────────

/// The documented neighbour rule, re-derived here on purpose: typed edges in
/// both directions, shared `cell_pin`, and `parent → ` label-prefix links.
fn neighbors_of(nodes: &[CoherenceNode], edges: &[TypedEdge], id: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let push = |out: &mut Vec<String>, x: String| {
        if !out.contains(&x) {
            out.push(x);
        }
    };

    for e in edges {
        if e.source_id == id {
            push(&mut out, e.target_id.clone());
        } else if e.target_id == id {
            push(&mut out, e.source_id.clone());
        }
    }

    if let Some(node) = nodes.iter().find(|n| n.id == id) {
        if let Some(cell_pin) = &node.cell_pin {
            for n in nodes {
                if n.id != id && n.cell_pin.as_ref() == Some(cell_pin) {
                    push(&mut out, n.id.clone());
                }
            }
        }
        if let Some(label) = &node.label {
            let prefix = format!("{label} → ");
            for n in nodes {
                if n.id == id {
                    continue;
                }
                if let Some(nlabel) = &n.label {
                    if nlabel.starts_with(&prefix) {
                        push(&mut out, n.id.clone());
                    }
                }
            }
        }
    }

    out
}

/// BFS mirroring the wave's own cutoffs (`depth > MAX_PROPAGATION_DEPTH`,
/// `impact.abs() < MIN_IMPACT && depth > 0`). No HashSet iteration escapes:
/// the result is returned unsorted and the caller sorts.
fn expected_wave(
    nodes: &[CoherenceNode],
    edges: &[TypedEdge],
    target: &str,
    source_sim: f32,
) -> Vec<String> {
    fn walk(
        nodes: &[CoherenceNode],
        edges: &[TypedEdge],
        id: &str,
        depth: usize,
        source_sim: f32,
        visited: &mut HashSet<String>,
        out: &mut Vec<String>,
    ) {
        if depth > MAX_PROPAGATION_DEPTH || visited.contains(id) {
            return;
        }
        visited.insert(id.to_string());

        let impact = source_sim * GAMMA.powi(depth as i32);
        if impact.abs() < MIN_IMPACT && depth > 0 {
            return;
        }
        out.push(id.to_string());

        for neighbor in neighbors_of(nodes, edges, id) {
            if !visited.contains(&neighbor) {
                walk(nodes, edges, &neighbor, depth + 1, source_sim, visited, out);
            }
        }
    }

    let mut visited = HashSet::new();
    let mut out = Vec::new();
    walk(nodes, edges, target, 0, source_sim, &mut visited, &mut out);
    out
}

/// A1: compare as a set sorted by `node_id`.
fn wave_ids(report: &OntologyDeltaReport) -> Vec<String> {
    let mut v: Vec<String> = report
        .affected_nodes
        .iter()
        .map(|d| d.node_id.clone())
        .collect();
    v.sort();
    v.dedup();
    v
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v.dedup();
    v
}

/// Apply the mutation to a copy of the base nodes — the "full" path. Only
/// `EmbeddingShift` changes node state (a `PropertySet` carries no embedding
/// change; `Relation*` ops change edges, not nodes).
fn mutated_nodes(nodes: &[CoherenceNode], mutation: &OntologyMutation) -> Vec<CoherenceNode> {
    let mut out = nodes.to_vec();
    if let MutationOp::EmbeddingShift { new_embedding, .. } = &mutation.operation {
        if let Some(n) = out.iter_mut().find(|n| n.id == mutation.target_node_id) {
            n.embedding = new_embedding.clone();
        }
    }
    out
}

fn coherences(nodes: &[CoherenceNode], edges: &[TypedEdge]) -> HashMap<String, f32> {
    let ctx = EvaluationContext::from_base(nodes, &[], edges);
    nodes
        .iter()
        .map(|n| (n.id.clone(), ctx.compute_coherence(&n.id)))
        .collect()
}

// ── The contract ─────────────────────────────────────────────────────────

/// Runs the incremental path and checks P1, P2, P3 and P7 against the two
/// independent computations. Returns the report for arm-specific assertions.
fn assert_contract(
    nodes: &[CoherenceNode],
    edges: &[TypedEdge],
    mutation: OntologyMutation,
    source_sim: f32,
) -> OntologyDeltaReport {
    let target = mutation.target_node_id.clone();

    let stored: HashMap<String, f32> = nodes
        .iter()
        .map(|n| (n.id.clone(), n.coherence_score))
        .collect();
    let mutated = mutated_nodes(nodes, &mutation);
    let full = coherences(&mutated, edges);

    let mut ctx = EvaluationContext::from_base(nodes, &[], edges);
    let report = evaluate_mutation(&mut ctx, mutation);

    // P3 — wave shape, as a set sorted by node_id.
    let got = wave_ids(&report);
    let want = sorted(expected_wave(nodes, edges, &target, source_sim));
    assert_eq!(
        got, want,
        "P3: affected_nodes must equal the independent BFS over the documented neighbour rule"
    );

    // P1 + P2.
    let mut summed = 0.0_f32;
    for d in &report.affected_nodes {
        let full_value = full[&d.node_id];
        assert!(
            (d.new_coherence - full_value).abs() < EPS,
            "P2: {} new_coherence {} != full recompute {}",
            d.node_id,
            d.new_coherence,
            full_value
        );
        assert!(
            (d.previous_coherence - stored[&d.node_id]).abs() < EPS,
            "P1: {} previous_coherence {} is not the stored score {}",
            d.node_id,
            d.previous_coherence,
            stored[&d.node_id]
        );
        summed += d.new_coherence - d.previous_coherence;
    }

    // P7 — the running sum, checked against the deltas it is built from.
    assert!(
        (report.net_coherence_delta - summed).abs() < EPS,
        "P7: net_coherence_delta {} != sum of deltas {}",
        report.net_coherence_delta,
        summed
    );

    report
}

// ── Arms ─────────────────────────────────────────────────────────────────

/// Arm: no-op. A7's zero-shift invariant — nothing is touched, nothing moves.
#[test]
fn no_op_shift_returns_an_empty_report() {
    let (nodes, edges) = local_graph();
    let mutation = shift("n0", unit_vector(0), unit_vector(0));
    let mut ctx = EvaluationContext::from_base(&nodes, &[], &edges);
    let report = evaluate_mutation(&mut ctx, mutation);

    assert!(
        report.affected_nodes.is_empty(),
        "A7: a zero-shift must produce no deltas, got {:?}",
        report.affected_nodes
    );
    assert_eq!(report.net_coherence_delta, 0.0);
    assert!(report.revision_walk.is_none(), "A7 returns no walk");
    assert!(
        ctx.shadow_nodes.is_empty(),
        "A7: the shadow frame must not be touched"
    );
}

/// Arm: local. Two nodes, no link — only the target is in the wave.
#[test]
fn local_mutation_affects_only_the_target() {
    let (nodes, edges) = local_graph();
    let report = assert_contract(
        &nodes,
        &edges,
        shift("n0", unit_vector(0), shifted()),
        cosine_sim(&unit_vector(0), &shifted()),
    );
    assert_eq!(
        wave_ids(&report),
        vec!["n0".to_string()],
        "an unlinked target must not drag its neighbour in"
    );
}

/// Arm: fanout. An edge and a shared cell pin both carry the wave.
#[test]
fn cell_pin_fanout_reaches_both_link_classes() {
    let (nodes, edges) = fanout_graph();
    let report = assert_contract(
        &nodes,
        &edges,
        shift("n0", unit_vector(0), shifted()),
        cosine_sim(&unit_vector(0), &shifted()),
    );
    assert_eq!(
        wave_ids(&report),
        vec!["n0".to_string(), "n1".to_string(), "n2".to_string()],
        "the edge carries n1, the shared pin carries n2"
    );
    assert!(
        report.affected_nodes.len() >= 2,
        "the fanout arm needs a real fanout"
    );
}

/// Arm: cycle. The DependsOn loop must be recorded and must terminate.
#[test]
fn depends_on_cycle_is_recorded_and_terminates() {
    let (nodes, edges) = cycle_graph();
    let report = assert_contract(
        &nodes,
        &edges,
        shift("n0", unit_vector(0), shifted()),
        cosine_sim(&unit_vector(0), &shifted()),
    );
    let walk = report
        .revision_walk
        .as_ref()
        .expect("the revision walk runs for every non-zero-shift mutation");
    assert!(
        !walk.cycles.is_empty(),
        "the closing edge of the DependsOn loop must be recorded"
    );
    assert!(!walk.truncated, "two nodes cannot hit the walk cap");
}

/// Arm: disconnected. An orphan target keeps the wave to itself.
#[test]
fn disconnected_mutation_does_not_leak() {
    let nodes = vec![
        make_node("orphan", unit_vector(0), 0.5, None),
        make_node("island", unit_vector(1), 0.5, None),
    ];
    let report = assert_contract(
        &nodes,
        &[],
        shift("orphan", unit_vector(0), shifted()),
        cosine_sim(&unit_vector(0), &shifted()),
    );
    assert_eq!(wave_ids(&report), vec!["orphan".to_string()]);
}

/// P8 — the gap MS-002 called hidden global work, pinned so a silent change in
/// wave coverage is visible. `n3` is identical to `n0` before the shift and
/// linked to nothing, so the wave never visits it; its coherence still moves,
/// because `compute_coherence` scans the whole effective graph.
#[test]
fn full_recompute_moves_a_node_the_wave_never_visits() {
    let (nodes, edges) = fanout_graph();
    let base = coherences(&nodes, &edges);

    let mutation = shift("n0", unit_vector(0), shifted());
    let mutated = mutated_nodes(&nodes, &mutation);
    let full = coherences(&mutated, &edges);

    let mut ctx = EvaluationContext::from_base(&nodes, &[], &edges);
    let report = evaluate_mutation(&mut ctx, mutation);
    let wave = wave_ids(&report);

    let moved_outside: Vec<&String> = full
        .iter()
        .filter(|(id, v)| (**v - base[*id]).abs() > EPS && !wave.contains(id))
        .map(|(id, _)| id)
        .collect();

    assert!(
        moved_outside.contains(&&"n3".to_string()),
        "the hidden global work must show up as a changed-but-unvisited node; got {moved_outside:?}"
    );
}

/// Arm: corrupted-incremental (negative). The incremental context drops the
/// n0–n1 edge, simulating a missed dependency. The oracle must reject the
/// wave and name the omitted node — a check that cannot fail is not a check.
#[test]
fn oracle_rejects_a_corrupted_incremental_wave() {
    let (nodes, edges) = fanout_graph();
    let mutation = shift("n0", unit_vector(0), shifted());

    let corrupted_edges: Vec<TypedEdge> = edges
        .iter()
        .filter(|e| !(e.source_id == "n0" && e.target_id == "n1"))
        .cloned()
        .collect();
    assert!(
        corrupted_edges.len() < edges.len(),
        "the corruption must actually remove an edge"
    );

    let mut ctx = EvaluationContext::from_base(&nodes, &[], &corrupted_edges);
    let report = evaluate_mutation(&mut ctx, mutation);
    let wave = wave_ids(&report);

    let want = sorted(expected_wave(
        &nodes,
        &edges,
        "n0",
        cosine_sim(&unit_vector(0), &shifted()),
    ));
    assert_ne!(
        wave, want,
        "the corrupted wave must differ from the oracle's expectation"
    );

    let omitted: Vec<String> = want
        .iter()
        .filter(|id| !wave.contains(id))
        .cloned()
        .collect();
    assert_eq!(
        omitted,
        vec!["n1".to_string()],
        "the oracle must name the dependency the corruption dropped"
    );
}
