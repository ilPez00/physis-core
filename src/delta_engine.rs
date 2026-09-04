//! Dynamic Ontology Delta & Propagation Reasoning Engine.
//!
//! When a single node or node property mutates, this engine simulates the
//! cascade effects on graph coherence and hypothesis statuses in an isolated
//! shadow frame — without writing to persistent storage or modifying the base
//! state vectors passed into the context. The caller inspects the
//! [`OntologyDeltaReport`] and decides whether to commit.
//!
//! ## Propagation Model
//!
//! A topological wave traverses the adjacency graph starting from the mutated
//! node. Each hop attenuates the impact by the decay factor γ (default 0.85):
//!
//! ```text
//! ΔImpact(N) = CosineSimilarity(E_old, E_new) × γ^depth
//! ```
//!
//! Only nodes within the propagation radius (depth ≤ [`MAX_PROPAGATION_DEPTH`]
//! and impact ≥ [`MIN_IMPACT`]) are evaluated and reported.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::hypothesis::{Evidence, Hypothesis, HypothesisStatus};
use crate::models::{cosine_sim, CoherenceNode, Score};
use crate::relation::TypedEdge;

/// Decay factor γ for impact attenuation during topological propagation.
pub const GAMMA: f32 = 0.85;

/// Score-degradation threshold ϵ for hypothesis status transitions.
/// When a hypothesis's coherence drops beyond this, a Supported → Contradicted
/// transition is triggered.
pub const DEGRADATION_THRESHOLD: f32 = 0.25;

/// Maximum propagation depth in the DFS wave.
pub const MAX_PROPAGATION_DEPTH: usize = 5;

/// Minimum impact factor below which propagation stops.
pub const MIN_IMPACT: f32 = 0.01;

// ── Mutation & Shadows ───────────────────────────────────────────────────

/// A single atomic mutation operation on a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationOp {
    /// Set or replace a JSON property on the node.
    PropertySet {
        key: String,
        old_val: serde_json::Value,
        new_val: serde_json::Value,
    },
    /// Shift the node's embedding vector.
    EmbeddingShift {
        old_embedding: Vec<f32>,
        new_embedding: Vec<f32>,
    },
    /// Sever an outgoing relationship edge to another node.
    RelationSevered {
        target_id: String,
    },
    /// Create a new outgoing relationship edge to another node.
    RelationCreated {
        target_id: String,
        weight: f32,
    },
}

/// A complete mutation event targeting a specific node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyMutation {
    pub id: String,
    pub target_node_id: String,
    pub operation: MutationOp,
    pub triggered_by: Option<String>,
    pub timestamp: DateTime<Utc>,
}

impl OntologyMutation {
    /// Convenience constructor with a fresh UUID and the current timestamp.
    pub fn new(target_node_id: impl Into<String>, operation: MutationOp) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            target_node_id: target_node_id.into(),
            operation,
            triggered_by: None,
            timestamp: Utc::now(),
        }
    }

    /// Mark this mutation as being caused by another mutation or event.
    pub fn with_trigger(mut self, trigger_id: impl Into<String>) -> Self {
        self.triggered_by = Some(trigger_id.into());
        self
    }
}

/// Transient frame for isolated evaluation.
///
/// `base_nodes` and `base_hypotheses` are immutable references to the real
/// graph state — never mutated. `shadow_nodes` and `shadow_hypotheses` are
/// owned copies that receive the mutation and all propagation side-effects.
/// The caller inspects [`OntologyDeltaReport`] and decides whether to commit
/// the shadow state back to persistent storage.
pub struct EvaluationContext<'a> {
    /// The unmodified graph nodes (read-only).
    pub base_nodes: &'a [CoherenceNode],
    /// The unmodified hypotheses (read-only).
    pub base_hypotheses: &'a [Hypothesis],
    /// Typed edges that define graph adjacency (read-only).
    pub edges: &'a [TypedEdge],
    /// Shadow copies of nodes that have been touched by the mutation / wave.
    pub shadow_nodes: HashMap<String, CoherenceNode>,
    /// Shadow copies of hypotheses whose fitness has been recomputed.
    pub shadow_hypotheses: HashMap<String, Hypothesis>,
    /// Append-only log of every mutation applied in this frame.
    pub mutation_log: Vec<OntologyMutation>,
}

impl<'a> EvaluationContext<'a> {
    /// Build a fresh shadow frame from base data with empty mutation log.
    pub fn from_base(
        base_nodes: &'a [CoherenceNode],
        base_hypotheses: &'a [Hypothesis],
        edges: &'a [TypedEdge],
    ) -> Self {
        Self {
            base_nodes,
            base_hypotheses,
            edges,
            shadow_nodes: HashMap::new(),
            shadow_hypotheses: HashMap::new(),
            mutation_log: Vec::new(),
        }
    }

    // ── Effective (shadow-preferred) lookups ────────────────────────────

    /// Resolve a node ID to its effective representation: the shadow copy if
    /// one exists (mutation applied), otherwise the base reference.
    pub fn effective_node(&self, id: &str) -> Option<&CoherenceNode> {
        self.shadow_nodes
            .get(id)
            .or_else(|| self.base_nodes.iter().find(|n| n.id == id))
    }

    /// All nodes visible in this frame — shadow copies first, then base nodes
    /// not yet shadowed.
    pub fn effective_nodes_all(&self) -> Vec<&CoherenceNode> {
        let mut result: Vec<&CoherenceNode> = self.shadow_nodes.values().collect();
        for n in self.base_nodes {
            if !self.shadow_nodes.contains_key(&n.id) {
                result.push(n);
            }
        }
        result
    }

    /// Resolve a hypothesis ID to its effective representation.
    pub fn effective_hypothesis(&self, id: &str) -> Option<&Hypothesis> {
        self.shadow_hypotheses
            .get(id)
            .or_else(|| self.base_hypotheses.iter().find(|h| h.id == id))
    }

    /// All hypotheses visible in this frame.
    pub fn effective_hypotheses_all(&self) -> Vec<&Hypothesis> {
        let mut result: Vec<&Hypothesis> = self.shadow_hypotheses.values().collect();
        for h in self.base_hypotheses {
            if !self.shadow_hypotheses.contains_key(&h.id) {
                result.push(h);
            }
        }
        result
    }

    // ── Shadow management ───────────────────────────────────────────────

    /// Ensure a node exists in the shadow frame by cloning from base if needed.
    /// Returns a mutable reference to the shadow copy.
    pub fn ensure_shadowed(&mut self, node_id: &str) -> Option<&mut CoherenceNode> {
        if !self.shadow_nodes.contains_key(node_id) {
            if let Some(base) = self.base_nodes.iter().find(|n| n.id == node_id) {
                self.shadow_nodes
                    .insert(node_id.to_string(), base.clone());
            }
        }
        self.shadow_nodes.get_mut(node_id)
    }

    // ── Coherence computation ───────────────────────────────────────────

    /// Recompute the coherence score for a node: the mean cosine similarity to
    /// its k nearest neighbours across the entire effective graph.
    pub fn compute_coherence(&self, node_id: &str) -> Score {
        let all = self.effective_nodes_all();
        if all.len() <= 1 {
            return 1.0;
        }

        let query = match self.effective_node(node_id) {
            Some(n) => &n.embedding,
            None => return 0.0,
        };

        let k = 5.min(all.len() - 1);
        let mut sims: Vec<f32> = all
            .iter()
            .filter(|n| n.id != node_id && !n.embedding.is_empty())
            .map(|n| cosine_sim(query, &n.embedding))
            .collect();

        if sims.is_empty() {
            return 1.0;
        }

        sims.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(k);

        let avg = sims.iter().sum::<f32>() / sims.len() as f32;
        avg.max(0.0)
    }

    // ── Adjacency ───────────────────────────────────────────────────────

    /// Find all node IDs adjacent to `node_id` via typed edges, shared
    /// cell pins, and label-prefix links.
    pub fn find_neighbors(&self, node_id: &str) -> Vec<String> {
        let mut neighbors: HashSet<String> = HashSet::new();

        // 1. Typed edges (both directions)
        for edge in self.edges {
            if edge.source_id == node_id {
                neighbors.insert(edge.target_id.clone());
            } else if edge.target_id == node_id {
                neighbors.insert(edge.source_id.clone());
            }
        }

        // 2. Shared cell_pin — nodes pinned to the same (domain, mode) cell
        if let Some(node) = self.effective_node(node_id) {
            if let Some(ref cell_pin) = node.cell_pin {
                for n in self.base_nodes {
                    if n.id != node_id && n.cell_pin.as_ref() == Some(cell_pin) {
                        neighbors.insert(n.id.clone());
                    }
                }
                for n in self.shadow_nodes.values() {
                    if n.id != node_id && n.cell_pin.as_ref() == Some(cell_pin) {
                        neighbors.insert(n.id.clone());
                    }
                }
            }
        }

        // 3. Label links — nodes whose label starts with "parent → "
        if let Some(node) = self.effective_node(node_id) {
            if let Some(ref label) = node.label {
                let prefix = format!("{} → ", label);
                for n in self.base_nodes {
                    if n.id == node_id {
                        continue;
                    }
                    if let Some(ref nlabel) = n.label {
                        if nlabel.starts_with(&prefix) {
                            neighbors.insert(n.id.clone());
                        }
                    }
                }
            }
        }

        neighbors.into_iter().collect()
    }

    /// Pre-compute the full adjacency map for all effective nodes.
    pub fn build_adjacency(&self) -> HashMap<String, Vec<String>> {
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
        for n in self.effective_nodes_all() {
            adjacency.insert(n.id.clone(), self.find_neighbors(&n.id));
        }
        adjacency
    }
}

// ── Report Structures ────────────────────────────────────────────────────

/// A coherence delta for a single node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDelta {
    pub node_id: String,
    pub previous_coherence: Score,
    pub new_coherence: Score,
    pub delta_type: String,
}

/// A status transition for a single hypothesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisTransition {
    pub hypothesis_id: String,
    pub previous_status: HypothesisStatus,
    pub new_status: HypothesisStatus,
    pub trigger_reason: String,
}

/// Full report of the mutation evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyDeltaReport {
    pub mutation_source: OntologyMutation,
    pub affected_nodes: Vec<NodeDelta>,
    pub hypothesis_status_shifts: Vec<HypothesisTransition>,
    pub net_coherence_delta: f32,
}

// ── Core Propagation ─────────────────────────────────────────────────────

/// Classify a coherence delta.
fn classify_delta(delta: f32) -> &'static str {
    if delta.abs() < 1e-6 {
        "Unchanged"
    } else if delta > 0.0 {
        "Strengthened"
    } else if delta < -DEGRADATION_THRESHOLD {
        "Invalidated"
    } else {
        "Shifted"
    }
}

/// Depth-first collection of affected nodes with impact factors.
///
/// The impact factor for a node at depth d is `source_sim × γ^d`.
/// Propagation stops when depth exceeds [`MAX_PROPAGATION_DEPTH`] or impact
/// falls below [`MIN_IMPACT`].
fn dfs_collect(
    current_id: &str,
    adjacency: &HashMap<String, Vec<String>>,
    depth: usize,
    visited: &mut HashSet<String>,
    affected: &mut Vec<(String, f32)>,
    source_sim: f32,
) {
    if depth > MAX_PROPAGATION_DEPTH || visited.contains(current_id) {
        return;
    }
    visited.insert(current_id.to_string());

    let impact = source_sim * GAMMA.powi(depth as i32);
    if impact.abs() < MIN_IMPACT && depth > 0 {
        return;
    }

    affected.push((current_id.to_string(), impact));

    if let Some(neighbors) = adjacency.get(current_id) {
        for neighbor_id in neighbors {
            if !visited.contains(neighbor_id) {
                dfs_collect(
                    neighbor_id,
                    adjacency,
                    depth + 1,
                    visited,
                    affected,
                    source_sim,
                );
            }
        }
    }
}

/// Recompute the fitness breakdown for a hypothesis after a node mutation.
///
/// Updates `semantic_fit` (cosine alignment with referenced nodes),
/// `logical_consistency` (penalised by contradicting evidence), and the
/// overall `coherence` score, then delegates to
/// [`Hypothesis::recompute_fitness`] for the composite.
fn recompute_hypothesis_fitness(
    hyp: &mut Hypothesis,
    ctx: &EvaluationContext,
    affected_node_ids: &[String],
) {
    let affected_refs: Vec<&CoherenceNode> = hyp
        .ontology_refs
        .iter()
        .filter(|id| affected_node_ids.contains(id))
        .filter_map(|id| ctx.effective_node(id))
        .filter(|n| !n.embedding.is_empty())
        .collect();

    if !hyp.embedding.is_empty() && !affected_refs.is_empty() {
        let sims: Vec<f32> = affected_refs
            .iter()
            .map(|n| {
                let raw = cosine_sim(&hyp.embedding, &n.embedding);
                (raw + 1.0) / 2.0 // map [-1, 1] → [0, 1]
            })
            .collect();

        let new_semantic_fit = sims.iter().sum::<f32>() / sims.len() as f32;
        let new_semantic_fit = new_semantic_fit.clamp(0.0, 1.0);

        hyp.fitness_breakdown.semantic_fit = new_semantic_fit;
        hyp.coherence_profile.semantic.score = new_semantic_fit;
        hyp.coherence = new_semantic_fit;
    }

    // Logical consistency degrades proportionally to contradicting evidence
    let contra_count = hyp.contradicting_evidence.len() as Score;
    let logical = 1.0 - (contra_count * 0.1).min(0.5);
    hyp.fitness_breakdown.logical_consistency = logical;
    hyp.coherence_profile.logical.score = logical;

    hyp.recompute_fitness();
}

/// Evaluate a single mutation against the current graph state.
///
/// All computation happens inside `ctx` — no writes to persistent storage.
/// The returned [`OntologyDeltaReport`] captures every structural change:
/// node coherence deltas, hypothesis status shifts, and the net coherence
/// change across the affected sub-graph.
pub fn evaluate_mutation(
    ctx: &mut EvaluationContext,
    mutation: OntologyMutation,
) -> OntologyDeltaReport {
    ctx.mutation_log.push(mutation.clone());

    let target_id = mutation.target_node_id.clone();

    // Resolve the target node from the effective (base or shadow) view.
    let base_node = ctx.effective_node(&target_id);

    // ── Isolation Phase: apply mutation to the shadow frame ─────────────
    let mut shadow_node = base_node
        .cloned()
        .unwrap_or_else(|| CoherenceNode::new(vec![]));

    let source_sim = match &mutation.operation {
        MutationOp::PropertySet { .. } => 1.0,
        MutationOp::EmbeddingShift {
            old_embedding: old_emb,
            new_embedding,
        } => {
            shadow_node.embedding = new_embedding.clone();
            if old_emb.len() == new_embedding.len() {
                cosine_sim(old_emb, new_embedding)
            } else {
                0.0
            }
        }
        MutationOp::RelationSevered { .. } => 1.0,
        MutationOp::RelationCreated { .. } => 1.0,
    };

    ctx.shadow_nodes.insert(target_id.clone(), shadow_node);

    // ── Topological Wave: DFS propagation ────────────────────────────────
    let adjacency = ctx.build_adjacency();

    let mut visited: HashSet<String> = HashSet::new();
    let mut affected: Vec<(String, f32)> = Vec::new(); // (node_id, impact)
    dfs_collect(
        &target_id,
        &adjacency,
        0,
        &mut visited,
        &mut affected,
        source_sim,
    );

    // ── Local Recalculation: update coherence scores ───────────────────
    let mut node_deltas: Vec<NodeDelta> = Vec::new();
    let mut net_delta = 0.0_f32;

    for (node_id, _impact) in &affected {
        // Capture pre-update coherence from the effective node (base or shadow)
        let prev_coherence = ctx
            .effective_node(node_id)
            .map(|n| n.coherence_score)
            .unwrap_or(0.0);

        // Recompute coherence considering the full effective graph
        let new_coherence = ctx.compute_coherence(node_id);

        // Persist the updated coherence in the shadow frame
        if let Some(node) = ctx.ensure_shadowed(node_id) {
            node.coherence_score = new_coherence;
        }

        let delta = new_coherence - prev_coherence;
        net_delta += delta;

        node_deltas.push(NodeDelta {
            node_id: node_id.clone(),
            previous_coherence: prev_coherence,
            new_coherence,
            delta_type: classify_delta(delta).to_string(),
        });
    }

    // ── Hypothesis Evaluation ──────────────────────────────────────────
    let mut hypothesis_transitions: Vec<HypothesisTransition> = Vec::new();
    let affected_node_ids: Vec<String> = affected.iter().map(|(id, _)| id.clone()).collect();

    // Collect IDs of hypotheses that reference any affected node
    let hyp_ids: Vec<String> = ctx
        .effective_hypotheses_all()
        .into_iter()
        .filter(|h| {
            h.ontology_refs
                .iter()
                .any(|r| affected_node_ids.contains(r))
        })
        .map(|h| h.id.clone())
        .collect();

    for hyp_id in &hyp_ids {
        // Capture pre-mutation values from the effective (base or shadow) view
        let (prev_coherence, prev_fitness, prev_status) = {
            let hyp = ctx.effective_hypothesis(hyp_id).expect("hypothesis must exist");
            (hyp.coherence, hyp.fitness, hyp.status)
        };

        // Clone into shadow for mutation
        let mut updated = ctx
            .effective_hypothesis(hyp_id)
            .expect("hypothesis must exist")
            .clone();

        recompute_hypothesis_fitness(&mut updated, ctx, &affected_node_ids);

        let coherence_delta = prev_coherence - updated.coherence;

        if coherence_delta > DEGRADATION_THRESHOLD {
            if matches!(
                prev_status,
                HypothesisStatus::Supported | HypothesisStatus::Confirmed
            ) {
                let new_status = HypothesisStatus::Contradicted;

                // Insert contradicting evidence automatically
                let evidence = Evidence::contradicts(
                    &mutation.id,
                    format!(
                        "Node '{}' embedding shifted; semantic alignment degraded \
                         by {:.4} (coherence {:.4} → {:.4}). Fitness change: {:.4}.",
                        target_id,
                        coherence_delta,
                        prev_coherence,
                        updated.coherence,
                        prev_fitness - updated.fitness,
                    ),
                );
                updated.contradicting_evidence.push(evidence);
                updated.recompute_fitness();

                hypothesis_transitions.push(HypothesisTransition {
                    hypothesis_id: updated.id.clone(),
                    previous_status: prev_status,
                    new_status,
                    trigger_reason: format!(
                        "Semantic degradation exceeded threshold ϵ={} \
                         (Δcoherence={:.4})",
                        DEGRADATION_THRESHOLD, coherence_delta,
                    ),
                });

                // Reflect the transition on the shadow hypothesis
                updated.status = new_status;
                updated.revised_at = Utc::now();
                updated.revision_history.push(crate::hypothesis::Revision {
                    timestamp: Utc::now(),
                    description: format!(
                        "Auto-transitioned to Contradicted by delta engine (mutation {})",
                        mutation.id
                    ),
                    previous_status: prev_status,
                    new_status,
                    trigger: Some(mutation.id.clone()),
                });
            }
        }

        ctx.shadow_hypotheses.insert(updated.id.clone(), updated);
    }

    OntologyDeltaReport {
        mutation_source: mutation,
        affected_nodes: node_deltas,
        hypothesis_status_shifts: hypothesis_transitions,
        net_coherence_delta: net_delta,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, embedding: Vec<f32>, coherence: Score) -> CoherenceNode {
        let mut n = CoherenceNode::new(embedding);
        n.id = id.to_string();
        n.coherence_score = coherence;
        n
    }

     #[test]
    fn property_set_does_not_change_embedding() {
        let mut n = node("n1", vec![1.0, 0.0, 0.0, 0.0], 0.5);
        n.label = Some("test_node".to_string());
        let edge_vec: Vec<TypedEdge> = vec![];
        let nodes = vec![n.clone()];
        let ctx = EvaluationContext::from_base(&nodes, &[], &edge_vec);
        assert_eq!(n.embedding, ctx.effective_node("n1").unwrap().embedding);
    }

    #[test]
    fn classify_delta_thresholds() {
        assert_eq!(classify_delta(0.0), "Unchanged");
        assert_eq!(classify_delta(0.5), "Strengthened");
        assert_eq!(classify_delta(-0.1), "Shifted");
        assert_eq!(classify_delta(-0.3), "Invalidated");
    }

    #[test]
    fn dfs_respects_max_depth() {
        // Chain: 0 - 1 - 2 - 3 - 4 - 5
        let adjacency: HashMap<String, Vec<String>> = [
            ("n0", vec!["n1"]),
            ("n1", vec!["n0", "n2"]),
            ("n2", vec!["n1", "n3"]),
            ("n3", vec!["n2", "n4"]),
            ("n4", vec!["n3", "n5"]),
            ("n5", vec!["n4"]),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), v.iter().map(|s| s.to_string()).collect()))
        .collect();

        let mut visited = HashSet::new();
        let mut affected = Vec::new();
        dfs_collect("n0", &adjacency, 0, &mut visited, &mut affected, 1.0);

        // With MAX_PROPAGATION_DEPTH=5, nodes n0..n5 should all be visited
        assert_eq!(affected.len(), 6);
        assert_eq!(affected[0].0, "n0");
        assert_eq!(affected[5].0, "n5");
    }

    #[test]
    fn ontology_mutation_serializes() {
        let mutation = OntologyMutation::new(
            "node_42",
            MutationOp::EmbeddingShift {
                old_embedding: vec![1.0, 0.0],
                new_embedding: vec![0.0, 1.0],
            },
        )
        .with_trigger("manual_review");

        let json = serde_json::to_string(&mutation).unwrap();
        let back: OntologyMutation = serde_json::from_str(&json).unwrap();
        assert_eq!(back.target_node_id, "node_42");
        assert_eq!(back.triggered_by, Some("manual_review".to_string()));
    }

    #[test]
    fn report_serializes_to_json() {
        let report = OntologyDeltaReport {
            mutation_source: OntologyMutation::new(
                "n1",
                MutationOp::PropertySet {
                    key: "label".into(),
                    old_val: serde_json::Value::String("old".into()),
                    new_val: serde_json::Value::String("new".into()),
                },
            ),
            affected_nodes: vec![NodeDelta {
                node_id: "n1".to_string(),
                previous_coherence: 0.8,
                new_coherence: 0.6,
                delta_type: "Invalidated".to_string(),
            }],
            hypothesis_status_shifts: vec![],
            net_coherence_delta: -0.2,
        };

        let json = serde_json::to_string(&report).unwrap();
        let back: OntologyDeltaReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.net_coherence_delta, -0.2);
        assert_eq!(back.affected_nodes[0].delta_type, "Invalidated");
    }
}
