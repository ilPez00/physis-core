//! Diffusion over the projected world graph.
//!
//! Latent reasoning here means one concrete thing: scores spread from seed
//! nodes across outgoing edges, decaying each step, with mass conserved.
//! No model, no sampling, no hidden state — a deterministic walk a caller
//! can replay and cite.

use std::collections::{HashMap, HashSet};

use crate::query::WorldGraph;

/// Spread `seeds` across `graph` for `steps` iterations.
///
/// Each node keeps `1 - decay` of its score and passes `decay` evenly to its
/// outgoing neighbors. `decay` must sit in `[0.0, 1.0]`; `steps == 0`
/// returns the seeds unchanged.
pub fn diffuse(
    graph: &WorldGraph,
    seeds: &HashMap<String, f32>,
    steps: usize,
    decay: f32,
) -> HashMap<String, f32> {
    assert!(
        (0.0..=1.0).contains(&decay),
        "decay must be in [0.0, 1.0], got {decay}"
    );
    let mut scores = seeds.clone();
    for _ in 0..steps {
        let mut next = HashMap::new();
        for (node, score) in &scores {
            *next.entry(node.clone()).or_default() += score * (1.0 - decay);
            let mut targets = HashSet::new();
            if let Some(outgoing) = graph.edges.get(node) {
                for objects in outgoing.values() {
                    targets.extend(objects.iter().cloned());
                }
            }
            if targets.is_empty() {
                // Sink: retained mass already accounts for the full score.
                continue;
            }
            let share = score * decay / targets.len() as f32;
            for target in targets {
                *next.entry(target).or_default() += share;
            }
        }
        scores = next;
    }
    scores
}

/// Scores sorted highest first for readable ranking.
pub fn ranked(scores: &HashMap<String, f32>) -> Vec<(String, f32)> {
    let mut out: Vec<(String, f32)> =
        scores.iter().map(|(k, v)| (k.clone(), *v)).collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::{Predicate, Proposition, Triple, WorldState};

    fn chain() -> WorldGraph {
        let world = WorldState {
            facts: vec![
                Proposition::observed(Triple::new("A", Predicate::Produces, "B"), "t"),
                Proposition::observed(Triple::new("B", Predicate::Causes, "C"), "t"),
            ],
            constraints: Vec::new(),
        };
        WorldGraph::from_world(&world, &[])
    }

    #[test]
    fn zero_steps_returns_seeds() {
        let graph = chain();
        let seeds = HashMap::from([("A".to_string(), 1.0)]);
        assert_eq!(diffuse(&graph, &seeds, 0, 0.5), seeds);
    }

    #[test]
    fn scores_spread_forward_and_conserve_mass() {
        let graph = chain();
        let seeds = HashMap::from([("A".to_string(), 1.0)]);
        let scores = diffuse(&graph, &seeds, 2, 0.5);
        let total: f32 = scores.values().sum();
        assert!((total - 1.0).abs() < 1e-5, "mass not conserved: {total}");
        assert!(scores.get("B").unwrap_or(&0.0) > &0.0);
        assert!(scores.get("C").unwrap_or(&0.0) > &0.0);
        let ranked = ranked(&scores);
        assert_eq!(ranked[0].0, "B");
    }

    #[test]
    #[should_panic(expected = "decay must be in [0.0, 1.0]")]
    fn decay_outside_unit_interval_panics() {
        let graph = chain();
        let seeds = HashMap::from([("A".to_string(), 1.0)]);
        diffuse(&graph, &seeds, 1, 1.5);
    }
}
