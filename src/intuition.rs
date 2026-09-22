//! Intuition is proposal; Physis is evaluation.
//!
//! A proposal names an action worth trying. It carries no guarantees: no
//! verified preconditions, no trusted prediction, no checked postconditions.
//! Those belong to the deterministic machinery in [`crate::transition`]
//! (`verify_preconditions`, `predict_delta`, execution, comparison, ensures).
//!
//! This split keeps fallible guessing outside the world-state core. The core
//! never trusts a proposal; it evaluates one.

use serde::{Deserialize, Serialize};

use crate::models::Score;
use crate::transition::{Action, Transition};
use crate::transform::WorldState;

/// A fallible guess at a useful transition.
///
/// Deliberately thinner than [`Transition`]: no `requires`, no `predicts`,
/// no `ensures`. An [`Intuition`] produces these; the evaluator decides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    pub goal: String,
    pub action: Action,
    pub rationale: String,
    pub confidence: Score,
    pub cost: f64,
}

/// Proposes candidate transitions. Never verifies them.
pub trait Intuition: Send + Sync {
    fn propose(&self, world: &WorldState, goal: &str, limit: usize) -> Vec<Proposal>;
}

impl Proposal {
    /// Wrap a proposal as an unevaluated [`Transition`].
    ///
    /// Empty `requires`/`ensures` and a default prediction mark it as
    /// not-yet-evaluated: the evaluator must fill or check these before
    /// anything is trusted or executed.
    pub fn as_unevaluated_transition(&self, id: impl Into<String>) -> Transition {
        Transition {
            id: id.into(),
            goal: self.goal.clone(),
            requires: Vec::new(),
            action: self.action.clone(),
            predicts: crate::transition::StateDelta::default(),
            ensures: Vec::new(),
            evidence_requirements: Vec::new(),
            confidence: self.confidence,
            cost: self.cost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::{Predicate, Proposition, Triple};
    use crate::transition::{predict_delta, verify_preconditions};

    struct FixedProposer;

    impl Intuition for FixedProposer {
        fn propose(&self, _world: &WorldState, goal: &str, limit: usize) -> Vec<Proposal> {
            vec![Proposal {
                goal: goal.to_string(),
                action: Action::Edit {
                    path: "/test.rs".to_string(),
                    old_body: "old".to_string(),
                    new_body: "new".to_string(),
                },
                rationale: "canned guess for the boundary test".to_string(),
                confidence: 0.4,
                cost: 0.1,
            }]
            .into_iter()
            .take(limit)
            .collect()
        }
    }

    #[test]
    fn proposals_carry_no_guarantees_until_evaluated() {
        let world = WorldState {
            facts: vec![Proposition::observed(
                Triple::new("file:A.rs", Predicate::Produces, "code"),
                "t",
            )],
            constraints: Vec::new(),
        };
        let proposals = FixedProposer.propose(&world, "demo", 3);
        assert_eq!(proposals.len(), 1);

        let unevaluated = proposals[0].as_unevaluated_transition("intuition-001");
        assert!(unevaluated.requires.is_empty());
        assert!(unevaluated.ensures.is_empty());

        // The evaluator runs on the same value the proposer produced.
        let evaluated_prediction = predict_delta(&world, &unevaluated);
        assert!(!evaluated_prediction.added.is_empty() || !evaluated_prediction.removed.is_empty());
        assert!(verify_preconditions(&world, &unevaluated));
    }
}
