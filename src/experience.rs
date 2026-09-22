//! Execution experience: remember what transitions actually did.
//!
//! The observation log records what was seen; [`crate::transition`] records
//! single executions. This module keeps the structured trace across runs so a
//! repeated goal+action can consult its own history before acting again.
//!
//! Case-based reuse here is exact and boring on purpose: same goal plus same
//! action shape recalls past outcomes and their success rate. Similarity
//! ranking can arrive when a measured retrieval need exists.

use serde::{Deserialize, Serialize};

use crate::models::Score;
use crate::transition::{Action, ComparisonResult, ExecutedTransition};

/// One remembered execution, stripped to what reuse needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub source_id: String,
    pub goal: String,
    pub action: Action,
    pub comparison: ComparisonResult,
    pub committed: bool,
    pub confidence: Score,
    pub cost: f64,
    pub recorded_at: chrono::DateTime<chrono::Utc>,
}

/// In-memory case store. Serializable, so persistence stays a caller choice.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExperienceStore {
    experiences: Vec<Experience>,
}

impl ExperienceStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one execution as experience.
    pub fn record(&mut self, executed: &ExecutedTransition) -> Experience {
        let experience = Experience {
            source_id: executed.transition.id.clone(),
            goal: executed.transition.goal.clone(),
            action: executed.transition.action.clone(),
            comparison: executed.comparison.clone(),
            committed: executed.committed,
            confidence: executed.transition.confidence,
            cost: executed.transition.cost,
            recorded_at: chrono::Utc::now(),
        };
        self.experiences.push(experience.clone());
        experience
    }

    /// Past runs of exactly this goal and action shape, newest first.
    pub fn recall_exact(&self, goal: &str, action: &Action, limit: usize) -> Vec<Experience> {
        let mut out: Vec<Experience> = self
            .experiences
            .iter()
            .rev()
            .filter(|e| e.goal == goal && e.action == *action)
            .take(limit)
            .cloned()
            .collect();
        out.truncate(limit);
        out
    }

    /// Past runs of this goal with the same action kind, newest first.
    pub fn recall_kind(&self, goal: &str, action: &Action, limit: usize) -> Vec<Experience> {
        let kind = action_kind(action);
        self.experiences
            .iter()
            .rev()
            .filter(|e| e.goal == goal && action_kind(&e.action) == kind)
            .take(limit)
            .cloned()
            .collect()
    }

    /// Fraction of the given experiences that committed. `None` when empty.
    pub fn success_rate(experiences: &[Experience]) -> Option<f64> {
        if experiences.is_empty() {
            return None;
        }
        let committed = experiences.iter().filter(|e| e.committed).count();
        Some(committed as f64 / experiences.len() as f64)
    }
}

/// Stable action shape name for kind-level recall.
pub fn action_kind(action: &Action) -> &'static str {
    match action {
        Action::Shell { .. } => "shell",
        Action::Fs { .. } => "fs",
        Action::Edit { .. } => "edit",
        Action::Graph { .. } => "graph",
        Action::Notebook { .. } => "notebook",
        Action::Praxis { .. } => "praxis",
        Action::Telemetry { .. } => "telemetry",
        Action::AgentCall { .. } => "agent-call",
        Action::MpcCall { .. } => "mpc-call",
        Action::Custom { .. } => "custom",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transition::{StateDelta, Transition};

    fn executed(goal: &str, action: Action, committed: bool) -> ExecutedTransition {
        let transition = Transition {
            id: format!("{goal}-001"),
            goal: goal.to_string(),
            requires: Vec::new(),
            action,
            predicts: StateDelta::default(),
            ensures: Vec::new(),
            evidence_requirements: Vec::new(),
            confidence: 0.7,
            cost: 0.2,
        };
        ExecutedTransition {
            transition,
            observed_delta: StateDelta::default(),
            comparison: if committed {
                ComparisonResult::ExactMatch
            } else {
                ComparisonResult::Conflict
            },
            preconditions_met: true,
            postconditions_met: committed,
            committed,
            trace: crate::transition::ExecutedTrace {
                before: crate::transform::WorldState::default(),
                after: crate::transform::WorldState::default(),
                action: Action::Shell {
                    cmd: "true".to_string(),
                    cwd: "/tmp".to_string(),
                },
                predicted_vs_observed: (StateDelta::default(), StateDelta::default()),
                observations: Vec::new(),
                errors: Vec::new(),
                verdict: ComparisonResult::ExactMatch,
            },
        }
    }

    #[test]
    fn exact_recall_and_success_rate() {
        let mut store = ExperienceStore::new();
        let action = Action::Edit {
            path: "/test.rs".to_string(),
            old_body: "old".to_string(),
            new_body: "new".to_string(),
        };
        store.record(&executed("demo", action.clone(), true));
        store.record(&executed("demo", action.clone(), false));

        let recalled = store.recall_exact("demo", &action, 10);
        assert_eq!(recalled.len(), 2);
        assert_eq!(ExperienceStore::success_rate(&recalled), Some(0.5));

        let other = Action::Shell {
            cmd: "echo hi".to_string(),
            cwd: "/tmp".to_string(),
        };
        assert!(store.recall_exact("demo", &other, 10).is_empty());
        assert!(ExperienceStore::success_rate(&[]).is_none());
    }

    #[test]
    fn kind_recall_groups_action_shapes() {
        let mut store = ExperienceStore::new();
        store.record(&executed(
            "demo",
            Action::Edit {
                path: "/a.rs".to_string(),
                old_body: "a".to_string(),
                new_body: "b".to_string(),
            },
            true,
        ));
        let recalled = store.recall_kind(
            "demo",
            &Action::Edit {
                path: "/other.rs".to_string(),
                old_body: "x".to_string(),
                new_body: "y".to_string(),
            },
            10,
        );
        assert_eq!(recalled.len(), 1);
    }
}
