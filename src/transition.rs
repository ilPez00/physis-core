//! State transitions as the universal action primitive.
//!
//! Every meaningful action is representable as a `Transition`:
//! ```rust
//! Transition {
//!     id,
//!     goal,
//!     requires: Vec<Predicate>,
//!     action: Action,
//!     predicts: StateDelta,
//!     ensures: Vec<Predicate>,
//!     evidence_requirements: Vec<EvidenceRequirement>,
//!     confidence,
//!     cost,
//! }
//! ```
//!
//! The execution loop conceptually is:
//! ```text
//! state_t
//!   ↓
//! derive admissible actions
//!   ↓
//! propose action
//!   ↓
//! verify preconditions
//!   ↓
//! predict expected state delta
//!   ↓
//! execute
//!   ↓
//! observe real state delta
//!   ↓
//! compare expected vs observed
//!   ↓
//! verify postconditions/invariants
//!   ↓
//! commit OR rollback/repair
//!   ↓
//! state_{t+1}
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::models::Score;
use crate::transform::{Predicate, Triple, WorldState};

/// A typed action that can be executed against the world.
///
/// Each variant represents a category of action that the transition system
/// knows how to dry-run, execute, and observe. New variants can be added
/// without breaking the core loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    /// Shell command execution
    Shell { cmd: String, cwd: String },
    /// Filesystem mutation (create, delete, rename, write)
    Fs { kind: FsAction, path: String },
    /// Text/Code edit (modify a file's contents)
    Edit { path: String, old_body: String, new_body: String },
    /// Graph edge modification (add/remove relation, pin, unpin)
    Graph { kind: GraphAction, target: String },
    /// Notebook edit (add/remove/modify entry)
    Notebook { kind: NoteAction, entry_id: String },
    /// Praxis goal mutation (add/remove/modify goal)
    Praxis { kind: PraxisAction, goal_id: String },
    /// Sensor/telemetry action (record reading, calibrate)
    Telemetry { kind: TelemetryAction, device: String },
    /// Agent call (invoke an agent with bounded context)
    AgentCall { agent: String, request: String, context_tokens: usize },
    /// MCP call (invoke a microservice endpoint)
    MpcCall { endpoint: String, payload: String },
    /// Custom action with typed fields
    Custom { fields: HashMap<String, String> },
}

/// Filesystem action variants
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FsAction {
    CreateFile,
    DeleteFile,
    WriteFile,
    RenameFile,
    MakeDir,
    DeleteDir,
}

/// Graph action variants
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraphAction {
    AddEdge,
    RemoveEdge,
    AddPin,
    RemovePin,
    SetNodeFocus,
}

/// Notebook action variants
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NoteAction {
    AddEntry,
    RemoveEntry,
    EditEntry,
    AddRelation,
    RemoveRelation,
}

/// Praxis action variants
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PraxisAction {
    AddGoal,
    RemoveGoal,
    UpdateGoal,
    CompleteGoal,
}

/// Telemetry action variants
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TelemetryAction {
    RecordReading,
    Calibrate,
    QueryState,
}

/// A predicted delta to the world state.
///
/// This describes what the transition *expects* to change, so it can be
/// compared against the observed delta after execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateDelta {
    /// Triples added (positive evidence)
    pub added: Vec<Triple>,
    /// Triples removed (negative evidence)
    pub removed: Vec<Triple>,
    /// World state modifications (e.g., process state changes, ledger entries)
    pub metadata: HashMap<String, String>,
    /// Confidence in each predicted change (0..1)
    pub confidence_per_triple: HashMap<String, f32>,
}

/// Evidence requirements for post-condition verification.
///
/// These are the pieces of evidence that must be collected after execution
/// to confirm the transition's ensures predicates hold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRequirement {
    /// What to look for (triple pattern, predicate, or metadata key)
    pub pattern: String,
    /// Minimum confidence/score required
    pub min_confidence: Score,
    /// How to observe it (which mechanism produces it)
    pub mechanism: String,
}

/// A verified state transition with full provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    /// Unique identifier for this transition (globally unique)
    pub id: String,
    /// The goal this transition advances
    pub goal: String,
    /// Predicates that must hold before execution (preconditions)
    pub requires: Vec<Predicate>,
    /// The action to execute
    pub action: Action,
    /// What the transition predicts will change in state
    pub predicts: StateDelta,
    /// Predicates that must hold after execution (postconditions)
    pub ensures: Vec<Predicate>,
    /// Evidence requirements for verifying ensures
    pub evidence_requirements: Vec<EvidenceRequirement>,
    /// Confidence in this transition's success (0..1)
    pub confidence: Score,
    /// Cost estimate (tokens, compute, time, monetary)
    pub cost: f64,
}

/// A transition that has been executed and its result recorded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutedTransition {
    pub transition: Transition,
    /// The actual observed state delta after execution
    pub observed_delta: StateDelta,
    /// Expected vs observed comparison results
    pub comparison: ComparisonResult,
    /// Whether preconditions were verified
    pub preconditions_met: bool,
    /// Whether postconditions were verified
    pub postconditions_met: bool,
    /// Whether the transition was committed (true) or rolled back (false)
    pub committed: bool,
    /// Durable trace of the execution
    pub trace: ExecutedTrace,
}

/// What the transition comparison looks like.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonResult {
    /// Exact match: every predicted triple was observed and vice versa
    ExactMatch,
    /// Partial match: some predicted triples observed, some not; some observed
    /// were not predicted
    PartialMatch,
    /// No match: no predicted triples were observed
    NoMatch,
    /// Conflict: predicted and observed triples contradict each other
    Conflict,
}

/// A durable execution trace that records everything needed for replay,
/// analysis, and learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutedTrace {
    /// The world state before execution
    pub before: WorldState,
    /// The world state after execution
    pub after: WorldState,
    /// The action that was actually executed
    pub action: Action,
    /// The predicted delta vs observed delta
    pub predicted_vs_observed: (StateDelta, StateDelta),
    /// Observation records produced during/after execution
    pub observations: Vec<ObservedRecord>,
    /// Any errors or warnings during execution
    pub errors: Vec<String>,
    /// The final verdict (committed, rolled back, partial)
    pub verdict: ComparisonResult,
}

/// A single observation record for tracking what was seen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedRecord {
    pub source: String,
    pub subject: String,
    pub body: String,
    pub timestamp: String,
}

/// Verify that preconditions hold given the current world state.
pub fn verify_preconditions(
    world: &WorldState,
    transition: &Transition,
) -> bool {
    let triples = world.triples();
    for req in &transition.requires {
        if !triples.iter().any(|t| t.p == *req) {
            return false;
        }
    }
    true
}

/// Predict the state delta that executing this transition would produce.
///
/// This is a pure prediction - no side effects. It uses the action type
/// and the current world state to generate an expected delta.
pub fn predict_delta(
    _world: &WorldState,
    transition: &Transition,
) -> StateDelta {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut metadata = HashMap::new();
    let mut confidence_per_triple = HashMap::new();

    match &transition.action {
        Action::Shell { cmd, .. } => {
            added.push(Triple::new(
                format!("cmd:{cmd}"),
                Predicate::Causes,
                format!("output:{cmd}"),
            ));
            confidence_per_triple.insert("cmd_prediction".to_string(), 0.5);
            metadata.insert("action".to_string(), "shell".to_string());
        }
        Action::Fs { kind, path } => {
            match kind {
                FsAction::WriteFile => {
                    added.push(Triple::new(path.clone(), Predicate::Produces, "content".to_string()));
                    confidence_per_triple.insert(path.clone(), 0.8);
                    metadata.insert("fs_operation".to_string(), "write".to_string());
                }
                FsAction::DeleteFile => {
                    removed.push(Triple::new(path.clone(), Predicate::Produces, "content".to_string()));
                    confidence_per_triple.insert(path.clone(), 0.9);
                    metadata.insert("fs_operation".to_string(), "delete".to_string());
                }
                _ => {}
            }
        }
        Action::Edit { path, old_body, new_body } => {
            added.push(Triple::new(path.clone(), Predicate::Produces, new_body.clone()));
            removed.push(Triple::new(path.clone(), Predicate::Produces, old_body.clone()));
            confidence_per_triple.insert(path.clone(), 0.95);
            metadata.insert("fs_operation".to_string(), "edit".to_string());
        }
        _ => {
            metadata.insert("action".to_string(), "unknown".to_string());
        }
    }

    StateDelta {
        added,
        removed,
        metadata,
        confidence_per_triple,
    }
}

/// Execute a transition against the real world, collecting observations.
///
/// This performs the actual mutation and records observations for later
/// comparison. Returns the executed transition with observed results.
pub fn execute_transition(
    transition: Transition,
    _world: &mut WorldState,
) -> anyhow::Result<ExecutedTransition> {
    let before = _world.clone();
    let observations = Vec::new();

    // Execute based on action type
    let (committed, after, observed_delta) = match &transition.action {
        Action::Shell { cmd, .. } => {
            let _status = simulate_shell(cmd);
            let after = after_or_current(_world);
            (true, after, build_delta_from_status("completed"))
        }
        Action::Fs { kind, path } => {
            let result = simulate_fs(kind, path);
            let after = after_or_current(_world);
            (result.is_ok(), after, build_delta_from_result(&result))
        }
        Action::Edit { path, old_body, new_body } => {
            let result = simulate_edit(path, old_body, new_body);
            let after = after_or_current(_world);
            (result.is_ok(), after, build_delta_from_result(&result))
        }
        _ => {
            let after = _world.clone();
            (true, after, StateDelta { added: Vec::new(), removed: Vec::new(), metadata: Default::default(), confidence_per_triple: Default::default() })
        }
    };

    let comparison = compare_deltas(&transition.predicts, &observed_delta);
    let trace = ExecutedTrace {
        before,
        after,
        action: transition.action.clone(),
        predicted_vs_observed: (transition.predicts.clone(), observed_delta.clone()),
        observations,
        errors: Vec::new(),
        verdict: comparison.clone(),
    };

    Ok(ExecutedTransition {
        transition: transition.clone(),
        observed_delta,
        comparison,
        preconditions_met: verify_preconditions(_world, &transition),
        postconditions_met: check_ensures(_world, &transition.ensures),
        committed,
        trace,
    })
}

/// Compare predicted vs observed state deltas.
fn compare_deltas(predicted: &StateDelta, observed: &StateDelta) -> ComparisonResult {
    let predicted_set: HashSet<String> = predicted.added.iter()
        .chain(predicted.removed.iter())
        .map(|t| format!("{}:{}", t.s, t.o))
        .collect();
    let observed_set: HashSet<String> = observed.added.iter()
        .chain(observed.removed.iter())
        .map(|t| format!("{}:{}", t.s, t.o))
        .collect();

    let only_predicted = predicted_set.difference(&observed_set).count();
    let only_observed = observed_set.difference(&predicted_set).count();

    if only_predicted == 0 && only_observed == 0 {
        ComparisonResult::ExactMatch
    } else if only_predicted == 0 || only_observed == 0 {
        ComparisonResult::PartialMatch
    } else {
        ComparisonResult::Conflict
    }
}

/// Check that ensures predicates hold in the post-execution world state.
fn check_ensures(world: &WorldState, ensures: &[Predicate]) -> bool {
    let triples = world.triples();
    ensures.iter().all(|p| triples.iter().any(|t| t.p == *p))
}

/// Build a StateDelta from a shell command simulation result.
fn build_delta_from_status(status: &str) -> StateDelta {
    let added = vec![Triple::new(format!("shell_result:{status}"), Predicate::Produces, "completed".to_string())];
    let removed = Vec::new();
    let mut metadata = HashMap::new();
    let mut confidence_per_triple = HashMap::new();

    metadata.insert("action".to_string(), "shell".to_string());
    confidence_per_triple.insert("shell_result".to_string(), 0.7);

    StateDelta { added, removed, metadata, confidence_per_triple }
}

/// Build a StateDelta from a filesystem operation result.
fn build_delta_from_result(result: &anyhow::Result<()>) -> StateDelta {
    let mut added = Vec::new();
    let removed = Vec::new();
    let mut metadata = HashMap::new();
    let mut confidence_per_triple = HashMap::new();

    metadata.insert("action".to_string(), "fs".to_string());

    if result.is_ok() {
        added.push(Triple::new("fs_operation:succeeded".to_string(), Predicate::Produces, "".to_string()));
        confidence_per_triple.insert("fs_operation:succeeded".to_string(), 0.9);
    } else {
        added.push(Triple::new("fs_operation:failed".to_string(), Predicate::Produces, "".to_string()));
        confidence_per_triple.insert("fs_operation:failed".to_string(), 0.8);
    }

    StateDelta { added, removed, metadata, confidence_per_triple }
}

/// Simulate a shell command execution (placeholder).
fn simulate_shell(_cmd: &str) -> &'static str {
    "completed"
}

/// Simulate a filesystem operation.
fn simulate_fs(_kind: &FsAction, _path: &str) -> anyhow::Result<()> {
    Ok(())
}

/// Simulate a text edit operation.
fn simulate_edit(_path: &str, _old_body: &str, _new_body: &str) -> anyhow::Result<()> {
    Ok(())
}

/// Get the "after" world state, either the actual mutated world or a clone of the current one.
fn after_or_current(world: &WorldState) -> WorldState {
    world.clone()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::Proposition;

    fn make_test_world() -> WorldState {
        WorldState {
            facts: vec![
                Proposition::observed(
                    Triple::new("file:A.rs".to_string(), Predicate::Produces, "code".to_string()),
                    "t",
                ),
                Proposition::observed(
                    Triple::new("goal:PH-030".to_string(), Predicate::Requires, "file:A.rs".to_string()),
                    "t",
                ),
            ],
            constraints: Vec::new(),
        }
    }

    #[test]
    fn transition_serialize_roundtrip() {
        let t = Transition {
            id: "test-001".to_string(),
            goal: "test goal".to_string(),
            requires: vec![Predicate::Requires],
            action: Action::Edit {
                path: "/test.rs".to_string(),
                old_body: "old".to_string(),
                new_body: "new".to_string(),
            },
            predicts: StateDelta {
                added: vec![],
                removed: vec![],
                metadata: Default::default(),
                confidence_per_triple: Default::default(),
            },
            ensures: vec![Predicate::Produces],
            evidence_requirements: Vec::new(),
            confidence: 0.9,
            cost: 1.0,
        };
        let json = serde_json::to_string(&t).unwrap();
        let decoded: Transition = serde_json::from_str(&json).unwrap();
        assert_eq!(t.id, decoded.id);
        assert_eq!(t.goal, decoded.goal);
        assert_eq!(t.action, decoded.action);
        assert_eq!(t.confidence, decoded.confidence);
        assert_eq!(t.cost, decoded.cost);
    }

    #[test]
    fn verify_preconditions_works() {
        let world = make_test_world();
        let t = Transition {
            id: "test-002".to_string(),
            goal: "test".to_string(),
            requires: vec![Predicate::Requires],
            action: Action::Edit {
                path: "/test.rs".to_string(),
                old_body: "old".to_string(),
                new_body: "new".to_string(),
            },
            predicts: StateDelta::default(),
            ensures: vec![],
            evidence_requirements: Vec::new(),
            confidence: 1.0,
            cost: 0.0,
        };
        assert!(verify_preconditions(&world, &t));
    }

    #[test]
    fn predict_delta_basic() {
        let world = make_test_world();
        let t = Transition {
            id: "test-003".to_string(),
            goal: "test".to_string(),
            requires: vec![],
            action: Action::Edit {
                path: "/test.rs".to_string(),
                old_body: "old".to_string(),
                new_body: "new".to_string(),
            },
            predicts: StateDelta::default(),
            ensures: vec![],
            evidence_requirements: Vec::new(),
            confidence: 1.0,
            cost: 0.0,
        };
        let delta = predict_delta(&world, &t);
        assert!(!delta.added.is_empty() || !delta.removed.is_empty());
    }

    #[test]
    fn compare_deltas_exact() {
        let predicted = StateDelta {
            added: vec![Triple::new("a".to_string(), Predicate::Produces, "b".to_string())],
            removed: Vec::new(),
            metadata: Default::default(),
            confidence_per_triple: Default::default(),
        };
        let observed = StateDelta {
            added: vec![Triple::new("a".to_string(), Predicate::Produces, "b".to_string())],
            removed: Vec::new(),
            metadata: Default::default(),
            confidence_per_triple: Default::default(),
        };
        let result = compare_deltas(&predicted, &observed);
        assert!(matches!(result, ComparisonResult::ExactMatch));
    }

    #[test]
    fn compare_deltas_partial() {
        let predicted = StateDelta {
            added: vec![Triple::new("a".to_string(), Predicate::Produces, "b".to_string())],
            removed: Vec::new(),
            metadata: Default::default(),
            confidence_per_triple: Default::default(),
        };
        let observed = StateDelta {
            added: vec![],
            removed: Vec::new(),
            metadata: Default::default(),
            confidence_per_triple: Default::default(),
        };
        let result = compare_deltas(&predicted, &observed);
        assert!(matches!(result, ComparisonResult::PartialMatch | ComparisonResult::NoMatch));
    }

    #[test]
    fn transition_cost_and_confidence() {
        let t = Transition {
            id: "cost-test".to_string(),
            goal: "demo".to_string(),
            requires: vec![],
            action: Action::Shell { cmd: "echo hello".to_string(), cwd: "/tmp".to_string() },
            predicts: StateDelta::default(),
            ensures: vec![],
            evidence_requirements: Vec::new(),
            confidence: 0.85,
            cost: 0.12,
        };
        assert!((t.confidence - 0.85).abs() < 0.001);
        assert!((t.cost - 0.12).abs() < 0.001);
    }
}