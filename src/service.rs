//! Shared service layer — one implementation, three shells (CLI, TUI, web).
//!
//! Built additively (2026-09-13): nothing in the engine reads this module, so
//! no existing behaviour changes whether or not it compiles in. The three
//! surfaces call the same functions here:
//!
//! - `physis-core ask` (CLI) → [`ask`]
//! - the studio's `/api/v1/ask` + `/ask` page (web) → [`ask`]
//! - `physis-core tui`'s ASK / LEDGER / WATCH / STATUS tabs (TUI) → [`ask`],
//!   [`ledger`], [`watch_tail`], [`status`]
//!
//! The web Ask console and the CLI share the persisted core through
//! `nodes.json` — the same file `studio.rs` reads and writes — so an ask run
//! from a browser and one from a terminal land in one substrate, and every ask
//! is appended to the observation log (`source: "model"`) so the epistemic
//! loop of `act`/`claim` can cite it.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::core::PhysisCore;
use crate::embed;
use crate::map;
use crate::notebook::{self, Answer, Draft};
use crate::observe::{self, Observation};
use crate::rag;

// ── Core persistence — same file, same shape, as studio.rs ──────────────────

/// Load the persisted engine state. Tolerates a missing or broken file the
/// same way `studio.rs` does: fresh state, never an error, so a read-only
/// surface cannot crash on state another surface corrupted.
pub fn load_core(data_dir: &Path) -> PhysisCore {
    let json = std::fs::read_to_string(data_dir.join("nodes.json")).unwrap_or_default();
    if json.trim().is_empty() {
        return PhysisCore::new();
    }
    PhysisCore::from_json(&json).unwrap_or_else(|_| PhysisCore::new())
}

/// Persist the engine state (write-through; callers decide when).
pub fn save_core(data_dir: &Path, core: &PhysisCore) -> anyhow::Result<()> {
    let _ = std::fs::create_dir_all(data_dir);
    std::fs::write(data_dir.join("nodes.json"), core.to_json()?)?;
    Ok(())
}

// ── Ask — the grounded-answer pipeline behind CLI / web / TUI ───────────────

fn default_corpus() -> String {
    "examples".to_string()
}

/// One ask request. `corpus` is a directory (markdown/txt), the same input
/// `notebook`/`context`/`chain` take.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AskRequest {
    pub query: String,
    #[serde(default = "default_corpus")]
    pub corpus: String,
    /// Token budget for the compiled context. Default mirrors `notebook`.
    #[serde(default)]
    pub budget: Option<usize>,
    /// Draft-and-fill: let the n-gram table draft what the retrieved context
    /// supports and mark the rest as gaps, instead of calling a generator.
    #[serde(default)]
    pub draft: bool,
    /// Probability floor below which the table declines to continue.
    #[serde(default)]
    pub confidence: Option<f32>,
}

/// What came back, plus the accounting that makes it auditable.
#[derive(Debug, Clone, Serialize)]
pub struct AskOutcome {
    pub query: String,
    pub corpus: String,
    /// Which embedder answered — `"random-projection"` or the ONNX model id.
    /// Surfaced for the same reason the CLI labels vectors: a lexical hit
    /// must not present itself as a semantic one.
    pub embedder: String,
    /// Set when `draft` was requested.
    pub draft: Option<Draft>,
    /// Set otherwise (synthesised when an oracle endpoint is configured,
    /// extractive with a recorded `degraded` reason when not).
    pub answer: Option<Answer>,
    /// `draft.table_share()` — the number that says whether the
    /// draft-and-fill architecture applies to this question at all.
    pub table_share: Option<f32>,
    /// Context tokens the ask actually consumed, for the ledger trail.
    pub context_tokens: usize,
    /// Sequence of the observation appended to the log, when it was recorded.
    pub recorded_seq: Option<u64>,
    pub elapsed_ms: f64,
}

/// Grounded answer over a local corpus — the pipeline `notebook` ships, with
/// the ask appended to the observation log so the epistemic loop can cite it.
pub fn ask(req: &AskRequest) -> anyhow::Result<AskOutcome> {
    let corpus_dir = PathBuf::from(&req.corpus);
    let docs = map::load_corpus(&corpus_dir)?;
    anyhow::ensure!(
        !docs.is_empty(),
        "no corpus documents under {}",
        corpus_dir.display()
    );
    let (embedder, kind) = embed::select(384);
    let budget = req.budget.unwrap_or(1200).max(64);
    let started = std::time::Instant::now();

    let texts: Vec<String> = docs.iter().map(|(_, b)| b.clone()).collect();
    let top_k = docs.len().clamp(3, 8);
    let (result, _) =
        rag::retrieve_from_texts(&texts, &req.query, embedder.as_ref(), budget, top_k);

    let mut outcome = if req.draft {
        let d = notebook::draft(&result, &req.query, 40, req.confidence.unwrap_or(0.0))?;
        let table_share = d.table_share();
        AskOutcome {
            query: req.query.clone(),
            corpus: req.corpus.clone(),
            embedder: kind.to_string(),
            draft: Some(d),
            answer: None,
            table_share: Some(table_share),
            context_tokens: result.total_tokens,
            recorded_seq: None,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        }
    } else {
        let a = notebook::answer(&docs, &req.query, embedder.as_ref(), budget)?;
        AskOutcome {
            query: req.query.clone(),
            corpus: req.corpus.clone(),
            embedder: kind.to_string(),
            draft: None,
            answer: Some(a),
            table_share: None,
            context_tokens: result.total_tokens,
            recorded_seq: None,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        }
    };

    append_ask_observation(&mut outcome);
    Ok(outcome)
}

/// Append the ask to the observation log. Append-only and never fatal: an
/// ask that succeeded but could not be recorded is still returned, with
/// `recorded_seq: None` saying so.
fn append_ask_observation(outcome: &mut AskOutcome) {
    let obs = Observation::new("model", format!("ask: {}", outcome.query))
        .with_body(if let Some(d) = &outcome.draft {
            format!(
                "tier=draft corpus={} context_tokens={} table_share={:.2} gaps={}",
                outcome.corpus,
                outcome.context_tokens,
                d.table_share(),
                d.gaps
            )
        } else if let Some(a) = &outcome.answer {
            format!(
                "tier={} corpus={} context_tokens={}",
                a.tier.label(),
                outcome.corpus,
                outcome.context_tokens
            )
        } else {
            format!("tier=unknown corpus={}", outcome.corpus)
        })
        .with_duration(outcome.elapsed_ms as u64)
        .by("physis-ask");
    let recorded = observe::append(&observe::log_path(), &mut [obs])
        .ok()
        .map(|(first, _)| first);
    outcome.recorded_seq = recorded;
}

// ── Ledger — what is believed, what contradicts, what was never checked ─────

/// A hypothesis, as the surfaces render it (no embeddings in the view).
#[derive(Debug, Clone, Serialize)]
pub struct HypoView {
    pub id: String,
    pub statement: String,
    pub status: String,
    pub fitness: f32,
    pub supporting: usize,
    pub contradicting: usize,
    /// Predictions made and never scored — the ledger's own open promises.
    pub open_predictions: Vec<String>,
    pub created_at: String,
}

/// A contradiction, both parties and both sources — never netted into one
/// number, for the reason `ground.rs` gives: a view that quietly picked a
/// side would be destroying the one signal a disagreement carries.
#[derive(Debug, Clone, Serialize)]
pub struct ContraView {
    pub id: String,
    pub claim_a: String,
    pub source_a: String,
    pub claim_b: String,
    pub source_b: String,
    pub resolution: String,
}

/// The whole ledger in one read-only snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct LedgerSnapshot {
    /// Fitness-descending — the beliefs that have survived most first.
    pub hypotheses: Vec<HypoView>,
    pub contradictions: Vec<ContraView>,
    pub open_predictions: usize,
}

fn resolution_str(r: &crate::contradiction::ResolutionStatus) -> String {
    use crate::contradiction::ResolutionStatus::*;
    match r {
        APreferred => "A preferred".to_string(),
        BPreferred => "B preferred".to_string(),
        Open => "open".to_string(),
        Contextual => "contextual (both valid)".to_string(),
        Superseded => "superseded".to_string(),
    }
}

/// Read-only ledger snapshot over the persisted core.
pub fn ledger(data_dir: &Path) -> anyhow::Result<LedgerSnapshot> {
    let core = load_core(data_dir);
    let mut hypotheses: Vec<HypoView> = core
        .hypotheses
        .values()
        .map(|h| HypoView {
            id: h.id.clone(),
            statement: h.statement.clone(),
            status: h.status.as_str().to_string(),
            fitness: h.fitness,
            supporting: h.supporting_evidence.len(),
            contradicting: h.contradicting_evidence.len(),
            open_predictions: h
                .predictions
                .iter()
                .filter(|p| p.correct.is_none())
                .map(|p| p.statement.clone())
                .collect(),
            created_at: h.created_at.to_rfc3339(),
        })
        .collect();
    hypotheses
        .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

    let contradictions = core
        .contradictions
        .iter()
        .map(|c| ContraView {
            id: c.id.clone(),
            claim_a: c.claim_a.claim.clone(),
            source_a: c.claim_a.source.clone(),
            claim_b: c.claim_b.claim.clone(),
            source_b: c.claim_b.source.clone(),
            resolution: resolution_str(&c.resolution),
        })
        .collect();

    let open_predictions = hypotheses.iter().map(|h| h.open_predictions.len()).sum();
    Ok(LedgerSnapshot {
        hypotheses,
        contradictions,
        open_predictions,
    })
}

// ── Status + watch — the TUI's other two tabs, and anyone else's ────────────

/// Engine counts in one struct (what `snapshot()` reports, plus ledger sizes).
#[derive(Debug, Clone, Serialize)]
pub struct StatusSummary {
    pub data_dir: String,
    pub total_nodes: usize,
    pub high_coherence: usize,
    pub mid_coherence: usize,
    pub low_coherence: usize,
    pub coherence_index: f32,
    pub certified_branches: usize,
    pub isolated_branches: usize,
    pub dream_cycles: usize,
    pub cluster_count: usize,
    pub outlier_count: usize,
    pub asserted_success: usize,
    pub asserted_inert: usize,
    pub hypotheses: usize,
    pub contradictions: usize,
    pub edges: usize,
    pub open_predictions: usize,
    /// Latest sequence in the observation log (`None` when the log is empty).
    pub last_observation_seq: Option<u64>,
}

pub fn status(data_dir: &Path) -> anyhow::Result<StatusSummary> {
    let core = load_core(data_dir);
    let snap = core.snapshot();
    let ledger = ledger(data_dir)?;
    Ok(StatusSummary {
        data_dir: data_dir.display().to_string(),
        total_nodes: snap.total_nodes,
        high_coherence: snap.high_coherence,
        mid_coherence: snap.mid_coherence,
        low_coherence: snap.low_coherence,
        coherence_index: snap.coherence_index,
        certified_branches: snap.certified_branches_count,
        isolated_branches: snap.isolated_branches_count,
        dream_cycles: snap.dream_cycle_count,
        cluster_count: snap.cluster_count,
        outlier_count: snap.outlier_count,
        asserted_success: snap.asserted_success,
        asserted_inert: snap.asserted_inert,
        hypotheses: core.hypotheses.len(),
        contradictions: core.contradictions.len(),
        edges: core.edges.len(),
        open_predictions: ledger.open_predictions,
        last_observation_seq: observe::last_seq(&observe::log_path()).ok(),
    })
}

/// Tail of the observation log — what the machine saw, most recent last.
pub fn watch_tail(limit: usize) -> anyhow::Result<Vec<Observation>> {
    observe::read_tail(&observe::log_path(), limit)
}
