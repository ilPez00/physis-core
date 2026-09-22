//! Progressive refinement retrieval (PH-101 Task 5).
//!
//! The packet's ladder — exact → structural → sparse → dense → ColBERT →
//! graph constraints — orchestrated over the existing [`crate::query`]
//! primitives, not reimplemented. Each rung runs only when its inputs exist;
//! ColBERT reports unavailable until weights exist on the machine, and the
//! graph rung runs causal traversal when a pattern is present. Results
//! dedupe with early-rung priority: a cheaper rung's hit outranks a dearer
//! rung's duplicate.

use serde::{Deserialize, Serialize};

use crate::query::{QueryPlan, QueryPlanner, QueryResult, RetrievalStrategy};
use crate::transform::{Triple, TriplePattern};

/// One ladder rung and what it returned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageHits {
    pub stage: String,
    pub strategy: Option<RetrievalStrategy>,
    pub triples: Vec<Triple>,
    /// Why this rung ran or skipped — the ladder never silently narrows.
    pub note: String,
}

/// The full ladder output: per-rung hits plus deduped triples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinedResult {
    pub stages: Vec<StageHits>,
    pub triples: Vec<Triple>,
}

/// The refinement ladder over a [`QueryPlanner`].
pub struct RefinementLadder<'a> {
    planner: &'a QueryPlanner,
    max_results: usize,
}

impl<'a> RefinementLadder<'a> {
    pub fn new(planner: &'a QueryPlanner, max_results: usize) -> Self {
        Self { planner, max_results }
    }

    /// Run the ladder. `pattern` enables exact/structural/causal rungs;
    /// `text` enables sparse/dense rungs.
    pub fn search(&self, pattern: Option<TriplePattern>, text: &str) -> RefinedResult {
        let mut stages = Vec::new();

        // Exact — needs a fully constant pattern.
        match &pattern {
            Some(p) if is_constant(p) => {
                stages.push(self.run_stage("exact", RetrievalStrategy::Exact, pattern.clone(), None));
            }
            Some(_) => stages.push(skipped("exact", "pattern has variables; structural rung covers it")),
            None => stages.push(skipped("exact", "no pattern supplied")),
        }

        // Structural — any pattern.
        match &pattern {
            Some(_) => stages.push(self.run_stage(
                "structural",
                RetrievalStrategy::Structured,
                pattern.clone(),
                None,
            )),
            None => stages.push(skipped("structural", "no pattern supplied")),
        }

        // Sparse (BM25) — needs text and a configured index.
        if text.trim().len() > 2 {
            match self.run_stage_checked("sparse", RetrievalStrategy::BM25, None, Some(text)) {
                Ok(hits) => stages.push(hits),
                Err(note) => stages.push(skipped("sparse", &note)),
            }
        } else {
            stages.push(skipped("sparse", "query text too short"));
        }

        // Dense — needs text and a configured embedder.
        if text.trim().len() > 2 {
            match self.run_stage_checked("dense", RetrievalStrategy::Embedding, None, Some(text)) {
                Ok(hits) => stages.push(hits),
                Err(note) => stages.push(skipped("dense", &note)),
            }
        } else {
            stages.push(skipped("dense", "query text too short"));
        }

        // ColBERT — named rung, honestly unavailable.
        stages.push(skipped("colbert", "no ColBERT weights on this machine"));

        // Graph constraints — causal traversal when a pattern is present.
        match &pattern {
            Some(_) => match self.run_stage_checked(
                "graph",
                RetrievalStrategy::Causal,
                pattern.clone(),
                None,
            ) {
                Ok(hits) => stages.push(hits),
                Err(note) => stages.push(skipped("graph", &note)),
            },
            None => stages.push(skipped("graph", "no pattern supplied")),
        }

        // Dedupe with early-rung priority.
        let mut seen = std::collections::HashSet::new();
        let mut triples = Vec::new();
        for stage in &stages {
            for t in &stage.triples {
                let key = format!("{}|{}|{}", t.s, t.p.as_str(), t.o);
                if seen.insert(key) {
                    triples.push(t.clone());
                }
            }
        }

        RefinedResult { stages, triples }
    }

    fn run_stage(
        &self,
        stage: &str,
        strategy: RetrievalStrategy,
        pattern: Option<TriplePattern>,
        text_query: Option<&str>,
    ) -> StageHits {
        match self.run_stage_checked(stage, strategy, pattern, text_query) {
            Ok(hits) => hits,
            Err(note) => skipped(stage, &note),
        }
    }

    fn run_stage_checked(
        &self,
        stage: &str,
        strategy: RetrievalStrategy,
        pattern: Option<TriplePattern>,
        text_query: Option<&str>,
    ) -> Result<StageHits, String> {
        let plan = QueryPlan {
            strategy,
            pattern,
            text_query: text_query.map(str::to_string),
            max_results: self.max_results,
            min_confidence: 0.0,
            causal_direction: Some(crate::query::CausalDirection::Forward),
            temporal_window: None,
        };
        let result: QueryResult = self
            .planner
            .execute(&plan)
            .map_err(|e| format!("{stage} rung failed: {e}"))?;
        Ok(StageHits {
            stage: stage.to_string(),
            strategy: Some(strategy),
            triples: result.triples,
            note: result.explanation,
        })
    }
}

fn skipped(stage: &str, note: &str) -> StageHits {
    StageHits {
        stage: stage.to_string(),
        strategy: None,
        triples: Vec::new(),
        note: note.to_string(),
    }
}

fn is_constant(pattern: &TriplePattern) -> bool {
    use crate::transform::PatElem;
    matches!(pattern.s, PatElem::Const(_))
        && pattern.p.is_some()
        && matches!(pattern.o, PatElem::Const(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{AtomicFact, WorldGraph};
    use crate::transform::{Predicate, Proposition, WorldState};

    fn planner() -> QueryPlanner {
        let world = WorldState {
            facts: vec![
                Proposition::observed(Triple::new("A", Predicate::Produces, "B"), "t"),
                Proposition::observed(Triple::new("B", Predicate::Causes, "C"), "t"),
            ],
            constraints: Vec::new(),
        };
        let facts = vec![AtomicFact {
            triple: Triple::new("A", Predicate::Produces, "B"),
            obs_seq: 1,
            source: "t".to_string(),
            confidence: 0.9,
            extracted_at: chrono::Utc::now(),
            method: "t".to_string(),
        }];
        let graph = WorldGraph::from_world(&world, &facts);
        let docs: Vec<String> = graph
            .facts
            .iter()
            .map(|f| format!("{} {} {}", f.triple.s, f.triple.p.as_str(), f.triple.o))
            .collect();
        QueryPlanner::new(graph).with_bm25(crate::query::Bm25Index::new(docs))
    }

    #[test]
    fn ladder_runs_all_rungs_and_dedupes() {
        let planner = planner();
        let ladder = RefinementLadder::new(&planner, 10);
        let pattern = TriplePattern {
            s: crate::transform::PatElem::Const("A".to_string()),
            p: Some(Predicate::Produces),
            o: crate::transform::PatElem::Const("B".to_string()),
        };
        let out = ladder.search(Some(pattern), "A Produces B");
        let stages: Vec<&str> = out.stages.iter().map(|s| s.stage.as_str()).collect();
        assert_eq!(stages, vec!["exact", "structural", "sparse", "dense", "colbert", "graph"]);
        // Dense has no embedder: skipped with a note, not an error.
        let dense = out.stages.iter().find(|s| s.stage == "dense").unwrap();
        assert!(dense.triples.is_empty());
        assert!(!dense.note.is_empty());
        // A→B found once despite matching on several rungs.
        assert_eq!(out.triples.iter().filter(|t| t.s == "A").count(), 1);
        // Causal graph rung walks A→B→C.
        assert!(out.triples.iter().any(|t| t.s == "B" && t.o == "C"));
    }

    #[test]
    fn text_only_search_skips_pattern_rungs_loudly() {
        let planner = planner();
        let ladder = RefinementLadder::new(&planner, 10);
        let out = ladder.search(None, "produces");
        for stage in ["exact", "structural", "graph"] {
            let s = out.stages.iter().find(|s| s.stage == stage).unwrap();
            assert!(s.triples.is_empty(), "{stage} must not run without a pattern");
            assert!(!s.note.is_empty());
        }
    }
}
