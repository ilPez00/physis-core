//! Atomic observation/triplet substrate and query planner.
//!
//! This module extends the symbolic Triple/Predicate substrate with:
//! - **Atomic facts**: Observations promoted to typed triplets with provenance
//! - **Graph projection**: WorldState projected as a queryable graph
//! - **Query planner**: Multi-strategy retrieval (exact, structured, BM25, embedding, causal, temporal)

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::embed::VectorEmbed;
use crate::models::{Score, cosine_sim};
use crate::observe::Observation;
use crate::transform::{Predicate, Triple, TriplePattern, PatElem, WorldState};

/// An atomic fact derived from an observation, carrying full provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AtomicFact {
    /// The symbolic triple
    pub triple: Triple,
    /// Source observation sequence
    pub obs_seq: u64,
    /// Source of the observation (fs, git, terminal, model, human, sensor...)
    pub source: String,
    /// Confidence in this fact (0..1)
    pub confidence: Score,
    /// When the fact was extracted
    pub extracted_at: chrono::DateTime<chrono::Utc>,
    /// Extraction method used
    pub method: String,
}

/// A graph projection of the world state for efficient traversal.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorldGraph {
    /// All triples as edges (subject -> predicate -> object)
    pub edges: HashMap<String, HashMap<Predicate, HashSet<String>>>,
    /// Reverse edges (object -> predicate -> subject)
    pub rev_edges: HashMap<String, HashMap<Predicate, HashSet<String>>>,
    /// All nodes in the graph
    pub nodes: HashSet<String>,
    /// Atomic facts backing the graph
    pub facts: Vec<AtomicFact>,
}

impl WorldGraph {
    /// Build a graph projection from a WorldState and its atomic facts.
    pub fn from_world(world: &WorldState, facts: &[AtomicFact]) -> Self {
        let mut edges: HashMap<String, HashMap<Predicate, HashSet<String>>> = HashMap::new();
        let mut rev_edges: HashMap<String, HashMap<Predicate, HashSet<String>>> = HashMap::new();
        let mut nodes = HashSet::new();

        // Add all triples from world state
        for t in world.triples() {
            nodes.insert(t.s.clone());
            nodes.insert(t.o.clone());
            edges
                .entry(t.s.clone())
                .or_default()
                .entry(t.p.clone())
                .or_default()
                .insert(t.o.clone());
            rev_edges
                .entry(t.o.clone())
                .or_default()
                .entry(t.p.clone())
                .or_default()
                .insert(t.s.clone());
        }

        // Add atomic facts
        for f in facts {
            let t = &f.triple;
            nodes.insert(t.s.clone());
            nodes.insert(t.o.clone());
            edges
                .entry(t.s.clone())
                .or_default()
                .entry(t.p.clone())
                .or_default()
                .insert(t.o.clone());
            rev_edges
                .entry(t.o.clone())
                .or_default()
                .entry(t.p.clone())
                .or_default()
                .insert(t.s.clone());
        }

        Self { edges, rev_edges, nodes, facts: facts.to_vec() }
    }

    /// Get all outgoing edges from a node.
    pub fn outgoing(&self, node: &str) -> Option<&HashMap<Predicate, HashSet<String>>> {
        self.edges.get(node)
    }

    /// Get all incoming edges to a node.
    pub fn incoming(&self, node: &str) -> Option<&HashMap<Predicate, HashSet<String>>> {
        self.rev_edges.get(node)
    }

    /// Get all neighbors (both directions).
    pub fn neighbors(&self, node: &str) -> HashSet<String> {
        let mut out = HashSet::new();
        if let Some(m) = self.edges.get(node) {
            for s in m.values() {
                out.extend(s.clone());
            }
        }
        if let Some(m) = self.rev_edges.get(node) {
            for s in m.values() {
                out.extend(s.clone());
            }
        }
        out
    }

    /// Find all paths of length up to `max_depth` from `start`.
    pub fn bfs_paths(&self, start: &str, max_depth: usize) -> Vec<Vec<(String, Predicate, String)>> {
        let mut paths = Vec::new();
        let mut queue = vec![(start.to_string(), vec![])];

        while let Some((current, path)) = queue.pop() {
            if path.len() >= max_depth {
                paths.push(path.clone());
                continue;
            }
            if let Some(outgoing) = self.edges.get(&current) {
                for (pred, targets) in outgoing {
                    for target in targets {
                        let mut new_path = path.clone();
                        new_path.push((current.clone(), pred.clone(), target.clone()));
                        queue.push((target.clone(), new_path));
                    }
                }
            }
        }
        paths
    }
}

/// Retrieval strategy for the query planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalStrategy {
    /// Exact triple pattern matching
    Exact,
    /// Structured pattern with variable binding
    Structured,
    /// BM25 lexical search over fact bodies
    BM25,
    /// Embedding cosine similarity
    Embedding,
    /// Causal chain traversal (Causes/Produces/Requires)
    Causal,
    /// Temporal ordering (Precedes)
    Temporal,
}

/// A query plan with a chosen strategy and parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    pub strategy: RetrievalStrategy,
    pub pattern: Option<TriplePattern>,
    pub text_query: Option<String>,
    pub max_results: usize,
    pub min_confidence: Score,
    pub causal_direction: Option<CausalDirection>,
    pub temporal_window: Option<TemporalWindow>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalDirection {
    Forward,   // Causes -> Effects
    Backward,  // Effects -> Causes
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalWindow {
    pub start: chrono::DateTime<chrono::Utc>,
    pub end: chrono::DateTime<chrono::Utc>,
}

/// Result of a query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub triples: Vec<Triple>,
    pub facts: Vec<AtomicFact>,
    pub strategy_used: RetrievalStrategy,
    pub score: Score,
    pub explanation: String,
}

/// The query planner selects and executes retrieval strategies.
pub struct QueryPlanner {
    graph: WorldGraph,
    embedder: Option<Box<dyn VectorEmbed>>,
    bm25_index: Option<Bm25Index>,
}

impl QueryPlanner {
    /// Create a new planner from a world graph.
    pub fn new(graph: WorldGraph) -> Self {
        Self {
            graph,
            embedder: None,
            bm25_index: None,
        }
    }

    /// Set an embedder for embedding-based retrieval.
    pub fn with_embedder(mut self, embedder: Box<dyn VectorEmbed>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Set a BM25 index for lexical retrieval.
    pub fn with_bm25(mut self, index: Bm25Index) -> Self {
        self.bm25_index = Some(index);
        self
    }

    /// Execute a query plan.
    pub fn execute(&self, plan: &QueryPlan) -> anyhow::Result<QueryResult> {
        match plan.strategy {
            RetrievalStrategy::Exact => self.execute_exact(plan),
            RetrievalStrategy::Structured => self.execute_structured(plan),
            RetrievalStrategy::BM25 => self.execute_bm25(plan),
            RetrievalStrategy::Embedding => self.execute_embedding(plan),
            RetrievalStrategy::Causal => self.execute_causal(plan),
            RetrievalStrategy::Temporal => self.execute_temporal(plan),
        }
    }

    /// Exact pattern matching.
    fn execute_exact(&self, plan: &QueryPlan) -> anyhow::Result<QueryResult> {
        let pattern = plan.pattern.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Exact strategy requires a pattern"))?;

        let mut results = Vec::new();
        let mut facts = Vec::new();

        // Match against all facts
        for fact in &self.graph.facts {
            if fact.confidence < plan.min_confidence {
                continue;
            }
            if pattern.match_against(&fact.triple).is_some() {
                results.push(fact.triple.clone());
                facts.push(fact.clone());
            }
        }

        // Also match against graph edges
        for (s, preds) in &self.graph.edges {
            for (p, objects) in preds {
                for o in objects {
                    let triple = Triple::new(s.clone(), p.clone(), o.clone());
                    if pattern.match_against(&triple).is_some() && !results.contains(&triple) {
                        results.push(triple);
                    }
                }
            }
        }

        results.truncate(plan.max_results);
        facts.truncate(plan.max_results);

        Ok(QueryResult {
            triples: results.clone(),
            facts,
            strategy_used: RetrievalStrategy::Exact,
            score: if results.is_empty() { 0.0 } else { 1.0 },
            explanation: format!("Exact pattern match: {} results", results.len()),
        })
    }

    /// Structured pattern with variable binding (uses find_homomorphisms).
    fn execute_structured(&self, plan: &QueryPlan) -> anyhow::Result<QueryResult> {
        use crate::transform::find_homomorphisms;

        let patterns = plan.pattern.as_ref()
            .map(|p| vec![p.clone()])
            .unwrap_or_default();

        if patterns.is_empty() {
            return Ok(QueryResult {
                triples: Vec::new(),
                facts: Vec::new(),
                strategy_used: RetrievalStrategy::Structured,
                score: 0.0,
                explanation: "Structured query requires at least one pattern".to_string(),
            });
        }

        let all_triples: Vec<Triple> = self.graph.facts.iter()
            .map(|f| f.triple.clone())
            .collect();

        let bindings = find_homomorphisms(&patterns, &all_triples);
        let binding_count = bindings.len();
        let mut results = Vec::new();

        for binding in &bindings {
            for pat in &patterns {
                if let Some(t) = pat.instantiate(binding) {
                    results.push(t);
                }
            }
        }

        results.truncate(plan.max_results);

        Ok(QueryResult {
            triples: results.clone(),
            facts: Vec::new(),
            strategy_used: RetrievalStrategy::Structured,
            score: if results.is_empty() { 0.0 } else { 0.8 },
            explanation: format!("Structured pattern match: {} bindings", binding_count),
        })
    }

    /// BM25 lexical search over fact bodies.
    fn execute_bm25(&self, plan: &QueryPlan) -> anyhow::Result<QueryResult> {
        let query = plan.text_query.as_ref()
            .ok_or_else(|| anyhow::anyhow!("BM25 strategy requires a text query"))?;

        let Some(index) = &self.bm25_index else {
            return Err(anyhow::anyhow!("BM25 index not configured"));
        };

        let results = index.search(query, plan.max_results);
        let facts: Vec<AtomicFact> = results.iter()
            .filter_map(|(idx, _)| self.graph.facts.get(*idx).cloned())
            .collect();
        let triples: Vec<Triple> = facts.iter().map(|f| f.triple.clone()).collect();

        Ok(QueryResult {
            triples,
            facts,
            strategy_used: RetrievalStrategy::BM25,
            score: if results.is_empty() { 0.0 } else { 0.7 },
            explanation: format!("BM25 search: {} results", results.len()),
        })
    }

    /// Embedding cosine similarity search.
    fn execute_embedding(&self, plan: &QueryPlan) -> anyhow::Result<QueryResult> {
        let query = plan.text_query.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Embedding strategy requires a text query"))?;

        let Some(embedder) = &self.embedder else {
            return Err(anyhow::anyhow!("Embedder not configured"));
        };

        let query_vec = embedder.embed(query);
        let mut seen = HashSet::new();
        let mut scored = Vec::new();

        for (subject, predicates) in &self.graph.edges {
            for (predicate, objects) in predicates {
                for object in objects {
                    let triple = Triple::new(subject.clone(), predicate.clone(), object.clone());
                    if !seen.insert(triple.clone()) {
                        continue;
                    }
                    let candidate = format!("{} {} {}", triple.s, triple.p.as_str(), triple.o);
                    let score = cosine_sim(&query_vec, &embedder.embed(&candidate));
                    scored.push((triple, score));
                }
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(plan.max_results);

        let mut triples = Vec::new();
        let mut facts = Vec::new();
        for (triple, _) in &scored {
            triples.push(triple.clone());
            facts.extend(self.graph.facts.iter().filter(|fact| {
                fact.triple == *triple && fact.confidence >= plan.min_confidence
            }).cloned());
        }

        Ok(QueryResult {
            triples,
            facts,
            strategy_used: RetrievalStrategy::Embedding,
            score: scored.first().map(|(_, score)| *score).unwrap_or(0.0),
            explanation: format!("Embedding search: {} results", scored.len()),
        })
    }

    /// Causal chain traversal.
    fn execute_causal(&self, plan: &QueryPlan) -> anyhow::Result<QueryResult> {
        let pattern = plan.pattern.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Causal strategy requires a pattern"))?;

        // Find seed nodes matching the pattern. Include both scored atomic facts
        // and world triples already projected into the graph edges.
        let mut seeds = Vec::new();
        for fact in &self.graph.facts {
            if fact.confidence < plan.min_confidence {
                continue;
            }
            if pattern.match_against(&fact.triple).is_some() {
                seeds.push(fact.triple.s.clone());
                seeds.push(fact.triple.o.clone());
            }
        }
        for (subject, predicates) in &self.graph.edges {
            for (predicate, objects) in predicates {
                for object in objects {
                    let triple = Triple::new(subject.clone(), predicate.clone(), object.clone());
                    if pattern.match_against(&triple).is_some() {
                        seeds.push(triple.s.clone());
                        seeds.push(triple.o.clone());
                    }
                }
            }
        }

        let direction = plan.causal_direction.unwrap_or(CausalDirection::Forward);
        let mut results = Vec::new();
        let mut visited = HashSet::new();

        for seed in seeds {
            if visited.contains(&seed) {
                continue;
            }
            visited.insert(seed.clone());

            let mut queue = vec![seed];
            while let Some(current) = queue.pop() {
                if let Some(outgoing) = self.graph.edges.get(&current) {
                    for (pred, targets) in outgoing {
                        if !pred.is_causal() {
                            continue;
                        }
                        let should_follow = match direction {
                            CausalDirection::Forward => true,
                            CausalDirection::Backward => false,
                            CausalDirection::Both => true,
                        };
                        if !should_follow {
                            continue;
                        }
                        for target in targets {
                            results.push(Triple::new(current.clone(), pred.clone(), target.clone()));
                            if !visited.contains(target) {
                                visited.insert(target.clone());
                                queue.push(target.clone());
                            }
                        }
                    }
                }

                // Also follow backward if needed
                if direction == CausalDirection::Backward || direction == CausalDirection::Both {
                    if let Some(incoming) = self.graph.rev_edges.get(&current) {
                        for (pred, sources) in incoming {
                            if !pred.is_causal() {
                                continue;
                            }
                            for source in sources {
                                results.push(Triple::new(source.clone(), pred.clone(), current.clone()));
                                if !visited.contains(source) {
                                    visited.insert(source.clone());
                                    queue.push(source.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        results.truncate(plan.max_results);

        Ok(QueryResult {
            triples: results.clone(),
            facts: Vec::new(),
            strategy_used: RetrievalStrategy::Causal,
            score: if results.is_empty() { 0.0 } else { 0.85 },
            explanation: format!("Causal traversal: {} edges", results.len()),
        })
    }

    /// Temporal ordering traversal.
    fn execute_temporal(&self, plan: &QueryPlan) -> anyhow::Result<QueryResult> {
        let pattern = plan.pattern.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Temporal strategy requires a pattern"))?;

        let window = plan.temporal_window.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Temporal strategy requires a temporal window"))?;

        let mut results = Vec::new();

        for fact in &self.graph.facts {
            if fact.confidence < plan.min_confidence {
                continue;
            }
            if fact.extracted_at < window.start || fact.extracted_at > window.end {
                continue;
            }
            if pattern.match_against(&fact.triple).is_some() {
                results.push(fact.triple.clone());
            }
        }

        // Also check for Precedes edges within window
        for fact in &self.graph.facts {
            if fact.triple.p == Predicate::Precedes
                && fact.extracted_at >= window.start
                && fact.extracted_at <= window.end
                && pattern.match_against(&fact.triple).is_some()
            {
                results.push(fact.triple.clone());
            }
        }

        results.truncate(plan.max_results);

        Ok(QueryResult {
            triples: results.clone(),
            facts: Vec::new(),
            strategy_used: RetrievalStrategy::Temporal,
            score: if results.is_empty() { 0.0 } else { 0.9 },
            explanation: format!("Temporal query: {} results", results.len()),
        })
    }

    /// Auto-plan: select best strategy based on query shape.
    pub fn auto_plan(&self, query: &str, pattern: Option<TriplePattern>, max_results: usize) -> QueryPlan {
        // Heuristic: if pattern provided, prefer structured/exact
        if let Some(ref pat) = pattern {
            if pat.p.is_some() && matches!(pat.s, PatElem::Const(_)) && matches!(pat.o, PatElem::Const(_)) {
                return QueryPlan {
                    strategy: RetrievalStrategy::Exact,
                    pattern,
                    text_query: None,
                    max_results,
                    min_confidence: 0.5,
                    causal_direction: None,
                    temporal_window: None,
                };
            }
            return QueryPlan {
                strategy: RetrievalStrategy::Structured,
                pattern,
                text_query: None,
                max_results,
                min_confidence: 0.5,
                causal_direction: None,
                temporal_window: None,
            };
        }

        // If causal predicates mentioned, use causal
        if query.contains("cause") || query.contains("produces") || query.contains("requires") {
            return QueryPlan {
                strategy: RetrievalStrategy::Causal,
                pattern: None,
                text_query: Some(query.to_string()),
                max_results,
                min_confidence: 0.5,
                causal_direction: Some(CausalDirection::Forward),
                temporal_window: None,
            };
        }

        // If temporal words, use temporal
        if query.contains("before") || query.contains("after") || query.contains("precedes") {
            return QueryPlan {
                strategy: RetrievalStrategy::Temporal,
                pattern: None,
                text_query: Some(query.to_string()),
                max_results,
                min_confidence: 0.5,
                causal_direction: None,
                temporal_window: Some(TemporalWindow {
                    start: chrono::Utc::now() - chrono::Duration::days(30),
                    end: chrono::Utc::now(),
                }),
            };
        }

        // Default: BM25 if available, else exact
        if self.bm25_index.is_some() {
            QueryPlan {
                strategy: RetrievalStrategy::BM25,
                pattern: None,
                text_query: Some(query.to_string()),
                max_results,
                min_confidence: 0.0,
                causal_direction: None,
                temporal_window: None,
            }
        } else {
            QueryPlan {
                strategy: RetrievalStrategy::Exact,
                pattern: None,
                text_query: Some(query.to_string()),
                max_results,
                min_confidence: 0.0,
                causal_direction: None,
                temporal_window: None,
            }
        }
    }
}

/// Minimal BM25 index for lexical retrieval over fact bodies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Index {
    docs: Vec<String>,
    doc_freq: HashMap<String, usize>,
    avg_dl: f32,
    k1: f32,
    b: f32,
}

impl Bm25Index {
    pub fn new(docs: Vec<String>) -> Self {
        let mut doc_freq = HashMap::new();
        let mut total_len = 0usize;

        for doc in &docs {
            let tokens: HashSet<String> = tokenize(doc);
            total_len += tokens.len();
            for token in tokens {
                *doc_freq.entry(token).or_default() += 1;
            }
        }

        let avg_dl = if docs.is_empty() { 0.0 } else { total_len as f32 / docs.len() as f32 };

        Self {
            docs,
            doc_freq,
            avg_dl,
            k1: 1.2,
            b: 0.75,
        }
    }

    fn score(&self, query: &str, doc_idx: usize) -> f32 {
        let doc = &self.docs[doc_idx];
        let doc_len = tokenize(doc).len() as f32;
        let query_terms = tokenize(query);

        let mut score = 0.0f32;
        let n = self.docs.len() as f32;

        for term in query_terms {
            let df = *self.doc_freq.get(&term).unwrap_or(&0) as f32;
            if df == 0.0 {
                continue;
            }
            // Use smoothed IDF to avoid negative scores with small corpora
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5).ln()).max(0.0);
            let tf = doc.to_lowercase().matches(&term.to_lowercase()).count() as f32;
            score += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_dl));
        }
        score
    }

    pub fn search(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = (0..self.docs.len())
            .map(|i| (i, self.score(query, i)))
            .filter(|(_, s)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(k);
        scored
    }
}

/// Simple whitespace tokenization.
fn tokenize(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(|s| s.to_lowercase())
        .filter(|s| s.len() > 2)
        .collect()
}

/// Promote observations to atomic facts.
pub fn observations_to_facts(observations: &[Observation]) -> Vec<AtomicFact> {
    observations.iter().map(|obs| {
        // Simple extraction: subject as entity, source as relation
        let triple = Triple::new(
            obs.subject.clone(),
            Predicate::Observes,
            obs.body.clone(),
        );
        AtomicFact {
            triple,
            obs_seq: obs.seq,
            source: obs.source.clone(),
            confidence: 0.9,
            extracted_at: obs.at,
            method: "direct".to_string(),
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::RandomProjectionEmbedder;
    use crate::transform::{Predicate, Triple, WorldState};
    use crate::observe::Observation;

    #[test]
    fn atomic_fact_from_observation() {
        let mut obs = Observation::new("fs", "file.rs");
        obs.seq = 1;
        obs.body = "content".to_string();

        let facts = observations_to_facts(&[obs]);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].triple.s, "file.rs");
        assert_eq!(facts[0].triple.p, Predicate::Observes);
        assert_eq!(facts[0].obs_seq, 1);
    }

    #[test]
    fn world_graph_build() {
        let world = WorldState {
            facts: vec![
                crate::transform::Proposition::observed(
                    Triple::new("A", Predicate::Produces, "B"), "src"
                ),
                crate::transform::Proposition::observed(
                    Triple::new("B", Predicate::Causes, "C"), "src"
                ),
            ],
            constraints: vec![],
        };

        let facts = vec![
            AtomicFact {
                triple: Triple::new("X", Predicate::Observes, "Y"),
                obs_seq: 1,
                source: "fs".to_string(),
                confidence: 0.9,
                extracted_at: chrono::Utc::now(),
                method: "test".to_string(),
            }
        ];

        let graph = WorldGraph::from_world(&world, &facts);
        assert!(graph.nodes.contains("A"));
        assert!(graph.nodes.contains("B"));
        assert!(graph.nodes.contains("C"));
        assert!(graph.nodes.contains("X"));
        assert!(graph.nodes.contains("Y"));

        let outgoing = graph.outgoing("A").unwrap();
        assert!(outgoing.get(&Predicate::Produces).unwrap().contains("B"));
    }

    #[test]
    fn bfs_paths() {
        let world = WorldState {
            facts: vec![
                crate::transform::Proposition::observed(
                    Triple::new("A", Predicate::Produces, "B"), "src"
                ),
                crate::transform::Proposition::observed(
                    Triple::new("B", Predicate::Causes, "C"), "src"
                ),
            ],
            constraints: vec![],
        };
        let graph = WorldGraph::from_world(&world, &[]);
        let paths = graph.bfs_paths("A", 2);
        assert!(!paths.is_empty());
        assert!(paths.iter().any(|p| p.iter().any(|(_, pred, _)| *pred == Predicate::Produces)));
    }

    #[test]
    fn query_planner_exact() {
        let world = WorldState {
            facts: vec![
                crate::transform::Proposition::observed(
                    Triple::new("A", Predicate::Produces, "B"), "src"
                ),
            ],
            constraints: vec![],
        };
        let facts = vec![];
        let graph = WorldGraph::from_world(&world, &facts);
        let planner = QueryPlanner::new(graph);

        let plan = QueryPlan {
            strategy: RetrievalStrategy::Exact,
            pattern: Some(TriplePattern {
                s: crate::transform::PatElem::Const("A".to_string()),
                p: Some(Predicate::Produces),
                o: crate::transform::PatElem::Var("x".to_string()),
            }),
            text_query: None,
            max_results: 10,
            min_confidence: 0.0,
            causal_direction: None,
            temporal_window: None,
        };

        let result = planner.execute(&plan).unwrap();
        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.triples[0].s, "A");
        assert_eq!(result.triples[0].p, Predicate::Produces);
        assert_eq!(result.triples[0].o, "B");
    }

    #[test]
    fn query_planner_embedding() {
        let target = Triple::new("A", Predicate::Produces, "B");
        let world = WorldState {
            facts: vec![
                crate::transform::Proposition::observed(target.clone(), "src"),
                crate::transform::Proposition::observed(
                    Triple::new("C", Predicate::Causes, "D"), "src"
                ),
            ],
            constraints: vec![],
        };
        let facts = vec![AtomicFact {
            triple: target.clone(),
            obs_seq: 7,
            source: "test".to_string(),
            confidence: 0.9,
            extracted_at: chrono::Utc::now(),
            method: "test".to_string(),
        }];
        let graph = WorldGraph::from_world(&world, &facts);
        let planner = QueryPlanner::new(graph)
            .with_embedder(Box::new(RandomProjectionEmbedder::new(32)));

        let plan = QueryPlan {
            strategy: RetrievalStrategy::Embedding,
            pattern: None,
            text_query: Some("A Produces B".to_string()),
            max_results: 2,
            min_confidence: 0.0,
            causal_direction: None,
            temporal_window: None,
        };

        let result = planner.execute(&plan).unwrap();
        assert_eq!(result.triples[0], target);
        assert!(result.facts.iter().any(|fact| fact.obs_seq == 7));
        assert!(result.score > 0.99);
    }

    #[test]
    fn query_planner_causal() {
        let world = WorldState {
            facts: vec![
                crate::transform::Proposition::observed(
                    Triple::new("A", Predicate::Produces, "B"), "src"
                ),
                crate::transform::Proposition::observed(
                    Triple::new("B", Predicate::Causes, "C"), "src"
                ),
                crate::transform::Proposition::observed(
                    Triple::new("C", Predicate::Produces, "D"), "src"
                ),
            ],
            constraints: vec![],
        };
        let graph = WorldGraph::from_world(&world, &[]);
        let planner = QueryPlanner::new(graph);

        let plan = QueryPlan {
            strategy: RetrievalStrategy::Causal,
            pattern: Some(TriplePattern {
                s: crate::transform::PatElem::Const("A".to_string()),
                p: None,
                o: crate::transform::PatElem::Var("x".to_string()),
            }),
            text_query: None,
            max_results: 10,
            min_confidence: 0.0,
            causal_direction: Some(CausalDirection::Forward),
            temporal_window: None,
        };

        let result = planner.execute(&plan).unwrap();
        // The forward causal walk from A should traverse the whole projected chain.
        assert_eq!(result.triples.len(), 3);
        assert!(result.triples.contains(&Triple::new("A", Predicate::Produces, "B")));
        assert!(result.triples.contains(&Triple::new("B", Predicate::Causes, "C")));
        assert!(result.triples.contains(&Triple::new("C", Predicate::Produces, "D")));
    }

    #[test]
    fn bm25_index_basic() {
        let docs = vec![
            "the quick brown fox".to_string(),
            "the lazy dog".to_string(),
            "brown fox jumps".to_string(),
        ];
        let index = Bm25Index::new(docs);
        let results = index.search("brown fox", 2);
        assert_eq!(results.len(), 2);
        // First result should be "the quick brown fox" or "brown fox jumps"
        assert!(results[0].1 > 0.0);
    }

    #[test]
    fn auto_plan_selects_strategy() {
        let world = WorldState::default();
        let graph = WorldGraph::from_world(&world, &[]);
        let planner = QueryPlanner::new(graph);

        // With pattern and constants -> Exact
        let plan = planner.auto_plan(
            "",
            Some(TriplePattern {
                s: crate::transform::PatElem::Const("A".to_string()),
                p: Some(Predicate::Produces),
                o: crate::transform::PatElem::Const("B".to_string()),
            }),
            10,
        );
        assert_eq!(plan.strategy, RetrievalStrategy::Exact);

        // Causal query -> Causal
        let plan = planner.auto_plan("what causes failure", None, 10);
        assert_eq!(plan.strategy, RetrievalStrategy::Causal);
    }
}