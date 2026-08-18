//! physis-core CLI.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use physis_core::classify::{CellClassifier, CellScore};
use physis_core::contradiction::{Contradiction, ContradictionParty, ResolutionStatus};
use physis_core::core::PhysisCore;
use physis_core::embed::{RandomProjectionEmbedder, VectorEmbed};
use physis_core::history::import_file as import_history_file;
use physis_core::hypothesis::{Evidence, EvidencePolarity, Hypothesis, HypothesisStatus, Prediction};
use physis_core::models::{
    Abstraction, Agency, FacetFilter, LifecyclePhase, PinEdit, Scale,
};
use physis_core::ontology::OntologyLoader;
use physis_core::praxis::parse_export as parse_praxis_export;
use physis_core::quality::QualityTracker;
use physis_core::rag::{count_tokens, RagChunk, RagCorpus, TokenFixedRetriever};
use physis_core::store;
use physis_core::vault::{collect_labels as collect_vault_labels, scan_git_log, scan_vault};

/// physis-core — embed, classify, cohere, learn from feedback.
#[derive(Parser)]
#[command(name = "physis-core", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Classify text against the semiotic grid.
    Classify { text: String },
    /// Show ontology stats.
    Ontology,
    /// Query ontology entries by orthogonal facets.
    Facet {
        /// Filter by lifecycle phase: DESIGN | FABRICATE | OPERATE | MAINTAIN | RETIRE
        #[arg(long)]
        lifecycle: Option<String>,
        /// Filter by agency: SELF | OTHER | AUTOMATED | COLLECTIVE
        #[arg(long)]
        agency: Option<String>,
        /// Filter by scale: PERSONAL | INTERPERSONAL | ORGANIZATIONAL | CIVIL
        #[arg(long)]
        scale: Option<String>,
        /// Filter by abstraction: CONCRETE | ABSTRACT
        #[arg(long)]
        abstraction: Option<String>,
        /// Sub-domain (case-insensitive substring)
        #[arg(long)]
        sub_domain: Option<String>,
        /// Sub-mode (case-insensitive substring)
        #[arg(long)]
        sub_mode: Option<String>,
        /// Filter by kind
        #[arg(long)]
        kind: Option<String>,
        /// Filter by domain
        #[arg(long)]
        domain: Option<String>,
        /// Filter by mode
        #[arg(long)]
        mode: Option<String>,
    },
    /// Scan a directory, registering each text file as a coherence node.
    Scan {
        /// Directory to scan recursively.
        dir: PathBuf,
    },
    /// Search recalled coherence nodes by similarity to a query.
    Search {
        /// Query text.
        query: String,
        /// Max results.
        #[arg(long, default_value_t = 10)]
        max: usize,
    },
    /// Edit a coherence node's content ("exchange parts") while preserving structure.
    NodeEdit {
        /// Node ID.
        id: String,
        /// New content (label).
        label: String,
        /// Pin domain (e.g. FABRICATE).
        #[arg(long)]
        domain: Option<String>,
        /// Pin mode (e.g. MAINTAIN).
        #[arg(long)]
        mode: Option<String>,
        /// Clear any existing pin.
        #[arg(long)]
        clear_pin: bool,
    },
    /// Delete a coherence node and its index entry.
    NodeDelete {
        /// Node ID.
        id: String,
    },
    /// Fixed-token budget search over recalled coherence nodes.
    NodeSearch {
        /// Query text.
        query: String,
        /// Token budget for packed results.
        #[arg(long, default_value_t = 600)]
        budget: usize,
        /// Max results.
        #[arg(long, default_value_t = 12)]
        max: usize,
    },
    /// Import a knowledge vault (Obsidian/markdown notes, plain text, git log).
    Vault {
        /// Vault directory to scan.
        dir: PathBuf,
        /// Max git commits to import (0 = skip git).
        #[arg(long, default_value_t = 50)]
        git: usize,
    },
    /// Import personal history (bookmarks HTML, history JSON, OPML, chat JSONL).
    History {
        /// File to import.
        file: PathBuf,
    },
    /// Backfill behavioral records from a Praxis life-log export.
    Praxis {
        /// JSON export file.
        file: PathBuf,
    },
    /// Show a coherence snapshot of the persisted node graph.
    Snapshot,
    /// Report an asserted verdict on a node by label.
    Assert {
        /// Node label (exact).
        label: String,
        /// Verdict: success|inert|failure.
        verdict: String,
    },
    /// Dream over low-coherence / failed nodes.
    Dream,
    /// Quality feedback loop.
    Quality {
        #[command(subcommand)]
        cmd: QualityCmd,
    },
    /// Epistemic hypothesis management.
    Hypothesis {
        #[command(subcommand)]
        cmd: HypothesisCmd,
    },
    /// Contradiction management and truth maintenance.
    Contradiction {
        #[command(subcommand)]
        cmd: ContradictionCmd,
    },
    /// View chronological epistemic audit log.
    Audit,
    /// Replay historical belief state for a subject at timestamp T.
    Replay {
        /// Subject ID or hypothesis ID.
        #[arg(long)]
        subject: String,
        /// Optional ISO 8601 timestamp.
        #[arg(long)]
        at: Option<String>,
    },
    /// Discover ontology gaps and candidate domains from unstructured corpus texts.
    Discover {
        /// Directory containing text files, or scan persisted nodes if omitted.
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Minimum cluster size for domain proposal.
        #[arg(long, default_value_t = 3)]
        min_cluster: usize,
    },
    /// Run the ontology studio web GUI.
    #[cfg(feature = "studio")]
    Studio {
        /// Port to listen on.
        #[arg(long, default_value_t = 3000)]
        port: u16,
        /// ONNX model directory (model.onnx + tokenizer.json). Optional — falls
        /// back to deterministic random projection when absent or unloadable.
        #[arg(long)]
        model: Option<String>,
    },
}

#[derive(Subcommand)]
enum QualityCmd {
    /// Show quality tracker summary.
    Summary,
    /// Report a failure with optional correct domain.
    Fail { feedback: String },
    /// Report a success for a cell (DOMAIN\x00MODE).
    Pass { cell: String },
}

#[derive(Subcommand)]
enum HypothesisCmd {
    /// List all registered hypotheses.
    List,
    /// Register a candidate hypothesis.
    Create {
        /// Statement of the hypothesis.
        statement: String,
        /// Initial confidence [0.0, 1.0].
        #[arg(long, default_value_t = 0.5)]
        confidence: f32,
    },
    /// Add supporting or contradicting evidence to a hypothesis.
    Evidence {
        /// Hypothesis ID.
        id: String,
        /// Evidence claim description.
        claim: String,
        /// Source identifier.
        #[arg(long, default_value = "cli")]
        source: String,
        /// Polarity: supporting (default) or contradicting.
        #[arg(long, default_value = "supporting")]
        polarity: String,
        /// Evidence weight [0.0, 1.0].
        #[arg(long, default_value_t = 0.8)]
        weight: f32,
    },
    /// Record a testable prediction on a hypothesis.
    Predict {
        /// Hypothesis ID.
        id: String,
        /// Prediction statement.
        statement: String,
        /// Expected outcome.
        #[arg(long)]
        expected: Option<String>,
    },
    /// Transition hypothesis lifecycle status.
    Transition {
        /// Hypothesis ID.
        id: String,
        /// New status: Candidate|Supported|Contradicted|Confirmed|Inert|Failed|Superseded|Isolated|Certified
        status: String,
        /// Reason for transition.
        #[arg(long, default_value = "cli transition")]
        reason: String,
    },
    /// Show structured explanation report for a hypothesis.
    Explain {
        /// Hypothesis ID.
        id: String,
    },
}

#[derive(Subcommand)]
enum ContradictionCmd {
    /// List all active and resolved contradictions.
    List,
    /// Record a contradiction between two conflicting claims.
    Record {
        /// First claim statement.
        claim_a: String,
        /// Source of first claim.
        source_a: String,
        /// Second claim statement.
        claim_b: String,
        /// Source of second claim.
        source_b: String,
    },
    /// Resolve a contradiction.
    Resolve {
        /// Contradiction ID.
        id: String,
        /// Resolution status: APreferred|BPreferred|Open|Contextual|Superseded
        status: String,
        /// Explanation for resolution.
        #[arg(long)]
        explanation: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    store::ensure_data_dir()?;
    match cli.command {
        Command::Classify { text } => run_classify(&text),
        Command::Ontology => run_ontology(),
        Command::Facet {
            lifecycle,
            agency,
            scale,
            abstraction,
            sub_domain,
            sub_mode,
            kind,
            domain,
            mode,
        } => run_facet(
            lifecycle,
            agency,
            scale,
            abstraction,
            sub_domain,
            sub_mode,
            kind,
            domain,
            mode,
        ),
        Command::Scan { dir } => run_scan(&dir),
        Command::Search { query, max } => run_search(&query, max),
        Command::NodeEdit {
            id,
            label,
            domain,
            mode,
            clear_pin,
        } => run_node_edit(&id, &label, domain, mode, clear_pin),
        Command::NodeDelete { id } => run_node_delete(&id),
        Command::NodeSearch { query, budget, max } => run_node_search(&query, budget, max),
        Command::Vault { dir, git } => run_vault(&dir, git),
        Command::History { file } => run_history(&file),
        Command::Praxis { file } => run_praxis(&file),
        Command::Snapshot => run_snapshot(),
        Command::Assert { label, verdict } => run_assert(&label, &verdict),
        Command::Dream => run_dream(),
        Command::Quality { cmd } => run_quality(cmd),
        Command::Hypothesis { cmd } => run_hypothesis(cmd),
        Command::Contradiction { cmd } => run_contradiction(cmd),
        Command::Audit => run_audit(),
        Command::Replay { subject, at } => run_replay(&subject, at.as_deref()),
        Command::Discover { dir, min_cluster } => run_discover(dir.as_deref(), min_cluster),
        #[cfg(feature = "studio")]
        Command::Studio { port, model } => run_studio(port, model),
    }
}

fn load_embedder() -> RandomProjectionEmbedder {
    RandomProjectionEmbedder::new(384)
}

fn load_quality() -> QualityTracker {
    QualityTracker::load_or_new(&store::quality_path())
}

fn run_classify(text: &str) -> anyhow::Result<()> {
    let ontology = OntologyLoader::load_all();
    let embedder = load_embedder();
    let classifier = CellClassifier::build(&ontology, &embedder);
    let quality = load_quality();

    let results = classifier.classify_text(text, &embedder);
    let adjusted: Vec<CellScore> = results
        .iter()
        .map(|r| {
            let key = format!("{}\x00{}", r.domain, r.mode);
            let mut r2 = r.clone();
            r2.score = quality.adjust_score(&key, r.score);
            r2
        })
        .collect();

    println!("Query: {text}");
    println!("Cells populated: {}", classifier.cell_count());
    for r in adjusted.iter().take(8) {
        let bar = (r.score * 20.0).round() as usize;
        println!(
            "  {:<11} × {:<10} {:5.3} {}{}",
            r.domain,
            r.mode,
            r.score,
            "█".repeat(bar),
            "░".repeat(20 - bar)
        );
    }
    if let Some((sim, domain, mode)) = classifier.best_entry_sim(&embedder.embed(text)) {
        println!("Best entry cosine: {sim:.3} ({domain}×{mode})");
    }
    Ok(())
}

fn run_ontology() -> anyhow::Result<()> {
    let ontology = OntologyLoader::load_all();
    println!("Ontology: {} classification entries", ontology.entry_count());
    println!("  human (grid) domains: {}", ontology.human_domains.len());
    println!("  custom (extra) domains: {}", ontology.custom_domains.len());
    println!("Categories:");
    for cat in ontology.categories() {
        let n = ontology
            .all_domains()
            .values()
            .filter(|d| d.category.as_deref() == Some(cat.as_str()))
            .count();
        println!("  {cat}: {n}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_facet(
    lifecycle: Option<String>,
    agency: Option<String>,
    scale: Option<String>,
    abstraction: Option<String>,
    sub_domain: Option<String>,
    sub_mode: Option<String>,
    kind: Option<String>,
    domain: Option<String>,
    mode: Option<String>,
) -> anyhow::Result<()> {
    let ontology = OntologyLoader::load_all();
    let filter = FacetFilter {
        lifecycle: lifecycle.as_deref().and_then(LifecyclePhase::parse_facet),
        agency: agency.as_deref().and_then(Agency::parse_facet),
        scale: scale.as_deref().and_then(Scale::parse_facet),
        abstraction: abstraction.as_deref().and_then(Abstraction::parse_facet),
        sub_domain,
        sub_mode,
        kind,
        domain,
        mode,
    };

    let mut matches = Vec::new();
    for (name, entry) in ontology.all_domains() {
        if let Some(ref k) = filter.kind {
            if entry.axis_kind.as_deref() != Some(k.as_str()) {
                continue;
            }
        }
        if let Some(ref d) = filter.domain {
            if !entry.domain.as_deref().unwrap_or_default().eq_ignore_ascii_case(d) {
                continue;
            }
        }
        if let Some(ref m) = filter.mode {
            if !entry.mode.as_deref().unwrap_or_default().eq_ignore_ascii_case(m) {
                continue;
            }
        }
        if entry.facets.matches(&filter) {
            matches.push((name, entry));
        }
    }

    println!("Facet filter matched {} entries:", matches.len());
    for (name, entry) in matches.iter().take(40) {
        println!(
            "  {:<32} {:<10} × {:<10} [{}]",
            name,
            entry.domain.as_deref().unwrap_or("—"),
            entry.mode.as_deref().unwrap_or("—"),
            entry.facets.simple()
        );
    }
    if matches.len() > 40 {
        println!("  ... and {} more", matches.len() - 40);
    }
    Ok(())
}

fn run_scan(dir: &std::path::Path) -> anyhow::Result<()> {
    let embedder = load_embedder();
    let mut core = load_core();
    let mut count = 0usize;
    for entry in walk(dir)? {
        let Ok(text) = std::fs::read_to_string(&entry) else { continue };
        let trimmed = text.trim();
        if trimmed.is_empty() || trimmed.chars().count() > 5000 {
            continue;
        }
        core.register_node_from_text(&format!("{}: {}", entry.display(), trimmed), &embedder);
        count += 1;
    }
    core.persist()?;
    let snap = core.snapshot();
    println!("Registered {count} files. Nodes: {}", snap.total_nodes);
    println!("Coherence index: {:.3}", snap.coherence_index);
    Ok(())
}

fn run_search(query: &str, max: usize) -> anyhow::Result<()> {
    let embedder = load_embedder();
    let core = load_core();
    let hits = core.search_text(query, &embedder, max);
    if hits.is_empty() {
        println!("No recalled nodes. Try `physis-core scan <dir>` first.");
        return Ok(());
    }
    println!("Top {max} matches for: {query}");
    for (i, (_, label, score)) in hits.iter().enumerate() {
        println!("  {}. {:.3}  {}", i + 1, score, label.chars().take(100).collect::<String>());
    }
    Ok(())
}

fn run_node_edit(
    id: &str,
    label: &str,
    domain: Option<String>,
    mode: Option<String>,
    clear_pin: bool,
) -> anyhow::Result<()> {
    let embedder = load_embedder();
    let mut core = load_core();
    let new_emb = embedder.embed(label);
    let pin = if clear_pin {
        PinEdit::Clear
    } else if let (Some(d), Some(m)) = (domain, mode) {
        PinEdit::Set(d, m)
    } else {
        PinEdit::Keep
    };
    let outcome = core.edit_node(id, label, new_emb, pin, Some("random-projection"))?;
    core.persist()?;
    println!(
        "Node {} updated → '{}' (asserted {:?}, coherence {:.3})",
        outcome.node_id, outcome.label, outcome.asserted, outcome.coherence_score
    );
    if let Some((d, m)) = outcome.pinned_cell {
        println!("  pinned cell: {}×{}", d, m);
    }
    Ok(())
}

fn run_node_delete(id: &str) -> anyhow::Result<()> {
    let mut core = load_core();
    if core.delete_node(id) {
        core.persist()?;
        println!("Node {} deleted", id);
    } else {
        println!("Node {} not found", id);
    }
    Ok(())
}

fn run_node_search(query: &str, budget: usize, max: usize) -> anyhow::Result<()> {
    let embedder = load_embedder();
    let core = load_core();
    let q_emb = embedder.embed(query);
    let candidates: Vec<(usize, String, Vec<f32>)> = core
        .nodes
        .values()
        .filter_map(|n| {
            let l = n.label.as_deref()?;
            if l.trim().is_empty() {
                return None;
            }
            Some((l.to_string(), n.embedding.clone()))
        })
        .enumerate()
        .map(|(i, (l, e))| (i, l, e))
        .collect();

    let corpus = RagCorpus {
        chunks: candidates
            .iter()
            .map(|(i, l, e)| RagChunk {
                id: *i,
                text: l.clone(),
                tokens: count_tokens(l),
                embedding: e.clone(),
            })
            .collect(),
    };
    let result = TokenFixedRetriever::new(budget, max).retrieve(&q_emb, &corpus);
    println!(
        "Fixed-token search: {} results ({} tokens, budget {})",
        result.chunks.len(),
        result.total_tokens,
        result.budget
    );
    for (i, c) in result.chunks.iter().enumerate() {
        println!("  {}. {:.3}  {} ({} tokens)", i + 1, c.score, c.text, c.tokens);
    }
    Ok(())
}

fn run_vault(dir: &std::path::Path, git: usize) -> anyhow::Result<()> {
    let embedder = load_embedder();
    let mut core = load_core();
    let mut docs = scan_vault(dir);
    if git > 0 {
        let commits = scan_git_log(dir, git);
        docs.extend(commits);
    }
    if docs.is_empty() {
        println!("No docs found in vault: {}", dir.display());
        return Ok(());
    }
    let pairs = collect_vault_labels(&docs);
    let before = core.nodes.len();
    for (label, text) in &pairs {
        let emb = embedder.embed(text);
        core.register_node_vec_labeled(emb, Some(label.clone()));
    }
    let added = core.nodes.len().saturating_sub(before);
    core.persist()?;
    println!(
        "Vault[{}]: {} docs → {} labels registered ({} deduped)",
        dir.display(),
        docs.len(),
        added,
        pairs.len().saturating_sub(added)
    );
    Ok(())
}

fn run_history(file: &std::path::Path) -> anyhow::Result<()> {
    let embedder = load_embedder();
    let mut core = load_core();
    let (docs, name) = import_history_file(file)?;
    let before = core.nodes.len();
    for doc in &docs {
        let emb = embedder.embed(&doc.body);
        core.register_node_vec_labeled(emb, Some(doc.title.clone()));
    }
    let added = core.nodes.len().saturating_sub(before);
    core.persist()?;
    println!(
        "History[{}]: {} items → {} nodes registered",
        name,
        docs.len(),
        added
    );
    Ok(())
}

fn run_praxis(file: &std::path::Path) -> anyhow::Result<()> {
    let embedder = load_embedder();
    let mut core = load_core();
    let text = std::fs::read_to_string(file)?;
    let records = parse_praxis_export(&text);
    let before = core.nodes.len();
    for r in &records {
        let emb = embedder.embed(&r.body);
        let id = core.register_node_vec_labeled(emb, Some(format!("praxis:{}:{}", r.kind, r.title)));
        core.assert_coherence(&id, r.status.as_score());
    }
    let added = core.nodes.len().saturating_sub(before);
    core.persist()?;
    println!(
        "Praxis[{}]: {} records → {} behavioral nodes asserted",
        file.display(),
        records.len(),
        added
    );
    Ok(())
}

fn run_snapshot() -> anyhow::Result<()> {
    let core = load_core();
    let snap = core.snapshot();
    println!("Coherence snapshot:");
    println!("  nodes:            {}", snap.total_nodes);
    println!("  coherence index:  {:.3}", snap.coherence_index);
    println!("  high/mid/low:     {}/{}/{}", snap.high_coherence, snap.mid_coherence, snap.low_coherence);
    println!("  certified branches: {}", snap.certified_branches_count);
    println!("  isolated branches:  {}", snap.isolated_branches_count);
    println!("  asserted s/i/f:   {}/{}/{}", snap.asserted_success, snap.asserted_inert, snap.asserted_failure);
    println!("  asserted index:   {:?}", snap.asserted_index);
    Ok(())
}

fn run_assert(label: &str, verdict: &str) -> anyhow::Result<()> {
    let mut core = load_core();
    let id = core
        .nodes
        .values()
        .find(|n| n.label.as_deref() == Some(label))
        .map(|n| n.id.clone())
        .ok_or_else(|| anyhow::anyhow!("no node with label '{label}'"))?;
    let score = match verdict.to_lowercase().as_str() {
        "success" => 1.0,
        "inert" => 0.0,
        "failure" => -1.0,
        other => anyhow::bail!("verdict must be success|inert|failure, got '{other}'"),
    };
    core.assert_coherence(&id, score);
    core.persist()?;
    println!("Asserted '{label}' as {verdict}");
    Ok(())
}

fn run_dream() -> anyhow::Result<()> {
    let mut core = load_core();
    let results = core.dream();
    if results.is_empty() {
        println!("No nodes need replaying right now.");
        return Ok(());
    }
    for d in &results {
        println!(
            "Dream {} tested {} node(s): outcome {:.1} prevented_failure={}",
            &d.dream_id[..8],
            d.nodes_tested.len(),
            d.outcome,
            d.prevented_failure,
        );
    }
    core.persist()?;
    Ok(())
}

fn run_quality(cmd: QualityCmd) -> anyhow::Result<()> {
    let mut quality = load_quality();
    match cmd {
        QualityCmd::Summary => {
            println!("Quality tracker: {} failures", quality.failures.len());
            println!("  cells penalized: {}", quality.cell_penalties.len());
            println!("  cells boosted:   {}", quality.cell_boosts.len());
            let mut penalties: Vec<_> = quality.cell_penalties.iter().collect();
            penalties.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            for (cell, p) in penalties.iter().take(10) {
                println!("  {} penalty={:.2}", cell.replace('\x00', " × "), p);
            }
            for f in quality.failures.iter().rev().take(5) {
                println!(
                    "  [{}] {} score={:.3} sev={:.1} feedback={}",
                    &f.id[..8],
                    f.top_cell.replace('\x00', " × "),
                    f.top_score,
                    f.severity,
                    f.feedback.chars().take(50).collect::<String>(),
                );
            }
        }
        QualityCmd::Fail { feedback } => {
            let ontology = OntologyLoader::load_all();
            let embedder = load_embedder();
            let classifier = CellClassifier::build(&ontology, &embedder);
            let centroids = classifier.cell_centroids();
            let f = quality.report_failure(&feedback, &centroids, None);
            println!(
                "Recorded failure → {} (score {:.3})",
                f.top_cell.replace('\x00', " × "),
                f.top_score
            );
            quality.save(&store::quality_path())?;
        }
        QualityCmd::Pass { cell } => {
            quality.report_success(&cell);
            quality.save(&store::quality_path())?;
            println!("Boosted cell '{}'", cell.replace('\x00', " × "));
        }
    }
    Ok(())
}

fn run_hypothesis(cmd: HypothesisCmd) -> anyhow::Result<()> {
    let mut core = load_core();
    let embedder = load_embedder();
    match cmd {
        HypothesisCmd::List => {
            println!("Registered hypotheses: {}", core.hypotheses.len());
            for h in core.hypotheses.values() {
                println!(
                    "  [{}] ({:?}) fitness={:.3} — {}",
                    &h.id[..8],
                    h.status,
                    h.fitness,
                    h.statement
                );
            }
        }
        HypothesisCmd::Create { statement, confidence } => {
            let emb = embedder.embed(&statement);
            let mut hyp = Hypothesis::new(&statement, emb);
            hyp.confidence = confidence;
            let id = core.register_hypothesis(hyp);
            core.persist()?;
            println!("Registered hypothesis [{id}]");
        }
        HypothesisCmd::Evidence {
            id,
            claim,
            source,
            polarity,
            weight,
        } => {
            let hyp = core
                .hypotheses
                .get_mut(&id)
                .ok_or_else(|| anyhow::anyhow!("hypothesis {id} not found"))?;
            let pol = match polarity.to_lowercase().as_str() {
                "contradicting" | "contra" | "refuting" => EvidencePolarity::Contradicts,
                _ => EvidencePolarity::Supports,
            };
            let ev = Evidence::new(claim, source).with_weight(weight);
            match pol {
                EvidencePolarity::Supports => hyp.add_supporting(ev),
                EvidencePolarity::Contradicts => hyp.add_contradicting(ev),
            }
            core.persist()?;
            println!("Added evidence to hypothesis [{id}]");
        }
        HypothesisCmd::Predict { id, statement, expected } => {
            let hyp = core
                .hypotheses
                .get_mut(&id)
                .ok_or_else(|| anyhow::anyhow!("hypothesis {id} not found"))?;
            let mut pred = Prediction::new(statement);
            if let Some(exp) = expected {
                pred.expected_outcome = Some(exp);
            }
            hyp.add_prediction(pred);
            core.persist()?;
            println!("Added prediction to hypothesis [{id}]");
        }
        HypothesisCmd::Transition { id, status, reason } => {
            let target_status = match status.to_lowercase().as_str() {
                "candidate" => HypothesisStatus::Candidate,
                "supported" => HypothesisStatus::Supported,
                "contradicted" => HypothesisStatus::Contradicted,
                "confirmed" => HypothesisStatus::Confirmed,
                "inert" => HypothesisStatus::Inert,
                "failed" => HypothesisStatus::Failed,
                "superseded" => HypothesisStatus::Superseded,
                "isolated" => HypothesisStatus::Isolated,
                "certified" => HypothesisStatus::Certified,
                other => anyhow::bail!("unknown status '{other}'"),
            };
            if core.transition_hypothesis(&id, target_status, &reason, None) {
                core.persist()?;
                println!("Hypothesis [{id}] transitioned to {:?}", target_status);
            } else {
                anyhow::bail!("hypothesis {id} not found");
            }
        }
        HypothesisCmd::Explain { id } => {
            let report = core
                .full_explanation_report(&id)
                .ok_or_else(|| anyhow::anyhow!("hypothesis {id} not found"))?;
            println!("{}", report.ascii_summary());
        }
    }
    Ok(())
}

fn run_contradiction(cmd: ContradictionCmd) -> anyhow::Result<()> {
    let mut core = load_core();
    match cmd {
        ContradictionCmd::List => {
            println!("Recorded contradictions: {}", core.contradictions.len());
            for c in &core.contradictions {
                println!(
                    "  [{}] status={:?} — '{}' ({}) vs '{}' ({})",
                    &c.id[..8],
                    c.resolution,
                    c.claim_a.claim,
                    c.claim_a.source,
                    c.claim_b.claim,
                    c.claim_b.source
                );
            }
        }
        ContradictionCmd::Record {
            claim_a,
            source_a,
            claim_b,
            source_b,
        } => {
            let ca = ContradictionParty::new(claim_a, source_a);
            let cb = ContradictionParty::new(claim_b, source_b);
            let contra = Contradiction::new(ca, cb);
            let id = core.record_contradiction(contra);
            core.persist()?;
            println!("Recorded contradiction [{id}]");
        }
        ContradictionCmd::Resolve { id, status, explanation } => {
            let c = core
                .contradictions
                .iter_mut()
                .find(|c| c.id == id || c.id.starts_with(&id))
                .ok_or_else(|| anyhow::anyhow!("contradiction {id} not found"))?;
            let res = match status.to_lowercase().as_str() {
                "apreferred" | "a" => ResolutionStatus::APreferred,
                "bpreferred" | "b" => ResolutionStatus::BPreferred,
                "contextual" => ResolutionStatus::Contextual,
                "superseded" => ResolutionStatus::Superseded,
                _ => ResolutionStatus::Open,
            };
            c.resolution = res;
            c.explanation = explanation;
            core.persist()?;
            println!("Contradiction [{}] resolved to {:?}", id, res);
        }
    }
    Ok(())
}

fn run_audit() -> anyhow::Result<()> {
    let core = load_core();
    println!("Epistemic audit trail: {} events", core.epistemic_audit.events.len());
    println!("{}", core.epistemic_audit.summary());
    Ok(())
}

fn run_replay(subject: &str, at: Option<&str>) -> anyhow::Result<()> {
    let core = load_core();
    let timestamp = match at {
        Some(t) => chrono::DateTime::parse_from_rfc3339(t)?.with_timezone(&chrono::Utc),
        None => chrono::Utc::now(),
    };
    let belief = core.reconstruct_belief_at(subject, timestamp);
    match belief {
        Some(status) => println!("Belief at {timestamp} for [{subject}]: {:?}", status),
        None => println!("No recorded belief for [{subject}] at {timestamp}"),
    }
    Ok(())
}

fn run_discover(dir: Option<&std::path::Path>, min_cluster: usize) -> anyhow::Result<()> {
    let embedder = load_embedder();
    let ontology = OntologyLoader::load_all();
    let classifier = CellClassifier::build(&ontology, &embedder);
    let mut texts = Vec::new();

    if let Some(d) = dir {
        for entry in walk(d)? {
            if let Ok(text) = std::fs::read_to_string(&entry) {
                for line in text.lines() {
                    let trimmed = line.trim();
                    if trimmed.len() > 10 {
                        texts.push(trimmed.to_string());
                    }
                }
            }
        }
    } else {
        let core = load_core();
        for node in core.nodes.values() {
            if let Some(ref l) = node.label {
                texts.push(l.clone());
            }
        }
    }

    if texts.is_empty() {
        println!("No texts available for ontology gap discovery.");
        return Ok(());
    }

    let config = physis_core::discovery::DiscoveryConfig {
        min_cluster,
        ..Default::default()
    };
    let report = physis_core::discovery::discover(&texts, &classifier, &embedder, &config);

    println!(
        "Ontology Gap Discovery: evaluated {} texts ({} unmapped)",
        report.total,
        report.unmapped
    );
    for prop in &report.proposals {
        println!(
            "\n  Proposed Domain: {} ({} × {}) — {} texts",
            prop.name, prop.domain, prop.mode, prop.samples.len()
        );
        println!("    Hints: {}", prop.hints.join(", "));
        for s in prop.samples.iter().take(2) {
            println!("    - {}", s);
        }
    }
    Ok(())
}

#[cfg(feature = "studio")]
fn run_studio(port: u16, model: Option<String>) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(physis_core::studio::run_with_model(port, model))
}

/// Load the persisted core (or a fresh one).
fn load_core() -> PhysisCore {
    let json = store::read_optional(&store::nodes_path());
    if json.trim().is_empty() {
        PhysisCore::new()
    } else {
        PhysisCore::from_json(&json).unwrap_or_else(|_| PhysisCore::new())
    }
}

fn walk(dir: &std::path::Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if is_text(&path) {
                out.push(path);
            }
        }
    }
    Ok(out)
}

fn is_text(path: &std::path::Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("txt" | "md" | "rs" | "py" | "json" | "toml" | "csv" | "log" | "xml" | "yml" | "yaml")
    )
}

trait PersistCore {
    fn persist(&self) -> anyhow::Result<()>;
}

impl PersistCore for PhysisCore {
    fn persist(&self) -> anyhow::Result<()> {
        let json = self.to_json()?;
        std::fs::write(store::nodes_path(), json)?;
        Ok(())
    }
}
