# Physis Core (`physis-core`)

[![Crates.io](https://img.shields.io/crates/v/physis-core.svg)](https://crates.io/crates/physis-core)
[![Documentation](https://docs.rs/physis-core/badge.svg)](https://docs.rs/physis-core)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**The lean, high-performance epistemic reasoning engine extracted from the Physis intelligence system.**

> *“An engine that maintains competing interpretations of reality, evaluates their coherence with observations, processes, evidence, and outcomes, and preferentially retains interpretations that continue to work.”*

---

## Table of Contents

- [Overview & Core Epistemic Thesis](#overview--core-epistemic-thesis)
- [Key Capabilities & Innovations](#key-capabilities--innovations)
- [Architectural Boundary: Core vs. Pro](#architectural-boundary-core-vs-pro)
- [Architecture & Epistemic Pipeline](#architecture--epistemic-pipeline)
- [The Semiotic Grid](#the-semiotic-grid)
- [Module Catalog](#module-catalog)
- [Installation & Cargo Features](#installation--cargo-features)
- [Rust API Usage & Code Examples](#rust-api-usage--code-examples)
  - [1. Competing Hypotheses & Evidence Attestation](#1-competing-hypotheses--evidence-attestation)
  - [2. Truth Maintenance & Contradiction Resolution](#2-truth-maintenance--contradiction-resolution)
  - [3. Epistemic Audit Stream & Historical Replay](#3-epistemic-audit-stream--historical-replay)
  - [4. Token-Fixed Budget RAG](#4-token-fixed-budget-rag)
  - [5. Unsupervised Ontology Gap Discovery](#5-unsupervised-ontology-gap-discovery)
- [CLI Reference](#cli-reference)
- [Embedded Studio Web Workbench](#embedded-studio-web-workbench)
- [Publishing to Crates.io & Integration](#publishing-to-cratesio--integration)
- [License](#license)

---

## Overview & Core Epistemic Thesis

Mainstream AI knowledge graphs, retrieval pipelines, and vector databases operate on naive assumptions:
1. **Premature Convergence**: They collapse ambiguous observations into a single winning embedding or statement.
2. **Destructive Updates**: Contradictory evidence either overwrites prior truth or creates silent hallucination loops.
3. **Static Semantics**: Ontologies are treated as frozen taxonomies, incapable of discovering operational gaps.
4. **Epistemic Amnesia**: They lack an immutable audit log of *why* an assertion was believed, *what* assumptions supported it, and *how* confidence evolved over time.

`physis-core` replaces these flaws with a **dynamic truth-maintenance and coherence-seeking architecture**:

- **Active Hypothesis Competition**: Multiple mutually exclusive interpretations are maintained in parallel.
- **Multidimensional Coherence**: Hypotheses are scored across 5 orthogonal axes (`Semantic`, `Ontological`, `Logical`, `Empirical`, `Process`).
- **Non-Destructive Tension**: Contradictions are explicitly modeled and preserved as first-class entities with contextual preferencing.
- **Time-Machine Replay**: An append-only epistemic event trail allows instantaneous deterministic reconstruction of beliefs at any prior timestamp $t$.
- **Evolutionary Feedback**: Contextual quality feedback loops penalize cells that generate false predictions, dynamically tuning subsequent classifications.

---

---

## Key Capabilities & Innovations

| Feature | Description | Implementation |
|---|---|---|
| **Competing Hypotheses** | Parallel candidate interpretations with explicit assumption tracking, prediction verification, and Bayes-like survival. | `physis_core::hypothesis` |
| **Semiotic Grid (5×14)** | 70 canonical axes mapping 5 philosophical domains across 14 operational modes with sub-domain facets. | `physis_core::classify` |
| **Truth Maintenance System** | Explicit conflict modeling between contradictory claims with confidence-weighted tension and contextual override. | `physis_core::contradiction` |
| **Epistemic Audit & Time Machine** | Cryptographically chained audit stream with point-in-time state reconstruction. | `physis_core::epistemic` |
| **Structured Explanations** | Generation of structured reports detailing supporting/contradicting evidence, historical precedents, and fitness breakdowns. | `physis_core::explanation` |
| **Ontology Gap Discovery** | Unsupervised semantic clustering over unclassified text to propose new domain/mode entries automatically. | `physis_core::discovery` |
| **Fixed-Token Budget RAG** | Strict token-budget bounded recall with Maximal Marginal Relevance (MMR) diversity discounting. | `physis_core::rag` |
| **Vault & History Ingest** | Extract structured knowledge nodes from Markdown vaults, Git history, Netscape bookmarks, browser history, and OPML feeds. | `physis_core::vault`, `physis_core::history` |
| **Exchange-Parts Node Editing** | Swap node content while preserving ID, verdict, edges, and provenance (`cell_pin` hybrid semantics). | `physis_core::core` |

---

## Architectural Boundary: Core vs. Pro

> **Guiding Principle**: *Core stays Core (the pure, dependency-light epistemic reasoning engine), and Pro stays Pro (the enterprise industrial monitoring and operational intelligence suite).*

| Dimension | `physis-core` (Open Source Engine) | `physis-pro` (Industrial Suite) |
|---|---|---|
| **Primary Focus** | Epistemic truth maintenance, coherence evaluation, competing hypotheses | Industrial shop-floor telemetry, backoffice automation, multi-tenant deployment |
| **Ontology Engine** | 70 canonical semiotic grid cells (5 Domains × 14 Modes) + 33 domain ontologies | Extended 370+ industrial, machine process, and agent workflow domains |
| **Conflict & Truth** | First-class `Contradiction` tracking, non-destructive polarity, temporal replay | Shop-floor anomaly escalation, quality failure loop, automated arbitration |
| **State Persistence** | Lean, dependency-light in-memory or single JSON snapshot (`~/.physis-core/`) | High-performance durable Sled DB + Cloud Spanner Graph mirror |
| **Multi-Tenancy** | Single session / embedded in-process | Isolated per-tenant `RuntimeState` mapped via `X-Physis-User` header |
| **Hardware & IoT** | Model-agnostic text and vector embeddings | MQTT, Modbus TCP/RTU, Serial, OPC-UA machine adapters |
| **Multimodal Sensory** | Extensible `VectorEmbed` trait (RandomProjection, ONNX) | Real-time `AuraFrame` sensory bus, Whisper-large voice, CLIP visual features |
| **LLM Integration** | Token-fixed budget retriever (MMR RAG) | Dynamic LLM Coherence Harness, multi-provider cascade, auto-revision loops |
| **User Interface** | Lightweight embedded Axum studio (`physis-core studio`) | Full glassmorphic Operations Console, Gantt scheduler, OEE dashboards |

---

## Architecture & Epistemic Pipeline

```text
                               ┌─────────────────────────────┐
                               │    Raw Observations / Text  │
                               └──────────────┬──────────────┘
                                              │
                                      [Vector Embedder]
                                              │
                                              ▼
┌───────────────────────────┐      ┌─────────────────────────────┐      ┌───────────────────────────┐
│     Semiotic Grid (70)    │ ◄─── │       Cell Classifier       │ ───► │  Unsupervised Discovery   │
│   5 Domains × 14 Modes    │      │  (Nearest-Centroid Scoring) │      │  (Gap Analysis & Cluster) │
└───────────────────────────┘      └──────────────┬──────────────┘      └───────────────────────────┘
                                                  │
                                                  ▼
                                   ┌─────────────────────────────┐
                                   │         PhysisCore          │
                                   └──────────────┬──────────────┘
                                                  │
                ┌─────────────────────────────────┼─────────────────────────────────┐
                ▼                                 ▼                                 ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐ ┌───────────────────────────────┐
│     Competing Hypotheses      │ │     Contradiction Engine      │ │    Epistemic Audit Stream     │
│  - Assumption tracking        │ │  - Polarity tension           │ │  - Append-only event log      │
│  - Multidimensional fitness   │ │  - Contextual resolution      │ │  - Time-machine state replay  │
│  - Empirical predictions      │ │  - Non-destructive conflict   │ │  - Root-cause explanation     │
└───────────────────────────────┘ └───────────────────────────────┘ └───────────────────────────────┘
                │                                 │                                 │
                └─────────────────────────────────┼─────────────────────────────────┘
                                                  │
                                                  ▼
                                   ┌─────────────────────────────┐
                                   │ Contextual Quality Feedback │
                                   │   & Reinforcement Penalty   │
                                   └─────────────────────────────┘
```

---

## The Semiotic Grid

The foundational taxonomy divides knowledge across **5 Ontological Domains** and **14 Semiotic Modes**:

| Domain | Philosophical Meaning | Canonical Focus |
|---|---|---|
| **Techne** | Craft, engineering, instrumentation | Tools, physical parameters, mechanics, code, hardware |
| **Episteme** | Scientific knowledge, causal models | Theories, empirical proofs, equations, verification |
| **Phronesis** | Practical wisdom, situational prudence | Operational decisions, safety trade-offs, risk management |
| **Polis** | Collective governance, organizational systems | Teams, contracts, regulatory compliance, backoffice workflows |
| **Soma** | Physical embodiment, biological state | Sensory streams, machine health, thermal profiles, ergonomics |

Cross-referenced across **14 Operational Modes**:
`Substance`, `Form`, `Relation`, `Quantity`, `Quality`, `Space`, `Time`, `Position`, `State`, `Action`, `Passivity`, `Purpose`, `Process`, `Genesis`.

---

## Module Catalog

- **`classify`**: Computes cosine proximity against domain/mode cell centroids with quality-penalty adjustments.
- **`coherence_dimensions`**: Five-factor coherence profiles and composite fitness aggregation.
- **`coherence_query`**: Declarative queries over coherence graph nodes, verdicts, and thresholds.
- **`contradiction`**: Truth-maintenance conflict records, tension weights, and non-destructive resolution states.
- **`core`**: Central `PhysisCore` orchestrating nodes, edges, hypotheses, Dreaming engine, and memory snapshots.
- **`discovery`**: Density-based ontology gap detection and new domain/mode proposal generation.
- **`embed`**: Zero-dependency deterministic `RandomProjectionEmbedder` and extensible `VectorEmbed` trait.
- **`embed_onnx`**: Optional ONNX runtime integration for high-accuracy embedding models (`all-MiniLM-L6-v2`, `bge-small`, etc.).
- **`epistemic`**: Immutable event sourcing for beliefs, hypothesis states, and temporal replay reconstruction.
- **`explanation`**: Structured explanatory justifications with supporting/contradicting evidence chains.
- **`history`**: Multi-format personal history parsers (Netscape HTML bookmarks, Chrome/Firefox JSON history, OPML, chat JSONL).
- **`hypothesis`**: Competing hypothesis data structures, evidence polarities, and revision histories.
- **`ontology`**: Embedded loader for 33 built-in domain ontologies (human grid, machine process, AI agents, office operations).
- **`praxis`**: Behavioral tracking records with success/inert/failure feedback loops.
- **`process`**: Industrial and operational state machines, task sequences, and cycle tracking; measurements flag their own deviation from a constraint's `[min, max]` band with a normalized 0..1 severity.
- **`provenance`**: Cryptographic SHA-256 provenance chains connecting source data to final inferences.
- **`quality`**: Quality feedback tracker with cell-level *and* per-agent penalties/boosts, plus contextual fitness weighting — an agent that keeps producing bad output for a domain gets demoted the same way a cell does.
- **`rag`**: Fixed-budget token retrievers with BPE-style approximate tokenization and diversity filtering.
- **`vault`**: Knowledge vault readers for Markdown hierarchies (frontmatter, headings) and Git log streams.
- **`studio`**: Embedded Axum web server providing an interactive browser UI and RESTful HTTP API.

---

## Installation & Cargo Features

Add `physis-core` to your `Cargo.toml`:

```toml
[dependencies]
# 1. Lean Engine Only (Zero heavy web/ML dependencies)
physis-core = { version = "0.1", default-features = false }

# 2. Complete Engine + CLI + Embedded Studio UI
physis-core = { version = "0.1" }

# 3. Complete Engine + ONNX Real Embeddings Runtime
physis-core = { version = "0.1", features = ["embed-onnx"] }
```

### Feature Flags

| Feature | Default | Dependencies | Purpose |
|---|---|---|---|
| `cli` | **Yes** | `clap` | Standalone CLI binary (`physis-core`) with subcommands. |
| `studio` | **Yes** | `axum`, `tokio` | Embedded Web GUI workbench and REST API server. |
| `embed-onnx` | No | `ort`, `tokenizers` | Hardware-accelerated ONNX semantic embeddings. |

---

## Rust API Usage & Code Examples

### 1. Competing Hypotheses & Evidence Attestation

```rust
use physis_core::{
    PhysisCore, RandomProjectionEmbedder, Hypothesis, HypothesisStatus,
    Evidence, EvidencePolarity, Prediction, VectorEmbed,
};

fn main() {
    let embedder = RandomProjectionEmbedder::new(64);
    let mut core = PhysisCore::new();

    // Register competing explanations for a production quality defect
    let emb_a = embedder.embed("Extrusion temperature too low causing delamination");
    let mut hyp_a = Hypothesis::new("Low nozzle temperature", emb_a);
    hyp_a.assumptions.push("Thermistor calibration is accurate".to_string());
    let id_a = core.register_hypothesis(hyp_a);

    let emb_b = embedder.embed("Filament moisture absorption causing steam bubbles");
    let hyp_b = Hypothesis::new("Wet filament spool", emb_b);
    let id_b = core.register_hypothesis(hyp_b);

    // Corroborate hypothesis A with thermocouple measurement
    let ev = Evidence {
        source: "thermal_camera_infrared".to_string(),
        polarity: EvidencePolarity::Supports,
        confidence: 0.95,
        claim: "Melt zone thermal gradient is 18C below target setpoint".to_string(),
        observed_at: Some(chrono::Utc::now()),
        embedding: vec![],
        context: vec!["nozzle_diameter: 0.4mm".to_string()],
    };
    core.hypotheses.get_mut(&id_a).unwrap().add_supporting_evidence(ev);

    // Commit to something falsifiable BEFORE the next run, then record what
    // actually happened. This — not hand-assigning a status — is what moves
    // fitness: `resolve_prediction` writes `Prediction.correct`, which feeds
    // the `predictive_success` term and the falsified-prediction penalty.
    if let Some(h) = core.hypotheses.get_mut(&id_a) {
        h.add_prediction(Prediction::new(
            "Raising the setpoint 18C eliminates delamination on the next run",
        ));

        // ... the next run happens, and it does not ...
        let resolved = h.resolve_prediction(0, "Delamination unchanged at +18C", false);
        assert!(resolved);
        assert_eq!(h.predictions[0].correct, Some(false));

        // Fitness fell on its own; nothing set it by hand.
        // Resolution is write-once — a second call is refused, not applied:
        assert!(!h.resolve_prediction(0, "actually it worked", true));

        // Status changes go through `transition_to`, which records the
        // before/after pair in the revision history.
        h.transition_to(HypothesisStatus::Failed, "prediction falsified", None);
    }

    // Anything still outstanding, with the index `resolve` wants:
    for (idx, pending) in core.hypotheses[&id_a].open_predictions() {
        println!("#{idx} still open: {}", pending.statement);
    }
}
```

### 2. Truth Maintenance & Contradiction Resolution

```rust
use physis_core::{
    PhysisCore, Contradiction, ContradictionParty, ResolutionStatus,
};

fn main() {
    let mut core = PhysisCore::new();

    // Create an explicit contradiction between two sensor claims
    let claim_1 = ContradictionParty {
        source: "sensor_flow_meter_a".to_string(),
        claim: "Coolant line pressure is 4.2 bar (Nominal)".to_string(),
        confidence: 0.88,
        context: vec!["sampled_at_manifold".to_string()],
    };

    let claim_2 = ContradictionParty {
        source: "sensor_pressure_transducer_b".to_string(),
        claim: "Coolant line pressure is 0.8 bar (Cavitation Risk)".to_string(),
        confidence: 0.94,
        context: vec!["sampled_at_impeller".to_string()],
    };

    let conflict = Contradiction::new(claim_1, claim_2, 0.85);
    let conflict_id = core.register_contradiction(conflict);

    // Later: resolve with contextual grounding without deleting the dissenting record
    core.resolve_contradiction(
        &conflict_id,
        ResolutionStatus::ResolvedPreferredB,
        "Transducer B is downstream of clogged line filter; cavitation verified",
    );
}
```

### 3. Epistemic Audit Stream & Historical Replay

```rust
use physis_core::{
    PhysisCore, EpistemicEvent, EpistemicEventType,
};

fn main() {
    let mut core = PhysisCore::new();
    let hyp_id = "hyp-550e8400-e29b-41d4-a716-446655440000";

    // Record belief lifecycle events
    let t0 = chrono::Utc::now();
    core.epistemic_audit.record(
        EpistemicEvent::new(EpistemicEventType::HypothesisGenerated, hyp_id, "Candidate proposed")
            .with_metric(0.50),
    );

    core.epistemic_audit.record(
        EpistemicEvent::new(EpistemicEventType::HypothesisSupported, hyp_id, "Telemetry verified")
            .with_metric(0.92),
    );

    // Replay: Query what the engine believed at timestamp t0
    let snapshot = core.epistemic_audit.replay_state_at(hyp_id, t0);
    assert_eq!(snapshot.status_at_time, "Candidate");
}
```

### 4. Token-Fixed Budget RAG

```rust
use physis_core::{RagCorpus, RagChunk, TokenFixedRetriever, RandomProjectionEmbedder, VectorEmbed};

fn main() {
    let embedder = RandomProjectionEmbedder::new(64);
    let mut corpus = RagCorpus::new();

    corpus.add_chunk(RagChunk::new("doc-1", "Standard operating maintenance procedure for turbine pumps. Check oil level.", embedder.embed("turbine pump maintenance")));
    corpus.add_chunk(RagChunk::new("doc-2", "Emergency shutdown protocol for pressure loss exceeding 2 bar.", embedder.embed("emergency shutdown pressure")));

    let retriever = TokenFixedRetriever::new();
    let query_vec = embedder.embed("pump oil check");
    
    // Retrieve maximum relevant context bounded by 50 tokens
    let result = retriever.retrieve_bounded(&corpus, &query_vec, 50, 0.70);
    println!("Retrieved {} chunks ({} tokens)", result.chunks.len(), result.total_tokens);
}
```

### 5. Unsupervised Ontology Gap Discovery

```rust
use physis_core::{discover, DiscoveryConfig, RandomProjectionEmbedder, OntologyLoader, PhysisConfig};

fn main() {
    let embedder = RandomProjectionEmbedder::new(64);
    let ontology = OntologyLoader::load_all(&PhysisConfig::default());

    let unmapped_corpus = vec![
        "Quantum qubit decoherence in dilution refrigerator".to_string(),
        "Microwave pulse calibration on superconducting transmon".to_string(),
        "Surface code lattice surgery for fault-tolerant logical qubits".to_string(),
    ];

    let config = DiscoveryConfig {
        coverage_threshold: 0.85,
        min_cluster_size: 2,
        max_clusters: 5,
    };

    let report = discover(&unmapped_corpus, &ontology, &embedder, &config);
    println!("Discovered {} candidate domain proposals", report.proposed_domains.len());
}
```

### 6. Process Deviation Detection

```rust
use physis_core::{ProcessConstraint, ProcessCycle, ProcessMeasurement};

fn main() {
    let constraint = ProcessConstraint {
        id: "spindle-temp".into(),
        name: "Spindle temperature".into(),
        metric: "temp_c".into(),
        min_value: None,
        max_value: Some(80.0),
        is_hard_constraint: true,
    };

    let mut cycle = ProcessCycle::default();
    cycle.measurements.push(ProcessMeasurement {
        id: "m-1".into(),
        metric: "temp_c".into(),
        value: 92.0,
        unit: "°C".into(),
        machine_or_source: "cnc-01".into(),
        timestamp: chrono::Utc::now(),
        is_nominal: false,
    });

    // Flags every measurement outside its matching constraint's band.
    let added = cycle.scan_deviations(&[constraint]);
    println!("{added} deviation(s): {:?}", cycle.deviations);
}
```

### 7. Per-Agent Quality Tracking

```rust
use physis_core::{QualityTracker, RandomProjectionEmbedder};

fn main() {
    let mut quality = QualityTracker::new(Box::new(RandomProjectionEmbedder::new(64)));

    // A cell-level penalty (existing) and an agent-level one (new) are tracked
    // separately, so a bad agent in an otherwise-healthy domain gets demoted
    // without punishing every other agent working that cell.
    quality.report_agent_failure("agent-euclid-worker-3");
    quality.report_agent_failure("agent-euclid-worker-3");
    quality.report_agent_success("agent-euclid-worker-3");

    println!(
        "penalty: {:.2}, boost: {:.2}",
        quality.agent_penalties.get("agent-euclid-worker-3").copied().unwrap_or(0.0),
        quality.agent_boosts.get("agent-euclid-worker-3").copied().unwrap_or(0.0),
    );
}
```

---

## CLI Reference

### `physis` — the front door

Installing this crate gives you two executables: `physis-core`, the engine CLI,
and `physis`, a thin front door over whichever Physis edition is present.

```sh
cargo install physis-core     # installs `physis-core` and `physis`
physis -h                     # one help screen covering both editions
```

`physis -h` lists Core's commands and Pro's, marks which side is installed, and
forwards everything else unchanged — `physis classify …` runs `physis-core`,
`physis doctor …` runs `physis-pro`. Subcommand `--help`, exit codes and stdio
are untouched, because the front door `exec`s the target rather than wrapping it.
Names that exist on both sides (`classify`, `scan`, `discover`, `quality`,
`facet`) resolve to Core, so they mean the same thing whether or not Pro is
installed.

```sh
physis upgrade                # what Pro adds, and where to get it
physis web                    # serve the Pro dashboards (needs Pro)
```

Pro is a separate, licensed product. Core does not link it — the two are joined
at runtime by locating the Pro executable on disk — so Core keeps working, and
keeps its Apache-2.0 licence, whether or not Pro is there. The studio shows the
same information under its **Pro / Upgrade** tab, served from `/api/edition`.

### `physis-core`

The `physis-core` CLI exposes full engine capabilities directly to the terminal:

```sh
# 1. Semiotic Classification
physis-core classify "Nozzle temperature dropped below glass transition point"

# 2. Inspect Loaded Ontology Axes & Entries
physis-core ontology --search "thermal"

# 3. Filter Ontology Entries by Facets
physis-core facet --lifecycle OPERATE --agency SELF --kind machine

# 4. Corpus Ingestion into Coherence Graph
physis-core scan /path/to/engineering/vault

# 5. Coherence Similarity Search
physis-core search "bearing fatigue" --limit 5

# 6. Report Asserted Verdict (Reinforcement Signal)
physis-core assert "Extrusion nozzle check" failure

# 7. Execute Dream Cycle (Dissent Replay)
physis-core dream

# 8. Unsupervised Ontology Gap Discovery
physis-core discover /path/to/unclassified/notes --min-cluster 3

# 9. Launch the Embedded Studio Web Workbench
physis-core studio --port 3000 --host 127.0.0.1
```

### `physis-core hypothesis` — the epistemic loop

Everything above classifies. This subcommand is the part that *keeps score*: it
records what you believed, what you predicted would follow, and — crucially —
what actually happened.

```sh
# Register a candidate. Prints the id; the first 8 chars are enough everywhere.
physis-core hypothesis create "Clogs correlate with ambient humidity" --confidence 0.55

# What do we already believe about this? Ranked, dead claims hidden.
physis-core hypothesis query "humidity clog" --limit 5

# Attest evidence, with a polarity and a weight.
physis-core hypothesis evidence a1b2c3d4 "Three clogs, all above 62% RH" --source "line-2 log" --weight 0.7
physis-core hypothesis evidence a1b2c3d4 "One clog at 31% RH" --contradicts --weight 0.4

# Commit to something falsifiable BEFORE you find out.
physis-core hypothesis predict a1b2c3d4 "Next clog occurs above 60% RH" \
    --expected "RH > 60 at the timestamp of the next clog event"

# What did I predict and never check?  Oldest first.
physis-core hypothesis open --older-than 14

# Record the outcome. Fitness moves. This is the step that makes the rest real.
physis-core hypothesis resolve a1b2c3d4 0 "Clogged at 34% RH" --wrong

physis-core hypothesis explain a1b2c3d4     # full structured report
physis-core hypothesis transition a1b2c3d4 failed --reason "prediction falsified twice"
```

**Ids are prefixes.** Every command that takes an id accepts any unambiguous
prefix of it, because that is what `list` and `query` print. An ambiguous prefix
is an error that shows you the candidates; it never silently picks one.

**`resolve` requires a verdict.** Exactly one of `--correct` or `--wrong` — the
command refuses if you pass neither or both, rather than defaulting. A default in
either direction would let a falsified prediction be recorded as successful by
omission, which is the one outcome this command exists to prevent.

**Resolution is write-once.** Re-resolving an already-resolved prediction is
refused. A record whose outcome can change on a second call is not a record.

**What resolving actually moves.** `Prediction.correct` feeds `predictive_success`,
which is 25% of the composite fitness, plus a separate penalty for falsified
predictions:

```text
fitness 0.500 -> 0.225    after one prediction resolved --wrong
fitness 0.500 -> 0.625    after one prediction resolved --correct
```

#### A caveat on `query` relevance scores

`query` ranks by cosine similarity in the embedding space, and **the default
embedder is `RandomProjectionEmbedder`** — hashed bag-of-words with no trained
weights. Its absolute scores are not meaningful: unrelated sentences routinely
score above 0.8. In one measured case, the query *"stripe payment webhooks in the
praxis backend"* scored **0.830** against a claim about `OntologyDeltaReport` and
ranked it first.

Treat the *ordering* as a weak hint and the *number* as noise. The CLI prints
this warning in its own output rather than burying it here, because the moment
you need to know is the moment you are reading a score. Build with
`--features embed-onnx` and supply `model.onnx` + `tokenizer.json` for
similarity that means something.

---

## Embedded Studio Web Workbench

Launch the studio with `physis-core studio --port 3000`:

- **Classify Workbench**: Live multi-cell classification, raw vs quality-penalized score comparisons, nearest entry details, and one-click feedback buttons (`✓ Success` / `✕ Failure`).
- **Semiotic Heatmap**: Interactive visual grid of the 5 domains × 14 modes with dynamic axis discovery and cell density mapping.
- **Ontology Editor**: Search, create, and modify domain entries, units, synonyms, and sub-domain facets with instantaneous re-indexing.
- **Corpus & Coherence Graph**: Browse labeled nodes, examine confidence links, and trigger Dream cycles over dissenting paths.
- **Gap Discovery Studio**: Run clustering over unmapped document collections and promote discovered domain clusters into first-class ontology nodes with a single click.
- **Quality & Feedback Matrix**: View active penalties, inspect failure records, and apply boosts to undo historical penalties.

### Environment

| Variable | Default | Effect |
| --- | --- | --- |
| `PHYSIS_CORE_DIR` | `$HOME/.physis-core` | State directory for `nodes.json`, `quality.json` and `custom_ontology.json`. The studio and the CLI both resolve the graph through it, which is why `physis-core scan` and the studio see the same nodes; set it to work on a per-project graph, mount a volume in a container, or keep a test run away from your real graph. |
| `PHYSIS_STUDIO_HOST` | `127.0.0.1` | Bind address. Loopback by default because the ingest and scan routes read arbitrary local paths — only widen it (e.g. `0.0.0.0`) where the port is not publicly reachable. |
| `PHYSIS_EMBEDDER` | auto-detect | `random-projection` selects the deterministic offline embedder: reproducible and coarse, reported as `semantic: false` but *not* as degraded, since it was asked for rather than fallen back to. |

---

## Publishing to Crates.io & Integration

`physis-core` is distributed as a standalone crate and as part of the `physis-pro` industrial suite.

### Publishing Verification

```sh
# Run clippy across all feature sets
cargo clippy --package physis-core --all-targets --all-features -- -D warnings

# Execute test suite
cargo test --package physis-core --all-targets --all-features

# Build documentation locally
cargo doc --package physis-core --all-features --no-deps --open

# Dry-run publish
cargo publish --package physis-core --dry-run
```

---

---

## Practical Examples

### Build a Quality Prediction Agent

```rust
use physis_core::{
    PhysisCore, RandomProjectionEmbedder, VectorEmbed,
    quality::QualityTracker,
    classify::{classify_text, ClassifierConfig},
    ontology::OntologyLoader,
    config::PhysisConfig,
};

fn main() -> anyhow::Result<()> {
    let embedder = RandomProjectionEmbedder::new(64);
    let ontology = OntologyLoader::load_all(&PhysisConfig::default());
    let mut core = PhysisCore::new();
    core.set_embedder_id("random-projection");

    // Load quality tracker (persisted penalties)
    let mut quality = QualityTracker::load("~/.physis-core/quality.json")?;

    // Ingest a production defect report
    let report = "Surface finish degradation on turned parts — chatter marks";
    let classification = classify_text(report, &ontology, &embedder, &ClassifierConfig::default());

    println!("Top cell: {}×{} ({:.2})", 
        classification.top_cell.domain, 
        classification.top_cell.mode, 
        classification.top_cell.score);

    // Apply quality penalty for this cell (learned from past failures)
    let penalty = quality.penalty_for_cell(&classification.top_cell.domain, &classification.top_cell.mode);
    println!("Learned penalty: {:.2}", penalty);

    // Register the defect with asserted failure
    let node_id = core.register_node_from_text(report, &embedder)?;
    core.assert_coherence(&node_id, -1.0);

    // Save updated quality tracker
    quality.save("~/.physis-core/quality.json")?;

    Ok(())
}
```

### Competing Hypotheses for Root Cause Analysis

```rust
use physis_core::{
    PhysisCore, Hypothesis, HypothesisStatus, Evidence, EvidencePolarity,
    RandomProjectionEmbedder, VectorEmbed,
};

fn main() {
    let embedder = RandomProjectionEmbedder::new(128);
    let mut core = PhysisCore::new();

    // Hypothesis A: Thermal issue
    let emb_a = embedder.embed("Nozzle temperature instability causes layer adhesion failure");
    let mut hyp_a = Hypothesis::new("Thermal instability", emb_a);
    hyp_a.assumptions.push("Thermistor reads true melt temperature".to_string());
    let id_a = core.register_hypothesis(hyp_a);

    // Hypothesis B: Material issue  
    let emb_b = embedder.embed("Wet filament causes steam bubbles and poor layer bonding");
    let mut hyp_b = Hypothesis::new("Moisture contamination", emb_b);
    hyp_b.assumptions.push("Filament stored in sealed container with desiccant".to_string());
    let id_b = core.register_hypothesis(hyp_b);

    // Evidence supporting A
    core.attach_evidence(&id_a, Evidence {
        source: "thermal_camera".to_string(),
        polarity: EvidencePolarity::Supports,
        confidence: 0.92,
        claim: "Nozzle temp oscillates ±8°C during print".to_string(),
        observed_at: Some(chrono::Utc::now()),
        embedding: vec![],
        context: vec!["material: PLA".to_string(), "nozzle: 0.4mm".to_string()],
    });

    // Evidence refuting B
    core.attach_evidence(&id_b, Evidence {
        source: "humidity_sensor".to_string(),
        polarity: EvidencePolarity::Refutes,
        confidence: 0.88,
        claim: "Filament chamber RH 12% — well below absorption threshold".to_string(),
        observed_at: Some(chrono::Utc::now()),
        embedding: vec![],
        context: vec!["chamber: dry_box".to_string()],
    });

    // Update fitness scores
    if let Some(h) = core.hypotheses.get_mut(&id_a) {
        h.status = HypothesisStatus::Supported;
        h.fitness = 0.91;
    }
    if let Some(h) = core.hypotheses.get_mut(&id_b) {
        h.status = HypothesisStatus::Refuted;
        h.fitness = 0.22;
    }

    // Best hypothesis wins
    let best = core.hypotheses.values().max_by(|a,b| a.fitness.partial_cmp(&b.fitness).unwrap());
    println!("Root cause: {} (fitness: {:.2})", best.unwrap().label, best.unwrap().fitness);
}
```

### Token-Budgeted RAG for LLM Context

```rust
use physis_core::{
    RagCorpus, RagChunk, TokenFixedRetriever, RandomProjectionEmbedder, VectorEmbed,
};

fn main() {
    let embedder = RandomProjectionEmbedder::new(64);
    let mut corpus = RagCorpus::new();

    // Ingest technical docs
    corpus.add_chunk(RagChunk::new(
        "sof-001", 
        "SOP-204: Spindle bearing replacement. Torque to 45Nm. Run-in 30min at 500RPM.",
        embedder.embed("spindle bearing replacement torque run-in")
    ));
    corpus.add_chunk(RagChunk::new(
        "sof-002",
        "SOP-312: Coolant filter change. Interval 500h. Use FN-7 filter only.",
        embedder.embed("coolant filter change interval")
    ));
    corpus.add_chunk(RagChunk::new(
        "sof-003",
        "SOP-108: Emergency stop reset. Verify guard closure. Press blue reset 3s.",
        embedder.embed("emergency stop reset guard")
    ));

    let retriever = TokenFixedRetriever::new();
    let query = embedder.embed("bearing torque procedure");

    // Hard budget: 200 tokens max, diversity threshold 0.7
    let result = retriever.retrieve_bounded(&corpus, &query, 200, 0.70);
    
    println!("Retrieved {} chunks, {} tokens", result.chunks.len(), result.total_tokens);
    for chunk in result.chunks {
        println!("  [{}] {}", chunk.id, chunk.text);
    }
}
```

### Ontology Gap Discovery — Find Missing Domains

```rust
use physis_core::{
    discover, DiscoveryConfig, RandomProjectionEmbedder, OntologyLoader, PhysisConfig,
};

fn main() {
    let embedder = RandomProjectionEmbedder::new(64);
    let ontology = OntologyLoader::load_all(&PhysisConfig::default());

    // Unclassified maintenance logs from a new machine type
    let logs = vec![
        "Laser power drift during micro-welding cycle".to_string(),
        "Beam focus position variance exceeds 15 microns".to_string(),
        "Argon shield gas flow instability at 12 L/min".to_string(),
        "Galvo scanner hysteresis on tight radius corners".to_string(),
    ];

    let config = DiscoveryConfig {
        coverage_threshold: 0.75,   // Flag texts below 0.75 similarity
        min_cluster_size: 2,        // Need at least 2 similar texts
        max_clusters: 8,
    };

    let report = discover(&logs, &ontology, &embedder, &config);

    println!("Gap report: {} texts analyzed, {} clusters found", 
        report.analyzed, report.proposed_domains.len());
    
    for (i, domain) in report.proposed_domains.iter().enumerate() {
        println!("  Proposal {}: {}×{} — {}", i+1, domain.domain, domain.mode, domain.rationale);
        println!("    Exemplars: {}", domain.exemplars.join(", "));
    }

    // Promote to ontology (serializes JSON for config/)
    println!("{}", serde_json::to_string_pretty(&report.proposed_domains)?);
}
```

### Contradiction Tracking — Non-Destructive Conflict Resolution

```rust
use physis_core::{
    PhysisCore, Contradiction, ContradictionParty, ResolutionStatus,
};

fn main() {
    let mut core = PhysisCore::new();

    // Sensor A says pressure nominal
    let claim_a = ContradictionParty {
        source: "pressure_transducer_primary".to_string(),
        claim: "System pressure 6.2 bar — within spec".to_string(),
        confidence: 0.91,
        context: vec!["location: manifold".to_string()],
    };

    // Sensor B says cavitation
    let claim_b = ContradictionParty {
        source: "acoustic_emission_sensor".to_string(),
        claim: "Cavitation signatures detected at impeller".to_string(),
        confidence: 0.96,
        context: vec!["location: pump_impeller".to_string()],
    };

    // Register explicit contradiction (both preserved)
    let conflict = Contradiction::new(claim_a, claim_b, 0.93);
    let conflict_id = core.register_contradiction(conflict);

    // Later: resolve with contextual grounding — dissent NOT deleted
    core.resolve_contradiction(
        &conflict_id,
        ResolutionStatus::ResolvedPreferredB,
        "Primary transducer upstream of clogged suction filter; cavitation confirmed by borescope",
    );

    // Audit trail shows full history
    let events = core.epistemic_audit.query(&conflict_id);
    for e in events {
        println!("[{}] {} — {}", e.timestamp, e.event_type, e.note);
    }
}
```

### CLI Workflows

```bash
# Classify a technical issue
physis-core classify "Coolant pump seal leaking at 200h interval"

# Search ontology for relevant entries
physis-core ontology --search "seal"

# Filter by facets (machine process domain)
physis-core facet --kind machine --lifecycle OPERATE

# Ingest a vault of engineering notes
physis-core scan ~/engineering-notes

# Search coherence graph for similar issues
physis-core search "pump seal" --limit 10

# Assert a failure verdict (reinforcement learning signal)
physis-core assert "Coolant pump seal leak" failure

# Run dream cycle to generate countermeasures
physis-core dream

# Discover ontology gaps in unclassified logs
physis-core discover ~/unclassified-logs --min-cluster 3

# Launch embedded studio UI
physis-core studio --port 3000
```

### Embedded Studio Web Workbench

```bash
# Start the studio (feature: studio, enabled by default)
physis-core studio --port 3000 --host 127.0.0.1

# Open http://127.0.0.1:3000 for:
# - Classify Workbench: live multi-cell classification with quality penalties
# - Semiotic Heatmap: interactive 5×14 grid with density mapping
# - Ontology Editor: create/modify domain entries with instant re-indexing
# - Corpus & Coherence Graph: browse nodes, examine confidence links
# - Gap Discovery Studio: cluster unmapped docs → promote to ontology
# - Quality Matrix: view penalties, inspect failures, apply boosts
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PHYSIS_CORE_DIR` | `$HOME/.physis-core` | State directory (`nodes.json`, `quality.json`, `custom_ontology.json`) |
| `PHYSIS_STUDIO_HOST` | `127.0.0.1` | Bind address (loopback by default — scan routes read local paths) |
| `PHYSIS_EMBEDDER` | auto-detect | `random-projection` for deterministic offline mode |

---

## License

`physis-core` is dual-licensed under the **Apache License, Version 2.0** ([LICENSE](LICENSE)).

