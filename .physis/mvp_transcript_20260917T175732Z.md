# MVP transcript — /home/gio/dev/physis-pro/physis-core

## startup

```
PHYSIS — physis-core

files          339
objects        2992
relations      0
sample map     repeats 9 · differences 8 · contradictions 1
index          ready

```


## input

```
what does this project do?
```


## intent: what does this project do?

```
PROJECT
 └─ examples
     └─ experiment47_cross_model_divergence.rs
         └─ examples/experiment47_cross_model_divergence.rs:841-880  ↔ 0.56  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
         └─ examples/experiment47_cross_model_divergence.rs:241-280  ↔ 0.55  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
         └─ examples/experiment47_cross_model_divergence.rs:961-1000  ↔ 0.55  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
     └─ impossible_machine_experiment.rs
         └─ examples/impossible_machine_experiment.rs:281-320  ↔ 0.55  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
     └─ perspective_invention_experiment.rs
         └─ examples/perspective_invention_experiment.rs:521-560  ↔ 0.55  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
 └─ src
     └─ propose.rs
         └─ src/propose.rs:161-200  ↔ 0.55  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
 └─ src/bin
     └─ world.rs
         └─ src/bin/world.rs:121-160  ↔ 0.55  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60

trail: ?

── CONTEXT ──
packed 2000/2000 tokens of 1204962 candidate tokens from 6 chunks (2761 candidates over 348 files, 2755 dropped)
   0.795  examples/experiment47_cross_model_divergence.rs:841-880  (460 tokens)
   0.795  examples/impossible_machine_experiment.rs:281-320  (386 tokens)
   0.792  examples/experiment47_cross_model_divergence.rs:241-280  (437 tokens)
   0.793  examples/experiment34_shared_substructure.rs:241-280  (471 tokens)
   0.781  src/transform.rs:441-480  (244 tokens)
   0.746  src/coverage.rs:441-442  (2 tokens)

── TELEMETRY ──
repository objects considered 2992
objects selected 6
candidate/raw context tokens 1204962
final context tokens 2000
model/provider used random-projection (extractive; oracle when configured)
agent calls 0
latency 412 ms
  [0.56] examples/experiment47_cross_model_divergence.rs  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
  [0.55] examples/impossible_machine_experiment.rs  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
  [0.55] examples/experiment47_cross_model_divergence.rs  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
  [0.55] examples/experiment47_cross_model_divergence.rs  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
  [0.55] examples/perspective_invention_experiment.rs  why: lexical 0.00 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60

```


## answer: what does this project do?

```
── ANSWER · EXTRACTIVE (n-gram decoder — fragments, not synthesis) ──

Box::new(move |r: &Row| coh(r)),             ), [1]
let v = x.iter().map(|r| (r[j] - mu[j]).powi(2)).sum::<f64>() / n;         sd[j] = v.sqrt().max(1e-9); [2]
let n = self.n_max as usize;         let mut worst = 0.0f64; [3]

!! DEGRADED: no PHYSIS_ORACLE_KEY / OPENROUTER_API_KEY / GROQ_API_KEY — synthesis needs a generator. A local ollama works: PHYSIS_ORACLE_URL=http://localhost:11434/v1 PHYSIS_ORACLE_MODEL=<model> PHYSIS_ORACLE_KEY=ollama

── SOURCES ──
  [1] examples/experiment47_cross_model_divergence.rs:841-880 (0.657)
      Box::new(move |r: &Row| coh(r)),
  [2] examples/experiment47_cross_model_divergence.rs:241-280 (0.653)
      let v = x.iter().map(|r| (r[j] - mu[j]).powi(2)).sum::<f64>() / n;
  [3] examples/impossible_machine_experiment.rs:281-320 (0.638)
      let n = self.n_max as usize;
  [4] examples/experiment34_shared_substructure.rs:241-280 (0.633)
      hi_i.len(),
  [5] src/transform.rs:441-480 (0.599)
      } else {
  [6] src/coverage.rs:441-442 (0.540)
      }

6 documents · context 2000 tokens (conventional retrieval 2000) · 15 ms
retrieval backend: random-projection

```


## input

```
where is retrieval implemented?
```


## intent: where is retrieval implemented?

```
PROJECT
 └─ 
     └─ CHANGELOG.md
         └─ CHANGELOG.md:41-80  ↔ 0.72  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.82 × 0.60
         └─ CHANGELOG.md:81-120  ↔ 0.72  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.82 × 0.60
     └─ DEMO.md
         └─ DEMO.md:561-600  ↔ 0.71  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
     └─ NEUROSEMANTIC_LAYER.md
         └─ NEUROSEMANTIC_LAYER.md:1-40  ↔ 0.70  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
     └─ PLANNING.md
         └─ PLANNING.md:441-480  ↔ 0.71  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.80 × 0.60
     └─ README.md
         └─ README.md:81-120  ↔ 0.71  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
         └─ README.md:121-160  ↔ 0.71  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
         └─ README.md:1-40  ↔ 0.70  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
     └─ SHIP_PLAN.md
         └─ SHIP_PLAN.md:41-72  ↔ 0.70  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
 └─ research
     └─ E56_POSITION_FIRST.md
         └─ research/E56_POSITION_FIRST.md:81-118  ↔ 0.70  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
 └─ src
     └─ lib.rs
         └─ src/lib.rs:41-80  ↔ 0.71  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.80 × 0.60
         └─ src/lib.rs:1-40  ↔ 0.70  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60

trail: ? › retrieval implemented?

── CONTEXT ──
packed 743/2000 tokens of 1204962 candidate tokens from 8 chunks (2761 candidates over 348 files, 2753 dropped)
   0.839  examples/demo-corpus/maint-04.md:1-1  (27 tokens)
   0.831  benchmarks/ground-truth/deploy-06.md:1-1  (20 tokens)
   0.821  PLANNING.md:561-565  (84 tokens)
   0.824  CHANGELOG.md:41-80  (504 tokens)
   0.839  examples/demo-corpus/maint-02.md:1-1  (27 tokens)
   0.838  examples/demo-corpus/maint-07.md:1-1  (27 tokens)
   0.838  examples/demo-corpus/maint-01.md:1-1  (27 tokens)
   0.838  examples/demo-corpus/maint-00.md:1-1  (27 tokens)

── TELEMETRY ──
repository objects considered 2992
objects selected 8
candidate/raw context tokens 1204962
final context tokens 743
model/provider used random-projection (extractive; oracle when configured)
agent calls 0
latency 334 ms
  [0.72] CHANGELOG.md  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.82 × 0.60
  [0.72] CHANGELOG.md  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.82 × 0.60
  [0.71] DEMO.md  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
  [0.71] README.md  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
  [0.71] README.md  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60

```


## answer: where is retrieval implemented?

```
── ANSWER · EXTRACTIVE (n-gram decoder — fragments, not synthesis) ──

Pump maintenance required: replace the seal on line 5, vibration rising after bearing wear. Schedule downtime tonight and file the shift report. [1]
Pump maintenance required: replace the seal on line 3, vibration rising after bearing wear. Schedule downtime tonight and file the shift report. [2]
Pump maintenance required: replace the seal on line 8, vibration rising after bearing wear. Schedule downtime tonight and file the shift report. [3]

!! DEGRADED: no PHYSIS_ORACLE_KEY / OPENROUTER_API_KEY / GROQ_API_KEY — synthesis needs a generator. A local ollama works: PHYSIS_ORACLE_URL=http://localhost:11434/v1 PHYSIS_ORACLE_MODEL=<model> PHYSIS_ORACLE_KEY=ollama

── SOURCES ──
  [1] examples/demo-corpus/maint-04.md:1-1 (0.882)
      Pump maintenance required: replace the seal on line 5, vibration rising after bearing wear. Schedule downtime tonight and file the shift rep
  [2] examples/demo-corpus/maint-02.md:1-1 (0.881)
      Pump maintenance required: replace the seal on line 3, vibration rising after bearing wear. Schedule downtime tonight and file the shift rep
  [3] examples/demo-corpus/maint-07.md:1-1 (0.881)
      Pump maintenance required: replace the seal on line 8, vibration rising after bearing wear. Schedule downtime tonight and file the shift rep
  [4] examples/demo-corpus/maint-01.md:1-1 (0.881)
      Pump maintenance required: replace the seal on line 2, vibration rising after bearing wear. Schedule downtime tonight and file the shift rep
  [5] examples/demo-corpus/maint-00.md:1-1 (0.880)
      Pump maintenance required: replace the seal on line 1, vibration rising after bearing wear. Schedule downtime tonight and file the shift rep
  [6] CHANGELOG.md:41-80 (0.853)
      - See the commit history for E55-E66: log-ordered world states, entity-linked
  [7] benchmarks/ground-truth/deploy-06.md:1-1 (0.844)
      Release 23 deployed to staging: config drift fixed, rollout verified, telemetry green, rollback documented.
  [8] PLANNING.md:561-565 (0.834)
      `dream_proposes_restoring_severed_connections`,

8 documents · context 743 tokens (conventional retrieval 743) · 9 ms
retrieval backend: random-projection

```


## input

```
inspect
```


## inspect CHANGELOG.md:41-80

```
--- CHANGELOG.md:41-80 (Passage) ---
provenance: CHANGELOG.md
parents:
  fs     .
  note   CHANGELOG.md
  kind   Passage
relations:
  (none)

source:

- See the commit history for E55-E66: log-ordered world states, entity-linked
  retrieval reported beside the order-blind arm, and set-operation transition
  labels against a shuffled-order null.

## 0.1.24

### Added — structural transform algebra + n-gram embedder scaffold

- **Transform algebra module** (`transform.rs`): homomorphism engine with
  TraceStep-gated apply; 4 lib tests plus a 4-test adversarial bench
  vs similarity/rule baselines.
- **SyntheticNGramEmbedder scaffold** (`embed_ngram.rs`) plus test-target fix.
- Version bump only; no API break.

## 0.1.23

### Added — W1 resource integration (PLAN.md Phase 24 wave 1, 2026-09-10)

**T4 Nixon Diamond benchmark seed** (`tests/epistemic_w1.rs`): the ATMS
`rules_nixon` scaled to Physis — p and ¬p coexist with their evidence, the
contradiction is first-class and open, both sides leave a replayable audit
footprint, and a later human verdict disprefers without destroying
(`nixon_diamond_retains_both_sides`).

**T3 evidence retraction with JTMS cascade**:
`hypothesis::Hypothesis::retract_evidence` re-derives a hypothesis without a
source — status follows the closed ingest rule, fitness recomputes, the
retraction is an audited revision, and `Certified` survives (authority
leg; the authority registry itself is T11). `delta_engine::retract_evidence_with_cascade`
walks the A2 DependsOn closure of the retracted hypothesis's references and
re-evaluates every selected dependent in the shadow frame — a proposal until
committed (`evidence_retraction_replays_to_state_without_it`).

**G4 per-link intake ids**: `Evidence::intake_id`, `ProvenanceLink::intake_id`
and `EpistemicEvent::intake_id` stamp every link and event with the ingest
episode that produced it; `explanation::ExplanationReport::cited_intake_ids`
and `provenance::ProvenanceChain::cited_intake_ids` resolve an explanation's
citations to the concrete raw intakes (`citations_resolve_to_intake_ids`).


```


## input

```
dependencies
```


## dependencies CHANGELOG.md

```
dependencies of CHANGELOG.md
  ↔ config/machine_ontology.json
  ↔ config/software_dev_ontology.json

```


## input

```
what changed recently?
```


## recent

```
d65858b feat(system): delegation planning, an MCP surface for it, and the reference-spectrum example
8448393 fix(doc): a shell snippet in a doc comment was compiled as Rust
6a450bd fix(examples): a feature-gated main stopped `cargo test` for the whole crate
75acec6 feat(system): predict backs off to a borrowed prior, gated by coverage
60d3b31 feat: declare prediction-from-the-record as a capability
1210b1a feat(system): a run reads the prior it is about to defy
2c3a2dd feat(system): predict — read the workspace's own record before acting
9188b6f fix(cli): `physis … | head` printed a panic after the output it was asked for
145609e feat(system): --include-dir, because the default exclusions hide published artifacts
194e2d1 feat: declare the capabilities, so the claim is checkable instead of narrative
65e3ccc fix(system): history printed every note twice
56b5d6f feat(system): a workspace interface people and agents share, and a packer that pays for itself
88d61ed feat(worldstate): navigational layer over the machine's own log
0203b46 feat(c3): grade the interpreter swap instead of hashing it
9798c43 feat(modes): weight the unowned supersenses by what the corpus actually does

```


## input

```
improve retrieval without changing its public API
```


## intent: improve retrieval without changing its public API

```
PROJECT
 └─ 
     └─ CHANGELOG.md
         └─ CHANGELOG.md:41-80  ↔ 0.79  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
     └─ DEMO.md
         └─ DEMO.md:561-600  ↔ 0.80  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.82 × 0.60
         └─ DEMO.md:1-40  ↔ 0.79  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.82 × 0.60
         └─ DEMO.md:81-120  ↔ 0.79  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
     └─ README.md
         └─ README.md:81-120  ↔ 0.79  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
         └─ README.md:1-40  ↔ 0.79  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
 └─ research
     └─ E56_POSITION_FIRST.md
         └─ research/E56_POSITION_FIRST.md:81-118  ↔ 0.73  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.83 × 0.60
 └─ src
     └─ coherence_query.rs
         └─ src/coherence_query.rs:1-40  ↔ 0.73  why: lexical 0.50 × 0.30 · recency 0.89 × 0.08 · semantic 0.85 × 0.60
     └─ lib.rs
         └─ src/lib.rs:41-80  ↔ 0.78  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
     └─ rag.rs
         └─ src/rag.rs:41-80  ↔ 0.75  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.74 × 0.60
     └─ system.rs
         └─ src/system.rs:1-40  ↔ 0.77  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.78 × 0.60
         └─ src/system.rs:121-160  ↔ 0.75  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.75 × 0.60

trail: ? › retrieval implemented? › retrieval its public API

── CONTEXT ──
packed 1992/2000 tokens of 1204962 candidate tokens from 8 chunks (2761 candidates over 348 files, 2753 dropped)
   0.866  research/impossible_machine/EXPERIMENT51_PLAN.md:41-62  (532 tokens)
   0.840  src/system_delegation.rs:121-160  (287 tokens)
   0.850  src/coherence_query.rs:1-40  (427 tokens)
   0.831  benchmarks/ground-truth/maint-07.md:1-1  (18 tokens)
   0.840  examples/experiment12_integrated_synthesis.rs:521-539  (500 tokens)
   0.822  benchmarks/ground-truth/extra-device-spec-A.md:1-1  (19 tokens)
   0.828  research/E53_TRANSFORM_ANALOGY.md:81-95  (207 tokens)
   0.571  .gitignore:1-1  (2 tokens)

── TELEMETRY ──
repository objects considered 2992
objects selected 8
candidate/raw context tokens 1204962
final context tokens 1992
model/provider used random-projection (extractive; oracle when configured)
agent calls 0
latency 317 ms
  [0.80] DEMO.md  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.82 × 0.60
  [0.79] DEMO.md  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.82 × 0.60
  [0.79] README.md  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
  [0.79] README.md  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
  [0.79] DEMO.md  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60

```


## plan: improve retrieval without changing its public API

```
── PLAN ──
intent: improve retrieval without changing its public API
scoped files: .gitignore, benchmarks/ground-truth/extra-device-spec-A.md, benchmarks/ground-truth/maint-07.md, examples/experiment12_integrated_synthesis.rs, research/E53_TRANSFORM_ANALOGY.md, research/impossible_machine/EXPERIMENT51_PLAN.md, src/coherence_query.rs, src/system_delegation.rs
agent: existing harness reads the context; no new framework.
(pass --yes to approve)

```


## input

```
improve retrieval without changing its public API --yes
```


## intent: improve retrieval without changing its public API

```
PROJECT
 └─ 
     └─ CHANGELOG.md
         └─ CHANGELOG.md:41-80  ↔ 0.79  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
     └─ DEMO.md
         └─ DEMO.md:561-600  ↔ 0.80  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.82 × 0.60
         └─ DEMO.md:1-40  ↔ 0.79  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.82 × 0.60
         └─ DEMO.md:81-120  ↔ 0.79  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
     └─ README.md
         └─ README.md:81-120  ↔ 0.79  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
         └─ README.md:1-40  ↔ 0.79  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
 └─ research
     └─ E56_POSITION_FIRST.md
         └─ research/E56_POSITION_FIRST.md:81-118  ↔ 0.73  why: lexical 0.50 × 0.30 · recency 0.98 × 0.08 · semantic 0.83 × 0.60
 └─ src
     └─ coherence_query.rs
         └─ src/coherence_query.rs:1-40  ↔ 0.73  why: lexical 0.50 × 0.30 · recency 0.89 × 0.08 · semantic 0.85 × 0.60
     └─ lib.rs
         └─ src/lib.rs:41-80  ↔ 0.78  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.79 × 0.60
     └─ rag.rs
         └─ src/rag.rs:41-80  ↔ 0.75  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.74 × 0.60
     └─ system.rs
         └─ src/system.rs:1-40  ↔ 0.77  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.78 × 0.60
         └─ src/system.rs:121-160  ↔ 0.75  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.75 × 0.60

trail: ? › retrieval implemented? › retrieval its public API › retrieval its public API

── CONTEXT ──
packed 1992/2000 tokens of 1204962 candidate tokens from 8 chunks (2761 candidates over 348 files, 2753 dropped)
   0.866  research/impossible_machine/EXPERIMENT51_PLAN.md:41-62  (532 tokens)
   0.840  src/system_delegation.rs:121-160  (287 tokens)
   0.850  src/coherence_query.rs:1-40  (427 tokens)
   0.831  benchmarks/ground-truth/maint-07.md:1-1  (18 tokens)
   0.840  examples/experiment12_integrated_synthesis.rs:521-539  (500 tokens)
   0.822  benchmarks/ground-truth/extra-device-spec-A.md:1-1  (19 tokens)
   0.828  research/E53_TRANSFORM_ANALOGY.md:81-95  (207 tokens)
   0.571  .gitignore:1-1  (2 tokens)

── TELEMETRY ──
repository objects considered 2992
objects selected 8
candidate/raw context tokens 1204962
final context tokens 1992
model/provider used random-projection (extractive; oracle when configured)
agent calls 0
latency 290 ms
  [0.80] DEMO.md  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.82 × 0.60
  [0.79] DEMO.md  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.82 × 0.60
  [0.79] README.md  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
  [0.79] README.md  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60
  [0.79] DEMO.md  why: lexical 0.75 × 0.30 · recency 0.98 × 0.08 · semantic 0.81 × 0.60

```


## act: improve retrieval without changing its public API

```
── PLAN ──
intent: improve retrieval without changing its public API
scoped files: .gitignore, benchmarks/ground-truth/extra-device-spec-A.md, benchmarks/ground-truth/maint-07.md, examples/experiment12_integrated_synthesis.rs, research/E53_TRANSFORM_ANALOGY.md, research/impossible_machine/EXPERIMENT51_PLAN.md, src/coherence_query.rs, src/system_delegation.rs
agent: existing harness reads the context; no new framework.

Δ

CHANGED
  (none)

AFFECTED
  (none)

STRUCTURE
  (no relation churn)

VALIDATION
  act: cd '/home/gio/dev/physis-pro/physis-core' && git -C .. status --short | head -n 12 exit Some(0) in 45 ms
     M src/main.rs
     M src/pack.rs
     M src/ported/act.rs
     M src/scanner.rs
    ?? .mvp-backup-20260917/
    ?? LIMITATIONS.md
    ?? MVP_STATUS.md
    ?? POST_MVP.md
    ?? docs/MVP_DEMO.md
    ?? scripts/mvp-demo.sh
  file-type invariants preserved (rs/md/json/toml shapes)

MAP
  0 objects remapped (2992 → 2992, incremental)
  0 semantic neighborhood(s) changed


```


## input

```
quit
```

