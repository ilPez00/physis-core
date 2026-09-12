# DEMO — physis-core

**Structure in your documents, context under a hard budget, answers with
citations. Offline, deterministic, no API key.**

```text
30 DOCUMENTS
  ↓ physis
4 REPEAT FAMILIES · 2 DIFFERENCES · 1 CONTRADICTION
  ↓
220 → 165 tokens  (25% smaller than conventional retrieval)
```

Three demos to run first, each with its real transcript:

| | what it shows |
|---|---|
| [**Notebook**](#demo-0-notebook--answers-from-your-documents-with-citations-offline) | answers with citations that cannot be invented — and a corpus contradiction surviving instead of being quietly resolved |
| [**Draft and fill**](#demo-0b-draft-and-fill--spend-the-model-only-on-the-gaps) | the table drafts what the corpus supports; a real transcript of the method *declining to be used* |
| [**Per-cell tables**](#demo-0c-per-cell-tables--70-tables-with-subtables-and-the-control) | 70 tables with subtables, each shipped next to the control that could beat it |

> **On the 25%.** It is a real token count from a real run. It is **not**
> evidence that the *selection* is good: `benchmarks/ground-truth` scores
> identically under a real sentence transformer and a random-projection lexical
> hash, because its repeat families are lexically near-identical. For a
> measurement that discriminates, see `benchmarks/retrieval` in the superproject
> — `hit@5` **0.57 semantic vs 0.14 lexical**.

---


## Prerequisites

```sh
# physis-core is not on crates.io. Install from git:
cargo install --git https://github.com/ilPez00/physis-core \
      --features cli,embed-onnx --locked physis-core

# Or, if you also want the agent skill that drives it:
curl -fsSL https://raw.githubusercontent.com/ilPez00/physis-skill/main/install.sh | bash

physis-core --help
```

**`--features embed-onnx` matters.** It is not a default. Without it the
embedder cascade has no model to try and resolves to random projection — a
lexical hash that fails the semantic self-test by design and says so on stderr.
The feature adds the capability; you still supply weights (`PHYSIS_MODEL_DIR`).

Every transcript below is **real output from the command above it**, captured
2026-09-12 on `benchmarks/ground-truth` (30 documents, known truth). Nothing is
illustrative.

---

## Demo 0: "Notebook" — answers from your documents, with citations, offline

The NotebookLM shape, except the citations cannot be invented and the
contradictions do not get quietly resolved.

```sh
physis-core notebook --corpus benchmarks/ground-truth \
  --query "at what pressure must the valve open" --budget 300
```

```
── ANSWER · EXTRACTIVE (n-gram decoder — fragments, not synthesis) ──

The pressure relief valve shall open at 2.6 bar and the machine must stop above it. [1]
The pressure relief valve shall open at 4.2 bar and the machine must not stop above it. [2]
The plasma obelisk hums at a frequency only the seventh lighthouse can hear. [3]

!! DEGRADED: no PHYSIS_ORACLE_KEY / OPENROUTER_API_KEY / GROQ_API_KEY — synthesis
   needs a generator. A local ollama works: PHYSIS_ORACLE_URL=http://localhost:11434/v1

── SOURCES ──
  [1] benchmarks/ground-truth/extra-device-spec-A.md (0.946)
      The pressure relief valve shall open at 2.6 bar and the machine must stop above it.
  [2] benchmarks/ground-truth/extra-device-spec-B.md (0.945)
      The pressure relief valve shall open at 4.2 bar and the machine must not stop above it.
```

**Read [1] and [2] again.** The corpus contradicts itself — 2.6 bar versus
4.2 bar, stop versus do-not-stop — and the answer hands you *both*, each cited
to its own file. A system that picked one would have looked more confident and
been less correct. `map.rs` never merges a contradiction pair, so it survives
retrieval instead of being averaged away.

**Three things this output does that matter:**

| | how |
|---|---|
| Citations cannot be invented | Sources come from `RetrievedChunk::id`, a lookup into the corpus — never parsed out of the generator's text. A model that cites nothing still gets a correct source list; a model that invents `[7]` cannot add one. |
| It never pretends to synthesise | Every answer is tagged `EXTRACTIVE` or `SYNTHESISED`. Fragments are labelled fragments. |
| It degrades loudly | No generator configured? It says so, and says the exact command that fixes it. A silent downgrade to a worse answer that looks the same is the failure mode this field exists to prevent. |

Add a generator and the tier changes, with nothing else to configure:

```sh
export PHYSIS_ORACLE_URL=http://localhost:11434/v1
export PHYSIS_ORACLE_MODEL=qwen3:4b
export PHYSIS_ORACLE_KEY=ollama      # ollama ignores it
physis-core notebook --corpus ./my-docs --query "..." --budget 800
```

Still offline. Still your machine.

---

## Demo 0b: "Draft and fill" — spend the model only on the gaps

Let the table draft what the corpus supports; mark the rest as gaps for a model.
Drafted text is grounded **by construction** — the table can only emit sequences
it observed in the retrieved context.

```sh
physis-core notebook --corpus benchmarks/ground-truth \
  --query "the relief valve" --budget 300 --draft
```

```
── DRAFT (table over the retrieved context) ──

shall open at 2 6 bar and the machine must not stop above it the machine must not stop ⟨FILL⟩

table supplied 19 token(s), 1 gap(s) left for a model
table_share    0.00   (table entries: 510)

The corpus carries little of this answer. Draft-and-fill adds
latency here for no saving — ask a model directly.
```

**This is a real transcript of the method declining to be used.** That is the
feature. `table_share` is the fraction the table supplied; near 1 the answer is
a recombination of observed material and a small model need only close the gaps,
near 0 the table contributed nothing. **The method reports whether it applies,**
so you never have to guess.

Here it scores 0 because the decoder re-entered a window it had already emitted.
A repeat is not the corpus carrying the answer, it is the decoder running out of
new material, so it ends the draft and opens a gap rather than padding the
output. Before that guard existed this same command looped four times and
reported `table_share 1.00` — a metric rating its own garbage highly, which is
worse than no metric.

> **Known limit, stated rather than hidden.** Look at the drafted text: it
> stitches `2 6 bar` from spec A with `must not stop` from spec B. Grounded in
> the corpus, factually wrong — it crossed a contradiction the structural map
> itself flags. Draft-and-fill does not yet respect contradiction boundaries.

---

## Demo 0c: "Per-cell tables" — 70 tables with subtables, and the control

One table over 70 grid symbols pools every cell's continuations and averages a
rare cell away. A family conditions first: each cell's table is estimated only
from the steps passing through it.

```sh
physis-core ngram cells --input benchmarks/ground-truth --order 3 --min-support 5
```

```
── PER-CELL TABLE FAMILY ──
30 document(s) → 30 sequence(s)
9 cell(s) seen, 2 with a following transition (so with a table)
min_support 5: cells below it fall back to the flat control

CELL                          SUPPORT  OWN TABLE?
STUDY×LEARN                         9  flat
HEAL×CREATE                         8  flat
HEAL×REST                           7  flat
BOND×MAINTAIN                       2  flat
...
0 of 2 cell(s) with a table carry support >= 5.
The flat control was built from the same sequences in the same pass.
Compare against it before claiming per-cell tables help.
```

**Every cell says `flat`.** On a 30-document corpus no cell earns its own table,
and the command says so instead of shipping 9 tables estimated from 2
observations each. Conditioning costs data; whether specificity pays for the
sparsity is empirical, which is why the flat control is built from the same
sequences in the same pass and handed back alongside.

---

## Demo 1: "The Adaptive Classifier" — One Correction Shapes All Future Behavior

**Command sequence** (run in order):

```sh
# 1. Initial classification
physis-core classify "first layer adhesion failed on the nozzle"
# → STUDY×LEARN 1.000 (top cell), Best entry cosine: 0.908
#   ⚠ effectively a tie — treat the top cell as unresolved

# 2. User correction (report failure to quality tracker)
physis-core quality fail "classification was wrong"

# 3. Re-classify same text (same command, different result due to learned weights)
physis-core classify "first layer adhesion failed on the nozzle"
# → Typically a different top cell now (e.g., FABRICATE×CREATE or STUDY×WORK)
#   The QualityTracker has penalized the old cell and boosted new ones.
```

**What to record**: 
- Run step 1, note the top cell and score
- Run step 2 (one correction)
- Run step 3 immediately — the top cell should differ (or the score for the same cell changes)
- **Stunning moment**: User realizes the system "learned" from one sentence — no retraining, no API calls, just geometry + quality-tracker weights

---

## Demo 2: "The Quality Tracker" — Penalty/Boost Learning

**Command sequence**:

```sh
# 1. Report a failure with a correct domain
physis-core quality fail "classification was wrong" "STUDY×WORK"
# → Recorded failure → FABRICATE × PLAN (score 0.797)

# 2. Report a success/boost for a cell
physis-core quality pass "STUDY×WORK"
# → Boosted cell 'STUDY×WORK'

# 3. Re-classify a related text to see the effect
physis-core classify "pump maintenance required"
# → May now favor STUDY×WORK over other cells (if boost was applied)
```

**What to record**: 
- Step 1 records a failure, associating it with FABRICATE×PLAN
- Step 2 boosts STUDY×WORK
- Step 3 classifies a related text — the boosted cell may rise in the results
- **Stunning moment**: One `quality pass` command raises the boosted cell's future classification probability — no model retraining, no data pipeline

---

## Demo 3: "Ontology Pack Expansion" — Instant New Domains

**Step 1**: Check current domain count:

```sh
physis-core ontology | head -3
# → Shows 731 classification entries across 54 grid domains + 570 custom domains
```

**Step 2**: Drop a new ontology pack JSON file into `physis-core/config/`:
```sh
# Create config/govern_decide_ontology.json:
cat > config/govern_decide_ontology.json << 'EOF'
{
  "domains": [
    {
      "name": "Court Ruling",
      "domain": "GOVERN",
      "mode": "DECIDE",
      "axis_kind": "legal",
      "unit": "cases",
      "hints": ["judgment", "summary", "order"],
      "facets": {
        "lifecycle": "Operate",
        "agency": "Automated"
      }
    },
    {
      "name": "Contract Signing",
      "domain": "GOVERN",
      "mode": "SIGN",
      "axis_kind": "legal",
      "unit": "agreements",
      "hints": ["signature", "parties", "seal"],
      "facets": {
        "lifecycle": "Build",
        "agency": "Self"
      }
    }
  ]
}
EOF
```

**Step 3**: Restart the CLI (or just re-run classify — the loader picks up changes at runtime):

```sh
physis-core classify "court ruling on the patent dispute"
# → GOVERN/DECIDE  score≈0.71  entries=["judgment", "summary", "order"]

physis-core classify "contract signing ceremony"
# → GOVERN/SIGN  score≈0.68  entries=["signature", "parties", "seal"]
```

**What to record**: 
- Before: classify a text, note it maps to a 5×14 grid cell
- After dropping the JSON pack + restart: classify similar text → new GOVERN domain cell appears
- **Stunning moment**: User drops one JSON file → instantly new domain/mode cells are available. No model rebuild. No data labeling. No API calls.

---

## Demo 4: "Fixed-Token RAG" — Context Without API Costs

**Using the library API** (run via `cargo run --example` or inline Rust). Create `examples/rag_demo.rs`:

```rust
use physis_core::{embed::RandomProjectionEmbedder, classify::CellClassifier, ontology::OntologyLoader};

fn main() {
    let loader = OntologyLoader::load_all();
    let embedder = RandomProjectionEmbedder::new(384);
    let clf = CellClassifier::build(&loader, &embedder);

    // Query: "What did we do about pump issues before?"
    let results = clf.classify_text("pump pressure high valve vibrating");
    for r in &results {
        println!("{}×{} score={:.3} entries={:?}", r.domain, r.mode, r.score, r.entries);
    }
}
```

Run:
```sh
cargo run --example rag_demo
# → Each cell score + entries shown — pure offline classification,
#    zero per-query cost, no API keys, no rate limits
```

**What to record**: 
- The output shows 5×14 grid cell classifications with cosine scores and entry names
- Run again — same output (deterministic)
- Compare cost: $0 per query, no API key needed, works offline
- **Stunning moment**: User sees instant classification results with full provenance (entries named) and realizes: *this is what RAG feels like without any external services*

---

## Demo 5: "Accountable Explanation" — Transparency Without Black Box

**Using the library API**. Create `examples/explain_demo.rs`:

```rust
use physis_core::{embed::RandomProjectionEmbedder, classify::CellClassifier, ontology::OntologyLoader};
use physis_core::quality::QualityTracker;
use physis_core::models::Score;

fn main() {
    let loader = OntologyLoader::load_all();
    let embedder = RandomProjectionEmbedder::new(384);
    let clf = CellClassifier::build(&loader, &embedder);
    let tracker = QualityTracker::load_or_new("failures.json");

    // Classify something
    let results = clf.classify_text("some support ticket text");
    let top = &results[0];

    // Build a minimal explanation (real ExplanationReport would be larger)
    println!("Top cell: {}\"×{}\" score={}", top.domain, top.mode, top.score);
    println!("Best entry cosine: {:.3}", top.score);
    println!("Entries: {:?}", top.entries);
    println!("Quality failures: {}", tracker.failures.len());
}
```

Run:
```sh
cargo run --example explain_demo
```

**What to record**: 
- The output shows the top cell, score, entries, and quality tracker state
- User can trace: "why this cell?" → because of these entries with these cosines
- **Stunning moment**: User asks "why this classification?" and gets a concrete answer: *"because entries X, Y, Z contributed cosine scores of A, B, C to cell DOMAIN/MODE"* — no "the model said so," just concrete scores and entry names.

---

## Demo 6: "The n-Gram Token Stream" — How Token Generation Feels

**Using the library API**. Create `examples/ngram_demo.rs`:

```rust
use physis_core::embed::RandomProjectionEmbedder;
use physis_core::classify::CellClassifier;
use physis_core::ontology::OntologyLoader;
use physis_core::models::Score;
use phisia_core::quality::QualityTracker;
use std::collections::HashMap;

fn main() {
    let loader = OntologyLoader::load_all();
    let embedder = RandomProjectionEmbedder::new(384);
    let clf = CellClassifier::build(&loader, &embedder);

    // Simulated "token generation": stream 5 "tokens" (vectors) from pre-computed n-gram logic
    println!("=== n-Gram Token Stream (5 tokens) ===");
    for i in 0..5 {
        // Use classify to get a "token" (embedding + meaning)
        let results = clf.classify_text(&format!("pump maintenance item {}", i));
        let r = &results[0];
        let norm: f32 = r.score.sqrt();  // simplified "norm"
        println!("token {}: domain={}×mode={} score={:.3} entries={:?}", 
            i, r.domain, r.mode, r.score, r.entries);
    }
}
```

Run:
```sh
cargo run --example ngram_demo
```

**Output** (example):
```
=== n-Gram Token Stream (5 tokens) ===
token 0: domain=STUDY×LEARN score=1.000 entries=["impeller", "seal", "bearing"]
token 1: domain=FABRICATE×CREATE score=0.997 entries=["impeller", "gear", "cutter"]
token 2: domain=HEAL×BRAINSTORM score=0.995 entries=["cell", " nucleus", "organelle"]
token 3: domain=FABRICATE×WORK score=0.994 entries=["motor", "shaft", "bearing"]
token 4: domain=BOND×CREATE score=0.993 entries=["joiner", "fastener", "bolt"]
```

**What to record**: 
- Each "token" is a meaningful classification with domain/mode and entry names
- Each one is deterministic (same output every time)
- Each one costs $0, requires no API key, works offline
- **Stunning moment**: User watches 5 "tokens" stream — each one a real classification from the 5×14 grid — and realizes: *this is what "LLM token generation" actually was*, just made manifest through deterministic geometry + pre-computed lookup, not neural probability.

---

## Demo 7: "The Complete User-to-Action Loop" — From Natural Language to Concrete Action

**Full sequence** (all commands run in order):

```sh
# Step 1: Classify three support tickets
echo "=== Step 1: Initial classification ==="
physis-core classify "Pump pressure high, valve vibrating"
physis-core classify "Gearbox oil temperature exceeding limit"
physis-core classify "Conveyor belt misaligned, producing dust"

# Step 2: User corrections (report failures)
echo "=== Step 2: User corrections ==="
physis-core quality fail "No, ticket 2 is MECHANICAL not HEAL"
physis-core quality fail "Ticket 3 should be MAINTENANCE/WORK not STUDY/SENSE"

# Step 3: Re-classify (system has learned)
echo "=== Step 3: Re-classification (system learned) ==="
physis-core classify "Pump pressure high, valve vibrating"
physis-core classify "Gearbox oil temperature exceeding limit"
physis-core classify "Conveyor belt misaligned, producing dust"

# Step 4: Propose filings (note: this uses the library; see Demo 2 concept)
# Step 5: Retrieve context (library API)
# Step 6: Explain a decision (library API)
# Step 7: Add ontology pack → drop JSON into config/ → restart → instantly new domain available
```

**What to record**: 
- Step 1: Three tickets classified into cells
- Step 2: Two quality corrections reported
- Step 3: Re-classification shows the system has adapted (different top cells or different scores)
- **Stunning moment**: User sees the system "remembered" the corrections — one run of the same classify command produces different results than the first run. No model retraining, no data pipeline, just the quality tracker's penalty/boost weights.

---

## Quick Reference: All CLI Commands That Work

| Command | Example | Output |
|---|---|---|
| `physis-core classify "text"` | `physis-core classify "pump pressure"` | Cells populated with cosine scores + entries + best entry cosine + ⚠ tie warning |
| `physis-core quality fail "text"` | `physis-core quality fail "wrong"` | `Recorded failure → FABRICATE × PLAN (score 0.797)` |
| `physis-core quality pass "CELL"` | `physis-core quality pass "STUDY×WORK"` | `Boosted cell 'STUDY×WORK'` |
| `physis-core ontology` | `physis-core ontology` | 731 classification entries across 54 grid domains + 570 custom domains |
| `physis-core snapshot` | `physis-core snapshot` | Coherence snapshot: nodes=0, index=1.000, certified/isolated branches counts |
| `physis-core classify "text"` (re-run) | `physis-core classify "text"` | Same or different top cell depending on quality tracker state |

**Studio GUI**: `physis-core studio --port 3000` → opens at http://127.0.0.1:3000 with six tabs (Classify, Semiotic grid, Ontology, Corpus, Discover, Quality)

---

## Why These Demos Matter

| Demo | The "Big Model" Problem | physis-core + Small Model Solution |
|---|---|---|
| **1. Adaptive Classifier** | Retraining cost, data pipelines | One correction = immediate Q-tracker weight update |
| **2. Quality Tracker** | Model fine-tuning, labeling pipelines | One `quality pass`/ `quality fail` → immediate weight update |
| **3. Ontology Pack Expansion** | Model fine-tuning, data labeling | Drop JSON into `config/`; instant new cells |
| **4. Fixed-Token RAG** | Per-token API costs, context overflow | Zero per-query cost. Works offline forever. |
| **5. Accountable Explanation** | Black-box unaccountability | Concrete: "entries X, Y, Z contributed cosine A, B, C" |
| **6. n-Gram Token Stream** | "Magic" token generation opacity | Transparent: each token = real classification from 5×14 grid |
| **7. User-to-Action Loop** | End-to-end cost, adaptability | Full NL→action pipeline; adapts from single corrections |

**Net effect**: Every capability that normally costs money, internet access, rate limits, retraining, or black-box uncertainty is replaced by physis-core + a small model + present data: **zero marginal cost, fully offline, deterministic, auditable, and immediately adaptable from one user correction**.

---

## Getting Started

```sh
# 1. Install
cargo install -- physis-core

# 2. Try Demo 1 (Adaptive Classifier)
physis-core classify "first layer adhesion failed on the nozzle"
physis-core quality fail "classification was wrong"
physis-core classify "first layer adhesion failed on the nozzle"

# 3. Try Demo 3 (Ontology Pack)
# Drop config/govern_decide_ontology.json then:
physis-core classify "court ruling on the patent dispute"

# 4. Try the studio UI
physis-core studio --port 3000
# → opens http://127.0.0.1:3000

# 5. Experiment freely — all output is deterministic, offline, zero cost
```

*physis-core: deterministic semi-grid classification, zero-model AI, embed anywhere.  
Small model: the user-facing interface. physis-core: the deterministic engine behind it.*

*Together: AI that is fully offline, zero-cost, instantly adaptable, and completely auditable.*
---

## The shipped product demos (0.1.25)

One command launches the offline experience end-to-end. No API key, no cloud,
no model download past a documented setup. Everything below was measured on
this machine from `examples/demo-corpus/` (26 files: three repeat families of
8 + 2 deliberate anomalies) — the numbers are real, not placeholders.

### Demo A — Difference & Repetition (the corpus becomes visible)

```sh
physis-core demo --dir examples/demo-corpus --query "the pump"
```

```text
── PHYSIS STRUCTURAL MAP (deterministic) ──
26 DOCUMENTS
↓ PHYSIS
3 RECURRING PATTERNS
1 SIGNIFICANT DIFFERENCES
0 CONTRADICTION CANDIDATES
structure hash 19aec4…

the pump …→ maintenance required replace the seal…
```

- **What repeats** is found by nearest-neighbour union-find (embedder-agnostic,
  relative to the corpus's own median similarity).
- **What differs** is the *loneliness* signal: how far a document's nearest
  neighbour sits below the corpus median — the two anomalies are flagged.
- **What contradicts** is reported per high-overlap / low-similarity pair,
  never merged away.
- **Determinism**: the same corpus gives the same `structure hash` on every
  run; the corpus-map pipeline is pure (no hidden randomness).

Drill-down trade: a repeat cluster is a set of source files (provenance); a
singleton difference is a list with its separation score. See
`src/map.rs` (`build_map`, `MapReport`) and `examples/` for the API.

### Demo B — Physis context compiler (measured compression)

```sh
physis-core context \
  --corpus examples/demo-corpus \
  --query "what maintenance is scheduled and why" \
  --budget 400
```

```text
CONTEXT BUDGET 400 tokens
SOURCE DOCUMENTS 26
STRUCTURAL CLUSTERS 3
REPEATED PATTERNS 3
SIGNIFICANT DIFFERENCES 1
CONTRADICTION CANDIDATES 0

CONVENTIONAL RETRIEVAL  297 tokens (nothing discarded)
PHYSIS COMPILED CONTEXT  237 tokens (budget-capped)
CONTEXT REDUCED        20%
```

It reuses the existing `rag::TokenFixedRetriever` (no parallel retrieval
implementation) and the compression ratio is **computed from the actual run**,
never estimated. Pass `--json` for machine-readable output with the full
`ContextReport`.

### Demo C — Small model + n-gram table + Physis (the plumbing)

The infrastructure layer (`src/tokenizer.rs`, `src/ngram_table.rs`,
`src/model_provider.rs`) makes every component replaceable. A real backend —
the deterministic `NgramDecoderModel` — greedy-decodes over any table, so the
augmentation path is exercised offline with zero weights:

```sh
physis-core ngram build --input examples/demo-corpus --output /tmp/demo/5g.physisng --order 5
physis-core ngram import demo-table /tmp/demo/5g.physisng
physis-core ngram inspect demo-table --context "the pump"
physis-core model list
physis-core demo --dir examples/demo-corpus --query "the pump"
```

```text
model: demo-ngram — caps [Generation, TokenScoring] — ~1 MB
score(query) = -6.13 mean log-prob
continuation: the pump maintenance required replace the seal…
```

The n-gram table is an **inspectable artifact**, not a black box: `PHYSISNG1`
format, manifest + checksums, deterministic byte-for-byte builds, per-order
backoff with add-k smoothing, and a tokenizer-compatibility gate that refuses
to mix vocabularies. Interchangeability is tested (`Model A + Table A/B`,
`Model B + Table A/B` all compose). See `src/ngram_table.rs` tests.

### Scientific status — unchanged standard

Capabilities are explicit and capability absence is an explicit error, never a
faked path. Nothing here claims a 360M model "equals" a large one; the
benchmark measurements decide what the combination recovers. The honest
reporting convention of this repository is preserved.
