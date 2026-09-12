# The neurosemantic layer

> Scope: this file covers the *retrieval* operation only — one of six (see
> `../docs/WHAT_PHYSIS_IS.md` §3). For the five harnesses that feed the shared
> coherence graph, see `../wiring/neurosemantic_wiring.md`.


The layer that sits between a corpus and a language model: it decides *what the
model reads*. Everything here is measured or labelled `NOT MEASURED`. Numbers
without a run behind them do not belong in this file — see the scientific
honesty gate in `SHIP_PLAN.md`.

## 1. What it is

A language model's answer is a function of its weights **and** its context. The
neurosemantic layer is physis-core's claim on the second argument: given a
corpus and a query, compile the context that a model of *any* size should read,
under a hard token budget, deterministically.

Three components, all in this crate:

| Component | Module | What it does |
|---|---|---|
| Token-fixed retrieval | `src/rag.rs` | Ranks chunks by cosine to the query, greedily packs until the next chunk would exceed `token_budget`. The context is guaranteed to fit the window regardless of corpus size. |
| Structural map | `src/map.rs` | Union-find repeat families, loneliness differences, contradiction pairs (never merged). Deterministic: same corpus ⇒ same `structure_hash`. |
| Context compiler | `map::compile_context` | Both of the above in one call. Returns baseline tokens (nothing discarded) and physis tokens (budget-capped), so the compression ratio has an honest denominator. |

## 2. The measured surface

Corpus `benchmarks/ground-truth` (30 docs, known truth), physis-core 0.1.26,
seed 7, run twice — once under a lexical hash, once under a real ONNX model:

| Metric | random-projection | MiniLM (ONNX) |
|---|---|---|
| Repeat recovery | 1.00 (3/3) | 1.00 (3/3) |
| Anomaly recall | 1.00 (2/2) | 1.00 (2/2) |
| Contradiction recall | 0.50 (1/2) | 0.50 (1/2) |
| Map deterministic | true | true |
| Baseline tokens | 220 | 221 |
| Physis tokens | 165 | 168 |
| Context reduction | 25.0% | 24.0% |
| Structure hash | `0890098a…` | `c79cee20…` |
| **Big-model oracle leg** | `not_configured` | `not_configured` |

`cargo test --lib` = 194 pass. `cargo clippy --lib --bins --all-features -D warnings`
clean.

## 3. The finding that matters: the benchmark cannot fail

Read the table again. **Every quality metric is identical** under a
random-projection lexical hash and under a real semantic model. Only the token
counts move (by 1–3) and the structure hash changes.

That is not a result about physis. It is a result about the benchmark: the
ground-truth corpus's repeat families are lexically near-identical, so a
bag-of-grams hash recovers them exactly as well as MiniLM does. A benchmark
whose score does not move when retrieval quality moves is not measuring
retrieval quality.

This is the same failure shape as the old dog/puppy semantic probe, one level
up — a guard that cannot fail. Until the corpus contains items that are
*semantically* related while *lexically* disjoint, the 100%/100%/24% row is a
statement about string overlap, not about meaning.

**Consequence for the thesis.** The claim "a small model reading physis-compiled
context matches a big model reading everything" is currently **untestable in
this repo**: the oracle leg is `not_configured`, and the benchmark that would
carry the context-quality half of the claim does not discriminate. Both are
open work, not results.

## 3b. What was actually fixed here

- **The CLI never used a real embedder.** `load_embedder()` returned
  `RandomProjectionEmbedder::new(384)` unconditionally, and `cmd_context`,
  `cmd_demo`, `cmd_benchmark` and `cmd_run` each built their own
  `RandomProjectionEmbedder::new(64)`. `OnnxEmbedder` was exported from
  `lib.rs` and called by nothing. Every number in `benchmarks/results/`
  predating this change was produced by a lexical hash — which is why the
  comparison above was possible to run at all.
  All four sites now call `embed::select`, which gates each ONNX candidate on
  `semantic_self_test` and labels the result.
- **`provenance.json` now records `embedder`.** A benchmark that does not say
  which embedder produced it is not reproducible.
- **Padding tax.** The cascade requests `max_length` 128, not 512: measured on
  MiniLM, 18.3 ms/embed at 128 versus 108.1 ms at 512, a 5.9× cost paid
  entirely in padding on short documents. Override with `PHYSIS_MODEL_MAX_LEN`.
- **Redundant embedding.** `compile_context` runs `retrieve_from_texts` twice
  and `build_map` once over the same corpus — ~800 embed calls for ~30 distinct
  texts. `MemoEmbedder` memoizes by exact text for the process lifetime.
  Combined effect on `context` over the 30-doc corpus: **86.2 s → 14.3 s**.

## 3c. Still not measured

- Answer quality against a reference model. `src/oracle.rs` is built —
  OpenAI-compatible transport, four fixed probe cases, token-Jaccard agreement,
  evidence-citation check — but reports `not_configured` without
  `PHYSIS_ORACLE_KEY` (or `OPENROUTER_API_KEY` / `GROQ_API_KEY`).
- Contradiction recall above 0.5. One of two known pairs is missed. Known,
  unfixed, and unchanged by the embedder.
- Compression on a corpus larger than 30 documents.

## 4. Running it

```bash
# structural map + compiled context, fixed budget
cargo run --features cli --bin physis-core -- context <dir> "<query>" --json

# the full benchmark; writes benchmarks/results/{run,metrics,provenance}.json
cargo run --features cli --bin physis-core -- benchmark

# config-driven pipeline: model and n-gram table swap without a source edit
cargo run --features cli --bin physis-core -- run --config demo.json \
  [--model <id>] [--ngram <id>] [--query "<q>"]
```

### Configuring the reference (oracle) model

```bash
export PHYSIS_ORACLE_URL=https://openrouter.ai/api/v1   # default
export PHYSIS_ORACLE_MODEL=openai/gpt-4o                # default
export PHYSIS_ORACLE_KEY=<key>                          # nothing runs without this
cargo run --features cli,http --bin physis-core -- benchmark
```

A local OpenAI-compatible endpoint works the same way, and is the honest way to
run the leg offline:

```bash
export PHYSIS_ORACLE_URL=http://localhost:11434/v1      # ollama
export PHYSIS_ORACLE_MODEL=qwen3:4b
export PHYSIS_ORACLE_KEY=ollama                         # ollama ignores it
```

Note that a 4B local model is a *peer*, not an oracle. Labelling its output a
big-model reference would be the exact dishonesty the gate exists to prevent.

## 5. Embedder

`embed::select` resolves a cascade: explicit override (`PHYSIS_EMBEDDER=random-projection`), then ONNX model
candidates (each gated by `semantic_self_test`, chosen via `PHYSIS_MODEL_DIR` or
`./models/bge-base-en-v1.5` then `./models`), then random projection as the
last resort. The self-test uses three lexically **disjoint** probe pairs
(`car`/`automobile`, `dog`/`puppy`, spindle-overheat/thermal-fault), each
required to beat its distractor by `SEMANTIC_PROBE_MARGIN` = 0.05.

Random projection **fails** this test by design, and is pinned to fail by
`random_projection_fails_the_semantic_probe`. A lexical hasher passes the old
single dog/puppy probe — `old_probe_is_not_the_semantic_criterion` exists to
prove that pair is no longer the criterion. There is no
`config/embedder.toml`; the cascade is the configuration.

ONNX weights are opt-in: `--features embed-onnx`, with `model.onnx` +
`tokenizer.json` in the model directory at runtime.

## 6. Licensing

physis-core is Apache-2.0 and has no licence gate. The licence gate belongs to
physis_pro (proprietary); `PHYSIS_ALLOW_DEV_LICENCE` is a physis_pro variable
and has no effect in this crate.
