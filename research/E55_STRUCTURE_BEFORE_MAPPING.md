# E55 — structure before mapping: NEGATIVE, and the null was pre-registered

Run 2026-09-14. Code: `physis-core/src/worldstate.rs`,
`physis-core/examples/experiment55_structure_before_mapping.rs`.
Corpus: `benchmarks/worldstate/corpus.jsonl`, sha256 `264c40a07303…`, 72
sentences, frozen by `make_corpus.py` before any arm was run.
Artifacts: `benchmarks/results/worldstate-e55-{all,train,heldout}.json` and
`…-{split}-sweep.json`.

## The question

Two representations of the same 72 sentences. Derive a mutual-kNN graph in each
**independently**, then recover which node of A is which node of B **from
structure alone**. Identity is the gold answer and is never shown to a matcher.

A = `bge-base-en-v1.5` via ONNX (`onnx-explicit`, `PHYSIS_MODEL_DIR`), B =
`RandomProjectionEmbedder(384)`, the word-unigram+bigram hash space. Genuinely
different constructions: one semantic, one lexical.

## Result — the hypothesis arm is at chance

Pre-registered configuration k=5, profile=12, 8 anchors, seed 20260914.

| split | arm | top1 | top3 | knn-pres | top1 95% CI |
|---|---|---|---|---|---|
| all (n=72) | random (null) | 0.028 | 0.042 | 0.054 | [0.000, 0.069] |
| | anchor (weak supervision, 8 pairs) | **0.236** | 0.333 | 0.185 | [0.139, 0.333] |
| | supervised (gold-label centroids) | 0.083 | 0.167 | 0.054 | [0.028, 0.153] |
| | **structural (the hypothesis)** | **0.014** | 0.028 | 0.115 | [0.000, 0.042] |
| train (n=49) | random | 0.020 | 0.061 | 0.081 | [0.000, 0.061] |
| | anchor | 0.306 | 0.429 | 0.198 | [0.184, 0.449] |
| | supervised | 0.082 | 0.265 | 0.035 | [0.020, 0.163] |
| | **structural** | **0.020** | 0.082 | 0.058 | [0.000, 0.061] |
| heldout (n=23) | random | 0.130 | 0.217 | 0.150 | [0.000, 0.261] |
| | anchor | 0.609 | 0.696 | 0.675 | [0.391, 0.783] |
| | supervised | 0.174 | 0.348 | 0.375 | [0.043, 0.348] |
| | **structural** | **0.043** | 0.087 | 0.250 | [0.000, 0.130] |

ARI between the two spaces' components with no mapping at all: **0.287** (all),
0.278 (train), 0.000 (heldout).

**The structural arm sits on its null in every split**, and in two of three it
is numerically *below* it. The pre-registered kill condition — "`structural ≈
cosine`/at null ⇒ stop and record" — fires.

## It is not a hyperparameter

Sensitivity sweep, same corpus, same seed, six configurations
(`worldstate-e55-*-sweep.json`):

| k | profile | ARI(no map) | anchor top1 | structural top1 |
|---|---|---|---|---|
| 3 | 6 | 0.203 | 0.236 | 0.056 |
| 3 | 24 | 0.203 | 0.236 | 0.028 |
| 5 | 6 | 0.287 | 0.236 | 0.014 |
| 5 | 24 | 0.287 | 0.236 | 0.014 |
| 10 | 6 | 0.477 | 0.236 | 0.042 |
| 10 | 24 | 0.477 | 0.236 | 0.056 |

Structural stays in 0.014–0.056 against a null of 0.028 across every setting.
Nothing was tuned to make the headline row look worse or better; the headline is
the configuration fixed before the first run.

## What the run does establish

1. **Cross-space information exists and is recoverable — with supervision.**
   Eight known anchor pairs take top-1 from 0.028 to 0.236 (0.609 on held-out).
   So A and B are not unrelated spaces; the failure is specific to recovering
   the correspondence *without* anchors.
2. **Sorted-neighbourhood descriptors are the thing that fails.** The same
   descriptor recovers identity perfectly when A and B are the same space
   (`identical_spaces_recover_identity`, top1 > 0.99), so the implementation is
   not broken — it is uninformative across genuinely different geometries at
   this corpus size.
3. **Partition agreement is *higher* here than the settled cross-embedder
   figure** (ARI 0.287 vs ≈ 0.10), which is a property of a 72-sentence corpus
   with one topic, not a refutation of the earlier result. It should not be
   quoted as an improvement.
4. **`supervised` is weak.** Gold transition-class centroids barely beat the
   null (0.083 vs 0.028). The transition label is too coarse to identify a
   sentence; that is a fact about the label, not about supervision.

## What it does not establish

- It does not show that structure-before-mapping is impossible. It shows that
  **this** descriptor, on **this** corpus size, recovers nothing. Spectral or
  iterative graph-matching (IsoRank-style propagation from a seed) was not
  tried; the pre-registered plan called for one descriptor and one run.
- It says nothing about H2 (position in text), H4 (navigation) or H5
  (downstream). Those arms do not depend on cross-space correspondence.
- n = 72 is small. The interval on the heldout split (23 items) is wide enough
  that only the *direction* of the anchor effect is safe to read.

## Consequence, per the pre-registered plan

The kill condition fires, so:

1. This file is the record, and `src/worldstate.rs`'s header carries the
   negative.
2. **H1 is not pursued further with this descriptor.** The next legitimate move
   on H1 is a seeded propagation matcher (anchor arm shows seeds work), and it
   must be run as a *new* pre-registration, not as a retune of this one.
3. **The program pivots to H2** — world states ordered by **position in text** —
   which was already the next item and does not depend on cross-space agreement.
   Its null (shuffle the order) is exact and cheap.

## Corpus shortfall, recorded rather than hidden

The benchmark spec calls for ≥15 adversarial pairs of each kind. The frozen
corpus has **4 `adv_lex` and 4 `adv_struct`** sentences (2 pairs each kind), plus
5 `agree_dup` negatives. That is sufficient for E55, which uses none of them,
and **insufficient for E57 (navigation)**, which is built on exactly those
cases. E57 must not run until the corpus is extended and re-frozen with a new
sha256.
