# E57 — "what does this supersede?": the world-model arm answers it, every single leg does not

Run 2026-09-14. Code `physis-core/examples/experiment57_supersession_retrieval.rs`,
corpus `benchmarks/worldstate/corpus-large.jsonl` (**GENERATED**, seed 20260914,
sha256 `2eaff7bb33d7…`, 1320 sentences, 150 queries), embedder
`bge-base-en-v1.5` via ONNX. Artifact `benchmarks/results/worldstate-e57.json`.

## The task

Each cycle plants a fact ("Lathe-0 holds coolant pressure within tolerance"),
then several sentences mentioning the same machine, then a **query that
supersedes the planted fact in different words** ("Pressure on the lathe-0 line
has fallen out of specification"). Gold is the planted sentence. The distractors
sit *between* gold and query, so recency is wrong by construction, and two per
cycle are near-paraphrases of the gold about a different attribute
(temperature, flow, the neighbouring machine).

## Result

| arm | n | top1 | top3 | top1 95% CI |
|---|---|---|---|---|
| cosine (order-blind) | 150 | 0.000 | 0.000 | [0.000, 0.000] |
| cosine + position | 150 | 0.013 | 0.060 | [0.000, 0.033] |
| recency only | 150 | 0.000 | 0.207 | [0.000, 0.000] |
| **entity + position + cosine** | 150 | **0.700** | **0.993** | **[0.627, 0.773]** |
| shuffled-position null | 150 | 0.000 | 0.007 | [0.000, 0.007] |

By distance to the gold: 0.710 (gap 3) · 0.727 (5) · 0.600 (7) · 0.688 (9) ·
0.792 (11). **Flat** — the arm does not decay with distance over this range.

Order-blind cosine is at **exactly zero** because the corpus contains 150
near-identical planted facts, one per cycle, and global similarity retrieves the
wrong machine's copy every time. That is the entity-disambiguation failure mode
in its purest form.

## What it licenses

On this corpus, "what does this supersede?" is answered only when **structure
(the entity link), order (earlier positions only) and geometry (similarity)**
are used together. Each leg alone scores at or near zero, and so does the
shuffled-position null. The gap between 0.700 and 0.013 is not a tuning
difference; it is the difference between having a world model and having a
retriever.

This is the same shape as E56 L3 and as item A's cascade result: **the legs are
complementary and must not be collapsed into one score.**

## What it does not license

1. **The corpus is generated.** Its structure is a property of
   `make_corpus_large.py`. The number closes an interval on a retrieval arm; it
   is not evidence about real text.
2. **The arm is handed gold entity links.** A deployed system must extract
   them, and extraction error would cut this number by an unmeasured amount.
   That is the next experiment, and until it runs, 0.700 is an upper bound on
   the pipeline rather than a measurement of it.
3. Top-3 at 0.993 says the *candidate set* is nearly always right — consistent
   with everything else in this repository about `propose` beating `judge`.

## Two discarded runs, kept because they are the method

The first two versions of this benchmark scored **1.000 at every distance**, and
both were thrown away rather than reported:

- **Run 1** — distractors were template sentences with no lexical overlap with
  the gold, so "same entity + most similar" won by a mile.
- **Run 2** — hard distractors were added, but the *query still reused the
  gold's exact wording* ("no longer holds …"), so the task remained near-
  duplicate matching.

Only after the query was paraphrased away from the gold did the arm drop to
0.700 and the benchmark become capable of failing. **A score of 1.000 is a bug
report about the benchmark, not a result** — the same rule that
`benchmarks/ground-truth` failed and `benchmarks/retrieval` was built to fix.
