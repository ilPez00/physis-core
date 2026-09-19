# Product boundary

> Research discovers which mechanisms work.
> Core implements the minimal reproducible mechanisms.
> Physis Product decides when, where and how to apply them automatically.
> Physis Pro operates them at organisational scale.

## Example 1 — redundancy and retrieval

- **Research:** does entity partitioning beat BM25 under redundant text?
- **Core:** `Bm25Index`, entity partition, position filter, candidate ranking.
- **Product:** detect corpus redundancy automatically and choose the pipeline.

## Example 2 — cheap representations

- **Research:** can corpus-trained counting representations approximate
  transformer ranking inside filtered candidate sets?
- **Core:** count tables, embedder trait, `rank_candidates`.
- **Product:** continuously benchmark cost/quality and switch representation.

## Example 3 — order and staleness

- **Research:** does order help identify superseded claims?
- **Core:** ordered observations, temporal validity, `supersedes`.
- **Product:** maintain a living organisational timeline, surface stale facts.

## Example 4 — contradiction

- **Research:** which contradiction mechanisms survive adversarial pairs?
  (Bound surface rules reached precision 1.000/recall 0.840 on the holdout and
  0 false positives on adversarial input; a small NLI model reached 1.000/0.960
  but fired wrongly on adversarial pairs — so Core ships the rules and the
  routed residual is a product decision.)
- **Core:** contradiction types, evidence polarity, rule harness + fixtures.
- **Product:** choose and operate the pipeline per customer corpus.

## Work packages

**RESEARCH (grant-safe):** benchmark datasets, retrieval and
structural-inference experiments, corpus-property measurement, deterministic
representations, provenance and replay mechanisms, reference implementations,
reproducibility infrastructure. Output is published.

**PRODUCT:** continuously applying those mechanisms to a customer's changing
systems and data — ingestion, connectors, profiling, strategy selection,
memory, workspace, deployment, scale.
