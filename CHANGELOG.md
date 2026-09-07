# Changelog

Notable changes to `physis-core`. This file starts at 0.1.15; earlier
releases predate it and are documented only by their git tags and commit
history.

## 0.1.17

### Added

- **`physis_core::coverage`** — scores whether a candidate cell would let the
  ontology place records it currently cannot. `discovery` already proposed new
  entries but had no way to say which proposals were worth keeping; this is
  that missing half, and deliberately the narrow half.

  `uncovered` lists the records the live ontology cannot place at or above a
  threshold. `candidate_gain` and `rank_candidates` score candidates by how
  many of those they would rescue, strongest first, ties broken on index so
  the ranking is reproducible.

  **Why the check is shaped as an operational question.** A long research
  track tried nine mechanisms for certifying discovered structure and all nine
  failed. Every one asked a representational question — is this split correct,
  does a lexicon recognise it, do many pairs agree on it — and answered it with
  a statistic computed over the same embedding space that produced the
  candidate. This instead asks whether adding the candidate changes what the
  ontology can place, against a corpus external to whatever produced it, so it
  cannot be satisfied by the geometry agreeing with itself.

  Measured over 5-fold cross-validation on 691 held-out records: re-adding an
  entry the ontology already contains scores **exactly zero on 2000 of 2000
  trials**, while genuine interpolations score above zero (Welch t = +12.11).
  In that run it cut 2000 candidates down to 141 worth reading.

  **Coverage is not correctness.** A candidate that swallows records into a
  wrong cell scores exactly like one capturing a real gap — both make the
  records classifiable. Use this to shrink the pile and then have a person read
  what survives; it is a filter, not an approver.

## 0.1.16

### Added

- **`physis_core::linkage`** — a new module answering "which `(domain, mode)`
  cells does real data bridge, and via which items". Each text contributes one
  bridge, between its top-scoring and second-scoring cell. That is the whole
  rule: no threshold, no `k`, no seed, and ties broken on the cell key rather
  than on iteration order.

  It deliberately does **not** discover cells. A long research track tried
  seven structurally distinct ways to derive stable groupings from embedding
  geometry — margin/silhouette gating, three kNN-consistency variants,
  cross-embedder corroboration, density peaks, cross-embedder split agreement,
  capacity-constrained training loss, and plain k-means — and all seven failed
  on real data. Re-derived clusters do not survive corpus growth (58% anchor
  overlap after +25% data) and nothing survives an embedder swap (ARI ~0.10).
  So this module takes the ontology's hand-authored cells as the fixed points,
  because they cannot drift, and measures only the links between them.

  No claim is made about any individual item being "genuinely cross-cutting";
  an earlier thresholded version degenerated to a 98% flag rate, which merely
  restates that domains overlap. The signal is aggregate — a cell pair that
  recurs as many different items' runner-up is meaningfully linked.

  `LinkageGraph::build` / `links` / `strongest` / `cross_domain`, with
  `CellLink::is_cross_domain` for the links a single-label classification
  cannot represent at all.

## 0.1.15

### Fixed

- **`OntologyLoader::classification_domains()` returned entries in a
  nondeterministic order.** It chained three `HashMap::values()` iterators,
  and Rust randomizes `HashMap` iteration order per process by design. Every
  process therefore saw the same ontology entries in a different order, which
  changed anything order-sensitive downstream — centroid accumulation,
  train/test splits, and tie-breaking between near-equal similarity scores.
  The result was classification and discovery output that varied run to run
  from identical inputs and identical model weights.

  Entries are now sorted by `(name, domain, mode)` before being returned.
  Consumers on ≤ 0.1.14 who observed unstable output should upgrade before
  investigating other causes; this is very likely the cause.

  Note on how this was found, since the wrong answer is a tempting one: we
  attributed it twice to floating-point nondeterminism in ONNX Runtime's
  multi-threaded reductions. That was wrong. A direct probe
  (`examples/probe_embedding_determinism.rs`, added in this release) shows
  embeddings are bit-identical across processes *and* across thread counts.

### Added

- **`OnnxConfig::intra_threads: Option<usize>`** — opt-in ONNX Runtime
  intra-op thread control. Default `None` preserves existing behavior (all
  cores). Intended for throughput/latency tuning and for constrained
  environments. Explicitly **not** a determinism control: output was measured
  bit-identical at every thread count.
- **`examples/probe_embedding_determinism.rs`** — verifies embedder
  determinism in-process and, by printing exact dims plus a checksum, across
  processes.
- **README: "Determinism & Reproducibility"** — states the guarantees, the
  0.1.15 ordering fix, and two things that are explicitly *not* guaranteed:
  cross-embedder stability (pin your embedder; it is part of your ontology's
  identity) and stability of re-derived clusters under corpus growth (persist
  anchor identity instead of re-deriving).

### Packaging

- `wordnet-db` / `wordnet-types` moved from `[dependencies]` to
  `[dev-dependencies]`. Nothing in `src/` uses them — they are used only by
  two research examples — so they were being forced on every downstream
  consumer for no benefit. **No API change**; consumers simply stop pulling
  them.
- Stray files excluded from the published `.crate`: `**/*.bak` (a
  `Cargo.lock.bak` and an ontology backup were being shipped),
  `sw_add_domains.py`, `STRIPE_LICENSE_GUIDE.md`.
