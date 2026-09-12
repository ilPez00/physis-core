# SHIP PLAN — physis-core 0.1.26 "put doors in the cathedral"

Status is measured, not assumed. A box is [x] only after `cargo test`/`cargo build`
proved it **and** the live binary demonstrated it.

## Directive 1 — Ship Engineer (product), then Directive 2 — Model/N-Gram Infra
Both operate on the same base; the ship plan below interleaves them by phase.

### Phase 1 — Reconnaissance  [x]
- Read README, PLANNING, DEMO, research status; 172-test baseline green.
- Current branch `research/perspective-invention` (1 ahead of origin).

### Phase 2 — Difference & Repetition map  [x]
- `src/map.rs`: union-find repeats (relative-to-corpus-median), loneliness
  differences, contradiction pairs (never merged), reproducibility hash.
- Tested: same corpus ⇒ identical hash; mandelbrot outlier is most different.

### Phase 3 — Context compiler  [x]
- `compile_context` (reuses `rag::TokenFixedRetriever`) + `physis-core context`.
- Measured on demo corpus: 284 → 197 tokens (30.6% real). `--json` output.

### Phase 4 — Small model (infra)  [x plumbing]
- `ModelProvider` trait, capabilities, `NgramDecoderModel` (deterministic
  greedy decoder), model registry (checksum-verified local + http backend).
- `Tokenzier` trait + compatibility gate. `BackoffTable` (orders 1–5,
  add-k, stupid-backoff), `PHYSISNG1` format, table registry.
- Interchangeability tested (Model A + Table A/B, Model B + Table A/B).

### Phase 5 — Benchmark  [x]
- Deterministic synthetic corpus with KNOWN ground truth (3 repeat families,
  2 anomalies, 2 contradiction pairs, 2 regime changes).
- `physis-core benchmark` runs task families and writes machine-readable
  artifacts to `benchmarks/results/`.
- Measured on `benchmarks/ground-truth/` (30 docs, known truth):
  repeat 100%, anomaly 100%, contradiction 50%, map deterministic true,
  context compression 25% (220→166 tokens).
- Honesty: big-model oracle leg is **not configured** on this machine — the
  harness reports it as such rather than inventing a number.

### Phase 6 — Studio exposure  [ ] (lightweight: map endpoint + selector)

### Phase 7 — Documentation  [x] (README product-first + DEMO.md)
- README rewritten product-first (hero numbers + three demos).
- DEMO.md already updated; new-module doc links added.

### Phase 8 — Hardening  [x]
- `cargo test --lib` 185 pass · `cargo clippy --all-features -D warnings` clean
· `cargo build --release --features cli,http` ok.

### Phase 9 — Ship  [x] (0.1.26 commit)
- Bump 0.1.25 → 0.1.26, update CHANGELOG, commit on the current branch.

## Scientific honesty gate (applies to every phase)
Capabilities without a measurement are labeled `NOT TESTED`, never implied.
`BIG MODEL LEG: NOT CONFIGURED — no claim made` is the default until a
reference model is wired and its numbers land in `benchmarks/results/`.
