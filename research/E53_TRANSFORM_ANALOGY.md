# E53 — Symbolic analogy vs a construction-matched null

**Status: SUPPORTED, replicated on two corpora.** First measured positive for
`transform.rs`, which `docs/WHAT_PHYSIS_IS.md` §5 had listed under "never put in
front of a null".

Run: `cargo run --release --example experiment53_transform_analogy [corpus-dir]`

## Why this experiment

`transform.rs` is a symbolic homomorphism engine — deterministic, **no
embeddings**, patterns matched by exact token equality with variable binding.
Because it touches no embedding it inherits none of the settled negatives of the
E1–E18 track, which is precisely why it needed its own control rather than
borrowing that track's conclusions in either direction.

## Task

Link prediction over a real dependency graph. `crate::X` appearing in module `M`
yields `(M, Requires, X)` — real data, verifiable by grep, not authored for the
experiment. One edge is held out at a time; the method ranks candidates for the
missing target. Only subjects retaining ≥ 2 other edges are scored: a subject
with no remaining context tests the prior, not the analogy.

## Method — the A→B path, made concrete

For held-out subject `M`: find modules `M′` sharing the most *other* targets
with `M` (that shared set **is** the homomorphism — a binding `M′ → M` justified
by edges they already share); each `M′` votes for its own targets that `M` lacks,
weighted by `overlap / √|known|`; rank by summed vote. No embeddings, no model,
no training.

## Arms

| arm | what it isolates |
|---|---|
| ANALOGY | the claim |
| POPULARITY | global in-degree — the majority baseline that beats naive methods embarrassingly often |
| PERMUTED | **the construction-matched control**: identical ANALOGY code over a degree-preserving edge shuffle. Both degree sequences survive, so only the *pairing* is destroyed. If ANALOGY merely exploits popularity, PERMUTED matches it. |
| RANDOM | the floor |

## Results (top-3 containment)

### physis-core/src — 36 modules, 121 edges, 34 targets, n=86

| arm | top-1 | top-3 | top-5 |
|---|---|---|---|
| **ANALOGY** | **0.267** | **0.500** | **0.605** |
| POPULARITY | 0.209 | 0.337 | 0.430 |
| RANDOM | 0.081 | 0.151 | 0.244 |
| ANALOGY on permuted (n=67) | 0.149 | 0.209 | 0.343 |

ANALOGY − permuted = **+0.291** · ANALOGY − popularity = **+0.163**

### physis_pro/src — 106 modules, 390 edges, 79 targets, n=294

| arm | top-1 | top-3 | top-5 |
|---|---|---|---|
| **ANALOGY** | **0.133** | **0.446** | **0.503** |
| POPULARITY | 0.095 | 0.303 | 0.439 |
| RANDOM | 0.017 | 0.061 | 0.105 |
| ANALOGY on permuted (n=238) | 0.105 | 0.256 | 0.353 |

ANALOGY − permuted = **+0.189** · ANALOGY − popularity = **+0.143**

## The internal check that makes this credible

On the permuted graph, ANALOGY **loses to POPULARITY** — 0.209 vs 0.343 on core,
0.256 vs 0.307 on pro. That is exactly the expected behaviour when structure has
been destroyed but degree preserved: with no real pairing left to read, a
structure-reading method should fall *below* a degree-reading one. It does, on
both corpora. The arms are behaving as designed rather than all drifting
together, which is what an artifact usually looks like.

## Honest limits

- **The effect shrinks with graph size**: +0.291 → +0.189 from 121 to 390 edges.
  Two points is not a trend, but the direction is the pessimistic one and should
  be assumed to continue until measured otherwise.
- **`n` differs between real and permuted arms** (86/67, 294/238), because
  permutation changes which subjects retain ≥ 2 edges. The arms are therefore
  scored over overlapping but non-identical item sets. The deltas are far larger
  than that asymmetry plausibly explains, but it is not a paired test.
- **One relation type.** Only `Requires`. Nothing here says analogy works for
  `Causes`, `Precedes`, or any other predicate.
- **Both corpora are Rust source written by the same author.** Replication
  across two trees is weaker evidence than replication across two domains.
- **This measures ranking, not generation.** It supports "the A→B path carries
  real structure". It does *not* yet show that draft-and-fill built on that path
  produces output passing a gate — that is the next experiment.

## What it licenses

`transform.rs` may now be described as **measured against a control on link
prediction, positive, replicated**. It may not yet be described as a generator.
