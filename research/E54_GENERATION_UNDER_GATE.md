# E54 — Does the A→B path *generate*? The compiler decides.

**Status: PARTIAL at top-1, SUPPORTED in the retry regime.** Structure is real — it beats its
construction-matched null by +0.150 — but at one shot it does **not** beat
simply ranking by popularity (+0.025, n=40). Not a shippable positive.

Run: `python3 research/e54_generate_under_gate.py --limit N`

## Why this experiment

E53 showed `transform`'s analogy carries real structure for **ranking**. Ranking
is not generation: a method can order candidates well and still never produce a
usable artifact. E53's own report said so and refused to call `transform` a
generator. This is the harder question, asked with a gate nobody can argue with.

## Task

A module `M` contains `use crate::T::{items};` that it genuinely depends on.
Delete it and `cargo check --lib` **fails** — verified per case, so every scored
item is known load-bearing (0 of 40 were skipped as redundant). The method is
then given the items and must supply the missing module `T`. It emits the line
and the compiler judges it: a wrong `T` means the items do not exist there and
the build stays red.

The gate is **binary and external**. No partial credit, no metric of our own
choosing, no way to grade generously.

## Results — n=40, physis-core/src, 482 s

| arm | gate passes | rate |
|---|---|---|
| **ANALOGY** | 16/40 | **0.400** |
| POPULARITY | 15/40 | 0.375 |
| PERMUTED (degree-preserving null) | 10/40 | 0.250 |

ANALOGY − PERMUTED = **+0.150** · ANALOGY − POPULARITY = **+0.025**

`src/` verified clean after the run.

## Reading it honestly

**What survived.** The +0.150 over the permuted control reproduces E53's core
claim under a far harsher test: the structure in the dependency graph is real,
it is not an artifact of degree, and a method reading it emits compilable code
more often than the same method reading a shuffled graph.

**What did not.** +0.025 over popularity is nothing at n=40. At one shot,
analogy is not measurably better than "guess the most-imported module".

**And this was predictable from E53.** E53's headline was top-3 (+0.163 / +0.143
over popularity), but its *top-1* column was already thin: 0.267 vs 0.209
(+0.058) on core, 0.133 vs 0.095 (+0.038) on pro. E54 scores top-1. +0.025 is
what E53's own top-1 numbers implied. **The advantage lives in top-3, not
top-1** — which is a statement about where the method is usable, not a rescue.

## What this licenses

`transform` may be described as **measured against a control on generation,
structure confirmed, advantage over the majority baseline NOT established at
one shot.** It may not be described as a generator that beats the trivial
approach.

## The retry regime — SUPPORTED

One shot is not how an agent works: it emits, compiles, and retries. Scoring the
same task with up to 3 attempts, n=15:

| arm | gate passes within 3 | rate | mean attempts when it passed |
|---|---|---|---|
| **ANALOGY** | 12/15 | **0.800** | 1.33 |
| POPULARITY | 10/15 | 0.667 | 1.10 |
| PERMUTED null | 7/15 | 0.467 | 1.29 |

ANALOGY − PERMUTED = **+0.333** · ANALOGY − POPULARITY = **+0.133**

**This was a pre-registered prediction, and it held.** The top-1 section above
states plainly that E53's thin top-1 column implied the advantage lives in
top-3, *before* this arm was run. It does.

So the honest two-line summary of E54: at one shot, analogy is not distinguishable
from guessing the most-imported module (+0.025). Allowed three compiler-checked
attempts, it passes 80% against a majority baseline's 67% and a shuffled graph's
47%, at a mean cost of 1.33 attempts.

**Caveats that keep this from being a headline.** n=15 for the retry arm against
n=40 for one-shot, so it is the weaker measurement of the two. Same corpus, same
single predicate. And on most individual cases ANALOGY and POPULARITY emit the
*same* candidate — the gap comes from a minority of cases where analogy's
ordering differs, which is consistent with the small one-shot delta.

## Limits

- **n=40, one corpus, one predicate.** E53 replicated across two trees; this has
  not.
- **Only the module is generated**, not the item list. A harder and more
  realistic task would generate both.
- **`cargo check`, not `cargo test`.** It proves the code compiles and the items
  resolve. It does not prove the import was the *right* one semantically — but
  since the items must exist in the named module, a wrong-but-compiling answer
  is rare rather than impossible.
