# E66 — is the domain axis structural, or just written down? Both axes carve; the counts are convention

Run 2026-09-14, from the question put directly: *why 70 and not n? The 14 modes
make sense — tool use, method, workflow, the operative half. Domains dictate
content and should be adaptable.*

`examples/experiment66_axis_fitness.rs`, artifact
`benchmarks/results/worldstate-e66-axes.json`. Same 731 ontology entries, same
`bge-base-en-v1.5` embeddings, same machinery as D4: leave-one-out self-recovery
against a **size-preserving label permutation**, relabelled three ways.

| axis | classes | self-recovery | null | ratio | normalised (f−n)/(1−n) | classes above null | failing |
|---|---|---|---|---|---|---|---|
| DOMAIN/MODE | 70 | 0.438 | 0.038 | 11.5× | 0.416 | 49 | 8 |
| DOMAIN only | 5 | 0.672 | 0.209 | 3.2× | **0.585** | 5 | 0 |
| MODE only | 14 | 0.536 | 0.121 | 4.4× | 0.472 | 14 | 0 |

Read the **normalised** column, not the ratio: a null scales with 1/k, so a
5-class axis has a high null by arithmetic and a 70-cell grid a low one. Once
that is divided out, the ordering is domain 0.585 > mode 0.472 > full 0.416.

## The answer, in two parts

**1. Neither axis is arbitrary.** Every class on both axes clears its own
permutation null — 5 of 5 and 14 of 14, no failures. The intuition that domains
are a convention laid over content while modes are the real structure is **not
supported**: by this measure the domain axis is if anything the *more*
recoverable of the two.

**2. Nothing here supports five, or fourteen.** The experiment scores the axes as
given; it cannot score the axes that were never written down. And the external
anchors point the other way from the internal coherence:

* against WordNet's noun supersenses, **35 of 70 cells are `disputed`** and the
  four largest cells sit *on* their null (D1);
* against WordNet's fifteen verb supersenses, **6 of 14 modes fail to carve**,
  `CREATE` owns nothing while `verb.creation` sits unowned, and `verb.social` —
  **12.2% of the corpus's verb mass** — is owned by no mode at all (D2a).

So: internal coherence says the axes are real; external anchors say the *counts*
are convention. Those are compatible, and together they are the argument for
making the axis size a parameter rather than a constant.

## What changed in the code

`src/semiotic_grid.rs`: `DEFAULT_DOMAINS` is the pack, `domains()` is the axis in
force (`PHYSIS_DOMAINS` overrides it, comma-separated), `cell_count()` replaces
the assumption of 70, and `is_canonical` / `fallback_anchors` follow the active
axis. `DOMAINS` stays as an alias so no caller breaks. Modes are deliberately
**not** overridable: they are the operative axis, and D2a's EXPAND / MERGE /
RETIRE list is a set of proposals with a measurement behind them, not an approved
change — Track D's rule is that the machine proposes and a person adjudicates.

`docs/WHAT_PHYSIS_IS.md` §1 and §2 are rewritten accordingly, including rule 5,
which used to say "say 70 tables" and now says 70 is the default pack.

A first attempt at the test set `PHYSIS_DOMAINS` directly and raced the two other
tests in the file — cargo runs tests in parallel threads and the environment is
process-wide, so one test's override became another's reality. The axis is now
split into a pure `domains_from(Option<&str>)` and a thin env reader, and the
tests exercise the pure half.

## Not licensed

1. Self-recovery over ontology *entries* is internal coherence over the grid's
   own seed text. It is not classification accuracy over documents, and D4 says
   so in its own header.
2. The experiment cannot evaluate an axis that does not exist. "Would seven
   domains carve better than five?" needs seven domains authored first — which is
   exactly what D2's proposal pipeline is for.
3. One embedder. Cross-embedder ARI ≈ 0.10 is a settled negative in this
   repository, so an axis result resting on one embedding space is a result about
   that space.
