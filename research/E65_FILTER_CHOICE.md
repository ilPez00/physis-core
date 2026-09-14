# E65 — choosing the filter from the corpus: the obvious statistic fails, the right one is ambiguity after weighting

Run 2026-09-14. `src/worldstate.rs` (`redundancy`, `lexical_margin`,
`filter_from_margin`), `examples/experiment65_redundancy.rs`, artifact
`benchmarks/results/worldstate-e65.json`.

E63 left the two filters swapping places between regimes. If redundancy decides,
a shell can measure it and choose — so this run tries to.

## First statistic: raw term overlap. It fails.

`redundancy` = mean over documents of the highest term-Jaccard to any other
document.

| corpus | redundancy | arm that won |
|---|---|---|
| documentation | 0.172 | bm25 + position (0.535 vs 0.209) |
| hand-written | 0.471 | entity link (0.357 vs 0.286) |
| **machine log** | **0.786** | **bm25 + position (0.447 vs 0.296)** |
| generated templates | 0.802 | entity link (0.700 vs 0.013) |

The log scores 0.786 — as redundant as the templates at 0.802 — and the lexical
filter wins on it anyway. The reason is visible once stated: log records are file
paths sharing long common prefixes (`/home/gio/dev/physis-pro/...`), so raw
overlap is enormous, while **BM25's IDF discards exactly those shared tokens**.
Overlap measures what the filter already ignores.

## Second statistic: how decisively the best lexical match wins

`lexical_margin` = mean of `(score1 − score2) / score1` over the best two other
documents, scoring each document as a query against the corpus. A corpus of
near-twins produces ties; diverse text produces a clear winner.

| corpus | margin | bm25+pos | entity link | predicted | correct? |
|---|---|---|---|---|---|
| generated templates | **0.044** | 0.013 | **0.700** | entity link | ✓ |
| documentation | 0.235 | **0.535** | 0.209 | bm25 | ✓ |
| machine log | 0.520 | **0.447** | 0.296 | bm25 | ✓ |
| hand-written (n=28) | 0.265 | 0.286 | 0.357 | bm25 | ✗ |

**Right on every corpus where the arms actually differ**, and wrong on the one
where they are statistically tied: the hand corpus's two arms are 0.071 apart
with intervals [0.143, 0.464] and [0.179, 0.536], which is no separation at all.
The three decisive cases have gaps of 0.687, 0.326 and 0.151 and are all called
correctly.

The boundary is set at **0.15**, in the wide empty gap between 0.044 and 0.235,
rather than fitted. **Four corpora is four points.** A threshold tuned more
finely than "the gap in the middle" would be false precision, and is refused.

## Wired into the shell

`physis-world inspect` now reports the statistic and the recommendation, and
`why` marks which arm the measurement favours for the world currently loaded. On
this machine's 3232-observation log the margin is **0.157** — barely above the
boundary, which the shell shows rather than hides. A world that sits on the line
is exactly the case where printing both arms, as the shell already does, is the
honest interface.

## Not licensed

1. Four corpora, one of them synthetic, one of them too small to separate its
   arms. The statistic is a reading, not a law.
2. The boundary is a gap, not a fit. A fifth corpus landing between 0.044 and
   0.235 would immediately test it, and none exists yet.
3. `lexical_margin` samples 150 documents by stride on large corpora; the effect
   of that sampling on the estimate is unmeasured.
