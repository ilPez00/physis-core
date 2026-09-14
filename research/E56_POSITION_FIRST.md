# E56 — position in text as the temporal coordinate: POSITIVE on two legs of three, and the third is the interesting one

Run 2026-09-14, immediately after E55's negative. Code:
`physis-core/src/worldstate.rs` (E56 section), example
`examples/experiment56_position_first.rs`. Corpus: the same frozen
`benchmarks/worldstate/corpus.jsonl`, sha256 `264c40a07303…`, 72 sentences.
Embedder: `bge-base-en-v1.5` via ONNX. Seed 20260914.
Artifacts: `benchmarks/results/worldstate-e56.json`, `…-gap3.json`, `…-gap5.json`.

**The temporal coordinate is the sentence's ordinal position, not a clock.**
Nothing in this experiment reads a timestamp; the corpus has none.

## L1 — transition derivation beats the shuffled-order null

Set operations over the state sequence (first-mention, repeated fact, changed
predicate over the same subject/object pair, re-mention after a gap of 6) derive
a transition label per position. It never reads the gold label. The null re-runs
**the whole derivation** on a permuted reading order and scores it against gold
in the original order — so if order carries nothing, the two agree.

```text
accuracy          0.486
majority baseline 0.333
shuffled null     0.291 ± 0.040     z = +4.88   (500 permutations)
```

Per-label recall: `emerge` 0.700 (n=20) · `persist` 0.444 (n=9) ·
`change` 0.375 (n=24) · `recur` 0.214 (n=14) · `disappear` 1.000 (n=5).

**Read the `disappear` row as a word list, not as a result.** Its recall is 1.000
because the derivation contains six declared lexical cues ("no longer", "is
removed", "is drained", "stops", "stands idle", "was removed") and the corpus
uses them. That row measures the cue list. The other five labels are set
operations, and they are where the +4.88 comes from.

`recur` at 0.214 is the weakest real row: the gap threshold (6 positions) is a
constant someone picked, and it was not swept. Do not report `recur` as measured
until it is.

## L2 — semantic neighbours are textual neighbours, weakly

Mean |Δpos| over mutual-kNN edges (k=5): **16.82** against a random-pair null of
**24.19** — ratio **0.695**. Embedding neighbourhood carries some positional
information. This is a descriptive number with no arm attached; it is reported
because it explains L3.

## L3 — "what preceded this?": order and similarity are complementary, and neither works alone

Gold is mechanical: the nearest earlier sentence sharing an entity. Four arms,
top-1 with a 2000-sample bootstrap interval.

| queries | arm | top1 | top3 | 95% CI |
|---|---|---|---|---|
| all 61 | cosine (order-blind) | 0.098 | 0.328 | [0.033, 0.180] |
| | cosine + position | 0.311 | 0.541 | [0.197, 0.426] |
| | **position only** | **0.475** | 0.574 | [0.361, 0.590] |
| | shuffled-position null | 0.115 | 0.213 | [0.049, 0.197] |

At first reading the position-only arm wins outright and the embedder is
surplus. **That reading is wrong, and the corpus construction is why:** the gold
is *the nearest* earlier mention, so an arm that always answers `p−1, p−2, p−3`
is handed a large share of it for free. The check is to drop every query whose
gold antecedent is nearer than a gap:

| gap | queries | cosine (blind) | cosine + position | position only | shuffled null |
|---|---|---|---|---|---|
| ≥3 | 28 | 0.071 | **0.286** | **0.000** | 0.107 |
| ≥5 | 23 | 0.087 | **0.304** | **0.000** | 0.130 |

**Position-only collapses to exactly zero** — by construction it cannot reach
past three — and order-blind cosine stays near its null. The combined arm is the
only one that answers the question at all, at 0.286–0.304 top-1 against a
shuffled-position null of 0.107–0.130.

**What this licenses:** on this corpus, "what preceded this?" is answered by
*similarity constrained by order*, and neither leg alone answers it. That is a
statement about complementarity, and it is the same shape as item A's finding
that retrieval and ranking are different problems and must not be mixed into one
score.

**What it does not license:** the intervals overlap ([0.143, 0.464] vs
[0.000, 0.250] at gap 3), n = 28, and the corpus is one topic. This is
directional evidence, not a settled result. It needs the larger corpus before it
is quoted as a number.

## Verdict on H2

> Representing a corpus as a sequence of world states ordered by **position in
> text** improves retrieval and state reconstruction over order-blind embedding
> similarity.

- **State reconstruction (L1): supported.** 0.486 vs a shuffled-order null of
  0.291 ± 0.040, z = +4.88, against a majority floor of 0.333.
- **Retrieval (L3): supported in direction, not in magnitude.** The combined arm
  beats both single-leg arms and the shuffled null once the recency artifact is
  removed, but the intervals overlap at n = 28.
- **L2** is descriptive and consistent with both.

H2 survives; it does not yet carry a headline number.

## Immediate consequences

1. The five transition predicates are worth building into `WorldState` as the
   `T` field, because the derivation measurably beats its null.
2. The gold for "what preceded" must be rebuilt without recency bias before E57
   — either by requiring a gap or by making distractors that share the entity.
   The gap filter here is a patch, not the fix.
3. Sweep `RECUR_GAP`. It is a constant that decides a whole label.
4. Corpus: 72 sentences is too few for the L3 interval to close. The 1000-sentence
   corpus in the plan is now on the critical path, not a scale-sanity extra.

## Relationship to E55

E55 killed the cross-representation correspondence arm (structural matching at
its null). E56 is the leg that does **not** depend on cross-space agreement, and
it is the leg that moved. Read together: on this corpus, the *ordering* of a
representation carries recoverable structure, while the *geometry alone* does not
transfer across representations.
