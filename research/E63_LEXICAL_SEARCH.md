# E63 — is the entity abstraction necessary at all? On real text, no. Under near-duplicate load, yes.

Run 2026-09-14, testing a direct challenge to the whole link layer: *maybe it is
a problem of scale, and linking with a plain search function, word per word, is
good enough.* Three new arms, all lexical, all free:

* **bm25 + position** — `rag::Bm25Index` over earlier positions only. A search
  engine, word per word, with order as the only structure.
* **shared word + cos** — link to any earlier position sharing a term, rank by
  embedding similarity.
* **rare word + cos** — the same, restricted to terms with document frequency
  ≤ 3 (the IDF intuition, made an arm rather than an argument).

Scored on all three regimes with the arms already in the series. Artifacts
`worldstate-e59-docs-lex.json`, `worldstate-e59-log-lex.json`,
`worldstate-e57.json`.

## Documentation (168 sentences, 43 queries, gold = author's backticks)

| arm | top1 | top3 | 95% CI | cost |
|---|---|---|---|---|
| gold links | 0.698 | 0.977 | [0.558, 0.837] | — |
| **bm25 + position** | **0.535** | **0.744** | [0.395, 0.674] | microseconds |
| rare word + cos | 0.488 | 0.558 | [0.349, 0.628] | microseconds |
| rules + interpreter | 0.488 | 0.698 | [0.349, 0.628] | 168 model calls |
| shared word + cos | 0.465 | 0.721 | [0.326, 0.605] | microseconds |
| interpreter alone | 0.465 | 0.558 | [0.326, 0.605] | 168 model calls |
| cosine (blind) | 0.349 | 0.581 | [0.209, 0.488] | — |
| best rule arm | 0.209 | 0.279 | [0.093, 0.326] | — |

**BM25 with a position restriction beats every extraction arm, including the
interpreter, at zero marginal cost.**

## Machine log (388 observations, 159 queries, gold = commit scope / argv0)

| arm | top1 | top3 | 95% CI |
|---|---|---|---|
| gold links | 0.528 | 0.730 | [0.447, 0.604] |
| rules + interpreter | 0.465 | 0.629 | [0.390, 0.541] |
| interpreter alone | 0.459 | 0.623 | [0.384, 0.535] |
| **bm25 + position** | **0.447** | 0.610 | [0.371, 0.522] |
| shared word + cos | 0.434 | 0.572 | [0.358, 0.509] |
| df only | 0.403 | 0.579 | [0.327, 0.478] |
| cosine (blind) | 0.340 | 0.472 | [0.270, 0.415] |
| rare word + cos | 0.314 | 0.321 | [0.245, 0.390] |

BM25 is **statistically indistinguishable from the interpreter** here
([0.371, 0.522] vs [0.390, 0.541]) and costs nothing.

## Generated supersession corpus (1320 sentences, 150 queries) — the hypothesis breaks

| arm | top1 | top3 |
|---|---|---|
| **entity + position + cosine** | **0.700** | 0.993 |
| cosine + position | 0.013 | 0.060 |
| **bm25 + position** | **0.013** | 0.053 |
| shared word + cos | 0.013 | 0.060 |
| cosine (blind) | 0.000 | 0.000 |

**BM25 scores exactly what plain cosine scores: 0.013 against the entity arm's
0.700.** Fifty-fold.

## The answer, and the mechanism

The hypothesis is **right on real text and wrong under near-duplicate load**, and
the reason is visible in the corpora rather than in the method.

* Documentation and the machine log are *lexically diverse*: what distinguishes
  the right antecedent from the wrong one is largely which words occur. A search
  function reads exactly that signal, and reads it better than a 7B model naming
  entities — because the model's names are lossy and the words are not.
* The generated corpus is 150 near-identical cycles. Every sentence shares almost
  all of its vocabulary with 149 others; the *only* discriminating token is the
  machine identifier, and the near-paraphrase distractors share more words with
  the query than the true gold does. BM25 therefore ranks a distractor first,
  exactly as cosine does. The entity link wins because it removes 149 cycles from
  consideration before ranking begins, and the embedder then reads *pressure vs
  temperature* inside the one cycle that remains.

So the entity abstraction is not doing "semantic understanding". It is doing
**partitioning under redundancy** — and where the corpus is not redundant, a
search function does the same job for free.

## What this changes

1. **The interpreter is not justified by these measurements.** Two regimes, and
   in neither does it beat a BM25 arm that costs microseconds. E61 and E62 stand
   as measurements; their practical conclusion is superseded here.
2. **The default should be lexical.** `bm25 + position` is the arm to ship, with
   entity links added where they exist rather than required.
3. **Redundancy is the deciding property**, and it is now measurable in advance:
   mean pairwise term overlap of a corpus predicts which arm to use. That is the
   next experiment and it is cheap.

## What it does not license

1. The generated corpus is synthetic and its redundancy is extreme by
   construction. Real corpora with that property exist — log lines, templated
   alerts, ticket text — but none has been measured here.
2. 43 and 159 queries, overlapping intervals throughout. The *ordering* is
   consistent across two regimes; the magnitudes are not settled.
3. Nothing here tests scale in the sense of corpus size. "Scale" in the question
   turned out to mean redundancy, which is a different axis, and that is the
   finding.
