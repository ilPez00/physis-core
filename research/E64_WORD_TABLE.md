# E64 — a lookup table of words instead of a model: on the arms that matter, indistinguishable, and 2400× cheaper to build

Run 2026-09-14, testing the question directly: *what if we had a lookup table of
words?* Code in `src/worldstate.rs` (`word_table_count`, `sentence_from_table`,
`vocabulary`), selected by `PHYSIS_E64_REP`. Artifacts
`worldstate-e59-{log,docs}-{model,table-model,table-count}.json`.

Three representations, same corpora, same arms:

| name | how a sentence gets a vector | model calls |
|---|---|---|
| `model` | the embedder runs on the whole sentence (E55–E63's default) | one per sentence, forever |
| `table-model` | the embedder runs **once per vocabulary word**; a sentence is the mean of its words' vectors | one per word, once |
| `table-count` | **random indexing with PPMI weighting** over a co-occurrence window, built from the corpus by counting | **none, ever** |

`table-count` uses no neural inference at any point. A word's vector is the
PPMI-weighted sum of deterministic ±1 signatures of the words it co-occurs with;
the signatures come from the word's own hash, so nothing needs storing and the
table regenerates anywhere.

## Machine log (388 observations, 159 queries)

| arm | model | table-model | **table-count** |
|---|---|---|---|
| gold links | 0.528 | 0.516 | **0.497** |
| gold links top3 | 0.730 | 0.755 | **0.742** |
| shared word + cos | 0.434 | 0.415 | 0.403 |
| df only | 0.403 | 0.384 | 0.371 |
| cosine (blind) | 0.340 | 0.321 | 0.333 |
| bm25 + position | 0.447 | 0.447 | 0.447 |
| **build time** | **12.15 s** | 47.06 s | **6.0 ms** |

## Documentation (168 sentences, 43 queries)

| arm | model | table-model | **table-count** |
|---|---|---|---|
| gold links | 0.698 | 0.674 | **0.674** |
| gold links top3 | 0.977 | 0.953 | **0.953** |
| shared word + cos | 0.465 | 0.349 | 0.372 |
| rare word + cos | 0.488 | 0.372 | 0.395 |
| cosine (blind) | 0.349 | 0.279 | 0.279 |
| bm25 + position | 0.535 | 0.535 | 0.535 |
| **build time** | **7.79 s** | 36.44 s | **3.3 ms** |

## What the numbers say

1. **On the filtered arms — the ones the shell uses — counting matches the
   transformer.** Gold-link top-1 differs by 0.031 (log) and 0.024 (docs), with
   intervals overlapping almost completely, and top-3 differs by 0.012 and 0.024.
   The table builds in **3–6 ms against 8–12 s**: roughly 2400×, and the gap
   widens with corpus size because the model is linear in sentences while the
   table is linear in tokens with a tiny constant.
2. **On unfiltered similarity the model is genuinely better** — blind cosine
   0.349 vs 0.279 on docs. Ranking everything against everything is where a
   sentence encoder earns its cost; ranking *inside a candidate set someone else
   filtered* is not.
3. **`table-model` is the worst of both.** It costs more to build than the model
   itself (every vocabulary word is a separate forward pass, 47 s vs 12 s) and
   scores no better than counting. Averaging a transformer's word embeddings
   discards what the transformer is for.
4. **BM25 is identical across all three**, by construction — it never touches a
   vector. That row is the control proving the harness changed only what it
   claimed to change.

## Found by shipping it, and it is the honest limit

Switching `physis-world`'s default to `table-count` broke free-text queries:
*"which experiment measured the null for the grid"* returned `strings.xml` and
two i18n YAML files. A query is short, its words are common, and it is scored
against everything — exactly the unfiltered case row 2 says the table is weakest
on.

The fix is not to restore the model: it is to stop using a vector space for the
query path at all. `retrieve` now ranks by **BM25**, which E63 measured as the
best arm on real text, and the same query returns the E20 sieve-floor commit, the
research-agenda commit and the E2 mapper commits. `PHYSIS_WORLD_QUERY=model`
remains, and that path is the one place in the shell where running a model is
defensible.

## What this means for the architecture

The navigational layer now runs with **no neural inference anywhere**: counts for
vectors, BM25 for filtering and for queries, position for order, and the entity
link only where redundancy makes partitioning necessary (E63). The whole shell
loads a 3232-observation world and answers in about two seconds, on a laptop,
with no model on disk required.

That is the strongest form of the project's own thesis to date — not that a model
is unnecessary in general, but that **navigation of one's own machine does not
need one**, and the cost of finding that out was a 512-dimensional counting
table.

## Not licensed

1. Two corpora, 43 and 159 queries, overlapping intervals. The ordering is
   consistent; the magnitudes are not settled.
2. The table is built **on the corpus being navigated** — it is not a general
   word table, and a fresh corpus with no repetition would give it nothing to
   count. Whether a table built on one corpus transfers to another is untested.
3. `window = 4`, `dim = 512`, PPMI floor 0 — none swept.
4. The generated near-duplicate corpus was not re-run under the table
   representations; E63's partitioning result is about links, not vectors, but
   the interaction is unmeasured.
