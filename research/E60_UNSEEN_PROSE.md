# E60 — the chunker on text it was not authored beside: it does not transfer, and a wrong link layer is worse than none

Run 2026-09-14. Corpus `benchmarks/worldstate/corpus-docs.jsonl` (sha256
`f19c1cb4d264…`, 168 sentences drawn from `docs/WHAT_PHYSIS_IS.md`,
`RESCOPE.md`, `research/E53_TRANSFORM_ANALOGY.md`,
`docs/plans/2026-09-13-research-agenda.md`). Builder
`benchmarks/worldstate/make_corpus_docs.py`. Same two scores as E59, via
`PHYSIS_CORPUS`. Artifacts `benchmarks/results/worldstate-e59-docs.json` and
`…-e59.json`.

**Where the gold comes from.** Backticked spans in the repository's own
Markdown. `act::bearing_on`, `TokenFixedRetriever`, `RECUR_GAP` — each is a
thing the prose is about, marked as a thing by an author documenting code, with
no knowledge of this extractor. The chunker never sees the backticks: they are
stripped from `text` and kept only in `entities`. Known bias, stated before the
numbers: those spans are identifier-shaped, which favours identifier rules and
disfavours the noun-phrase chunker.

## Extraction

| extractor | docs P | docs R | docs F1 | hand P | hand R | hand F1 |
|---|---|---|---|---|---|---|
| id+cap | 0.387 | 0.088 | 0.144 | 0.972 | 0.343 | 0.507 |
| df | 0.286 | 0.119 | 0.168 | 0.420 | 0.804 | 0.551 |
| noun phrase | 0.224 | 0.123 | 0.159 | 0.488 | 0.699 | **0.575** |
| **code ids** (new, E60) | **0.832** | 0.379 | **0.520** | 0.000 | 0.000 | 0.000 |
| np + code | 0.480 | 0.462 | 0.471 | 0.488 | 0.699 | 0.575 |

**The chunker does not transfer.** F1 0.575 on the corpus it was authored beside,
**0.159** on prose it was not. That is the number E59 said it could not have
until this ran.

The `code ids` rule was added *after* seeing that failure, so it is scored on
both corpora deliberately: it is the best extractor on docs (P 0.832) and emits
**exactly nothing** on the hand corpus, leaving `np + code` identical to `np`
there. A rule added after a failure has to be checked where it could do damage;
this one is additive-safe.

## Downstream — antecedent at gap ≥ 3

| links | docs top1 (43 q) | docs top3 | hand top1 (28 q) | hand top3 |
|---|---|---|---|---|
| gold links | **0.698** | 0.977 | 0.393 | 0.714 |
| **cosine (blind, no links)** | **0.349** | 0.581 | 0.071 | 0.250 |
| df only | 0.302 | 0.465 | 0.321 | 0.500 |
| code ids | 0.209 | 0.233 | 0.000 | 0.000 |
| np + code | 0.209 | 0.279 | 0.357 | 0.393 |
| noun phrase | 0.093 | 0.093 | 0.357 | 0.393 |

**On unseen prose every extracted-link arm is worse than using no links at
all.** Blind cosine 0.349 beats the best extracted arm 0.302 and the
best-extraction arm 0.209. On the hand corpus the ordering is the opposite:
blind cosine 0.071, extracted links 0.321–0.357.

## The finding, and it corrects the direction of the last two

Gold links score **0.698** here — the highest world-model number in the whole
series, on real prose. So the ceiling is not the problem: **the extractor is.**
And the failure mode is the opposite of E58's.

- **E58 (generated, highly redundant):** links with too *little precision*
  connect everything, the filter stops discriminating, arm 0.700 → 0.013.
- **E60 (docs, low redundancy):** links with too *little recall* drop the true
  antecedent out of the candidate set entirely, so the filter cannot rank what
  it never admitted. `code ids` has precision 0.832 and still scores 0.209,
  because recall is 0.379 — a filter that never sees the answer cannot return it.

So the honest statement, third revision:

> An entity filter helps only when it is both specific enough not to link
> everything and complete enough not to hide the answer. Which half kills it
> depends on the corpus: redundancy punishes low precision, sparsity punishes
> low recall. And when neither holds, **no filter beats a bad filter** —
> plain similarity is the safer default.

## What this licenses

Nothing about the chunker. It licenses one thing about the architecture: with
correct links the arm reaches 0.698 on prose written for humans, so the
world-model layer is worth the link investment — but the link layer must be
measured on the target text, and a link layer transplanted from another corpus
should be assumed harmful until measured.

## What it does not license

1. 43 queries, intervals [0.558, 0.837] for the gold arm and [0.209, 0.488] for
   blind cosine. These overlap for every extracted arm.
2. The gold is backticks, which is an annotation of *code entities*, not of every
   thing the prose is about. Sentences about "the grid" or "the null" have no
   gold and are excluded — the corpus is biased toward technical nouns.
3. `min_df`, `max_len` and the code rule's shape are unswept.
