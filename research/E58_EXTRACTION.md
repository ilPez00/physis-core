# E58 — what extraction costs: nothing, or everything, depending on one rule

Run 2026-09-14. Code `src/worldstate.rs` (extraction section),
`examples/experiment58_extraction.rs`. Artifact
`benchmarks/results/worldstate-e58.json`. Corpora: hand-authored
`corpus.jsonl` (72) and generated `corpus-large.jsonl` (1320, seed 20260914).
Embedder `bge-base-en-v1.5`.

E57's winning arm read the corpus's **gold entity annotation**. A deployed
system has text. E58 replaces the annotation with links extracted by three
declared, model-free rules and measures the drop.

## Extraction quality

| corpus | rules | P | R | F1 |
|---|---|---|---|---|
| hand (72) | all three | 0.426 | 0.809 | 0.558 |
| generated (1320) | all three | 0.162 | 0.966 | 0.277 |
| hand | **id** (identifier-shaped) | **0.969** | 0.304 | 0.463 |
| hand | cap (capitalised mid-sentence) | 1.000 | 0.039 | 0.075 |
| hand | df (repeated content token) | 0.420 | 0.804 | 0.551 |
| hand | id+cap | 0.972 | 0.343 | 0.507 |
| generated | **id** | **0.928** | **0.966** | **0.947** |
| generated | cap | 0.000 | 0.000 | 0.000 |
| generated | df | 0.163 | 0.966 | 0.278 |
| generated | id+cap | 0.928 | 0.966 | 0.947 |

## End-to-end: E57's arm on extracted links

150 queries, gold = the superseded sentence.

| link source | top1 | top3 | 95% CI | vs gold links |
|---|---|---|---|---|
| gold links | 0.700 | 0.993 | [0.627, 0.773] | — |
| extracted (all three rules) | 0.013 | 0.060 | [0.000, 0.033] | **−0.687** |
| **extracted (id only)** | **0.700** | **0.993** | [0.627, 0.773] | **+0.000** |
| extracted (id+cap) | 0.700 | 0.993 | [0.627, 0.773] | +0.000 |

Per distance, id-only: 0.710 / 0.727 / 0.600 / 0.688 / 0.792 at gaps 3–11 — the
same profile as the gold-link run, position by position.

## The finding

**Adding a high-recall rule destroyed the task it was feeding.** The `df` rule
lifts entity recall to 0.966 and drops precision to 0.163, because it admits
`pressure`, `tolerance`, `line` — tokens every cycle shares. An entity filter
built on those links everything to everything, so the arm collapses to plain
cosine+position, which is exactly where it lands: 0.013, the same number that
arm scored in E57.

So the downstream task does not want a good extractor in the F1 sense. **It
wants a specific one.** Recall of generic tokens is worse than useless here: it
is actively destructive, and F1 hides that completely — 0.277 vs 0.947 looks
like a moderate difference and is the difference between 0.013 and 0.700.

This is the same lesson as item B's false-alarm accounting and as C2's F1
caveat, arriving from a third direction: **the aggregate metric is not the
operational one.** Pick the metric the downstream step actually consumes.

## What it licenses

On the generated corpus, the pipeline is **end-to-end complete with no gold
annotation and no model in the extraction step**: identifier-shaped tokens are
enough, and E57's 0.700 survives intact. The "0.700 is an upper bound" caveat
written into E57 is therefore lifted *for this corpus*.

## What it does not license

1. **It does not transfer to the hand corpus.** There the id rule has precision
   0.969 and recall **0.304** — real text names things as `the coolant pump`,
   `Filter-A`, `Rossi`, and only the first kind is identifier-shaped. The
   end-to-end claim holds where entities are machine ids and is untested where
   they are noun phrases.
2. The generated corpus's entities *are* identifier-shaped by construction, so
   the id rule is close to reading the generator's mind. That is why the number
   is reported next to the hand corpus's 0.304 and not alone.
3. `min_df = 3` was not swept. It is the df rule's only parameter and the df
   rule is the one that failed; sweeping it could move 0.163 but cannot make a
   generic token specific.

## Narrowed by E59 (2026-09-14, same day)

The claim above — that a high-recall rule destroys the downstream task — is
corpus-dependent and was stated too generally. On the hand corpus the same `df`
rule scores 0.321 top-1, third of six arms and well above blind cosine. The
mechanism is **corpus redundancy**: 150 sentences of one template mean a link on
`pressure` joins all of them. See `E59_NOUN_PHRASES.md`. Both results stand.

## Next

The honest next step is **noun-phrase extraction on the hand corpus** — the case
where identifiers do not exist — scored the same way, with the end-to-end arm
beside it. Until that runs, the pipeline is measured end-to-end only on text
that names its entities the way a machine would.
