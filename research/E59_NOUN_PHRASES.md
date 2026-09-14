# E59 — noun-phrase links on text that has no identifiers, and a correction to E58's lesson

Run 2026-09-14. Code `src/worldstate.rs` (`extract_noun_phrases`,
`merge_entities`), `examples/experiment59_noun_phrases.rs`. Corpus: the
hand-authored `corpus.jsonl` (72 sentences, sha256 `264c40a07303…`). Embedder
`bge-base-en-v1.5`. Artifact `benchmarks/results/worldstate-e59.json`.

E58 left the pipeline end-to-end complete where entities are machine ids and
untested where they are noun phrases. This is that test. The chunker uses no POS
tagger: a determiner, then up to three tokens that are not stopwords, not
`-ing`/`-ed`, and not a trailing `-s` *after* the phrase has a head — the last
rule because position is what separates `the parts` (plural noun) from `the
coolant pump feeds` (verb), and the first version of the file, which put `s` in
the verb-suffix list, chunked the latter as one three-word entity. Each phrase
emits **both** the full key and its head (`coolant-pump` and `pump`), which is
what lets a later `the pump` link without a coreference model.

## Extraction against gold

| extractor | P | R | F1 |
|---|---|---|---|
| id+cap | 0.972 | 0.343 | 0.507 |
| df (repeated content token) | 0.420 | 0.804 | 0.551 |
| **noun phrase** | 0.488 | 0.699 | **0.575** |
| np + id+cap | 0.488 | 0.699 | 0.575 |

`np + id+cap` is identical to `np` because the chunker already emits
identifier-shaped and mid-sentence-capitalised tokens; the merge arm is
redundant and is reported so nobody adds it again.

## Downstream: antecedent at gap ≥ 3, 28 queries

| links | top1 | top3 | 95% CI |
|---|---|---|---|
| gold links | 0.393 | 0.714 | [0.214, 0.571] |
| **noun phrase** | **0.357** | 0.393 | [0.179, 0.536] |
| df only | 0.321 | 0.500 | [0.143, 0.500] |
| id+cap only | 0.214 | 0.250 | [0.071, 0.393] |
| cosine (blind) | 0.071 | 0.250 | [0.000, 0.179] |

Noun-phrase links recover most of the gold-link arm — 0.357 against 0.393 — and
every link arm beats order-blind cosine. **n = 28 and the intervals overlap
heavily; this is directional, not a measurement.** Top-3 is where the gold links
still separate (0.714 vs 0.393), which says the chunker's links are good enough
to find the right candidate first and not good enough to keep the right
candidate in a short list.

## The correction to E58

E58 concluded that a high-recall, low-precision link rule *destroys* the
downstream task: `df` links took the supersession arm from 0.700 to 0.013. Here
the same `df` rule scores 0.321 — third of six, and well above blind cosine. So
the E58 lesson is real but was stated too generally. The corrected version:

> High-recall generic links collapse the entity filter **when the corpus
> contains many near-identical cycles** — that is, exactly when disambiguation
> is what the filter is for. The generated corpus has 150 sentences of the form
> "X holds pressure within tolerance"; a link on `pressure` connects all of
> them. The hand corpus has one topic told once, so a generic link is merely
> loose rather than fatal.

Both runs stand; the mechanism is corpus redundancy, not recall as such. This is
the kind of over-generalisation the repo rule about updating a failure comment
rather than deleting it exists to catch, and E58's record now points here.

## What this licenses

A model-free chunker carries the link layer on hand-written prose well enough to
beat order-blind retrieval and to approach gold links at top-1. That is the
weakest possible positive and it is the one the design needs: the world-model
arm does not require an annotated corpus.

## What it does not license

1. n = 28 queries. Nothing here is a number to quote; the interval spans 0.18 to
   0.54.
2. Precision 0.488 means half the extracted keys are not entities. The arm
   tolerates that on this corpus; E58 shows a corpus where it would not.
3. The chunker is English-specific, determiner-dependent, and was written
   against this corpus's sentences. It has not been run on any text it was not
   authored beside — which is the next thing to do, and the honest reading of
   every number above until then.
