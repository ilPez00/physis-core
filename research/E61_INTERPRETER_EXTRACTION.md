# E61 — the model enters the pipeline, at one place, and it flips the sign

Run 2026-09-14. Script `research/e61_interpreter_extraction.py`, scorer
`examples/experiment59_noun_phrases.rs` via `PHYSIS_E59_EXTRA`. Corpus
`benchmarks/worldstate/corpus-docs.jsonl` (168 sentences of this repository's own
prose, gold = the author's backticked spans). Interpreter
**`qwen2.5:7b-instruct-q4_K_M` on omo, local, temperature 0**. Artifacts
`benchmarks/results/worldstate-e59-docs-llm.json` and `…-llmfb.json`.

E60 left the link layer as the measured bottleneck: gold links 0.698, rule links
0.209, **no links at all 0.349**. This is the first experiment in which a model
takes part, and it takes part at exactly one step — naming what a sentence is
about.

Two conditions, both fixed before the run:

* **all** — the interpreter extracts links for every sentence, replacing rules.
* **fallback** — the interpreter is asked only where the rules produced nothing
  (29 of 168), and its output is merged with the rules.

## Extraction

| extractor | P | R | F1 |
|---|---|---|---|
| code ids (E60 best rule) | 0.832 | 0.379 | 0.520 |
| np + code | 0.480 | 0.462 | 0.471 |
| **interpreter** | 0.442 | 0.642 | 0.524 |
| **rules + interpreter** | 0.428 | **0.823** | **0.563** |

## Downstream — antecedent at gap ≥ 3, 43 queries

| links | top1 | top3 | 95% CI |
|---|---|---|---|
| gold links | 0.698 | 0.977 | [0.558, 0.837] |
| **rules + interpreter (all)** | **0.488** | **0.698** | [0.349, 0.628] |
| **interpreter alone (all)** | **0.465** | 0.558 | [0.326, 0.605] |
| cosine (blind, no links) | 0.349 | 0.581 | [0.209, 0.488] |
| rules + interpreter (**fallback only**) | 0.326 | 0.488 | [0.186, 0.465] |
| df only | 0.302 | 0.465 | [0.163, 0.442] |
| np + code (E60 best rule arm) | 0.209 | 0.279 | [0.093, 0.326] |
| interpreter, fallback only | 0.093 | 0.093 | [0.023, 0.186] |

## Two findings, and the second one is the one to keep

**1. The interpreter flips the sign of E60.** For the first time on prose the
extractor was not written against, a link arm *beats using no links at all*:
0.488 against 0.349. It closes 57% of the gap between rules (0.209) and gold
(0.698), and its top-3 (0.698) equals gold's top-1. The intervals overlap
([0.349, 0.628] vs [0.209, 0.488]) at n = 43, so this is a direction with a
mechanism, not a settled magnitude.

**2. Calling the model only where the rules failed does not work — it is worse
than not calling it.** The fallback condition scores 0.326, *below* blind cosine
and below the all condition by 0.162. The reason is the useful part: the rules'
problem on this corpus is not only the 29 sentences where they produce nothing,
it is the wrong links they produce everywhere else. Patching the gaps leaves the
bad links in place, and a filter is only as good as its worst links.

So the cheap architecture — cheap rules first, model only on misses — is
**refuted here**, which is the opposite of what a cost-conscious design would
assume. If the model is called at all, it must be called on everything.

## Narrowed by E62 (2026-09-14, same day)

Finding 2's strong form — that fallback coverage is worse than *no links at all*
— is regime-specific. On the machine log (159 queries) the fallback arm scores
0.377 against blind cosine's 0.340, i.e. better. What survives both regimes is
the weaker claim: **partial interpreter coverage always loses to full coverage**
(0.326 vs 0.488 on docs; 0.377 vs 0.465 on the log). See `E62_MACHINE_LOG.md`.

## Determinism

Temperature 0, one fixed prompt, model and endpoint recorded next to the
artifact. Spot-checked by re-running the first 15 sentences: **15/15 identical**.
The scorer reads the saved file, never the model, so every scored number here is
regenerable without the endpoint being up.

## What this does not license

1. n = 43 queries; every CI overlaps its neighbour. The ordering is stable
   across conditions, the magnitudes are not.
2. One corpus, one model, one prompt. A 7B at temperature 0 named things well
   enough here; nothing says a different prompt or a different 7B would.
3. Extraction F1 went *up* (0.563) while precision went *down* (0.428). The arm
   tolerated that here; E58 measured a corpus where lower precision was fatal.
   Which regime a corpus is in must be checked, not assumed.
4. Cost is not measured. 168 local calls took minutes; the rule path is
   microseconds. Nothing here says the improvement is worth that on a real log —
   `physis-world` runs over 3232 observations, which is 19× this corpus.

## Next

The shell (`physis-world`) still uses rules only. The honest way to wire this in
is an explicit, off-by-default `--interp` that says what it costs, plus the same
two scores measured on the machine log rather than on documentation — because
the log is 94% filenames and commit messages, which is a third text regime and
neither of the two measured so far.
