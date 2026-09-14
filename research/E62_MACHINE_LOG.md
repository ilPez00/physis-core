# E62 — the third text regime: the machine's own log, and the interpreter result replicates

Run 2026-09-14. Corpus `benchmarks/worldstate/corpus-log.jsonl` (sha256
`90539ad9baa8…`), built by `make_corpus_log.py` from
`~/.physis-core/observations.jsonl`: **388 observations — 234 git, 154
terminal**, 478 gold entity mentions, **159 scored queries** (the largest n in
the series). Interpreter `qwen2.5:7b-instruct-q4_K_M`, local, temperature 0.
Artifacts `worldstate-e59-log.json`, `…-log-llm.json`, `…-log-llmfb.json`.

## Where gold comes from, and why it is not circular

Only `git` and `terminal` observations are used, and each one's gold entity is
the annotation **its own source already carries**:

* `git` — the conventional-commit scope (`fix(ci):` → `ci`, `feat(fs/graph):` →
  `fs`, `graph`) plus dotted filenames in the message;
* `terminal` — the program invoked plus path-like arguments.

`fs` observations are **excluded deliberately**: their subject *is* a path, so a
path-derived gold would be the same string the `code ids` rule extracts and the
score would measure nothing. No rule in `worldstate.rs` targets commit scopes or
program names.

## Downstream — antecedent at gap ≥ 3, 159 queries

| links | top1 | top3 | 95% CI |
|---|---|---|---|
| gold links | 0.528 | 0.730 | [0.447, 0.604] |
| **rules + interpreter** | **0.465** | 0.629 | [0.390, 0.541] |
| interpreter alone | 0.459 | 0.623 | [0.384, 0.535] |
| df only | 0.403 | 0.579 | [0.327, 0.478] |
| rules + interpreter (**fallback only**) | 0.377 | 0.453 | [0.308, 0.453] |
| cosine (blind, no links) | 0.340 | 0.472 | [0.270, 0.415] |
| np + code (best rule arm) | 0.296 | 0.321 | [0.233, 0.371] |
| code ids | 0.088 | 0.088 | [0.044, 0.138] |

Extraction F1: interpreter 0.288, rules+interpreter 0.265, best rule 0.145. Every
extractor scores badly against this gold, because commit scopes and program names
are not what any of them look for — **and the downstream ordering is unchanged by
that**, which is the point of scoring both.

## What replicates and what does not

**Replicates (E61 → E62, different regime, n 43 → 159):** interpreter links beat
using no links at all — 0.465 vs 0.340 — and land within one interval of gold
(0.528). Two corpora, two golds, same direction, and the interval here is
tighter than anything earlier in the series.

**Replicates:** partial coverage is worse than full. Asking the model only where
the rules came up empty scores 0.377 against 0.465.

**Does NOT replicate:** on documentation, partial coverage was worse than *no
links at all* (0.326 vs 0.349). On the log it is better than blind (0.377 vs
0.340). So the strong form of E61's claim was regime-specific; the surviving
form is the weaker one — **partial interpreter coverage always loses to full
coverage**, and whether it also loses to no links depends on the corpus.

**New here:** `df` — the crudest rule, repeated content tokens — is the best
non-interpreter arm on the log (0.403), beating blind cosine. In a log the
repeated tokens *are* the recurring subjects. E58 found the same rule fatal on a
redundant generated corpus. Three regimes, three different best rules: there is
no corpus-independent extractor in this series, and that is now measured rather
than suspected.

## Wired into the shell

`physis-world` reads an interpreter cache from `PHYSIS_WORLD_INTERP`, off by
default, keyed by observation `seq` (positions shift as a log grows; seqs do
not). Converter: `benchmarks/worldstate/links_to_cache.py`. When the cache
covers under half the log the shell prints the measured warning rather than
silently running a configuration both experiments scored as worse.

On the current 3232-observation log the cache covers 377, so the warning fires —
correctly. `locate ontology` gains the `refactor(ontology):` commit that rules
alone miss entirely.

## Not licensed

1. 159 queries is the best n here and still leaves overlapping intervals between
   `rules + interpreter` and `gold`.
2. The log is this machine's, this week, mostly one repository. A different
   developer's log is a fourth regime.
3. **Cost is still unmeasured.** 388 interpreter calls took roughly eight
   minutes on omo; covering all 3232 observations is ~an hour, and the log grows.
   Nothing here says that trade is worth it — only that the links are better.
