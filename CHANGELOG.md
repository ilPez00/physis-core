# Changelog

Notable changes to `physis-core`. This file starts at 0.1.15; earlier
releases predate it and are documented only by their git tags and commit
history.

## 0.1.23

### Added — W1 resource integration (PLAN.md Phase 24 wave 1, 2026-09-10)

**T4 Nixon Diamond benchmark seed** (`tests/epistemic_w1.rs`): the ATMS
`rules_nixon` scaled to Physis — p and ¬p coexist with their evidence, the
contradiction is first-class and open, both sides leave a replayable audit
footprint, and a later human verdict disprefers without destroying
(`nixon_diamond_retains_both_sides`).

**T3 evidence retraction with JTMS cascade**:
`hypothesis::Hypothesis::retract_evidence` re-derives a hypothesis without a
source — status follows the closed ingest rule, fitness recomputes, the
retraction is an audited revision, and `Certified` survives (authority
leg; the authority registry itself is T11). `delta_engine::retract_evidence_with_cascade`
walks the A2 DependsOn closure of the retracted hypothesis's references and
re-evaluates every selected dependent in the shadow frame — a proposal until
committed (`evidence_retraction_replays_to_state_without_it`).

**G4 per-link intake ids**: `Evidence::intake_id`, `ProvenanceLink::intake_id`
and `EpistemicEvent::intake_id` stamp every link and event with the ingest
episode that produced it; `explanation::ExplanationReport::cited_intake_ids`
and `provenance::ProvenanceChain::cited_intake_ids` resolve an explanation's
citations to the concrete raw intakes (`citations_resolve_to_intake_ids`).

**G6 point-in-time queries**: `temporal::TemporalValidity::is_valid_at` now
reads the `expired_at` system leg (P0 landed it fields-only), and
`epistemic::EpistemicAuditTrail::point_in_time_status_at` gates the audited
status by the validity window — the surface can say None where history still
replays the state; invalidate-don't-delete visible on the surface
(`point_in_time_matches_audit_trail`).

**G5 hybrid retrieval beside cosine**: `rag::Bm25Index` (dependency-free
Okapi BM25, k1 = 1.5, b = 0.75), `rag::fuse_rrf` (k = 60 — the same constant
llm_wiki's `search_project` ships), `rag::rank_hybrid` and
`rag::measure_hybrid_vs_cosine`. The shipping decision is *derived* from the
measured verdict; a non-positive measurement ships the fused path disabled
(`hybrid_fusion_vs_cosine_baseline`, `bm25_index_scores_are_deterministic_and_repeatable`).

## 0.1.22

### Added — the epistemic revision track (Atlas + Graphiti patterns, P0 + P1)

The delta engine gains the epistemic-revision machinery specified in
`PLANNING.md` §7 and `../docs/ATLAS_GRAPHITI_INTEGRATION_PLAN.md`:
patterns, protocols and test shapes taken from Atlas (Ripple, AGM,
adjudication) and Graphiti (temporal legs, ingest discipline) — never
dependencies. Offline Rust over sled, no new crates. Every item ships
behind a named gate test in `tests/epistemic_p0.rs` / `tests/epistemic_p1.rs`.

**P0 — invariants + history (no scoring change):**

- **A7 no-perturbation invariant** (`delta_engine::evaluate_mutation`): a
  zero `EmbeddingShift` returns an empty wave without touching shadow state
  — fitness bit-identical, zero transitions
  (`zero_shift_wave_leaves_fitness_untouched`).
- **G7 invalidate-don't-delete** (`hypothesis::current_hypotheses`,
  `hypothesis::hypotheses_including_history`): superseded hypotheses stay
  history-readable and out of current queries; retention/compaction stated
  before the gate — no automatic deletion, audited compaction deferred
  (`superseded_items_stay_queryable_for_history`).
- **G1 `expired_at` leg** (`temporal::TemporalValidity`): the
  system-invalidation time, distinct from `valid_until`; fields only,
  serde-default keeps old stored JSON deserialising
  (`temporal_triple_serialises`).

**P1 — revision correctness:**

- **A2 DependsOn-only revision walk** (`EvaluationContext::depends_on_walk`,
  `RevisionWalk` on `OntologyDeltaReport::revision_walk`): hypothesis
  revision is selected by the declared DependsOn closure (BFS, hop counts,
  cycle recording, depth cap 5, node cap 5000 with flagged truncation), not
  by the breadth wave — mere arrival no longer shifts a hypothesis. Sparse
  graphs fall back to the breadth-affected selection, logged via
  `fallback_breadth_used`
  (`midchain_revision_revises_exact_dependents`,
  `depends_on_walk_records_cycles_and_terminates`).
- **A6 named fitness weights** (`hypothesis::FITNESS_WEIGHT_*`,
  `Hypothesis::fitness_term_breakdown`): the composite weights and penalty
  schedules are published frozen constants; every recompute names its terms
  and the contributions sum to the fitness wherever the clamp does not bite
  (`fitness_recompute_reports_term_breakdown`).
- **A5 adjudication routing** (`delta_engine::route_transition`,
  `AdjudicationDecision` on `OntologyDeltaReport::adjudications`): proposed
  demotions are routed, not blindly applied — `AutoApply` between ϵ and
  ϵ + `ADJUDICATION_STRATEGIC_FLOOR` (0.15, Atlas's number),
  `StrategicReview` beyond it (proposal + rationale + `ResolutionStatus::Open`,
  status unchanged), `CoreProtected` for `Certified` beliefs (flagged,
  never auto-demoted). Rationale record only; no queue surface
  (`large_delta_routes_to_open_with_rationale`,
  `certified_is_core_protected`,
  `small_delta_auto_applies_with_decision_recorded`,
  `route_transition_boundary_table`).

- **G3 episode + watermark trail** (`epistemic.rs`): `EpistemicEvent` gains
  the two clocks — `asserted_at` (episode reference time, when the source
  asserted the fact) vs `timestamp` (arrival / transaction time), and
  `reconstruct_status_at` replays in **assertion order**: out-of-order
  intake replays to the in-order state
  (`late_evidence_replays_to_same_state`). Per-stream `HighWaterMark`s
  advance on both clocks and flag late episodes via `note_intake` →
  `IntakeReceipt` (`watermark_advances_and_flags_late_episodes`). Pre-G3
  trails (no `asserted_at`, no watermarks) replay byte-for-byte unchanged;
  serde-default keeps old stored JSON deserialising
  (`legacy_trails_replay_by_arrival_unchanged`,
  `trail_serialises_with_g3_fields_and_reads_old_json`).
- **23.6 temporal dream** (new module `dream.rs`):
  `dream_over_history(&trail, &mutations, lookback)` replays the audit
  trail in assertion order and **proposes edits to retired branches** —
  re-activate a retained superseded/failed/inert hypothesis whose pattern
  re-presented itself in later observations, restore a severed DependsOn
  connection whose target re-confirmed, retire a connection whose target
  accumulated ≥ `RETIRE_AFTER_CONTRADICTIONS` (2) contradictions. Every
  `RetrospectiveProposal` cites the historical event/mutation ids that
  motivated it; the dream takes the trail by shared reference and cannot
  write — proposals only, the caller decides
  (`dream_proposes_reactivation_from_retained_branches`,
  `dream_proposes_restoring_severed_connections`,
  `dream_proposes_retiring_repeatedly_contradicted_connections`,
  `dream_stays_silent_without_reconfirmation`, `dream_never_writes`).

### Changed

- **Behavior**: a demotion with Δcoherence > ϵ + 0.15 no longer auto-applies.
  `test_delta_engine::hypothesis_cascade_embedding_shift` was updated to the
  new contract (its scenario is a full reversal, Δ = 1.0 → StrategicReview).
- The `hypothesis` re-export list now includes the fitness constants;
  `delta_engine` re-exports `RevisionWalk`, `WalkStep`, `AdjudicationRoute`,
  `AdjudicationDecision`, `route_transition`, `MAX_REVISION_WALK_NODES`,
  `ADJUDICATION_STRATEGIC_FLOOR`.

**Honesty constraints carry**: the machinery proposes, carries, and defers;
it does not verify. No number moved (cosine 0.519 ≈ chance; propose top-3
0.712 proposes, never verifies).

## 0.1.21

### Changed — Gate 0: the corpus was regenerated, and the domain axis finally has a definition

A 2026-09-07 audit (see 0.1.19) found **43.3% of a deterministic sample filed in
the wrong cell**. This release fixes the cause rather than the symptom.

**The root cause was a missing definition.** The fourteen modes have been
documented since they were designed — the activity-energy axis, DESTROY
completing the Greimas opposition against CREATE, PLAN converging against
BRAINSTORM. The five *domains* were defined nowhere in the codebase. 730 entries
had been filed against five words whose meaning was never written down.

Worse, the 70 anchors that stand in for that definition were each phrased in a
single register: HEAL spoke only about bodies ("rest day, recover, sleep
deeply"), CONSTRUCT only about building sites ("pour concrete, frame the wall"),
BOND only about friendships. That is a human-daily-life vocabulary, and the
corpus it has to classify is mostly machine telemetry, agent architectures,
office documents and semiotics. `Coolant & Lubrication` in HEAL/REST was not
obviously wrong; it was wrong *against an anchor that says "sleep deeply"* and
right against the idea those words stood in for.

**Added: `docs/GRID_AXES.md`**, the missing contract. The domain axis says what
is acted on — HEAL=condition, CONSTRUCT=structure, FABRICATE=output,
BOND=relation, STUDY=knowledge — and the mode axis says what act. It also
separates WALK from MAINTAIN, which the old anchors had made near-synonyms
(`CONSTRUCT/WALK` "Steady Upkeep" against `CONSTRUCT/MAINTAIN` "Building
Maintenance" shared almost all their meaning): WALK is the rhythm of normal
operation, MAINTAIN is work done against decay.

**All 70 anchors rewritten** to that contract, each spanning human, machine and
organisational registers, with hints kept mode-pure and domain-pure — the
register varies in the noun, never the verb. The classifier embeds `name +
hints` and nothing else, so a cell that only speaks one register cannot attract
the others' records.

Removing the register cheat could only have made cells more confusable. Measured,
it did the opposite:

| anchor separation | old | new |
|---|---|---|
| mean off-diagonal cosine | 0.6885 | **0.6631** |
| pairs above 0.85 | 25 | **10** |
| pairs above 0.90 | 5 | **0** |
| mean nearest-neighbour cosine | 0.8489 | **0.8207** |

**All 661 non-anchor entries re-filed** against the contract by a rater who was
shown name, category and hints with the existing assignment withheld. **513
(77.6%) moved. Occupied cells went 35/70 to 57/70.**

Independent of the rater — neither filing was made with embeddings, so the
embedder is a separate instrument — the share of entries whose own cell anchor
sits in the worse half of all 70 cells:

| | old anchors | new anchors |
|---|---|---|
| **old filing** | 31.8% | 31.8% |
| **new filing** | 18.2% | **16.2%** |

The new filing wins even judged against the *old* anchors, which is the stronger
test. On the 585 entries whose hints do not narrate their old cell, 34.5% →
16.9%, and the share whose own anchor is the nearest of all 70 goes 6.0% →
16.8%. Note what this table also says: **the anchor rewrite contributes almost
nothing to fit** (31.8% → 31.8%); its value is separation and register coverage,
and the re-filing does the work.

**What is NOT claimed.** The twelve discovery and soundness-checking mechanisms
evaluated against the old corpus are still UNTESTED — they have not been re-run
on this one. And the re-audit under the original protocol (6.6% clearly
misfiled, against 43.3%) is a **self-grade**: the rater who scored it also
produced the filing. It is published for checking, not offered as independent
evidence; the two numbers above are.

Full method, controls and raw records:
`research/perspective-discovery/FINAL_REPORT.md` (Stage 11),
`refiling_gate0.tsv` (all 661 before/after), `cell_adjudication_regenerated.tsv`.

## 0.1.20

### Changed — `rank_candidates` orders by lift, not by the rescue count

`coverage::rank_candidates` sorted candidates on `CandidateGain::count()`, the
number of records crossing the confidence threshold. It now sorts on a new
continuous field, `CandidateGain::lift`:

```
lift = Σ over records of max(0, cos(candidate, record) − live(record))
```

**Measured, not preferred.** 1,638 candidate proposals scored against 6,234
operational records, with a 10% hold-out of real ontology entries as ground
truth. Ordering by lift puts a genuine missing concept in **12 of the first 25**
rows — precision 0.480 against a 0.132 base rate, exact hypergeometric
p = 2.3e-5 — where ordering by `count()` reaches 0.320.

Four controls sit on the random floor at the head of the list: proposal
popularity (p = 0.43), the proposer's own geometry (p = 0.23), parent similarity
(p = 0.23), and how empty that region of the ontology is (p = 0.66). The last is
the important one — **the ontology's own geometry ranks its own holes at
chance**, so the ordering information is genuinely coming from the operational
records.

**Why the old ordering was weak: the threshold rule goes nearly inert on
operational text.** Records written by a running operation already match the
live ontology strongly (median best-entry cosine 0.741 on the corpus measured),
and a candidate interpolated between two existing entries can rarely beat every
known entry for any record — at that median only 272 of 1,638 candidates rescued
even one. The 0.1.17 cross-validation looked healthier because its records were
held-out ontology *entries*, which sit further out and leave the bar reachable.
Nothing in the API said so. Lift has no bar and degrades smoothly.

`count()` and `is_useful()` are unchanged and remain the *filter* — that is the
half the t = +12.11 duplicate control validated. Filter on `count()`, order on
`lift`.

### Added

- `CandidateGain::lift` (`f32`). A duplicate of an existing entry scores exactly
  `0.0`, for the same reason it rescues nothing: the live score is already a
  maximum over all entries, so a copy of one cannot exceed it. Covered by a
  test, along with the degenerate empty-ontology case (lift stays finite).

### Breaking

- `CandidateGain` no longer derives `Eq` — it now holds a float. `PartialEq`,
  `Debug`, `Clone` and `Default` are unchanged, so `assert_eq!` and `==` still
  work; only `Eq`-bound generic contexts (`HashSet`, `BTreeMap` keys) are
  affected.
- Anything relying on `rank_candidates` returning count-descending order will
  see a different order. Sort the returned `Vec` on `count()` yourself if you
  need the old behaviour.

### Documented — what ranking does *not* buy

The ordering was validated against **missing** concepts (a hold-out), never
against **wrong** ones. Asked instead to score entries against the cell they
were actually filed in, the same cosine machinery detects real misfilings at
**AUC 0.410**, with chance inside the confidence interval. Coverage finds gaps;
it cannot find mistakes, and a ranked worklist is still a worklist. Also stated
in the module docs and README: the hold-out concepts that validated the ranking
carry heavy corpus traffic (median max-record cosine 0.793), so this ranks gaps
the operation already talks about — a concept nobody writes about produces no
lift and cannot surface at all; and only 75% of candidates lift any record, the
rest tying at zero.

Full experiment and controls: `research/perspective-discovery/FINAL_REPORT.md`,
Stage 10, in the physis-pro superproject.

## 0.1.19

### Changed — the 0.1.18 reasoning was wrong; the conclusion was not

- **0.1.18 said twelve mechanisms "were evaluated against controls" and failed.
  That overstated what had been shown.** The evaluation's ground truth was never
  checked. It has now been.

  A deterministic sample of 60 of the 659 loaded ontology entries, adjudicated
  against each cell's own anchor definition, finds **43.3% clearly filed in the
  wrong cell** (31.7% clearly right, 25.0% marginal). `CONSTRUCT/WORK` — "pour
  concrete, frame the wall, lay bricks" — contains Access Control, Booking &
  Logistics, Database Engineering and a Peirce sign category. `HEAL/REST` —
  "rest day, recover, sleep deeply" — contains Firstness, Power Regulation and
  Semiconductor Cleanroom Environment.

  The audit that rejected those mechanisms injects 35–114 misfilings and asks
  each scorer to find them. At a 43.3% ambient rate the same 572 entries already
  hold ~248 real misfilings **labelled correct** — a signal-to-noise ratio of
  1:7 at the stride the published result used. A scorer that genuinely detects
  bad filing is *penalised* for flagging them.

  **So those mechanisms are UNTESTED, not refuted.** The experiment lacked the
  power to distinguish a working scorer from a broken one.

  **The crate's status is unchanged and better supported.** It does not do what
  it claims — not because twelve approaches were disproven, but because the cell
  assignments it classifies against are unreliable, which makes classification
  against them unreliable whatever the mechanism.

  Adjudication data and the re-test plan (Stage 8) live in the parent project at
  `research/perspective-discovery/`.

## 0.1.18

### Changed — status notice, against our own work

- **`becoming` does not detect meaning change, on an external benchmark.**
  Added to the status notice rather than left in a research log, because the
  README listed `becoming` among the primitives that work and a reader could
  reasonably have taken that as covering the task it is named for.

  The statistic is sound and stays: its runs test separates `AAAAAABBBBBB` from
  `ABABBABAABAB`, which no clusterer can. What fails is every attempt to give
  it the sense partition it cannot produce. Measured on **SemEval-2020 Task 1**
  (37 lemmas, ground truth by the task organisers — the previous evaluation was
  8 terms chosen by us on our own documentation, which could not have
  falsified anything): driving `becoming` from n-gram signature families gives
  **51.4% accuracy against a 56.8% majority-class baseline** and Spearman
  **0.183** against the graded gold, where published SOTA is 66.5% and 0.518.
  Below a baseline that never reads the text.

  Diagnosed rather than only scored: substitutability yields a median **57
  families per term**, `classify_labeled` reads the top two, so 31 of 37 terms
  return `Split` and `Stable` never fires. Granularity was the missing
  property, not exactness.

- **Relation typing by substitutability is dead too.** Ranking a word's true
  Greimas dual against eleven distractors over 12M tokens of CCOHA recovers it
  **0 of 12 times** (chance expects 1.0). Frequency dominates the rankings.
  This was the second *symbolic* mechanism to fail after nine geometric ones,
  so the wall is not a property of embeddings: words filling the same slot are
  near-synonyms, frequency peers and antonyms alike.

- **The salvage, named specifically.** n-gram signature families are real and
  survive their controls. Longest-match-with-backoff signatures recur where
  fixed trigrams do not (**92.0% vs 34.6%** median recurrence), and the long
  ones are not chance: **41.4%** of real signatures reach length ≥ 4 against
  **2.6%** when the same tokens are resampled independently, same geometry and
  same frequencies. Families form without collapsing (largest family 21.1%,
  35/37 terms). That is a usable collocation vocabulary — and explicitly not a
  sense inventory.

  A caution that cost a stage: the **recurrence rate itself is mostly a
  frequency artifact**. The shuffled control reaches 78.7% of the 92.0%
  headline. Only signature length survives it.

- **The crate is marked NOT CURRENTLY FUNCTIONAL for its stated purpose**, in
  both the package description and the README. Nothing is removed and no API
  changes; what changes is the claim being made.

  The engine exists to discover ontological structure and to audit whether
  records are filed soundly. Eleven mechanisms for that were built and tested
  against controls, and all eleven failed. The last surviving one — a
  structural, label-free misfiling detector — had been reported as beating a
  geometric baseline at AUC 0.654 vs 0.619. That result came from **one**
  arbitrary injected perturbation. Resampled across eleven, it wins 1 of 11 and
  loses to plain cosine distance (paired *t* = −3.08 on a held-out half,
  −6.06 overall). The claim is withdrawn rather than quietly dropped.

  Two numbers worth carrying elsewhere. A vocabulary-overlap scorer measured
  **+0.121 AUC, 11/11, *t* = +12.97** against the entries it was built from and
  **−0.009, *t* = −1.81** on a held-out half — a 0.130 swing from contamination
  alone, and it would have read as a landmark result without a train/test
  split. And the harness noise floor is **~0.074 AUC**, so single-run margins
  below that carry no information; several earlier reported results sit inside
  it.

  What continues to work is unaffected and tested: ontology loading and the
  5×14 semiotic grid, `linkage`, `coverage`, `becoming`, `process`. For
  deciding whether an entry belongs in a cell, on the data tested, cosine
  similarity over embeddings is better and simpler — use that instead.

  Full measurements: `research/perspective-discovery/FINAL_REPORT.md` in the
  parent project.

## 0.1.17

### Added

- **`physis_core::coverage`** — scores whether a candidate cell would let the
  ontology place records it currently cannot. `discovery` already proposed new
  entries but had no way to say which proposals were worth keeping; this is
  that missing half, and deliberately the narrow half.

  `uncovered` lists the records the live ontology cannot place at or above a
  threshold. `candidate_gain` and `rank_candidates` score candidates by how
  many of those they would rescue, strongest first, ties broken on index so
  the ranking is reproducible.

  **Why the check is shaped as an operational question.** A long research
  track tried nine mechanisms for certifying discovered structure and all nine
  failed. Every one asked a representational question — is this split correct,
  does a lexicon recognise it, do many pairs agree on it — and answered it with
  a statistic computed over the same embedding space that produced the
  candidate. This instead asks whether adding the candidate changes what the
  ontology can place, against a corpus external to whatever produced it, so it
  cannot be satisfied by the geometry agreeing with itself.

  Measured over 5-fold cross-validation on 691 held-out records: re-adding an
  entry the ontology already contains scores **exactly zero on 2000 of 2000
  trials**, while genuine interpolations score above zero (Welch t = +12.11).
  In that run it cut 2000 candidates down to 141 worth reading.

  **Coverage is not correctness.** A candidate that swallows records into a
  wrong cell scores exactly like one capturing a real gap — both make the
  records classifiable. Use this to shrink the pile and then have a person read
  what survives; it is a filter, not an approver.

## 0.1.16

### Added

- **`physis_core::linkage`** — a new module answering "which `(domain, mode)`
  cells does real data bridge, and via which items". Each text contributes one
  bridge, between its top-scoring and second-scoring cell. That is the whole
  rule: no threshold, no `k`, no seed, and ties broken on the cell key rather
  than on iteration order.

  It deliberately does **not** discover cells. A long research track tried
  seven structurally distinct ways to derive stable groupings from embedding
  geometry — margin/silhouette gating, three kNN-consistency variants,
  cross-embedder corroboration, density peaks, cross-embedder split agreement,
  capacity-constrained training loss, and plain k-means — and all seven failed
  on real data. Re-derived clusters do not survive corpus growth (58% anchor
  overlap after +25% data) and nothing survives an embedder swap (ARI ~0.10).
  So this module takes the ontology's hand-authored cells as the fixed points,
  because they cannot drift, and measures only the links between them.

  No claim is made about any individual item being "genuinely cross-cutting";
  an earlier thresholded version degenerated to a 98% flag rate, which merely
  restates that domains overlap. The signal is aggregate — a cell pair that
  recurs as many different items' runner-up is meaningfully linked.

  `LinkageGraph::build` / `links` / `strongest` / `cross_domain`, with
  `CellLink::is_cross_domain` for the links a single-label classification
  cannot represent at all.

## 0.1.15

### Fixed

- **`OntologyLoader::classification_domains()` returned entries in a
  nondeterministic order.** It chained three `HashMap::values()` iterators,
  and Rust randomizes `HashMap` iteration order per process by design. Every
  process therefore saw the same ontology entries in a different order, which
  changed anything order-sensitive downstream — centroid accumulation,
  train/test splits, and tie-breaking between near-equal similarity scores.
  The result was classification and discovery output that varied run to run
  from identical inputs and identical model weights.

  Entries are now sorted by `(name, domain, mode)` before being returned.
  Consumers on ≤ 0.1.14 who observed unstable output should upgrade before
  investigating other causes; this is very likely the cause.

  Note on how this was found, since the wrong answer is a tempting one: we
  attributed it twice to floating-point nondeterminism in ONNX Runtime's
  multi-threaded reductions. That was wrong. A direct probe
  (`examples/probe_embedding_determinism.rs`, added in this release) shows
  embeddings are bit-identical across processes *and* across thread counts.

### Added

- **`OnnxConfig::intra_threads: Option<usize>`** — opt-in ONNX Runtime
  intra-op thread control. Default `None` preserves existing behavior (all
  cores). Intended for throughput/latency tuning and for constrained
  environments. Explicitly **not** a determinism control: output was measured
  bit-identical at every thread count.
- **`examples/probe_embedding_determinism.rs`** — verifies embedder
  determinism in-process and, by printing exact dims plus a checksum, across
  processes.
- **README: "Determinism & Reproducibility"** — states the guarantees, the
  0.1.15 ordering fix, and two things that are explicitly *not* guaranteed:
  cross-embedder stability (pin your embedder; it is part of your ontology's
  identity) and stability of re-derived clusters under corpus growth (persist
  anchor identity instead of re-deriving).

### Packaging

- `wordnet-db` / `wordnet-types` moved from `[dependencies]` to
  `[dev-dependencies]`. Nothing in `src/` uses them — they are used only by
  two research examples — so they were being forced on every downstream
  consumer for no benefit. **No API change**; consumers simply stop pulling
  them.
- Stray files excluded from the published `.crate`: `**/*.bak` (a
  `Cargo.lock.bak` and an ontology backup were being shipped),
  `sw_add_domains.py`, `STRIPE_LICENSE_GUIDE.md`.
