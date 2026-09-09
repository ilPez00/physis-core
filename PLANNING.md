# physis-core Expansion Plan

> ## 0. GATE — read before acting on §1 (added 2026-09-09)
>
> **§1 proposes adding 6 domains and 10 modes. Mode expansion is BLOCKED by
> measurement. Domain expansion is gated on a pre-flight that has never been
> run for it.** The evidence is in the parent project
> (`research/RESEARCH_STATUS.md`, `research/mapper/`), and it was not available
> when §1 was written.
>
> ### Why mode expansion is blocked
>
> | finding | measurement |
> |---|---|
> | **The existing 14 modes already contain 5 starved classes** | BRAINSTORM **n = 1**, DESTROY 4, LIFT 11, PLAY 12, GUIDE 13 (E22) |
> | A class below ~20 entries is *unmeasured*, not weak | a well-populated mode cut to 3 entries scores F1 **0.06**; n ≈ 20 → F1 0.26–0.49 (E7) |
> | The pre-flight already STOPs the *current* grid | 35 of 57 classes below 6 training entries (E8) |
> | **The mode axis is not a coherent object at any granularity** | merging makes it worse (−0.187 at k = 5 vs domain's −0.007, E27); clustering finds *subject matter*, and the current labels beat every discovered carve (E29) |
>
> §1 would take mode from 14 to 24 classes on a corpus that cannot populate 14.
> **Adding classes to an incoherent axis compounds the incoherence**; it does not
> resolve it. The standing rule from the research track is **populate first,
> expand second**, and it is not satisfied.
>
> ### Why domain expansion is only gated, not blocked
>
> Domain is in much better shape and the difference is measured: it is a clean
> low-rank object — **4 dimensions carry it losslessly** (E26), all 5 classes are
> well populated (82–140 entries), and a proposer reaches **top-2 0.837** on it
> against 0.678 for mode (E21). Going to 11 domains is *plausible*, and untested.
>
> **Before adding any domain:** run `research/mapper/preflight.py`, honour it
> (house rule 4, overridden once and recorded as a mistake), and re-check that
> the domain axis still compresses to *k−1* dimensions. If it stops compressing,
> the new domains are not carving anything.
>
> ### What to fix in the existing grid first
>
> Both come from the confusion structure, not from taste (E22):
>
> 1. **`FABRICATE` ↔ `CONSTRUCT` confuse at 0.22/0.21, symmetric** — one
>    distinction wearing two names, and the cleanest merge candidate in the grid.
>    Merge, or write the sentence that separates them.
> 2. **`WORK`/`CREATE`/`MAINTAIN`/`SENSE` hold 68% of entries with ample
>    population and do not separate.** This is where the mode axis actually
>    fails, and no representation fixes four definitions that carve nothing
>    apart. It is definitional work and it is a human call — E29 showed the
>    machine has nothing further to offer here.
> 3. **`WALK`** has n = 27, comparable to LEARN (36) and PLAN (35), and scores
>    top-1 **0.18**. The corpus gave it a fair trial and it failed. Redefine or
>    retire — and unlike the starved classes, the evidence supports saying so.
>
> **Caveat on transfer:** these numbers were measured on the parent project's
> 658-entry mapper corpus against a blind re-filing, using the same 5×14 grid.
> The grid is shared; the corpus is not. The sparsity and coherence findings are
> about the grid and transfer directly. Absolute accuracies would differ on a
> different corpus.

## 1. Domain & Mode Expansion

The base 5×14 semiotic grid (HEAL/CONSTRUCT/FABRICATE/BOND/STUDY × 14 modes) is intentionally a **base layer**. New dimensions are added as **additive packs**, not by bloating the grid.

### Missing Domains (to be added as ontology packs)
- `GOVERN` — rule, law, adjudicate, allocate
- `EXCHANGE` — trade, buy/sell, negotiate value
- `EXPRESS` — communicate, signal, art-as-language
- `DEFEND` — secure, protect, pre-empt harm
- `MEANING` — ritual, worship, transcend, commemorate
- `EXPLORE` — navigate, orient, discover the unknown

### Missing Modes (to be added as ontology pack extensions)
- `DECIDE` / `WILL` — agency & intention
- `REASON` — deliberate, infer
- `REMEMBER` — recall, archive
- `MEASURE` — assess, diagnose
- `TEACH` — impart
- `PERSUADE` — negotiate, convince
- `IMAGINE` — envision pre-creation
- `SHARE` — give, distribute
- `PROTEST` — resist, rebuke
- `CELEBRATE` — mourn, ritualize affect

**Expansion approach:**
- Each new domain/mode pair is released as an **ontology JSON pack** that consumers drop into `config/`.
- The grid stays at 5×14; new cells are reachable via **facets** (`sub_domain`, `sub_mode`) or by selecting an axis-kind pack (`office`, `machine`, `agent` already supported).
- Pack authors add entries with `domain: "GOVERN"` and `mode: "DECIDE"` etc., and users can classify against them.

### Proposed Canonical Ordering (view hint, not constraint)

Domains:
```
HEAL < BOND < STUDY < CONSTRUCT < FABRICATE < GOVERN < EXCHANGE < EXPRESS < DEFEND < MEANING < EXPLORE
```

Modes (expanded):
```
REST < SENSE < REMEMBER < LIFT < MOVE < WORK < DECIDE < REASON < CREATE < IMAGINE < TEACH < GUIDE < PLAY < BRAINSTORM < MAINTAIN < PLAN < PROTEST < CELEBRATE
```

---

## 2. Ontology Pack Structure

Each ontology pack is a JSON file placed in `physis-core/config/`. Packs follow the same schema as built-in ontologies:

```json
{
  "name": "Governance",
  "category": "civic",
  "domain": "GOVERN",
  "mode": "DECIDE",
  "axis_kind": "legal",
  "unit": "cases",
  "hints": ["statute", "precedent", "appeal"],
  "facets": {
    "sub_domain": "litigation",
    "sub_mode": "adjudicate",
    "lifecycle": "Operate",
    "agency": "Automated",
    "scale": "Organizational",
    "abstraction": "Concrete"
  }
}
```

### Pack Naming Convention
- File name: `<domain>_<mode>_ontology.json`, e.g., `govern_decide_ontology.json`
- Pack may contain many entries; each entry follows `OntologyEntry` schema.

### Adding a Pack
1. Create JSON file with domain/mode values outside the original 5×14 grid.
2. Ensure `facets` are used for any extra dimensions.
3. Drop the file in `physis-core/config/`.
4. The `OntologyLoader` automatically loads it via `include_str!` or `load_from_str` at runtime.
5. Users can then classify text and get hits in the new domain/mode cell, with facet filtering via `FacetFilter`.

### Example Pack: `govern_decide_ontology.json`
```json
{
  "domains": [
    {
      "name": "Court Ruling",
      "category": "legal",
      "domain": "GOVERN",
      "mode": "DECIDE",
      "axis_kind": "legal",
      "unit": "cases",
      "hints": ["judgment", "summary", "order"],
      "facets": {
        "lifecycle": "Operate",
        "agency": "Automated"
      }
    },
    {
      "name": "Contract Signing",
      "category": "legal",
      "domain": "GOVERN",
      "mode": "SIGN",
      "axis_kind": "legal",
      "unit": "agreements",
      "hints": ["signature", "parties", "seal"],
      "facets": {
        "lifecycle": "Build",
        "agency": "Self"
      }
    }
  ]
}
```

The pack can be loaded with:
```rust
let loader = OntologyLoader::load_from_str(&include_str!("../config/govern_decide_ontology.json")).unwrap();
```

---

## 2b. What to port from `physis-pro` (added 2026-09-09)

The boundary in the README holds — *Core stays Core* — so this is a short list,
and it is short on purpose. A pro module earns a place in core only if it is
**epistemic**, **dependency-light**, and **measured**.

### Port: the filing proposer

**The one positive the research track has produced.** Every certification
mechanism was refuted (E12–E18, all at chance on real misfilings), because they
asked the machine to *judge*. Asked instead to *propose* — narrow 57 cells to a
shortlist a person chooses from — the same geometry works:

| at 436 human decisions | top-1 | top-3 | top-5 |
|---|---:|---:|---:|
| **proposer** | 0.447 | **0.712** | 0.791 |
| permuted (the null) | 0.036 | 0.136 | 0.233 |
| `old_cell`→`new_cell` lookup | 0.184 | 0.368 | 0.474 |

Two pieces, both tiny:

1. **Hint weighting at α = 0.5** — `v = normalise(v_name + 0.5·(v_hints −
   v_name))`. One line, and worth **+0.054 top-3** over the current
   representation, unanimous across 24 paired splits, *t* = +12.13 (E24/E28).
   This is a change to how core already embeds entries and costs nothing.
2. **Top-k cell proposal from accumulated decisions** — nearest-centroid over
   cells built from filings a user has already confirmed. Pure vector arithmetic;
   no new dependency.

**Ship it as a proposer, never as a validator.** Standing rule: *coverage filters
and ranks; it does not verify.* The same applies here — the shortlist is for a
person to choose from, and core's own README notice is right that the audit
claim does not hold.

### Do not port

The industrial suite (MQTT/Modbus/OPC-UA, OEE, Gantt, shifts, Spanner, RBAC,
licensing, multi-tenancy, the operations console) — that is the boundary working
as designed. Also not the `mapper` research harness: it is experiment code, it
belongs with its data in the parent project, and core should carry the *result*
rather than the apparatus.

### Also: the README's status notice needs one correction

The notice says twelve mechanisms were tested and **none was shown to work**.
That was true when written and is now imprecise in one direction:

- **Unchanged and still correct:** nothing here certifies that a filing is
  *sound*. Every scorer sits at chance on real misfilings (cosine 0.519,
  CONSTRAINT 0.536, ANCESTRY 0.534), and ANCHORVOC's apparent win was an
  entry-length artifact.
- **New, and it should be stated:** *proposing* a filing, given decisions a
  person has already made, does work and survives a construction-matched null.

The notice's honesty is the reason to keep it accurate in both directions.

## 3. Getting Started Guide

See `GETTING_STARTED.md` for a step‑by‑step guide.

---

## 4. Stripe License Selling Guide

See `STRIPE_LICENSE_GUIDE.md` for integrating Stripe to sell physis-core licenses.

---

## 5. Pricing & Advertisement Ideas

### Pricing Tiers

| Tier | Price (monthly) | Features |
|------|-----------------|----------|
| **Indie** | $29 / mo | Single‑user license, deterministic embeddings, full CLI + studio access, standard ontology (33 domains). |
| **Pro** | $99 / mo | Single‑user + ONNX embedder support (`--features embed-onnx`), priority ontology pack updates, seat‑limited (up to 3 seats), quality‑tracker export/import. |
| **Enterprise** | $299 / mo | Unlimited seats, custom ontology packs, dedicated support, on‑premises deployment, API access for internal tooling, SLA on uptime. |

*All tiers include a 14‑day free trial (via Stripe trial mode). Discounts available for annual subscriptions (2 months free).*

### Advertisement Channels

1. **Rust Community** – post on r/rust, r/programming, Rust subreddit, Rust Lang weekly newsletter; highlight the zero‑model AI aspect.
2. **AI/ML Newsletters** – AI Weekly, The Sequence, Synced Review; emphasize deterministic embeddings for reproducible research.
3. **Product‑Ops & SRE Forums** – discuss incident classification, feedback loops; target audience for SRE/incident‑management tools.
4. **Design & UX Blogs** – articles on semiotic grids for sense‑making; target designers exploring mental‑model mapping.
5. **Content Marketing** – write a 3‑part series: (a) "Why deterministic embeddings matter", (b) "How the semiotic grid structures knowledge", (c) "Monetizing with Stripe: a developer's guide".
6. **Twitter / X Threads** – short demos: `physis-core classify "..."` with screenshots of the studio heatmap.
7. **Conference Sponsorship** – small booth or lightning talk at Rust Conf, Strange Loop, or AI/ML meet‑ups.
8. **Partner Integrations** – offer a simple SDK for Node.js/Python that wraps physis-core classification; cross‑promote.

### Referral Program

- Give existing customers a unique referral code.
- Upon a successful purchase by a new user, both get one month free (or a 20% discount on renewal).
- Track referrals via Stripe's metadata or your own database.

### Conversion Funnel

1. **Awareness** – tweet/demo + link to GETTING_STARTED.md.
2. **Interest** – visitor reads getting‑started, tries classification.
3. **Consideration** – opens studio, experiments with ontology packs.
4. **Decision** – clicks "Buy License" (Stripe Checkout).
5. **Retention** – periodic email with new packs, quality‑tracker tips, feature updates.

---
## 6. Marketing Plan (executed after the free‑sample launch)

### 6.1 Positioning & Tagline
- **Tagline:** “Deterministic semi‑grid classification, zero‑model AI, embed anywhere.”
- **Core message:** Offline, explainable, learn‑from‑feedback classification without third‑party APIs.

### 6.2 Target Segments (priority)
| Segment | Pain point | Why physis‑core |
|---|---|---|
| SRE / Incident response | Log‑snippet overload, need feedback loop | Offline, penalty/boost learning, no data export |
| Technical writers / Doc‑ops | Auto‑tag thousands of specs, need explainable tags | 5×14 grid + ontology packs, deterministic |
| Rust systems developers | Want tiny embeddable “brain”, no GPU/runtime | Model‑free random‑projection, compile‑anywhere |
| Product‑ops / SaaS founders | Need issue‑routing/tagging, avoid per‑token costs | Flat‑fee licence, unlimited calls |

### 6.3 Launch funnel
| Phase | Goal | Tactics | Owner | ETA |
|---|---|---|---|---|
| **Pre‑launch buzz** | Build waiting list & early interest | • Teaser tweet thread (deterministic AI) <br>• Posts on r/rust, r/programming <br>• Blog: “Why deterministic embeddings matter” | Founder/Marketing | Week 1‑2 |
| **Free‑sample release** | Get hands‑in‑the‑door usage | • Publish download at `praxisweb.xyz/physis/free-sample` <br>• “7‑day trial” CTA <br>• GitHub Discussions for Q&A | Founder | Week 3 |
| **Beta‑program** | Collect feedback, refine packs & UI | • Invite 10‑15 users (SRE, doc‑ops, Rust devs) <br>• Offer discounted annual licence for feedback <br>• Weekly sync calls | PM / Community lead | Week 4‑6 |
| **Commercial launch** | Convert trial → paid licence | • Stripe product page (Indie/Pro/Enterprise) <br>• Email drip “trial ending” + upgrade incentive <br>• LinkedIn / Twitter ads targeting SRE & dev‑tool handles | Founder / Sales | Week 7‑8 |
| **Post‑launch growth** | Expand reach, add packs | • Release first ontology pack (e.g., `govern_decide`) <br>• Partner with SRE tooling companies <br>• SEO for “semiotic grid classification” | Community lead | Month 2‑3 |

### 6.4 Messaging framework
| Message | Audience | Channel |
|---|---|---|
| “Run AI on your laptop, no GPU, no internet.” | Rust developers, SRE | Twitter, Reddit, Hacker News |
| “Classify support tickets, then teach the system from your mistakes.” | SRE, incident‑management | Blogs, webinars, conference talks |
| “Tag your technical docs with a 5×14 domain‑mode grid, no model files.” | Technical writers | Newsletters, Medium, Dev.to |
| “Predictable $29‑$299/mo licence – unlimited classifications.” | SaaS founders, product‑ops | Email outreach, LinkedIn ads |

### 6.5 Metrics to track
- **Download count** of the free sample.
- **Trial‑to‑paid conversion rate** (target 5‑10 %).
- **Churn** after first year.
- **Ontology‑pack adoption** (how many users add custom packs).
- **Community contributions** (new packs, bug reports).

### 6.6 Budget (first 3 months)
| Item | Cost (USD) |
|---|---|
| Paid ads (Twitter/Linke​In) | $200 |
| Conference sponsorship (small booth) | $500 |
| Design of trial key graphic | $150 |
| **Total** | **≈ $850** (plus variable Stripe fees) |

### 6.7 Success criteria (by month 3)
- ≥ 500 downloads of the free sample.
- ≥ 30 trial users who upgrade to a paid licence.
- At least one community‑contributed ontology pack merged into the repo.
- Positive feedback NPS > 30.

---
*This plan builds on the pricing, advertisement, and referral concepts already captured in `PLANNING.md`. All tasks are deliberately small enough to be completed in a single work‑day, allowing rapid iteration and early revenue.*

---

## 7. Epistemic revision track — Atlas + Graphiti (2026-09-09)

> Detail: `../docs/ATLAS_GRAPHITI_INTEGRATION_PLAN.md` (§§0–6). Background:
> `../docs/resources.md` §§9–11. Index with per-item gates; adds no domain or mode, touches neither §0 nor §1.

### 7.0 Principle (offline Rust/sled, no new deps)

Take patterns, protocols, test shapes — never dependencies. No graph DB, no LLM
service, no Python runtime, no phone-home: offline Rust over sled + in-process
embeddings. **§2b applies throughout: coverage filters and ranks; it does not
verify.** Anything claiming a filing is *sound* on geometry alone is BLOCKED by
E14–E18 (cosine 0.519, CONSTRAINT 0.536, ANCESTRY 0.534).

### 7.1 Atlas takes → core symbols (gate per item, in order)

- **A7 — no-perturbation invariant.** `delta_engine.rs` (`evaluate_mutation`,
  `classify_delta`) + `hypothesis.rs` (`recompute_fitness`): zero-shift wave
  leaves fitness bit-identical, emits no transition. Gate: `zero_shift_wave_leaves_fitness_untouched` passes before any other 7.x item.
- **A2 — DependsOn-only walk.** `relation.rs` (`TypedEdge`, `RelationType`) +
  `delta_engine.rs` (`find_neighbors`, `build_adjacency`, `dfs_collect`,
  `MAX_PROPAGATION_DEPTH=5`): declared DependsOn edges only, justification-hop
  counting, cycle recording; kills the GAMMA=0.85 arrival wave. Gate: `midchain_revision_revises_exact_dependents` (exact dependents, BFS order).
- **A6 — named weights + term breakdown.** `delta_engine.rs` (GAMMA,
  DEGRADATION_THRESHOLD, MIN_IMPACT, MAX_PROPAGATION_DEPTH) +
  `hypothesis.rs` (`recompute_fitness` 0.20/0.15/0.15/0.25/0.25 + penalties):
  every recompute names its terms. Gate: `fitness_recompute_reports_term_breakdown`; tuning as separate scored configs — GATED on frozen defaults.
- **A1 — Ripple reassessment pass.** `delta_engine.rs` (`evaluate_mutation`,
  `OntologyDeltaReport`) + `hypothesis.rs` (`transition_to`,
  `recompute_fitness`) + `epistemic.rs` (`FitnessShifted`, `StatusTransition`):
  dependents recomputed per step, proposals before writes. Gate: seeded chains
  where the walk must beat the cosine wave — GATED on that comparison, else the walk ships as a named flag.
- **A5 — adjudication routing.** `contradiction.rs` (`Contradiction::new`,
  `ResolutionStatus::Open`) + `epistemic.rs` (`ContradictionDetected`):
  small-Δ recomputes; large-Δ/contradiction/high-authority stays Open with
  rationale, explicit approve/reject/adjust/synthesize only. Gate: `large_delta_routes_to_open_with_rationale` (record before any queue surface).
- **A3 — AGM shape gate.** `hypothesis.rs` (`HypothesisStatus` ×9,
  `transition_to`) + `epistemic.rs` (`StatusTransition`): shape taxonomy as
  tests, no Cypher; K*4/K*6 weak-form. Gate: `agm_shape_gate` at published N/N
  AND each test fails on ≥1 naive implementation — GATED on that mutation check.
- **A4 — hash-chained provenance.** `provenance.rs` (`ProvenanceLink`,
  `ProvenanceChain::summary`) + `explanation.rs` (`provenance_chain`,
  `human_readable_summary`): content hash + previous-link hash per link; verify
  reports the first break with sequence coordinates. Gate: `ledger_tamper_breaks_chain_at_sequence`.

### 7.2 Graphiti takes → core symbols (gate per item, in order)

- **G7 — invalidate-don't-delete.** `hypothesis.rs` (`Superseded`) +
  `contradiction.rs` + `temporal.rs` (`valid_until`): superseded stays
  history-readable with timestamps, excluded from current queries. Gate:
  `superseded_items_stay_queryable_for_history`; retention/compaction stated first — GATED on that note.
  **Retention/compaction story (stated 2026-09-09, before data gets large):**
  nothing in physis-core deletes automatically — the same rule as telemetry,
  by architecture rather than config. Superseded/Failed hypotheses accumulate
  in sled until an explicit, separately-audited compaction entry point exists
  (deferred to P2, alongside G3/G6, so it can ride the audit trail instead of
  preceding it). When it lands, compaction may remove only items whose full
  revision history has been exported to the provenance ledger first (G4's
  intake ids), and every removal records its own ledger event. Until then the
  bounded-growth mechanism is `current_hypotheses` keeping live sets small —
  history grows, and that is stated, not hidden.
- **G1 — expired_at leg (fields only first).** `temporal.rs`
  (`TemporalValidity`, `permanent`/`from`/`during`, `is_valid_at`,
  `overlaps`): system invalidation leg + backfill-vs-arrival split. Gate:
  `temporal_triple_serialises`. Dead weight until queries filter on it — fields-only until G6.
- **G2 — (resolved, invalidated, new) ingest triple.** `delta_engine.rs`
  (`evaluate_mutation`, `OntologyMutation`, `OntologyDeltaReport`) +
  `hypothesis.rs` (`add_supporting_evidence`, `add_contradicting_evidence`):
  deterministic judgment (overlap+recency, every call logged) before fitness
  update. Gate: `ingest_returns_resolved_invalidated_new`; GATED on 50 hand-built pairs above chance — below that, flag-only, never auto-invalidates.
- **G3 — episode + watermark trail.** `epistemic.rs` (`EpistemicAuditTrail`,
  `EpistemicEvent`, `history_for`) + `history.rs`: ordered replayable intake
  with arrival-vs-assertion times + per-stream high-water mark. Gate: `late_evidence_replays_to_same_state`.
- **G4 — per-link intake ids.** `provenance.rs` + `explanation.rs`: every
  derived link cites intake-record ids; newest-N + first-seen cap stated, never
  silent. Gate: seeded-chain explanations cite intake ids, not source names.
- **G6 — point-in-time queries.** `epistemic.rs` (`reconstruct_status_at`) +
  `temporal.rs` (`is_valid_at`): "believed at T?" / "valid during [a,b]?" from
  the triple + trail; status encoding normalized here. Gate:
  `point_in_time_matches_audit_trail` — GATED on beating current `is_valid_at` filtering; parity means cut.
- **G5 — BM25+RRF beside cosine.** `rag.rs` (`TokenFixedRetriever`,
  `TokenCounter`) + `propose.rs`: RRF fusion, budget packing and
  propose-does-not-verify kept, cosine-only kept as baseline. Gate:
  `hybrid_fusion_vs_cosine_baseline` — GATED on a frozen-baseline probe-set
  comparison; a negative ships recorded (disabled), not failed. Any *verifies* claim is BLOCKED by E14–E18.

### 7.3 NOT taking (with reason per item)

- Graph DB substrate (Neo4j/FalkorDB/Cypher/APOC) — breaks the offline contract.
- LLM extraction/judging/reranking — nondeterministic, key-gated, the 0.519 failure mode; BLOCKED as a judge by §2b, allowed nowhere near verdicts.
- Telemetry default-on — no phone-home by architecture, not config.
- Suspect inference code (hyperresolution mutating nogoods; TODO-flagged nogood order) — author-flagged non-standard.
- Full AGM K* tail as proof; paid rerankers first; dead surface + BMB 1.000 — weak-form tests / portable BM25+RRF first / honest cells only.

### 7.4 Order P0/P1/P2

- **P0 (invariants + history):** A7, G7, G1-fields. No scoring change — the point.
  **DONE 2026-09-09** (`tests/epistemic_p0.rs`, all three gates green; G7's
  retention/compaction story stated in §7.2 before the gate).
- **P1 (revision + ingest):** A2, A6, G2, A5-rationale-only.
  **A2 + A6 + A5 DONE 2026-09-09** (`tests/epistemic_p1.rs`):
  `midchain_revision_revises_exact_dependents` (DependsOn-only BFS walk,
  cycles recorded, same-cell bystander excluded, logged breadth fallback for
  sparse graphs), `fitness_recompute_reports_term_breakdown` (frozen named
  weights, per-term contributions sum to the composite), and the A5 routing
  (`large_delta_routes_to_open_with_rationale`, `certified_is_core_protected`,
  `small_delta_auto_applies_with_decision_recorded`,
  `route_transition_boundary_table`): ϵ + `ADJUDICATION_STRATEGIC_FLOOR`
  (0.15, Atlas's number) splits AutoApply from StrategicReview; Certified is
  CoreProtected; every proposed demotion records an `AdjudicationDecision`
  with rationale and `ResolutionStatus::Open` on the report — rationale
  record only, no queue surface. **Remaining:** G2 (ingest triple — GATED on
  50 hand-built pairs above chance, else flag-only).
- **P2 (time, retrieval, gate):** G3, G4, G6, G5, A3, A4. Each item ships only
  behind its §7.1/§7.2 gate; a failed gate blocks the next item, never bends it.
  **G3 DONE 2026-09-09** (`tests/epistemic_p2.rs`): `EpistemicEvent` carries
  the two clocks — `asserted_at` (episode reference time) vs `timestamp`
  (arrival / transaction time, the T18 two-clock watermark folded in);
  `reconstruct_status_at` replays in **assertion order**, so out-of-order
  intake replays to the in-order state
  (`late_evidence_replays_to_same_state`); per-stream `HighWaterMark`s
  advance on both clocks and flag late episodes via
  `note_intake` → `IntakeReceipt` (`watermark_advances_and_flags_late_episodes`);
  pre-G3 trails replay unchanged, serde-default keeps old JSON
  deserialising (`legacy_trails_replay_by_arrival_unchanged`,
  `trail_serialises_with_g3_fields_and_reads_old_json`). **Remaining in
  P2:** G4, G6, G5, A3, A4. **A1 note:** unscheduled by design — its gate
  (beat the cosine wave on seeded chains) is open question Q1's probe;
  mechanics ship when the probe answers.

### 7.5 Falsifiers (no number without its null)

A1 fails if mid-chain accuracy does not beat the cosine wave. A5/G7 fail if
pairs are unrecoverable after retraction. G1/G6 fail if belief-at-T is no
better than `is_valid_at` filtering. G5 fails if fusion does not beat
cosine-only; A3 fails on tautologies; A4 fails if verify survives a payload
edit. Every claim carries its null (permuted-labels or cosine-only) — §2b's proposer (top-3 0.712 vs 0.136 null) is the model.

### 7.6 Formal semantics track — operator separation + provisional fixed point (2026-09-09)

> Source: `../physis_resources.txt` §"Formal semantics track" (Triune
> Continuum × Kripke × Physis). Framing for this section's machinery; plan
> detail in `../PLAN.md` Phase 23. Adds no domain or mode; touches neither
> §0 nor §1.

**The separation this section states:** semantic construction (Γ:
`embed` → `classify` → `propose`, additive, partial — never forces
TRUE/FALSE) is not epistemic revision (R: `transition_to` →
`recompute_fitness` → the epistemic trail), and neither is truth.
`delta_engine.rs` is the one place Γ and R still share a function
(`evaluate_mutation` both computes semantic proposals and derives status
transitions); 23.1 is the work of making that boundary explicit, not of
inventing new machinery.

**Symbol audit for `h = ⟨e, τ, π, σ, ρ, κ⟩`:** every component already has a
home in this crate — e = embedding; τ = `ontology_refs`/`cell_pin`;
π = `provenance.rs` + `Revision` (G4 closes it); σ = `HypothesisStatus` ×9 +
fitness; ρ = `revision_history` (G3 closes it); κ = `TemporalValidity`
(P0 added the `expired_at` leg; G6 closes it). The tuple is an audit lens,
not a new struct — no new type is planned for it.

**Standing non-claim:** this crate is not a Kripke truth theory. Its
"fixed point" is *locally stable under currently available evidence and
context* (PHYSIS_STEP stage 10: report `locally stable` vs `revised`),
never metaphysical truth — and distinct from E20's geometric fixed points
(density maxima in the similarity graph): same word, different object,
cross-reference only.

**P3 items, each behind its gate (order: after P2, which supplies the G3/G6
vocabulary these invariants need to be honest):**

- **23.1** `gamma_report_carries_no_status_writes` — semantic proposals ride
  the report; status changes derivable only through the R-side trail.
  Mutation-tested: a naive implementation that writes status in
  `evaluate_mutation` must fail it.
- **23.2** `interpretations_never_shrink_on_ingest` — `I_t ⊆ I_{t+1}`;
  deprecation is state/provenance (the Γ-side face of G7), never deletion.
- **23.3** `stable_status_matches_zero_delta_and_evidence_window` — the
  provisional fixed-point status; falsifier: claims stability while any
  κ-window evidence is unresolved.
- **23.4** replay candidates from Failed/Inert/Contradicted structures;
  proposals cite the ledger events that motivated them; nothing replayed
  is deleted (the dream hook).
- **23.5** the critical experiment (in `../PLAN.md` §22.5/23.3) — the paper.
