# Core/Pro split audit — `ilPez00/physis-core` at `d65858b`

Date: 2026-09-14. Scope: the `physis-core` checkout at
`/home/gio/dev/physis-pro/physis-core`, HEAD `d65858b` (207 commits,
clean tree except untracked runtime `.physis/`). Method: every `src/*.rs`
module classified by its module-header statement of purpose **and** its
actual callers (`crate::<module>::` references in `src/`, `examples/`,
`tests/` plus CLI subcommand wiring in `src/main.rs`). A component used
only by experiments is research even when it lives in `src/`; a component
that manages workspaces, agents, deployments or external integrations is
product even if it was born as an experiment.

Conventions: **core** = future open research kernel · **pro** = move to
private product · **research** = stays research-only, not Core API ·
**history** = evidence only, do not carry forward.

## A. Observation / evidence primitives — CORE

| component | current path | actual callers | class | rationale | action |
|---|---|---|---|---|---|
| observation log | `src/observe.rs` | `act.rs`, `ground.rs`, `system.rs`, `system_delegation.rs`, `bin/world.rs`, `main.rs`, system tests | core | append-only observations are the primitive every product surface and every experiment shares | keep minimal; product backends stay out |
| local JSON store | `src/store.rs` | `observe.rs`, `studio.rs`, `main.rs`, `bin/physis.rs`, `bin/world.rs` | core | file-local persistence, no backend, no account | keep; forbid networked backends here |
| epistemic audit trail | `src/epistemic.rs` | `core.rs`, `observe.rs`, `act.rs`, epistemic tests | core | replayable belief timeline = the ledger primitive | keep |
| hypothesis model | `src/hypothesis.rs` | `core.rs`, `act.rs`, `act_recall.rs`, `epistemic.rs`, `ground.rs`, `observe.rs`, 6 more | core | evidence for/against + status derivation, used by both research and graph | keep |
| provenance chains | `src/provenance.rs` | `core.rs`, hypothesis paths | core | evidence lineage | keep |
| temporal validity | `src/temporal.rs` | 5 in-src users | core | when-is-it-true intervals | keep; extend to valid-time windows per literature, do not add scheduling |
| contradiction representation | `src/contradiction.rs` | `core.rs`, `main.rs`, act paths | core | conflicting-claim representation + resolution status | keep |
| retrospective proposals | `src/dream.rs` | `core.rs` (`dream_over_history`) | core | replay over retired branches, write-gated by design | keep; the one primitive with no commodity twin |
| typed relations | `src/relation.rs` | `core.rs`, delta paths | core | DependsOn/causal/typed edges the midchain walk needs | keep |
| delta propagation | `src/delta_engine.rs` | `core.rs`, hypothesis paths | core | dependent-fact revision along declared edges | keep; ChainEdit-style ripple logic belongs here, not in product |
| coherence query | `src/coherence_query.rs`, `src/coherence_dimensions.rs` | `core.rs`, `explanation.rs`, `hypothesis.rs`, `transform.rs` | core | epistemic-question interface + graded profiles | keep |
| explanation | `src/explanation.rs` | `core.rs`, `coherence_query.rs` | core | structured why-reports with precedents | keep |
| process cycles | `src/process.rs` | `core.rs`, `studio_lab.rs` | core | process-cycle types are state, not orchestration | keep types; any scheduler built on them is pro |

## B. Retrieval primitives — CORE

| component | current path | actual callers | class | rationale | action |
|---|---|---|---|---|---|
| token-fixed RAG + BM25 | `src/rag.rs` | 8 in-src users incl. `system.rs` | core | deterministic ranking + hard budget = the commodity half, openly kept | keep; truncation-budget null (E-series follow-up) lands here |
| embedder trait + random projection | `src/embed.rs` | 18 in-src + 69 examples | core | pluggable interface + deterministic fallback | keep; ONNX loader stays optional |
| ONNX embedder | `src/embed_onnx.rs` | `main.rs`, 58 examples | core | local weights, no account | keep behind `embed-onnx` |
| n-gram tables | `src/ngram_table.rs`, `src/embed_ngram.rs` | `main.rs` (NGram cmd), model plumbing | core | count-table representations the track proved competitive | keep; license *metadata* fields (`license: String`) are data provenance, not a licence gate |
| tokenizer abstraction | `src/tokenizer.rs` | 4 in-src users | core | infrastructure | keep |
| model provider indirection | `src/model_provider.rs` | 4 in-src users | core | never hardcodes a model; `license` field is model-card metadata | keep |
| fixed-budget context compiler | `src/map.rs` (`compile_context`) | 4 in-src users | core | the measured 35% primitive | keep; add quality-at-budget null |
| structural map (repeat/difference) | `src/map.rs` (map half) | same file | core | deterministic structure, same-corpus ⇒ same hash | keep |
| semiotic classification | `src/classify.rs` | 7 in-src + 8 examples | core | grid scoring primitive (data stays research) | keep engine; grid *content* is research data |
| propose / coverage | `src/propose.rs`, `src/coverage.rs` | chain, discovery, 2 examples | core | the two primitives that beat construction-matched nulls (+0.531 top-3, *t* = +12.11) | keep |
| ontology loader | `src/ontology.rs` | classify, 37 examples | core | loader mechanics | keep loader; built-in grid JSONs are research data |
| ontology data | `config/*.json` (30+ ontologies) | loader, grid experiments | research | authorial content under test, not mechanism | publish as reproducibility data, not API |
| discovery / linkage / becoming | `src/discovery.rs`, `src/linkage.rs`, `src/becoming.rs` | chain, coverage, 1–2 examples each | research | measured but not shipped; becoming has an external benchmark pending | research; promote only on external win |
| transform / transplant | `src/transform.rs`, `src/transplant.rs` | 1 example each (`experiment53`) | research | null tests never run (RESCOPE gap) | research; F1 null design already specified |
| chain harness | `src/chain.rs` | `main.rs` (Chain cmd) | research | composes mechanisms with their nulls; the honesty instrument | keep as research harness, not product reporting |
| grid fitness | `src/grid_fitness.rs` | 3 examples | research | D-series instrument that *refuted* cells | research |
| coherence engine state | `src/core.rs` (`PhysisCore`) | 76 examples, studio, act paths | core | graph + branches + audit in one inspectable state | keep; note studio/process coupling to cut at the boundary |
| data models | `src/models.rs` | 27 in-src + 43 examples | core | nodes, entries, grid types | keep, prune product fields at split |
| quality feedback tracker | `src/quality.rs` | `studio.rs`, `main.rs` (Quality cmd), e2e | **pro** | only callers are product surfaces; operational learning across users is §IV.G product | move; keep the *measurement* (penalty math) as research note if published |

## C. Workspace / agent operating environment — PRO

| component | current path | actual callers | class | rationale | action |
|---|---|---|---|---|---|
| shared workspace interface | `src/system.rs` | `system_cli.rs`, `system_delegation.rs`, `system_mcp.rs`, `main.rs` (System cmd), tests | pro | human/agent shared workspace, folder projections, history windows — §IV.A product | move whole; BM25/observation reuse stays via core traits |
| workspace CLI + MCP | `src/system_cli.rs`, `src/system_mcp.rs` | `main.rs`, system tests | pro | agent delegation surface + MCP stdio adapter for the workspace service | move |
| delegation planning / reservations | `src/system_delegation.rs` | `main.rs`, `system_mcp.rs`, tests | pro | task reservation, worktree leases, dispatch review — operational coordination, not mechanism | move; the `observe`-backed record it writes stays readable via core |
| system TUI | `src/bin/system_tui.rs` | standalone binary | pro | mouse/keyboard workspace navigation = finished operational UI | move |
| act / act_recall | `src/act.rs`, `src/act_recall.rs` | `main.rs` (Act/ActRecall cmds), studio surfaces, worldstate | pro | action execution checked against belief = the agentic loop; read by product surfaces, never by research primitives | move; `PhysisCore` reads inside them become a core *reader* API, not a move of the graph |
| prediction / borrowed priors | `src/nlq.rs` (world layer only), predict paths in `main.rs` | `main.rs`, `bin/world.rs` | pro | machine-history prediction service + cross-workspace prior borrowing = §IV.G production service | move service; publish the predict-before-acting *measurement* as research |
| notebook grounded answers | `src/notebook.rs` | `main.rs` (Notebook cmd), oracle paths | pro | automatic organizational context compilation over a corpus = §IV product | move; BM25 + packing it uses stay core |
| big-model oracle leg | `src/oracle.rs` | `notebook.rs`, `bench.rs` | pro | hosted-model calls are provider integration, not mechanism | move; null-control *protocol* stays research |
| ground (common ground) | `src/ground.rs` | `main.rs` (Ground cmd) | pro | shared human/machine conceptual state is the workspace product's memory | move |

## D. Retrieval-adjacent surfaces — check each

| component | current path | actual callers | class | rationale | action |
|---|---|---|---|---|---|
| natural-language verb mapper | `src/nlq.rs` | **no callers** (measured cue parser for `physis-world` verbs, never wired) | research | measured but unwired; a front door is product when shipped | research until wired, then pro |
| run-pipeline config | `src/config_run.rs` | `main.rs` (Run cmd) | research | one configuration mechanism for the run pipeline | research; operational scheduler built on it is pro |
| behavioral bench | `src/bench.rs` | `main.rs` (Benchmark cmd), `act_recall.rs` | research | Directives 1+2 compression/structure benchmark harness | research |
| behavioral harnesses E67+ | `benchmarks/behaviour/` (pro repo) + `src/bench.rs` | bench paths | research | the E-series evidence trail | research; results JSONs are provenance artifacts |
| retrieval benchmark | `benchmarks/retrieval/` | docs cross-links | research | hit@5 0.57 vs 0.14 is grant evidence | keep public; safe to redistribute |
| ground-truth corpus | `benchmarks/ground-truth/` | chain tests | research | repeat/anomaly fixtures incl. the 35% measurement corpus | keep public; provenance artifact |

## E. Machine / fleet, UI, enterprise — PRO or absent

| component | current path | actual callers | class | rationale | action |
|---|---|---|---|---|---|
| worldstate navigational layer | `src/worldstate.rs`, `src/bin/world.rs` | 6 world experiments + the binary | research | E55 structure-before-mapping instrument; machine-log navigation is product when operated | research; an operated `physis-world` service is pro |
| studio web app + lab + communities | `src/studio.rs`, `src/studio_lab.rs`, `src/studio_communities.rs`, `src/studio/*`, axum `studio` feature | `main.rs` (Studio cmd), `/api/edition` | pro | finished dashboards, semiotics lab, explorer = §IV.C product UI | move; keep tiny diagnostic UIs only if a paper needs them |
| CLI aggregator | `src/main.rs` (2686 lines, 40+ subcommands), `src/bin/physis.rs` front door | everything | split | the CLI is the product's front door; the *research* CLI is `chain`, `classify`, embed, bench verbs | split: minimal research CLI stays, workspace/product verbs move |
| importers (browser/vault/git/praxis) | `src/history.rs`, `src/vault.rs`, `src/praxis.rs` | `studio.rs`, `main.rs` (Vault/History/Praxis cmds) | **pro** | connectors + personal ingestion pipelines; only callers are product surfaces (vault's one research caller, `history.rs`, is itself product-bound) | move; define connector *traits* in core at most |
| edition detection | `src/edition.rs` | `bin/physis.rs`, `studio.rs` | pro | Pro-upgrade path detection; carries the upgrade URL + Pro binary names | move; future core must not advertise a specific commercial product |
| licence machinery | `STRIPE_LICENSE_GUIDE.md`, `Cargo.toml` exclude, PLANNING.md §4 | none in code | pro/history | no licence *gate* exists in code (verified: `license` fields are model-card metadata); the guide sells a "commercial license" for Apache code | retire guide; fix docs per §VIII below |
| auth/RBAC/tenants/billing/SSO | — | absent | pro | nothing to move; record absence so nothing is invented later | n/a |
| SSH/fleet/systemd | — | absent from this repo (lives in Pro/Aion) | pro | record that the boundary already holds here | n/a |

## F. Research-only tracks — RESEARCH, not Core API

| component | current path | actual callers | class | rationale | action |
|---|---|---|---|---|---|
| E-series harnesses | `examples/experiment*.rs` (~45 files), `research/` E-docs | each other + reports | research | the apparatus, incl. negative results (E1 divergence, RH/number-theory, Fourier, impossible-machine, JEPA binding exp17) | keep as history + reproducibility; never API |
| semiotic grid research | grid D-series commits, `grid_fitness.rs`, mode inventory | 3 examples | research | the grid is the most revised artifact in the repo (D1 62% disputed) | research data + instruments |
| WordNet anchoring | `experiment22/23`, `wordnet-db` dev-dep | dev only | research | deliberately not a runtime dep | keep that way |
| structural machines | `src/machines.rs` | 1 example | research | declared-≠-called by its own header's rule | research; ProofStatus/refusal types may graduate to core only with a second caller |
| mapper exports | `examples/mapper_*.rs` | research pipeline | research | corpus construction history | research |
| generated corpora | `benchmarks/ground-truth/`, word-table artifacts | tests, benches | history | benchmark-construction history; keep for reproducibility | history, redistribute-safe subset |
| `direction.rs`, `claim_identity.rs` | `src/direction.rs`, `src/claim_identity.rs` | `main.rs` / `act_recall.rs` + 1 example | research | measured but product-called; promote only with independent callers | research |
| dead code | `src/experiments.rs` (no callers), `src/config_run.rs` (CLI only) | none | history | unreferenced | do not carry forward |

## G. Ambiguous — decided with reasons

| component | decision | reason |
|---|---|---|
| `src/core.rs` `PhysisCore` | **core, but narrowed** | the graph is the shared substrate; but its studio/process/act couplings are product edges — the split cuts edges, not the node store |
| `src/worldstate.rs` | research, not pro | it is an E-series instrument (E55) wearing a service binary; the *operated* service would be pro |
| `src/quality.rs` | pro | penalty math is publishable; the tracker with users and loops is operational learning |
| `src/process.rs` | core types, pro schedulers | types are state; anything that runs them on a schedule is product |
| `src/observe.rs` | core | used by pro surfaces but depended on by nothing product-specific; append-only log is the primitive |
| connectors (`history`/`vault`/`praxis`) | pro, not core | connectors' commercial value is maintenance; core keeps at most traits |
| `physis system` predict | pro service, open measurement | the borrowed-prior *store* is production (§IV.G); the predict-before-acting *protocol* is publishable |
| adaptive pipeline selection | **absent — keep it absent** | no auto-selection engine exists in this repo; the split must not create one by accident in core |

## H. Do-not-carry-forward list (future public Core must NOT include)

`system*.rs`, `system_tui`, `studio*.rs`, `worldstate` service binary,
`history`/`vault`/`praxis` importers, `notebook`, `oracle`, `ground`, `act*`,
`quality` tracker, `edition.rs`, `STRIPE_LICENSE_GUIDE.md`, `experiments.rs`,
`config/*.json` grid content (as API), generated corpora (as API), Stripe/
licensing docs, Pro upgrade URLs, `target/`, `.physis/` runtime store.

## I. Secrets / hygiene (verified 2026-09-14)

No API keys, tokens, or private keys found in tracked files (pattern scan for
`sk-live/test`, `ghp_`, `github_pat_`, AWS keys, PEM private keys: clean).
Untracked: `.physis/system/observations.jsonl` (local runtime store — must
not be committed; add to `.gitignore`). `target/` is ignored. `config/*.bak`
is excluded from the published crate.
