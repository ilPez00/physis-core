# Core austerity audit — capability by capability

Date 2026-09-14, against `d65858b`. Rule applied per capability:

> Does this exist primarily to make a scientific/mechanical claim inspectable
> and reproducible? If yes → Core. If its value comes from convenience, UX,
> automation, persistence, integration, orchestration, deployment, multi-user
> operation or commercial workflow → Product/Pro.

Nothing is deleted. Every "move" below is a physical relocation into the
product repository (`physis-pro`), which already depends on Core — one
implementation, no forks.

| capability | keep core | move pro | research only | delete | reason |
|---|---:|---:|---:|---:|---|
| observation record (`observe`) | ✅ | | | | append-only primitive every claim rests on |
| hypothesis / evidence / contradiction / status | ✅ | | | | central epistemic primitives |
| temporal validity + supersession + replay | ✅ | | | | order and validity are Physis's claim |
| provenance chains | ✅ | | | | "where did this come from" must always answer |
| delta propagation over typed relations | ✅ | | | | revision of dependents is the mechanism |
| retrospective dream proposals | ✅ | | | | write-gated replay; no commodity twin |
| deterministic serialization / `PhysisCore` state | ✅ | | | | inspectable engine state |
| BM25 + deterministic ranking + tie-breaking | ✅ | | | | measured primitive |
| fixed-budget context packing (`compile_context`) | ✅ | | | | the 35% claim lives here |
| structural map (repeats / differences) | ✅ | | | | deterministic, same corpus ⇒ same hash |
| embedder trait + random-projection fallback | ✅ | | | | offline default; explicit, not magic |
| optional local ONNX embedder | ✅ | | | | local weights only, feature-gated |
| count / n-gram tables | ✅ | | | | research-justified cheap representation |
| semiotic classification engine | ✅ | | | | scoring mechanism (grid data is research) |
| propose / coverage | ✅ | | | | the two primitives that beat matched nulls |
| chain harness + matched nulls + NOT MEASURED | ✅ | | | | capability to prove Core wrong |
| benchmark fixtures / controls / result provenance | ✅ | | | | reproducibility is the product of Core |
| minimal CLI (`map`, `retrieve`, `pack`, `benchmark`, `demo`, epistemic verbs) | ✅ | | | | one help screen |
| local JSON store for observations | ✅ | | | | file-local, no backend, no account |
| studio / GUI / dashboards | | ✅ | | | finished UI is product |
| TUI workspace navigator | | ✅ | | | polished interaction |
| MCP workspace server | | ✅ | | | protocol access ≠ scientific necessity |
| workspace interface + delegation + reservations | | ✅ | | | human/agent coordination is operation |
| action execution (`act`) | | ✅ | | | Core is not a shell-execution platform |
| watchers (fs/terminal/browser/process/agent) | | ✅ | | | continuous collection is integration |
| notebook (synthesised answers, draft+fill) | | ✅ | | | application workflow |
| oracle (hosted big-model leg) | | ✅ | | | provider integration (research *protocol* stays as docs) |
| quality tracker with user loops | | ✅ | | | operational learning; penalty math stays measurable |
| importers: browser history / vault / praxis | | ✅ | | | connectors' value is maintenance |
| edition detection / Pro upgrade path | | ✅ | | | commercial routing |
| prediction service + borrowed priors (workspace path) | | ✅ | | | production prediction (§IV.G) |
| worldstate navigational service (`physis-world`) | | ✅ | | | E55 apparatus stays in research; operated service is product |
| NLQ verb mapper (unwired) | | | ✅ | | measured but never wired; research until shipped |
| E-series experiment harnesses | | | ✅ | | apparatus, incl. negatives |
| semiotic grid discovery / axis experiments | | | ✅ | | data under revision, not API |
| WordNet anchoring experiments | | | ✅ | | dev-dependency only, by design |
| transform / transplant (null tests unrun) | | | ✅ | | research until a null says otherwise |
| impossible-machine / RH / Fourier probes | | | ✅ | | research history |
| generated corpora + benchmark construction history | | | ✅ | | reproducibility artifacts |
| `experiments.rs` (zero callers) | | | | ✅ | dead runtime surface, not evidence |
| `STRIPE_LICENSE_GUIDE.md` | | ✅ (history) | | | misleading for Apache Core; relocated, not deleted |
| `target/`, `.physis/` runtime store, `*.bak` | | | | ✅ | artifacts, never sources |

## Metrics (measured at `d65858b`, before the move)

- stable runtime LOC: **35,968** across `src/*.rs`
- direct non-optional dependencies: **8** (serde, serde_json, chrono, uuid, anyhow, rand, sha2, libc)
- optional dependencies: **8** (clap, axum, tokio, ureq, ratatui, crossterm, ort, tokenizers)
- CLI subcommands: **~45** (plus nested groups)
- default features: `cli`, `studio`

Targets after the move: default features `cli` (+ optional `embed-onnx`); no
HTTP server, TUI or GUI dependency in Core; CLI that fits one help screen;
zero product orchestration code in the stable library.
