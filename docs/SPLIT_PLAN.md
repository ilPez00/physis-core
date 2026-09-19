# Split plan — exact mappings

Source of truth for classification: `docs/CORE_PRO_SPLIT_AUDIT.md` (call-site
verified at `d65858b`). Nothing here creates public replacements; prepare
locally/private first.

## PRIVATE: `physis/` (product application)

Product application, orchestration, connectors, organization/workspace layer,
agent execution, adaptive strategy selection, UI, hosted service, enterprise
functionality.

Receives from this repo: `src/system*.rs`, `src/bin/system_tui.rs`,
`src/act*.rs`, `src/ground.rs`, `src/notebook.rs`, `src/oracle.rs`,
`src/quality.rs` (tracker), `src/history.rs`, `src/vault.rs`, `src/praxis.rs`,
`src/edition.rs`, workspace/product CLI verbs from `src/main.rs`,
`src/bin/physis.rs` front-door product paths. `PhysisCore` graph and
`observe` log are *referenced*, not forked — the product depends on the
kernel's reader API.

## PRIVATE: `physis-pro/` (enterprise extensions)

Enterprise extensions, deployment, industrial integrations, authentication,
multi-tenancy, scaling, monitoring, premium connectors. Already exists as the
superproject here; receives the enterprise-absent record (auth/RBAC/tenants/
billing/SSO are greenfield, not moves).

## FUTURE PUBLIC: `physis-core/` (minimal Rust research kernel)

Observation/evidence primitives (`observe`, `epistemic`, `hypothesis`,
`provenance`, `temporal`, `contradiction`, `dream`, `relation`,
`delta_engine`, `coherence_query`, `explanation`, `process` types); retrieval
primitives (`rag`, `embed` trait + fallback, optional `embed_onnx`,
`ngram_table`, `tokenizer`, `model_provider`, `map` compiler + structural map,
`classify` engine, `propose`, `coverage`, `ontology` loader, `chain` as
harness); narrowed `PhysisCore` graph + `models`; minimal research CLI
(`chain`, `classify`, embed, bench verbs) with machine-readable JSON.
Research data (grid JSONs, corpora) ships as data, not API. Connector traits
at most — no maintained connectors.

## OPTIONALLY PUBLIC: `physis-research/`

Experiment writeups (E-series reports incl. negatives), benchmark harnesses
(`chain`, `bench`, conatus-style fixtures with nulls), redistribution-safe
datasets (`benchmarks/retrieval`, ground-truth subset), reproducibility
artifacts (`structure_hash` protocol, result JSONs), matched-null helpers and
known-bad control arms.

## Existing repository: `ilPez00/physis-core` → retired historical repository

Final commit adds only retirement/protection documentation (see section X).
No history rewrite. Tags `v0.1.13`–`v0.1.24` retained. Remote branches
(`agent/PH-019`–`PH-022`, `epistemic-revision-p0`, `fix/mmr-starves-budget`,
`research/*`) retained for provenance.

## Deliberately NOT copied into the future public Core

See audit §H. In one line: the workspace product, the studio, the importers,
the prediction/prior service, the quality tracker, licensing/packaging docs,
Pro upgrade paths, generated corpora as API, grid content as API, runtime
stores, build artifacts.

## Grant-safe surface vs commercial moat

Grant-safe: benchmark datasets, retrieval/structural experiments,
corpus-property measurement, deterministic representations, provenance/replay
mechanisms, reference implementations, reproducibility infrastructure.
Commercial moat: continuous ingestion, connectors, auto-profiling and
pipeline selection, organization memory, shared workspace, hosted service,
managed deployment, enterprise search, temporal org memory, multi-user
provenance, workflow automation, agent execution, dashboards, scale,
monitoring, industrial adapters, support, security, compliance, private
deployment, SLAs.
