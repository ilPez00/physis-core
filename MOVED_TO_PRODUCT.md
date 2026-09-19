# Moved to product

Every item below **left Core by relocation, not deletion**. Destination:
`physis-pro` (the product repository, which depends on `physis-core`). Where a
thing is listed as *history*, it was moved to the product repo's history area
rather than kept in Core.

Why this file exists: so that nobody re-adds product features to Core six
months from now "helpfully".

| feature | source (Core) | destination (Product) | note |
|---|---|---|---|
| Studio GUI + lab + communities | `src/studio.rs`, `src/studio_lab.rs`, `src/studio_communities.rs`, `src/studio/*` | `physis-pro/src/ported/studio*` | JSON/CSV/DOT output replaces visualisation in Core |
| Workspace service | `src/system.rs`, `src/system_cli.rs` | `physis-pro/src/ported/system*` | human/agent shared state |
| Delegation + reservations | `src/system_delegation.rs` | `physis-pro/src/ported/system_delegation.rs` | planning, leases, dispatch |
| MCP workspace adapter | `src/system_mcp.rs` | `physis-pro/src/ported/system_mcp.rs` | protocol access stays in product |
| TUI navigator | `src/bin/system_tui.rs` | `physis-pro/src/ported/tui.rs` | polished interaction |
| Action execution | `src/act.rs` | `physis-pro/src/ported/act.rs` | Core keeps the belief/ledger side it reads |
| Action recall harness (product path) | `src/act_recall.rs` | `physis-pro/src/ported/act_recall.rs` | measurement writeup stays in research |
| Common ground | `src/ground.rs` | `physis-pro/src/ported/ground.rs` | shared human/machine memory |
| Notebook | `src/notebook.rs` | `physis-pro/src/ported/notebook.rs` | Core keeps `retrieve`/`pack`/`sources` |
| Big-model oracle leg | `src/oracle.rs` | `physis-pro/src/ported/oracle.rs` | provider integration |
| Quality tracker | `src/quality.rs` | `physis-pro/src/ported/quality.rs` | operational learning; penalty math documented as measurement |
| Importers | `src/history.rs`, `src/vault.rs`, `src/praxis.rs` | `physis-pro/src/ported/{history,vault,praxis}.rs` | connectors: Core may define traits at most |
| Edition / upgrade routing | `src/edition.rs` | `physis-pro/src/ported/edition.rs` | commercial routing belongs to product |
| Watchers | `Watch`/`Observed` CLI + `src/observe.rs` collection paths | `physis-pro/src/ported/` (with `system`) | Core keeps `Observation` type + log format |
| World-state navigator | `src/worldstate.rs`, `src/bin/world.rs` | `physis-pro/src/ported/worldstate.rs` | E55 apparatus remains in `research/` |
| NLQ verb mapper | `src/nlq.rs` | `physis-pro/src/ported/nlq.rs` | unwired; research note retained |
| Run-pipeline config | `src/config_run.rs` | `physis-pro/src/ported/config_run.rs` | product configuration machinery |
| Product CLI verbs | `src/main.rs` (`System`, `Vault`, `History`, `Praxis`, `Quality`, `Watch`, `Observed`, `Act`, `Ground`, `Notebook`, `Studio`, `ActRecall`, notes group) | `physis-pro/src/ported/*` (no separate `cli.rs`; each verb dispatches from `physis-pro/src/main.rs`'s own `Commands` enum) | **corrected 2026-09-15, verified by grep, not assumed:** only `System` (`ported::system_cli::SystemArgs`) and `Notebook` (`ported::notebook::answer`) actually call their moved module. `Vault`/`History`/`Praxis`/`Studio` are live CLI commands but still call Pro's pre-existing `PhysisApp::run_{vault_import,history_import,praxis_backfill,studio}`, not `ported::{vault,history,praxis,studio}` — those moved files build but are not yet the code path. `Quality` is live but calls the pre-existing 152-line `physis-pro/src/quality.rs` facade, not the 833-line moved `ported::quality.rs` (only reached today via `tests/ported_quality_penalty.rs`, not from any CLI command) — this is a real duplicate-implementation risk, not just an unrouted file: two `QualityTracker`s exist, "one implementation, no fork" is not yet true for it. `Act`, `ActRecall`, `Ground` have **no** `physis-pro` CLI command at all. `Ground` is fully unrouted (zero references anywhere in `src/`, `tests/`, `examples/`). `Act`/`ActRecall` are reachable only via `cargo run --example {floor_spread,claim_identity}`, not from the product CLI or `tests/` — see "Declared but not yet routed" below for the corrected, examples/-inclusive grep (an earlier pass of this note checked only `src/main.rs` and `tests/`, missed `examples/`, and is corrected there rather than silently left). `Observed` is not and was never meant to be a `physis-pro` CLI verb — it is `physis-core`'s own `observed` subcommand (added 2026-09-15, this integration session, reading the primitives this file already says Core keeps: `observe::log_path`/`read`/`by_source`); this row's grouping of it with the Pro verbs is itself the documentation error being corrected here. See "Declared but not yet routed" below for the full list including `tui`/`world`. |
| Product binaries | `src/bin/physis.rs` front-door product paths, `physis-world` | `physis-pro` binaries | Core ships a minimal research CLI |
| Stripe licence guide | `STRIPE_LICENSE_GUIDE.md` | `physis-pro/docs/history/STRIPE_LICENSE_GUIDE.md` | it sold a licence for Apache code; retained as history with a correcting header |

## Deliberately NOT moved (they make the claims reproducible)

BM25, deterministic ranking, fixed-budget packing, structural map, structural
hash, propose/coverage, matched nulls, permutation controls, NOT MEASURED
reporting, observation/hypothesis/contradiction/temporal/provenance/delta
primitives, embedder trait + deterministic fallback, count tables, chain
harness, benchmark fixtures and result provenance.

## Declared but not yet routed (2026-09-15, verified by grep + `cargo check --lib`)

The split relocated these files into `physis-pro/src/ported/`; relocation is
not the same claim as integration, and the two should not be conflated. As of
this integration session, `physis-pro`'s live command dispatch does **not**
reach the following — reported so the gap is explicit rather than implied by
a destination column that just says the file exists:

- **`tui.rs`, `world.rs`** — zero callers anywhere; `cargo check --lib`
  itself flags every private item in both (`App`, `World`, `Screen`, `Clip`,
  their `main()` entry points, etc.) as dead code. Fully unrouted, not
  reachable from any binary.
- **`ground.rs`** — zero references anywhere in `src/`, `tests/` or
  `examples/` (grepped for `ported::ground`). No `cargo check` warning only
  because its items are `pub` (library-surface items aren't flagged dead by
  default) — pub-and-unused is not the same as wired-and-working.
- **`act.rs`, `act_recall.rs`** — **corrected 2026-09-15, second pass:** an
  earlier draft of this note said these had "zero references" the same as
  `ground.rs`; that checked only `src/main.rs` and `tests/` and missed
  `examples/`, which is exactly the scope-is-the-false-positive-source trap
  the `physis` skill's own liveness check (`physis-check calls`/`sweep`)
  warns about, so the correction is recorded rather than left standing.
  `act_recall.rs` is called from two real, `cargo run --example`-able
  entrypoints (`examples/floor_spread.rs`, `examples/claim_identity.rs`), and
  `act.rs` from those same examples plus internally from `act_recall.rs`.
  Neither is reachable from the product CLI (`src/main.rs`) or exercised by
  `tests/` — but "reachable only from an example binary, not the product" is
  a materially different, narrower gap than "fully unrouted, not reachable
  from any binary" (`tui.rs`/`world.rs`'s actual status), and conflating the
  two would itself be the kind of imprecise claim this file exists to avoid.
- **`vault.rs`, `history.rs`, `praxis.rs`, `studio.rs`, `quality.rs`** — each
  has a live `physis-pro` CLI command (`Vault`, `History`, `Praxis`,
  `Studio`, `Quality`), but every one of those commands calls a *different*,
  pre-existing Pro implementation (`PhysisApp::run_vault_import` /
  `run_history_import` / `run_praxis_backfill` / `run_studio`, and the
  152-line `physis-pro/src/quality.rs` facade), not the moved file. The moved
  copies compile and (for `quality.rs` only) are exercised by one test
  (`tests/ported_quality_penalty.rs`), but none is the code path a user
  actually invokes. This is the specific case this file warns against in its
  own opening line ("so that nobody re-adds product features to Core six
  months from now") inverted: the risk here is two live product
  implementations of the same feature, not a re-added one in Core.

Not fixed in this session — out of the bounded integration-gate scope that
produced this correction; recorded so it is not silently treated as done.

## Assistant-facing surfaces intentionally removed from Core

MCP server, TUI, GUI, delegation, watchers, action execution, notebook,
prediction service, cross-workspace prior store, connectors. A Core user gets a
library, a small CLI and JSON — not an assistant.

## Routing completed (2026-09-16, verified by running the binaries)

The 2026-09-15 note above is left standing as the record of what was true that
day. This section is what changed, and how it was checked — by executing the
product's own binaries, not by grepping a file for a destination column.

`physis-pro` now dispatches to the moved module for every verb that has one:

| moved module | `physis-pro` route | how it was checked |
|---|---|---|
| `ported::ground` | `physis-pro ground [--json]` | `tests/ported_routes.rs`, empty ledger and JSON key-set |
| `ported::act_recall` | `physis-pro act-recall [--top --seed --json]` | same suite; artifact written to `benchmarks/results/act-recall.json` |
| `ported::nlq` | `physis-pro nlq <question> [--json]` | same suite; the emitted command names `physis-pro-world` |
| `ported::config_run` | `physis-pro run --config <json>` | same suite; pipeline banner + the config's own corpus/query |
| `ported::world` | `physis-pro-world` (**binary**) | same suite; `observe` leaves the tree unchanged, `inspect` reads the sandboxed log |
| `ported::tui` | `physis-pro-system-tui` (**binary**, `tui` feature) | same suite; `--help` without a TTY |
| `ported::system_cli`, `ported::notebook`, `ported::act`, `ported::studio` | `system` / `notebook` / `act` / `studio-server` | already routed on 2026-09-15; `system`, `notebook`, `act`, `studio-server` suites |

`world` and `tui` are binaries rather than verbs because that is what Core
shipped (`physis-world`, `physis-system-tui`); their `main()`s moved into
`ported/` intact and stayed private and unreachable until two one-line shims
(`src/bin/world.rs`, `src/bin/system_tui.rs`) called them, exactly as
`src/bin/physis.rs` calls `front_door::main`. No licence gate was added to
either: Core had none, and adding one would be a new product decision rather
than a relocation. That decision is recorded here so it is reviewable.

Feature gating (Day 1 item 3) was also completed: `ported::{studio,
studio_communities, studio_lab}` are behind `feature = "web"`, `ported::tui`
behind `tui`, and `ported::system_cli` behind `cli`. Before that,
`cargo check --no-default-features --lib` in `physis-pro` failed on `axum`,
`crossterm` and `clap` imports from moved files; it now compiles with zero
errors. The product library had been unbuildable without its default features
since the split, and the moved modules were the cause.

### Still a duplicate, deliberately not switched

`vault.rs`, `history.rs` and `praxis.rs` remain two implementations each: the
moved copies under `ported/`, and `physis-pro`'s own pre-existing importers
(`src/{vault,history,praxis}.rs`) that back the live `vault` / `history` /
`praxis` CLI commands and their `POST /api/v1/*/import` routes. They are not
byte-identical — both sides carry documentation the other lacks — and they feed
different stores (Pro's scanner/`PhysisApp` graph versus the Core store the
moved studio reads). Switching the live commands to the moved copies would move
data paths, which the plan forbids for this gate, and merging them blind would
pick a winner without a parity test. The moved copies are *reachable* rather
than orphaned: `ported::history` and `ported::praxis` are called from
`ported::studio` (`studio-server`), and `ported::vault` from both of those.

`quality.rs` is no longer a duplicate: `physis-pro/src/quality.rs` is a thin
facade over `ported::quality::QualityTracker` (a newtype `Deref`, plus the
Pro-side `VectorEmbed` adapter), so the engine has one implementation. The
2026-09-15 note calling it a real duplicate-implementation risk is now stale in
that one respect and was checked by reading the facade, not assumed.

