# Experiment 63 Spec — Median Estimator at N=1e7 (precision check)

Status: SPEC FROZEN 2026-09-20, NOT RUN. Companion to E62.
E62 (N=1e6) AGREEMENT: the median of R(k)/S_unit(k) recovered 2C2
(1.395/1.451 vs 1.32032±0.15) where the unweighted mean missed
(1.470/1.487). E63 asks whether that agreement holds at N=1e7.

## Objective

Run E62's median estimator at N=1e7. The E62 CI [1.279, 1.660]
comfortably covers 1.32032, so this is a **precision check**, not a
direction check — the direction and value were already established at
N=1e6. If the N=1e7 fit stays inside ±0.15 the agreement is
confirmed at higher resolution; if it drifts out, the estimator is
imprecise and a wider kmax or larger N is needed.

## Frozen (unchanged from E62)

- Estimator: median of R(k)/S_unit(k) over even k in {2..40}.
- R(k) = N(k)/E(k), E(k) geometric-density independence expectation.
- S_unit(k) = prod_{odd p|k} (p-1)/(p-2), S_unit(2)=1.
- Splits: discovery (20, N/2], held-out (N/2, N].
- Nulls: N1 Cramer + N2 geometric (seed 20260912).
- Gates G1/G2/G3: identical thresholds to E62 (G3 band ±0.15).
- 1/2 never in generation, scoring, or ranking.

## Only change from E62

- n_max: 1_000_000 → 10_000_000. Everything else byte-identical
  (sieve, kmax, seed, gates, output schema).

## Expected outcomes (all valid, pre-registered)

| fit at N=1e7 | reading |
|---|---|
| inside ±0.15 of 1.32032 | AGREEMENT — confirmed at higher resolution |
| 1.35–1.45, moved from E62 | PARTIAL — estimator stable but imprecise; needs wider kmax |
| ~1.395, same as E62 | AGREEMENT — no N-dependence; estimator converged |
| outside band, drifted from E62 | MISS — agreement was a low-N artifact |

## Statistics

- Bootstrap CI for the median (B=200, LCG stream).
- Report discovery/hold-out separately; do not pool across N.
- Report Δ vs E62 (N=1e6) so the N-dependence is visible, not hidden.

## Artifacts (to be produced on run; absent now)

- `experiment63_median_n1e7.rs` example.
- JSON output + `EXPERIMENT63_RESULTS.md`.
- Run: `cargo run -p physis-core --release --example experiment63_median_n1e7`.

Status repeated: SPEC FROZEN, NOT RUN. Nothing above is evidence.