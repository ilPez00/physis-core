# Experiment 64 Spec — Median Estimator, Wider kmax (N=1e6)

Status: SPEC FROZEN 2026-09-20, NOT RUN. Companion to E62/E63.
E62 (N=1e6, kmax=40) and E63 (N=1e7, kmax=40) both PASS G3 with the
median estimator (1.395/1.451 and 1.459/1.470 vs 1.32032±0.15). Both
CIs cover 1.32032 comfortably, so the point estimate is robust but
imprecise. E64 asks whether the precision limiter is the bin count.

## Objective

Run the median estimator at N=1e6 with kmax=80 (40 even bins instead
of 20). If the fit tightens toward 1.32032 and the CI narrows, the
fixed 20-bin set was the precision bottleneck and the constant is
confirmed at higher precision. If nothing changes, the median over
even k is kmax-insensitive and a different approach is needed.

## Frozen (unchanged from E62)

- Sieve, seed 20260912, splits discovery (20, N/2] / holdout (N/2, N].
- R(k) = N(k)/E(k), E(k) geometric-density independence expectation.
- S_unit(k) = prod_{odd p|k} (p-1)/(p-2), S_unit(2)=1.
- Estimator: median of R(k)/S_unit(k) over even k in {2..kmax}.
- Nulls: N1 Cramer + N2 geometric (seed 20260912).
- Gates G1/G2/G3: identical thresholds to E62 (G3 band ±0.15).
- 1/2 never in generation, scoring, or ranking.

## Only change from E62

- kmax: 40 → 80. Everything else byte-identical
  (sieve, n_max, seed, gates, output schema).

## Expected outcomes (all valid, pre-registered)

| fit at N=1e6, kmax=80 | reading |
|---|---|
| inside ±0.15, closer than E62's 1.395 | AGREEMENT — precision improved; bin count was the limiter |
| inside ±0.15, same as E62 | AGREEMENT — kmax-insensitive; median stable, precision capped |
| 1.40–1.47, wider CI | PARTIAL — more bins add variance; needs N=1e7 at kmax=80 |
| outside ±0.15 | MISS — wider kmax destabilises; revert to kmax=40 |

## Statistics

- Bootstrap CI for the median (B=200, LCG stream).
- Report discovery/hold-out separately; do not pool across N.
- Report Δ vs E62 (kmax=40) so the kmax-dependence is visible.

## Artifacts (to be produced on run; absent now)

- `experiment64_median_kmax80.rs` example.
- JSON output + `EXPERIMENT64_RESULTS.md`.
- Run: `cargo run -p physis-core --release --example experiment64_median_kmax80`.

Status repeated: SPEC FROZEN, NOT RUN. Nothing above is evidence.