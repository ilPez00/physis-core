# Experiment 61 Spec — Convergence Test (N=1e7)

Status: SPEC FROZEN 2026-09-20, NOT RUN. Companion to E60.
E60 (N=1e6) returned PARTIAL: G1+G2 pass, G3 miss (fit 1.47/1.49 vs
1.32032±0.15). This spec asks the only remaining question with the same
estimator: is the G3 miss a finite-range normalisation artifact, or a
systematic estimator bias?

## Objective

Run E60's exact estimator at N=1e7. If the fitted 2C₂ moves toward
1.32032, the miss is a finite-N artifact and the Hardy-Littlewood
constant is confirmed at higher resolution. If it stays ~1.47, the
estimator is biased and a different fit is needed.

## Frozen (unchanged from E60)

- Estimator: 2C₂_hat = mean over even k in {2..40} of R(k)/S_unit(k),
  where S_unit(k) = prod_{odd p|k} (p-1)/(p-2), S_unit(2)=1.
- R(k) = N(k)/E(k), E(k) = geometric-density independence expectation
  (same E52/E60 definition).
- Splits: discovery (20, N/2], held-out (N/2, N].
- Nulls: N1 Cramer + N2 geometric (seed 20260912).
- Gates G1/G2/G3: identical thresholds to E60 (G3 band ±0.15).
- 1/2 never in generation, scoring, or ranking.

## Only change from E60

- n_max: 1_000_000 → 10_000_000. Everything else byte-identical
  (sieve, kmax, seed, gates, output schema).

## Expected outcomes (all valid, pre-registered)

| fit at N=1e7 | reading |
|---|---|
| inside ±0.15 of 1.32032 | AGREEMENT — finite-N artifact confirmed, H-L constant verified |
| 1.35–1.45, moved from 1.47 | PARTIAL — converging, needs N=1e8 |
| ~1.47, unchanged | MISS — estimator biased (1/ln p normalisation not the cause); different fit required |

## Statistics

- Same bootstrap CI as E60 (B=200, LCG stream).
- Report discovery/hold-out separately; do not pool across N.

## Artifacts (to be produced on run; absent now)

- `experiment61_convergence.rs` example.
- JSON output + `EXPERIMENT61_RESULTS.md`.
- Run: `cargo run -p physis-core --release --example experiment61_convergence`.

Status repeated: SPEC FROZEN, NOT RUN. Nothing above is evidence.