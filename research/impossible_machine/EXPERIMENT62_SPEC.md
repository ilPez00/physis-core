# Experiment 62 Spec — Estimator Search (N=1e6)

Status: SPEC FROZEN 2026-09-20, NOT RUN. Companion to E60/E61.
E60 (N=1e6) and E61 (N=1e7) both MISS G3 with the unweighted mean
estimator, and the fit moves AWAY from 2C2 with N (1.470 -> 1.498).
E60's own results doc names the next step: the fit overshoots because
the estimator is an unweighted mean over 20 fixed k-bins, and the
large-k bins are thin. E62 asks whether a different estimator on the
SAME data recovers 2C2 — isolating estimator from N.

## Objective

Fit 2C2 at N=1e6 with three estimators on identical data, and report
all three. If any lands in 1.32032±0.15, the estimator was the problem
and the Hardy-Littlewood constant is quantitatively confirmed.

## Frozen (unchanged from E60)

- Sieve, kmax=40, seed 20260912, splits discovery (20, N/2] / holdout (N/2, N].
- R(k) = N(k)/E(k), E(k) geometric-density independence expectation.
- S_unit(k) = prod_{odd p|k} (p-1)/(p-2), S_unit(2)=1.
- Nulls: N1 Cramer + N2 geometric (seed 20260912).
- Gates G1/G2/G3: identical thresholds to E60 (G3 band ±0.15).
- 1/2 never in generation, scoring, or ranking.

## Estimators (all three reported)

1. **Unweighted mean** (E60/E61 baseline): 2C2_hat = mean_{even k in 2..40} R(k)/S_unit(k).
2. **E(k)-weighted pooled** (E62a): 2C2_hat = sum N(k)/S_unit(k) / sum E(k).
3. **Median** (E62b): median_{even k in 2..40} R(k)/S_unit(k).

E62b is the estimator of choice: it needs no variance model and is
immune to the erratic multiples-of-6 bins, where R(k) is inflated by
primes congruent to ±1 mod 6 (R(6)=3.34, R(12)=3.21 vs R(2)=1.43).

## Expected outcomes (all valid, pre-registered)

| outcome | reading |
|---|---|
| median inside ±0.15 of 1.32032 | AGREEMENT — estimator was the problem; H-L constant confirmed |
| median 1.35–1.45, closer than mean | PARTIAL — converging, needs N=1e7 with the median |
| median ~1.47, same as mean | MISS — no estimator recovers 2C2 at this N; re-examine S_unit |

## Statistics

- Bootstrap CI for the median (B=200, LCG stream).
- Report discovery/holdout separately; do not pool across N.
- Report all three estimators in the output, not just the winner.

## Artifacts (to be produced on run; absent now)

- `experiment62_weighted_fit.rs` example.
- JSON output + `EXPERIMENT62_RESULTS.md`.
- Run: `cargo run -p physis-core --release --example experiment62_weighted_fit`.

Status repeated: SPEC FROZEN, NOT RUN. Nothing above is evidence.