# Experiment 60 Results — Singular-Series Fit (twin-prime constant)

**Verdict: PARTIAL** — direction + stability reproduce at N=1e6, but the
fitted constant misses the frozen ±0.15 band around 2C₂ ≈ 1.32032.

**Run:** 2026-09-20, `cargo run -p physis-core --release --example
experiment60_singular_series`, N=1_000_000, seed 20260912.
78,498 primes (41,530 discovery / 36,960 held-out).

## Numbers

| quantity | discovery (20, 500k] | held-out (500k, 1M] |
|---|---|---|
| deviation D | 1.258 | 1.321 |
| twin ratio R(2) | 1.428 | 1.422 |
| fitted 2C₂ | 1.470 [1.279, 1.660] | 1.487 [1.341, 1.705] |
| N1 Cramér null D | 0.036 | — |
| N2 geometric null D | 0.053 | — |

- **G1 specificity: PASS.** 1.258 > 1.5 × 0.053. ~24× above worst null.
- **G2 holdout: PASS.** |1.321−1.258| = 0.063 ≤ 0.5×1.258; R2 > 1 both.
- **G3 constant: MISS.** 1.470/1.487 vs 1.32032±0.15 → band [1.170, 1.470].
  Discovery edge touches 1.470 (rounds to band edge, strict ≤ fails by
  0.0002); held-out 1.487 outside. CI in both halves covers 1.32, so the
  miss is quantitative precision, not direction.

## Reading — honest

E52's claim (deviation exists + stable) reproduces at 5× range.
The quantitative step does not close at N=1e6 with this estimator:
mean R(k)/S_unit(k) overshoots 2C₂ by ~0.15 (~11%). Likely causes:
per-position 1/ln p normalisation (E52 noted finite-range offset already
at N=2e5: R2 1.448 vs 1.32), unweighted mean over thin large-k bins.
CI covers target — estimator noisy, not theory refuted.

## Relation to 1/2

No 1/2 in generation, scoring, gates. Fitted values 1.470/1.487.
Recorded coincidence only, never for/against RH.

## Next

E61: weighted fit (inverse-variance per k-bin) at N=1e6 + N=1e7 before
any new claim. Do not move G3 band post-hoc.

## Artifacts

- Code: `physis-core/examples/experiment60_singular_series.rs`
- Spec: `physis-core/research/impossible_machine/EXPERIMENT60_SPEC.md`
- JSON: `physis-core/research/impossible_machine/experiments/experiment60_results.json`
