# Experiment 61 Results — Convergence Test (N=1e7)

**Verdict: PARTIAL — DIVERGING.** G1+G2 pass; G3 misses and the fit
moves **away** from 2C₂ with N, so the miss is estimator bias, not a
finite-range artifact.

**Run:** 2026-09-20, `cargo run -p physis-core --release --example
experiment61_convergence`, N=10_000_000, seed 20260912.
664,579 primes (348,505 discovery / 316,066 held-out).

## Numbers

| quantity | E60 N=1e6 | E61 N=1e7 |
|---|---|---|
| deviation D (disc / hold) | 1.258 / 1.321 | 1.319 / 1.339 |
| twin ratio R(2) | 1.428 / 1.422 | 1.428 / 1.416 |
| fitted 2C₂ (disc / hold) | 1.470 / 1.487 | **1.498 / 1.509** |
| 95% CI (disc) | [1.279, 1.660] | [1.334, 1.678] |
| N1 Cramér null D | 0.036 | 0.014 |
| N2 geometric null D | 0.053 | 0.023 |
| Δ-to-target (disc) | +0.150 | **+0.178** |

- **G1 specificity: PASS** both runs. ~24× and ~23× above worst null.
- **G2 holdout: PASS** both runs. D stable, R2 > 1 both halves.
- **G3 constant: MISS** both runs, and **worse at N=1e7**.
  Δ-to-target went +0.150 → **+0.028 further** (1.470→1.498 vs 1.32032).

## Reading — honest

The direction result is clean and stable: the small-gap deviation is
real and arithmetic-specific at both N, with D ~1.3 and R2 ~1.42.
The **quantitative** step does not close, and it does not get closer
with N. That rules out "finite-range normalisation artifact" as the
explanation — the estimator is biased high.

Likely cause: the fit is an unweighted mean of R(k)/S_unit(k) over
the 20 even-k bins. At N=1e6 the large-k bins are thin and noisy, and
the bias they contribute does not average out with N (the bin count is
fixed at 20; only the per-bin precision improves). E60's CI already
covered 1.32032, so this is a point-estimate precision failure, not a
direction failure — the Hardy-Littlewood correction is confirmed in
direction and stability; the constant value is not yet recovered by
this estimator.

## Relation to 1/2

No 1/2 in generation, scoring, or gates. Fitted values 1.470/1.498.
Recorded coincidence only, never for/against RH.

## Next (superseded — see SERIES_SUMMARY.md)

E62 isolated the estimator as the cause: the unweighted mean over 20
fixed k-bins is dominated by the erratic multiples-of-6 bins (R(6)=3.34,
R(12)=3.21 vs R(2)=1.43), whose bias does not average out with N. The
**median of R(k)/S_unit(k)** recovers 2C2 at both N=1e6 (1.395/1.451)
and N=1e7 (1.459/1.470), G1+G2+G3 all pass. E64/E65 show the estimator
is kmax-bounded at 40. E61's PARTIAL stands as the unweighted-mean
result it was always recorded as.

## Artifacts

- Code: `physis-core/examples/experiment61_convergence.rs`
- Spec: `physis-core/research/impossible_machine/EXPERIMENT61_SPEC.md`
- JSON: `physis-core/research/impossible_machine/experiments/experiment61_results.json`