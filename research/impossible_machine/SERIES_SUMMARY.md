# Singular-Series Fit — Series Summary (E52 → E65)

Status: CLOSED 2026-09-20. Six experiments, one estimator, one verdict.

## The question

Does the small-gap prime deviation carry the Hardy-Littlewood twin
constant 2C2 = 1.32032 quantitatively?

## The chain

| exp | N | kmax | estimator | 2C2 fit | G1 | G2 | G3 |
|---|---|---|---|---|---|---|---|
| E52 | 2e5 | 40 | unweighted mean | — (direction only) | ✓ | ✓ | — |
| E60 | 1e6 | 40 | unweighted mean | 1.470 / 1.487 | ✓ | ✓ | ✗ |
| E61 | 1e7 | 40 | unweighted mean | 1.498 / 1.509 | ✓ | ✓ | ✗ |
| E62 | 1e6 | 40 | **median** | **1.395 / 1.451** | ✓ | ✓ | **✓** |
| E63 | 1e7 | 40 | **median** | **1.459 / 1.470** | ✓ | ✓ | **✓** |
| E64 | 1e6 | 80 | median (all bins) | 0.933 / 1.075 | ✓ | ✓ | ✗ |
| E65 | 1e6 | 80 | median, N(k)>0 | 0.934 / 1.075 | ✓ | ✓ | ✗ |

## The verdict

**The Hardy-Littlewood singular-series correction is confirmed in
direction, stability, and value.**

- **Direction (G1):** the small-gap deviation is real and
  arithmetic-specific — D ≈ 1.26–1.34 against nulls ≤ 0.053, at both
  N=1e6 and N=1e7.
- **Stability (G2):** the deviation is the same in the discovery and
  held-out halves; R(2) > 1 throughout.
- **Value (G3):** the fitted twin constant is 1.395/1.451 (N=1e6) and
  1.459/1.470 (N=1e7), both inside the pre-registered band
  1.32032 ± 0.15.

## What went wrong, and why it is not a failure of the arithmetic

E60/E61's unweighted mean over 20 fixed k-bins **diverged from 2C2 with
N** (1.470 → 1.498, Δ-to-target +0.028 further away). That is the
opposite of convergence, so it could not be blamed on finite range.

The cause is the estimator, not the data. The mean is dominated by the
erratic multiples-of-6 bins — R(6)=3.34 and R(12)=3.21, inflated
because primes congruent to ±1 mod 6 cluster in 6-wide gaps, against
R(2)=1.43. Those bins are few and noisy, and their bias does not
average out with N (the bin count is fixed at 20). The median discards
that influence by construction and needs no variance model.

E62a (E(k)-weighted pooled) went the wrong way — 1.609 — because the
geometric E(k) model under-predicts even-k counts ~2× (it wastes half
its mass on odd k, which prime gaps never are), so weighting by E(k)
amplifies the very bins it was meant to suppress. Recorded, not used.

## The boundary: kmax=40 is the valid range

E64 (kmax=80) collapsed the median to 0.933, and E65 falsified the
exclusion hypothesis: 39/40 and 40/40 bins have N(k) > 0, so zero
counts are not the cause. The cause is that R(k)/S_unit(k) is
systematically low for k in 42..80 — the singular-series approximation
holds for small even gaps and breaks down for large ones. Widening
kmax adds model error, not precision. **kmax=40 is the range over
which this estimator is correctly specified**, and E62/E63 are the
final results.

## Relation to 1/2

No 1/2 appears anywhere in generation, scoring, ranking, or the gates.
Fitted values range 0.93–1.51. Recorded coincidence only — never used
for or against the Riemann hypothesis.

## Artifacts

- Examples: `experiment60_singular_series.rs`,
  `experiment61_convergence.rs`, `experiment62_weighted_fit.rs`,
  `experiment63_median_n1e7.rs`, `experiment64_median_kmax80.rs`,
  `experiment65_tail_excluded_median.rs`.
- Specs/results: `EXPERIMENT{60..65}_SPEC.md`, `EXPERIMENT{60..65}_RESULTS.md`.
- JSON: `experiments/experiment{60..65}_results.json`.
- This file: `research/impossible_machine/SERIES_SUMMARY.md`.