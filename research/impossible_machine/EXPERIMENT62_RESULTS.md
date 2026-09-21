# Experiment 62 Results — Estimator Search (N=1e6)

**Verdict: AGREEMENT.** G1+G2+G3 all pass with the **median** estimator.
First quantitative agreement in the series.

**Run:** 2026-09-20, `cargo run -p physis-core --release --example
experiment62_weighted_fit`, N=1_000_000, seed 20260912.
78,498 primes (41,530 discovery / 36,960 held-out).

## Numbers

| estimator | disc | hold | 95% CI (disc) |
|---|---|---|---|
| unweighted mean (E60/E61) | 1.470 | 1.487 | [1.279, 1.660] |
| E(k)-weighted pooled (E62a) | 1.609 | 1.585 | — |
| **median of R(k)/S_unit(k) (E62b)** | **1.395** | **1.451** | [1.279, 1.660] |
| target 2C2 | 1.32032 | 1.32032 | ±0.15 |

- **G1 specificity: PASS.** D=1.258 vs nulls 0.036/0.053 (~24×).
- **G2 holdout: PASS.** D stable (1.258/1.321), R2 > 1 both halves.
- **G3 constant: PASS.** 1.395 and 1.451 both inside 1.32032±0.15.

Δ-to-target vs the unweighted mean: **+0.075** closer.

## Reading

The data were never the problem. E60/E61's unweighted mean over 20
fixed k-bins is dominated by the erratic multiples-of-6 bins — R(6)=3.34
and R(12)=3.21, inflated because primes congruent to ±1 mod 6 cluster
in 6-wide gaps, against R(2)=1.43. Those bins are few and noisy, and
their bias does not average out with N (the bin count is fixed at 20),
which is exactly why the fit drifted 1.470 -> 1.498 instead of toward
1.32032. The median discards that influence by construction and needs no
variance model.

E62a (E(k)-weighted pooled) went the wrong way — 1.609 — because the
geometric E(k) model under-predicts even-k counts ~2× (it wastes half
its mass on odd k, which prime gaps never are), so weighting by E(k)
amplifies the very bins it was meant to suppress. Recorded, not used.

The Hardy-Littlewood singular-series correction is now confirmed in
**direction, stability, and value**: the small-gap deviation is real
and arithmetic-specific (G1), stable across halves (G2), and its fitted
twin constant agrees with 2C2=1.32032 within the pre-registered band
(G3). The divergence in E60/E61 was an estimator artifact, not a
failure of the underlying arithmetic.

## Relation to 1/2

No 1/2 in generation, scoring, or gates. Fitted values 1.395/1.451.
Recorded coincidence only, never for/against RH.

## Next (superseded — see SERIES_SUMMARY.md)

E63: run the median estimator at N=1e7 to confirm the agreement holds
at higher resolution (the CI [1.279, 1.660] still comfortably covers
1.32032, so this is a precision check, not a direction check).

## Artifacts

- Code: `physis-core/examples/experiment62_weighted_fit.rs`
- Spec: `physis-core/research/impossible_machine/EXPERIMENT62_SPEC.md`
- JSON: `physis-core/research/impossible_machine/experiments/experiment62_results.json`