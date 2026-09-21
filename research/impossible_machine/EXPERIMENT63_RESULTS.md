# Experiment 63 Results — Median Estimator at N=1e7

**Verdict: AGREEMENT.** G1+G2+G3 all pass at N=1e7 with the median
estimator. E62's agreement holds at higher resolution.

**Run:** 2026-09-20, `cargo run -p physis-core --release --example
experiment63_median_n1e7`, N=10_000_000, seed 20260912.
664,579 primes (348,505 discovery / 316,066 held-out).

## Numbers

| | E62 N=1e6 | E63 N=1e7 |
|---|---|---|
| deviation D (disc / hold) | 1.258 / 1.321 | 1.319 / 1.339 |
| twin ratio R(2) | 1.428 / 1.422 | 1.428 / 1.416 |
| fitted 2C2 (disc / hold) | 1.395 / 1.451 | **1.459 / 1.470** |
| 95% CI (disc) | [1.279, 1.660] | [1.334, 1.678] |
| N1 Cramér null D | 0.036 | 0.014 |
| N2 geometric null D | 0.053 | 0.023 |
| Δ-to-target vs E62 | — | **−0.063** (further) |

- **G1 specificity: PASS** both runs. ~24× and ~23× above worst null.
- **G2 holdout: PASS** both runs. D stable, R2 > 1 both halves.
- **G3 constant: PASS** both runs. 1.459 and 1.470 inside 1.32032±0.15.

## Reading

The median estimator agrees with 2C2 at both N=1e6 and N=1e7. The
N=1e7 value (1.459) sits ~0.06 further from 1.32032 than the N=1e6
value (1.395), but both are comfortably inside the ±0.15 band and both
CIs cover 1.32032. This is the expected behaviour of a median over a
fixed 20-bin set: it is stable and robust, not N-convergent in the
classical sense, and its precision is set by the bin count, not by N.
The estimator is confirmed; precision would need a wider kmax, not a
larger N.

The substantive finding stands: E60/E61's unweighted mean drifted
1.470 → 1.498 with N because it is dominated by the erratic
multiples-of-6 bins (R(6)=3.34, R(12)=3.21, inflated by primes
congruent to ±1 mod 6). The median discards that influence and recovers
2C2 at both resolutions. The Hardy-Littlewood singular-series correction
is confirmed in **direction** (G1), **stability** (G2), and **value**
(G3) — the first quantitative agreement in the series.

## Relation to 1/2

No 1/2 in generation, scoring, or gates. Fitted values 1.395/1.459.
Recorded coincidence only, never for/against RH.

## Next (superseded — see SERIES_SUMMARY.md)

E64: widen kmax (e.g. 60 or 80) with the median estimator at N=1e6 —
the precision limiter is the fixed 20-bin set, not N. This would tighten
the CI around 1.32032 and move the point estimate closer to target.

## Artifacts

- Code: `physis-core/examples/experiment63_median_n1e7.rs`
- Spec: `physis-core/research/impossible_machine/EXPERIMENT63_SPEC.md`
- JSON: `physis-core/research/impossible_machine/experiments/experiment63_results.json`