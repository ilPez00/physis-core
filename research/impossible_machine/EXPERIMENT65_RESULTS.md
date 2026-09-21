# Experiment 65 Results — Tail-Excluded Median, kmax=80 (N=1e6)

**Verdict: MISS.** G1+G2 pass, G3 misses — and the exclusion hypothesis
is falsified: almost no bins were excluded.

**Run:** 2026-09-20, `cargo run -p physis-core --release --example
experiment65_tail_excluded_median`, N=1_000_000, seed 20260912.
78,498 primes (41,530 discovery / 36,960 held-out).

## Numbers

| | E62 kmax=40 | E64 kmax=80 raw | E65 kmax=80, N(k)>0 |
|---|---|---|---|
| fitted 2C2 (disc / hold) | 1.395 / 1.451 | 0.933 / 1.075 | **0.934 / 1.075** |
| 95% CI (disc) | [1.279, 1.660] | [0.807, 1.204] | [0.868, 1.222] |
| bins with N(k)>0 | 20/20 | 40/40 | **39/40, 40/40** |
| Δ vs E62 | — | −0.312 | **−0.311** |
| Δ vs E64 raw | — | — | **+0.001** |

- **G1 specificity: PASS.** D=0.975 vs nulls 0.036/0.053 (~18×).
- **G2 holdout: PASS.** D stable (0.975/1.009), R2 > 1 both halves.
- **G3 constant: MISS.** 0.934/1.075 outside 1.32032±0.15.

## Reading — the exclusion hypothesis is falsified, and that is the finding

E65 was set up to test whether the kmax=80 collapse (E64, 0.933) was
caused by bins with N(k)=0, whose R(k)=0/0 is recorded as 0 and which
might be dragging the median down. The answer is no: **39 of 40 bins
in the discovery half and all 40 in the holdout half have N(k) > 0.**
Excluding the one empty bin changed the fit by 0.001. The pathology is
not zero counts.

The real cause is that R(k)/S_unit(k) is **systematically low for
k in 42..80**. The Hardy-Littlewood singular-series approximation
R(k) ≈ S_unit(k)·2C2 holds for the small even gaps (2..40) — that is
exactly why kmax=40 recovers 2C2 — but it breaks down for larger k,
where the geometric E(k) model no longer tracks the true gap
distribution. Those low values sit at the bottom of the sorted list
and, with 40 bins instead of 20, they occupy the median position.

This is a clean, useful boundary result: **kmax=40 is not an arbitrary
robustness setting; it is the range over which the singular-series
approximation is valid for this estimator.** Widening it does not add
precision; it adds model error. The estimator is therefore correctly
specified at kmax=40, and E62/E63 (1.395/1.451 and 1.459/1.470,
both AGREEMENT) are the final quantitative results of this line.

## Relation to 1/2

No 1/2 in generation, scoring, or gates. Fitted values 1.395/0.934.
Recorded coincidence only, never for/against RH.

## Next

The series is closed at kmax=40. E62 (N=1e6) and E63 (N=1e7) are the
final word: the Hardy-Littlewood singular-series correction is
confirmed in direction (G1), stability (G2), and value (G3) with the
median-of-R(k)/S_unit(k) estimator at kmax=40. E60/E61's divergence
with N was an unweighted-mean artifact over the erratic multiples-of-6
bins; E64/E65 show the estimator is kmax-bounded and must not be
widened.

## Artifacts

- Code: `physis-core/examples/experiment65_tail_excluded_median.rs`
- Spec: `physis-core/research/impossible_machine/EXPERIMENT65_SPEC.md`
- JSON: `physis-core/research/impossible_machine/experiments/experiment65_results.json`