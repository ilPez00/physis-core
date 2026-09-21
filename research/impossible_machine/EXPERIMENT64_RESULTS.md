# Experiment 64 Results — Median Estimator, kmax=80 (N=1e6)

**Verdict: MISS.** G1+G2 pass, G3 misses — and the fit collapses
**below** 2C2, in the opposite direction from E60/E61's overshoot.

**Run:** 2026-09-20, `cargo run -p physis-core --release --example
experiment64_median_kmax80`, N=1_000_000, seed 20260912.
78,498 primes (41,530 discovery / 36,960 held-out).

## Numbers

| | E62 kmax=40 | E64 kmax=80 |
|---|---|---|
| deviation D (disc / hold) | 1.258 / 1.321 | 0.975 / 1.009 |
| twin ratio R(2) | 1.428 / 1.422 | 1.428 / 1.422 |
| fitted 2C2 (disc / hold) | 1.395 / 1.451 | **0.933 / 1.075** |
| 95% CI (disc) | [1.279, 1.660] | [0.807, 1.204] |
| N1 Cramér null D | 0.036 | 0.036 |
| N2 geometric null D | 0.053 | 0.053 |
| Δ-to-target vs E62 | — | **−0.312** (below) |

- **G1 specificity: PASS.** D=0.975 vs nulls 0.036/0.053 (~18×).
- **G2 holdout: PASS.** D stable (0.975/1.009), R2 > 1 both halves.
- **G3 constant: MISS.** 0.933 and 1.075 both outside 1.32032±0.15.

## Reading — the kmax=80 result is informative, not a failure

Doubling the bin count did not tighten the estimate; it broke it, and
it broke it **downward**. The mechanism is the same one that made E62a
(E(k)-weighted pooled) overshoot to 1.609, but mirrored:

for k in 42..80 the geometric E(k) is tiny, so a bin holding N(k)=0 or
1 gives R(k) = 0 or a very large number. With 40 such bins instead of
20, **half of them sit at R(k)=0**, and the median is dragged to ~0.93.
The kmax=40 median (1.395) survives because the 20 even bins in
2..40 are all reasonably populated; the kmax=80 median is corrupted by
the sparse tail.

This is a clean negative result with a definite operational
conclusion: **kmax=40 is the robust setting for this estimator at
N=1e6.** The precision limiter is therefore *not* the bin count in the
sense that widening it helps — it is the discrete, biased nature of the
large-k bins, which a median over raw R(k) cannot handle. Precision
would need a different treatment of the tail (e.g. truncating R(k) at
the k where E(k) falls below a threshold, or a shrinkage estimator),
not more bins.

The substantive finding is unaffected: at kmax=40 the median estimator
recovers 2C2 at both N=1e6 (E62, 1.395/1.451) and N=1e7 (E63,
1.459/1.470), G1+G2+G3 all pass. E64 shows the agreement is
kmax-sensitive and should not be pushed further with this estimator.

## Relation to 1/2

No 1/2 in generation, scoring, or gates. Fitted values 1.395/0.933.
Recorded coincidence only, never for/against RH.

## Next (superseded — see SERIES_SUMMARY.md)

E65: tail-truncated median at kmax=80 — exclude bins with E(k) below a
data-driven threshold (e.g. E(k) < mean(E)/10) before taking the
median. If that recovers a kmax-insensitive fit inside ±0.15, the
precision bottleneck is solved; if not, close the series at kmax=40
with E62/E63 as the final word.

## Artifacts

- Code: `physis-core/examples/experiment64_median_kmax80.rs`
- Spec: `physis-core/research/impossible_machine/EXPERIMENT64_SPEC.md`
- JSON: `physis-core/research/impossible_machine/experiments/experiment64_results.json`