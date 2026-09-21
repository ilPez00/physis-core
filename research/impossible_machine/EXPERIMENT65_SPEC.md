# Experiment 65 Spec — Tail-Excluded Median, kmax=80 (N=1e6)

Status: SPEC FROZEN 2026-09-20, NOT RUN. Companion to E64.
E64 (kmax=80) MISS: the median over raw R(k) collapsed to
0.933/1.075 because the sparse large-k bins (k=42..80) hold N(k)=0 or
1, so R(k) is 0 or a very large number, and half of them sit at 0,
dragging the median down. E62/E63 (kmax=40) remain AGREEMENT at
1.395/1.451 and 1.459/1.470.

E65 asks whether excluding the *uninformative* bins — those with no
observed events at all, where R(k) is undefined rather than measured —
recovers a kmax-insensitive fit. This is not a hyperparameter: a bin
with N(k)=0 contributes zero observed gaps of size k and cannot
estimate R(k), so dropping it is a measurement decision, not a tuning
choice.

## Objective

Run the median of R(k)/S_unit(k) at N=1e6, kmax=80, over bins with
N(k) > 0 only. Compare with E62 (kmax=40, all bins: 1.395/1.451) and
E64 (kmax=80, all bins: 0.933/1.075).

## Frozen (unchanged from E62)

- Sieve, seed 20260912, splits discovery (20, N/2] / holdout (N/2, N].
- R(k) = N(k)/E(k), E(k) geometric-density independence expectation.
- S_unit(k) = prod_{odd p|k} (p-1)/(p-2), S_unit(2)=1.
- Nulls: N1 Cramer + N2 geometric (seed 20260912).
- Gates G1/G2/G3: identical thresholds to E62 (G3 band ±0.15).
- 1/2 never in generation, scoring, or ranking.

## Only change from E62/E64

- kmax=80, and the median is taken over even k in {2..80} with N(k) > 0.
  Everything else byte-identical (sieve, n_max, seed, gates, output
  schema). Both the excluded-bin count and the raw kmax=80 median are
  reported so the comparison with E64 is visible.

## Expected outcomes (all valid, pre-registered)

| fit at N=1e6, kmax=80, N(k)>0 | reading |
|---|---|
| inside ±0.15 of 1.32032 | AGREEMENT — tail exclusion fixes it; precision improvable |
| 1.35–1.45, between E62 and E64 | PARTIAL — partial recovery; needs a shrinkage estimator |
| ~0.93, same as E64 | MISS — exclusion alone insufficient; N(k)=0 is not the only pathology |
| outside ±0.15, above 1.45 | MISS — tail exclusion overshoots; revert to kmax=40 |

## Statistics

- Bootstrap CI for the median (B=200, LCG stream), over the N(k)>0 bins.
- Report discovery/hold-out separately; do not pool across N.
- Report n_bins_used (N(k)>0) vs n_bins_total, so the exclusion rate is
  on the record rather than hidden.

## Artifacts (to be produced on run; absent now)

- `experiment65_tail_excluded_median.rs` example.
- JSON output + `EXPERIMENT65_RESULTS.md`.
- Run: `cargo run -p physis-core --release --example experiment65_tail_excluded_median`.

Status repeated: SPEC FROZEN, NOT RUN. Nothing above is evidence.