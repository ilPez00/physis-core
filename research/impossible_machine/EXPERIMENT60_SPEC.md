# Experiment 60 Spec — Singular-Series Fit (twin-prime constant)

Status: SPEC FROZEN 2026-09-20, NOT RUN. No numbers below are results.
Supersedes the Pro-side `EXPERIMENT53_SPEC.md` draft name (E53 taken by
transform-analogy in physis-core/examples). Design mirrors E52 conventions;
frozen before any held-out look.

## Objective

Fit empirical prime-gap ratio R(k) against the full Hardy-Littlewood
singular-series prediction at N=1e6. Target: twin constant 2C2 ≈ 1.32032.
Valid outcomes: agreement / partial / miss. E52 showed direction-exists +
stable; E60 asks whether the quantitative value matches.

## Frozen observable, splits, nulls (E52 family)

- Observable: R(k) = N(k)/E(k) for even k in {2..40} (E52 definition:
  geometric-density independence expectation). One scalar per k.
- Splits: discovery primes in (20, N/2], held-out in (N/2, N], N=1_000_000,
  deterministic sieve (no RNG in prime generation).
- Nulls: N1 Cramer random primes at density 1/ln x (seed 20260912) +
  N2 iid geometric gaps (same per-position density). Both must give D ≈ 0.
- Estimator sees primes only; no zeta zeros, no critical-line input.

## Gates (frozen before held-out)

- G1 specificity: discovery D > 1.5 × max(nulls).
- G2 holdout: |D_hold − D_disc| ≤ 0.5 × D_disc AND R2 > 1 in BOTH halves.
- G3 constant agreement: fitted 2C2_hat within ±0.15 of 1.32032
  (half-width frozen; ≈11% band) → agreement. Outside → miss.
- Never tune thresholds after seeing held-out values.

## Statistics

- Bootstrap 95% CI over independent k-windows (20 even-k bins,
  B=200 resamples, seed-derived LCG stream); resampling unit = k-bin.
- Effect size: (2C2_hat − 1) / CI-halfwidth reported alongside every claim.
- Record low-sample warnings honestly (large-k bins thin at N=1e6).

## Relation to 1/2

MUST state: no 1/2 appears in generation or scoring. Series, R(k),
nulls, and all three gates defined without the critical exponent.
A fitted value near 1/2 is recorded numerical coincidence, never
support for or against RH.

## Artifacts (to be produced on run; absent now)

- `experiment60_singular_series.rs` example (primes → R(k) → fit).
- JSON output: R(k) profiles, null values, fitted constant, CI, gates.
- Results doc `EXPERIMENT60_RESULTS.md` with discovery/held-out tables.
- Run: `cargo run -p physis-core --release --example
  experiment60_singular_series` at N=1e6.

Status repeated: SPEC FROZEN, NOT RUN. Nothing above is evidence.