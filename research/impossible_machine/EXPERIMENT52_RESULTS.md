# Experiment 52 Results — Prime Pair-Correlation (Hardy–Littlewood test)

**Verdict: PARTIAL-POSITIVE** — the first *object-level* property in the
E50/E51 line that is simultaneously **strongly arithmetic-specific** and
**out-of-sample stable**.

**The question:** E50/E51 could not find any geometric or cumulative-distribution
quantity that is both robust across representations and specific to the primes.
E52 asks whether there is *any* object-level law of the primes itself that is
specific and stable — using the one deviation Cramér-based reasoning *cannot*
reproduce: the **small-gap correlation** (the Hardy–Littlewood singular series).

**The answer: yes.** The deviation of small even-gap frequencies from the
Cramér independence model is ~12× the worst independence null and survives a
blind first-half/second-half split.

## Design

- Observable (pre-frozen): `R(k) = N(k)/E(k)` for even `k ∈ {2..40}`,
  `E(k)` = geometric-density expectation `Σ_p q_p(1-q_p)^{k-1}`, `q_p = 1/ln p`,
  `N(k)` = observed count. Deviation `D = RMS of (R(k)−1)`.
- Split (frozen): discovery = primes in `(20, N/2]`, held-out = `(N/2, N]`,
  `N = 200 000`, deterministic LCG seed `20260912`.
- Nulls (independence models, both must give `D ≈ 0`):
  - **N1** Cramér random primes at density `1/ln x`, same pipeline.
  - **N2** iid geometric gaps with per-position density.
- Views: **π/θ jumps** (consecutive-prime gaps) and **ψ/von-Mangoldt jumps**
  (primes + prime powers). Same primes ⇒ the two views necessarily agree
  (measured corr 0.999) — that is reported as a *caveat*, not a discovery.

## Results

| quantity | discovery (20, 100k] | held-out (100k, 200k] |
|---|---|---|
| deviation D | **1.222** | **1.258** |
| twin ratio R(2) | **1.448** | **1.448** |
| pi↔psi view corr | 0.999 | — |
| N1 Cramér null D | 0.096 | — |
| N2 geometric null D | 0.105 | — |

- **Specificity:** `D = 1.22` vs worst null `0.105` ⇒ ~12× separation. The gates
  (pre-frozen: `D > 1.5 × max_null`) pass strongly.
- **Out-of-sample stability:** held-out `D = 1.258` reproduces discovery
  `1.222` (`Δ = 0.036 ≤ 0.5·D`); twin ratio `1.45 > 1` in both halves.
- **Twin constant:** measured `R(2) ≈ 1.45`, consistent with the
  Hardy–Littlewood `2C₂ ≈ 1.32` excess (slight offset from the `1/ln p`
  normalisation and finite range).

## What this means — honestly

- E50/E51's negative was about **geometry** and **cumulative distributions**:
  there the arithmetic-specific part was representation-bound and the
  transferable part null-equivalent. That negative **stands**.
- E52 shows the small-`k` **correlation** of consecutive primes is different in
  kind: it is specific (independence nulls give `≈0`, real primes give `≈1.2`)
  and it is out-of-sample stable. Cramér's model structurally cannot produce it,
  because Cramér independence *is* the null.
- The "representation-covariance" claim is deliberately weak here: the π/θ and
  ψ views count the same primes, so every counting-function view trivially
  recovers the identical law. The substantive result is **specificity +
  out-of-sample stability**, the combination the E50/E51 framework demanded and
  did not find elsewhere.

## Relation to 1/2

No candidate targets or uses 1/2. Observed `D = 1.222/1.258`, twin ratios
`1.448/1.448`. No connection to 1/2 manufactured.

## Next

Formalise: fit the empirical `R(k)` to the full singular series
`∏_p ...` prediction at larger `N` (10⁶, 10⁷) to test the numeric value of the
twin constant (→ `2C₂ ≈ 1.32`) rather than only the >1 direction, and to
estimate the deviation's dependence on range. This moves from "a correction
exists and is stable" to "the correction is quantitatively the Hardy–Littlewood
constant".

## Artifacts

- Code: `physis-core/examples/experiment52_pair_correlation.rs`
- Results: `physis-core/research/impossible_machine/experiments/experiment52_results.json`
- Plan/packet: `packets/PH-052-prime-pair-correlation.md`