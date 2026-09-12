# EXPERIMENT 51 — REPRESENTATION-COVARIANT PROBABILITY LAWS

## SYSTEM ROLE
Act as the experimental core of Physis. E50 has now produced a strong negative result:
> Arithmetic-specific structure was found in individual representations, but the structure did not survive a blind change of representation.
The geometric ontology is therefore not currently representation-independent.
Do **not** attempt to rescue E50 by adding more geometric transformations.
Move one abstraction level upward.
The new question is:
> **Does the underlying arithmetic object induce probability laws or statistical relations that remain invariant across representations, even when their geometric descriptions differ?**

This experiment must be capable of returning **NO**.
Do not search for confirmation.
Do not target \(1/2\).
Do not assume that a common probability law exists.

---

## CENTRAL HYPOTHESIS
Let $X$ be the underlying arithmetic object.
Let $R_1(X), R_2(X), \ldots, R_n(X)$ be independently motivated arithmetic representations.
Each representation induces one or more statistical objects:
$$P_i = \mathcal P(R_i(X)).$$
Test whether there exists a low-complexity representation-independent law $L(X)$ such that:
$$L(R_i(X)) \approx L(R_j(X))$$
across independently chosen representations.

---

## PHASES & IMPLEMENTATION PLAN

### Phase I — Reconstruct the Representations
Include direct representations ($\psi(x)$, $\theta(x)$, $\pi(x)$, $M(x)/\sqrt{x}$) and derived representations ($\Lambda(n)$, prime-gap sequences, normalized prime gaps, Möbius sequences, recurrence/return sequences). Record the dependency graph and avoid treating algebraically dependent representations as independent evidence.

### Phase II — Construct Statistical Observables
Construct statistical observables across families:
- Distributional laws ($P(X>x)$, $P(X\le x)$, $P(\Delta X > g)$)
- Return and recurrence statistics
- Gap statistics and normalized variants
- Correlation structure ($C(k)$, partial correlation, covariance scaling)
- Fluctuation laws ($Z(x) = \frac{R(x) - \mathbb E[R(x)]}{\sigma_R(x)}$)
- Extreme-value behavior and conditional laws ($P(A \mid B)$)

### Phase III — Discover Candidate Laws & Complexity Penalty
Search restricted interpretable library (Gaussian, exponential, Poisson, geometric, power-law, log-normal, stretched exponential, extreme-value families, recurrence distributions, empirical copulas, conditional relations). Penalize heavily for complexity (MDL principle).

### Phase IV — Three Levels of Invariance
- **Level A**: Exact / direct invariance ($P_i \approx P_j$).
- **Level B**: Canonically normalized invariance ($N(P_i) \approx N(P_j)$ with mathematically justified, pre-fixed normalizations).
- **Level C**: Relational invariance (preservation of relations such as $A = f(B)$ or $A/B = c$ or $\operatorname{Corr}(A,B) = c$).

### Phase V — Null Controls
- **Null 1**: Shuffled arithmetic data (destroys ordering, preserves marginals).
- **Null 2**: Cramér / randomized arithmetic null (preserves baseline density while removing prime correlations).
- **Null 3**: Synthetic stochastic processes matching first-order statistics.

### Phase VI — Critical Contrast & Held-Out Blind Holdout
Candidates must satisfy **both** cross-representation transfer ($L_{\text{prime}}(R_i) \approx L_{\text{prime}}(R_j)$) **and** arithmetic specificity ($L_{\text{prime}} \not\approx L_{\text{null}}$).
Partition representations into Discovery ($R_1, R_2, \ldots$) and Held-out ($R_k, R_{k+1}, \ldots$). Freeze candidate family, normalization, parameters, and scoring before revealing held-out representations.

### Phase VII — Stopping Rule & Verdict
Clean failure is declared if no candidate simultaneously satisfies transfer, specificity, and held-out survival.
