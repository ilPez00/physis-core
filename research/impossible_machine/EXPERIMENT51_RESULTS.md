# Experiment 51 Results — Representation-Covariant Probability Laws

**Verdict: FAILED — cleanly, at the probability level, and with the failure mode
already predictable from E50.**

**The question:** does the underlying arithmetic object induce *probability
laws / statistical relations* that remain invariant across independent
representations, even though the *geometric* descriptions differ (E50's
negative)?

**The answer: no candidate law is simultaneously** (a) transferable across
independent representations, (b) arithmetic-specific against all three nulls,
and (c) held-out-surviving. **0 of 5 candidates clear all three gates.** The
shared statistical structure that *does* transfer is present in the nulls too —
it is generic to cumulative-counting series, not a property of primes; and the
representations split exactly the way E50 predicted.

## Pipeline integrity

- Representations reconstructed on a common `u = ln x` grid (10 → 200 000),
  `n_samples = 2048`, deterministic LCG seed `20260912` (same family as E48–E50).
- Discovery representations: **ψ(x)−x**, **θ(x)−x** (+ three nulls).
- Held-out representations (constructed blind, frozen before evaluation):
  **π(x) − x/ln x**, **Mertens M(x)/√x**.
- Nulls: **shuffled-ψ** (marginals kept, order destroyed), **Cramér** random
  primes at density 1/ln x through the identical ψ−x pipeline, **synthetic
  iid-Gaussian** matching ψ's first-order statistics.
- 1/2 absent from the entire discovery process.
- Five representation-agnostic observables, so cross-representation "transfer"
  is well-defined (a scalar on every series), none referencing 1/2:
  C1 excess kurtosis, C2 lag-1 increment autocorrelation, C3 tail exponent,
  C4 mean one-sigma return interval, C5 normalised consecutive displacement.

## Discovery phase — transfer, but it is shared with the nulls

| candidate | discovery (ψ, θ) | spread | shuff | Cramér | synthG | transfer | specific |
|---|---|---|---|---|---|---|---|
| C1 excess kurtosis | 5.919 | 7.718 | 9.778 | 5.544 | −0.135 | **NO** | no |
| C2 lag-1 ACF | −0.188 | 0.001 | −0.480 | −0.016 | −0.508 | **YES** | no |
| C3 tail exponent | 1.229 | 0.248 | 1.105 | 1.596 | 3.376 | **YES** | no |
| C4 return interval | 0.561 | 0.069 | 0.596 | 0.586 | 0.747 | **YES** | no |
| C5 displacement | 0.498 | 0.003 | 0.622 | 0.491 | 0.802 | **YES** | no |

ψ and θ *agree about almost everything* — including about quantities that the
nulls reproduce. Four of five candidates transfer across the discovery pair;
**none distinguishes primes from all three nulls.**

## Held-out phase — the agreement again evaporates

| candidate | discovery-value | held-out π | held-out Mertens | held-out survives |
|---|---|---|---|---|
| C1 | 5.919 | 4.186 | 0.102 | **NO** |
| C2 | −0.188 | 0.271 | 0.028 | **NO** (sign flips) |
| C3 | 1.229 | 0.967 | 3.113 | **NO** |
| C4 | 0.561 | 0.500 | 0.734 | **NO** |
| C5 | 0.498 | 0.635 | 0.668 | **NO** |

The discovery-pair agreement does not survive the blind change of
representation: π and Mertens drift outside the pre-frozen margin (C2 even flips
sign). This is the E50 false-positive-avoidance mechanism firing again, one
## The failure, precisely localized

- The probabilistic statistics that **transfer** (C2–C5: short-range dependence,
  tail-shape, recurrence, local fluctuation) are **null-equivalent** — they are
  properties of additive cumulative-counting series in general, reproduced by
  the Cramér and synthetic-Gaussian controls.
- The prime-specific notion that survives is the *separation from Cramér width*,
  but it does **not** transfer as a common law: what is specific is bound to
  each representation, and what transfers is generic.

**Phase VI condition (transfer AND specificity AND held-out) fails for every
candidate. No representation-covariant probability law is BOTH robust AND
arithmetic-specific.** This is E50's conclusion restated at the level the plan
deliberately moved to (return times, gap statistics, fluctuations, conditionals)
— and it is the "much smaller corpse" E50's successor note predicted for the
geometric-ontology hypothesis.

## The four conclusions (A–D)

**A. Probability-law invariance: NO.** No distributional, recurrence, correlational,
or fluctuation law is invariant across independent representations.

**B. Transfer and specificity are anti-correlated at the probability level too.**
What transfers across ψ, θ (and generalises nowhere else) is generic; what is
prime-specific (separation from nulls) is locked to a single representation.

**C. Cramér's model holds at the law level.** Prime fluctuations follow the
distribution a 1/ln x random process predicts — so observed "laws" are
*consequences of prime density*, not independent arithmetical invariants. (This
is exactly why density-based pipelines like Cramér's cannot over-predict the
structure: the predictable part is not arithmetically specific.)

**D. No oracle was hypothesised or targeted.** The experiment returned NO as
designed (it was explicitly capable of doing so) and no connection to 1/2 was
manufactured.

## Phase VII — relation to 1/2

Observed discovery candidate values: C1=5.919 C2=−0.188 C3=1.229 C4=0.561
C5=0.498. **None targets or lands on 1/2. Recorded honestly: nothing to
report; no connection manufactured.**

## Final adversarial question, answered

> Does the underlying arithmetic object induce probability laws that are the
> same no matter which arithmetic function you sample?

**Not from this direction.** The representation-invariant probability structure
is generic cumulative-counting behaviour; the prime-specific structure is not a
shared law. The hypothesis that "some probability law" survives a blind change
of representation is not supported.

## Next experiment

The failure mode names its own successor: the shared quantity that *does*
survive across representations and *is* specific to reference arithmetic would
have to be a **relational invariant defined within a single representation that
the nulls provably lack** — e.g. Hardy–Littlewood-style pair-correlation of
primes (the 1/ln x density predicts independence; primes violate it with a
small, sign-definite, representation-free correction) tested against a
Cramér null at the same object. If that relational deviation survives the held-out
representations *and* beats the Cramér width, it would be the first genuinely
representation-covariant arithmetic law; if not, the geometric-ontology
hypothesis is closed at this resolution.

## Artifacts

- Code: `physis-core/examples/experiment51_probability_laws.rs`
- Results: `physis-core/research/impossible_machine/experiments/experiment51_results.json`
- Plan: `physis-core/research/impossible_machine/EXPERIMENT51_PLAN.md`
abstraction level up.