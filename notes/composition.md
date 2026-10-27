# Composition and number theory

Figures 21, 31, 32, 33, 36, 37, 39, 40, 43. Scripts in `scripts/`: `eta_compose`,
`prime_sieve`, `zeta_compose`, `factorize`, `landscape`, `dna_algebra` (a shared module whose
`__main__` draws figure 37), `lfunctions`, `bsd`, `hsln_two_rings`, and `hsln_hecke_exercise`
(no figure). Every grade below cites the audit's evidence ledger
(`rmorleysmith/fugue`, `Fugue/02-threads/evidence-ledger.md`) by row id, and every Lean name is
in `rmorleysmith/fugue-formal`.

The family began with one word, "compose", used for two things, and ended with the discovery
that they are two different rings on the same list of numbers. Read in order, the figures are
the path from the first use to that distinction.

## The figures, in order

**21 — eta quotients** (`eta_compose.py`). A form f_r(τ) = ∏_{d|N} η(dτ)^{r_d} is described by
its exponent vector r, and multiplying two such forms adds the vectors: f_r · f_s = f_{r+s},
exactly, with the weight ½ Σ r_d adding. Newman's congruences and Ligozat's cusp-order
inequalities are linear in r, so the exponent vectors of holomorphic forms on Γ₀(N) form a
monoid under addition (middle panel, level 11; (2,2) is η(τ)²η(11τ)², the weight-2 form of the
curve 11a1, and (2,2) + (2,2) = (4,4) is again a form). The right panel is the price: η has no
zeros in ℍ, so no eta quotient does either (`Fugue.etaQuotient_ne_zero`); every zero is at a
cusp, and E₄, which vanishes at ρ, is out of reach. This was the family's first and simplest
composition rule, and it was a rule for *multiplying forms*: the Cauchy product of q-expansions.

**31 — the sieve as a product** (`prime_sieve.py`). The rule "the zero set of a product is the
union of the zero sets", applied to the combs sin(πx/p) for p ∈ {2, 3, 5, 7}: the product
vanishes at every integer with a prime factor ≤ 7, so the survivors in [11, 100] are the primes.
`Fugue.sieve_2357` states exactly this (D9). It is Eratosthenes' sieve, not a new mechanism
(D10); the figure is kept because it is the first time the family touched the integers.

**32 — the Euler product, coefficient by coefficient** (`zeta_compose.py`). The dual of 31:
multiply "keep the powers of p" instead of "drop the multiples of p". Each Euler factor
(1 − p^{−s})^{−1} is the vector with 1 at the powers of p; multiplied in the Dirichlet ring
(the product with Σ_{de=n}), the result after the primes ≤ 48 is the all-ones vector on
n ≤ 48, the window of ζ(s) = Σ n^{−s}. The bottom panel shows each row as the indicator of the
smooth numbers. Lean: `Fugue.EulerCompose.prod_eulerFactor` (the product over S is the
S-smooth indicator), `prod_eulerFactor_apply_of_le` (all ones for n ≤ N), `smooth_tendsto_zeta`,
and Mathlib's analytic `riemannZeta_eulerProduct` (D1). Nothing about zeros or Re s ≤ 1.

**33 — divisibility detectors** (`factorize.py`). The inverse direction: a roots-of-unity
detector per prime, D_p(n) = (1/p) Σ_k e^{2πikn/p} = [p | n] (`Fugue.sum_exp_eq`,
`Fugue.detector_iff`, D12), applied in order with division whenever one fires. The script is
kept for one sentence: composing per-prime divisibility detectors is trial division
(`Fugue.first_spike_is_minFac`). A detector set finds only the primes it contains, which is the
√N limit of trial division and not an explanation of anything (D11). The June version did not
terminate and used a different detector set from the one it described (D15); both are fixed.

**36 — a census of the integers** (`landscape.py`). The terrain under 31–33: for every
n ≤ 10⁶, the largest prime factor (strata n = k·p under the prime diagonal), the smoothness
exponent (Dickman-type), and ω(n) against a Gaussian (Erdős–Kac, D19). π(10⁶) = 78498 is
asserted. The Gaussian uses the sample σ ≈ 0.99, not √(log log N) ≈ 1.62; the panel is a
picture of the theorem, not a test.

**37 — the Dirichlet ring** (`dna_algebra.py`). The named vectors δ, ζ, μ, Λ, χ₋₄ and four
identities, each verified to machine precision on n ≤ 80 and each a Lean theorem:
ζ * μ = δ (`Fugue.zeta_mul_moebius`, C4); ζ * ζ = d(n) (`zeta_mul_zeta_eq_sigma_zero`, C5);
4·(ζ * χ₋₄)(n) = r₂(n), the number of ways n is a sum of two squares (`Fugue.card_sq_add_sq`,
lifted to Z_{ℤ²}(s) = 4ζ(s)L(s, χ₋₄) by `Fugue.Epstein.epstein_square`; C6); and log · μ = Λ, i.e. the log-derivative of ζ
normalised by ζ^{−1} is the von Mangoldt function (`Fugue.pmul_log_mul` shows log is a
derivation, C9). The figure's titles still use the June word "DNA" and say "proven in Lean"
as a blanket; the module's header is owned elsewhere and the figure is reproduced as the
module draws it. Two June readings of this figure are refuted in the ledger: that the
E8 theta coefficients are σ₃ = ζ^{*4} (they are ζ * Id₃; C8), and that δ is a "do-nothing"
vector in both rings (it is the shift q in the Cauchy ring; C3, figure 43).

**39 — characters as a family of generators** (`lfunctions.py`). Replace the all-ones vector
by a Dirichlet character and the same ring does the same things: the Euler product
∏_p (1 − χ(p)p^{−s})^{−1} returns χ(n) (complete multiplicativity, C11); the direct series at
s = 1 lands on the class number formula for χ₋₄, χ₋₃, χ₋₇, χ₈ to ~1e−7 (D6; `Fugue.LFunction_chi4_one`
proves L(1, χ₋₄) = π/4); L(2, χ₋₄) is Catalan's constant by definition (D7); and
6·(ζ * χ₋₃)(n) counts the Eisenstein-integer lattice points of norm n (`Fugue.card_hexagonal`,
`Fugue.Epstein.epstein_hexagonal`; C7, D8), the ℚ(√−3) twin of figure 37's sum of two
squares. Nothing about the zeros of L(s, χ).

**40 — a weight-2 coefficient vector from an elliptic curve** (`bsd.py`). The a_p = p + 1 −
#E(𝔽_p) of five curves by point counting, their local Euler factors multiplied in the same
ring, and L(E, 1) by the smoothed sum (1 + w) Σ (a_n/n) e^{−2πn/√N} that the functional
equation gives. Checks: a_p of 11a1 equal the LMFDB newform 11.2.a.a at fifteen primes
(modularity seen in coefficients; also C16, since that newform is the eta quotient (2,2) of
figure 21); L(11a1, 1) and L(14a1, 1) match LMFDB to 1e−16 (D16); L(389a1, 1) cancels to
1e−16 with w = +1, so nothing forces it (D17). The rank-0 direction is Kolyvagin's theorem
seen numerically; everything else about BSD is untouched, and the "0.306" a June caption
attached to 389a1 is L′(37a1, 1), which no script computes (D18).

**43 — two rings on one coefficient list** (`hsln_two_rings.py`). The resolution. The same
vector ζ squares to n − 1 under the Cauchy product and to d(n) under the Dirichlet product;
at n = 1 that is 0 against 1 (`Fugue.cauchy_ne_dirichlet`, C1), and over a field the two rings
are not isomorphic (`Fugue.not_ringEquiv_powerSeries`). δ = [n = 1] is the identity of one and
the shift X of the other (`Fugue.delta_is_X`, C3). Each ring has its own derivation, log n for
Dirichlet and n for Cauchy. The multiplicative vectors are a submonoid under the Dirichlet
product and not closed under addition (`Fugue.multiplicativeSubmonoid`,
`Fugue.zeta_add_zeta_not_multiplicative`, C11). The June log's "the same operation in two
realisations" is withdrawn (C2).

**(no figure) — primes determine all coefficients** (`hsln_hecke_exercise.py`). Ramanujan's
τ(n) for n < 2600 as exact integers; Hecke's recursion τ(p^{r+2}) = τ(p)τ(p^{r+1}) − p¹¹τ(p^r)
and Mordell's multiplicativity checked; τ(N) rebuilt from the prime values alone for every
composite N < 2600; and τ(n) ≡ σ₁₁(n) (mod 691) for all n (`Fugue.Tau691.tau_congr_691`). The
theorem is a determination, primes → all coefficients, and runs one way only (H14).

## The bridges

1. **Two rings on one coefficient list.** Figure 21 composes forms (Cauchy product: zeros
   union, weights add, exponents add); figures 32, 37, 39, 40 compose L-functions (Dirichlet
   product: Euler factors multiply). Figure 43 shows the two products disagree on the all-ones
   vector at n = 1 and are not isomorphic rings. A Hecke eigenform's coefficient vector sits in
   both, and that is the whole of the bridge.

2. **Euler product = unique factorisation, coefficient by coefficient.** Figure 32's grid is
   `prod_eulerFactor`: the product over a set of primes is the indicator of the numbers smooth
   over that set, and the all-ones row is the fundamental theorem of arithmetic. Figures 31 and
   33 are the same fact read as a sieve and as trial division.

3. **Eta quotients make composition linear.** In the Cauchy ring, eta quotients are the family
   where multiplication is addition of exponent vectors and admissibility is a linear
   feasibility check (C14). The cost is C15: no interior zeros.

4. **Characters as a family of generators.** The Dirichlet ring's product, inverse and
   log-derivative act on any multiplicative vector; ζ is one, each χ is another, and
   ζ * χ_D is the coefficient vector of ζ_{ℚ(√D)}, which counts lattice points (figures 37 and
   39, `epstein_square`, `epstein_hexagonal`).

5. **a_p of an elliptic curve as a weight-2 coefficient vector.** Point counts give a_p,
   modularity says the vector is a newform's, the eta quotient (2,2) of figure 21 is that
   newform for 11a1, and the same ring product assembles L(E, s) (figure 40). The Hecke
   exercise says how much of such a vector is redundant: everything beyond the primes.

## Table

| Script | Figure | Theorem | Lean name | What is not claimed |
|---|---|---|---|---|
| `eta_compose.py` | 21 | η ≠ 0 on ℍ; Gordon–Hughes–Newman and Ligozat conditions are linear in the exponent vector | `Fugue.etaQuotient_ne_zero` | no interior zeros, so not a composition rule for general forms (C14, C15, C16) |
| `prime_sieve.py` | 31 | Sieve of Eratosthenes: sieving by primes ≤ √N leaves the primes in (√N, N] | `Fugue.sieve_2357` | not a new way of finding primes (D9, D10) |
| `zeta_compose.py` | 32 | Euler (1737): ∏_p (1 − p^{−s})^{−1} = ζ(s), Re s > 1 | `Fugue.EulerCompose.prod_eulerFactor`, `prod_eulerFactor_apply_of_le`, `smooth_tendsto_zeta`, `zeta_eulerProduct` | nothing about continuation, zeros or RH (D1) |
| `factorize.py` | 33 | Σ_{k<p} e^{2πikn/p} = p·[p \| n]; first spike is the least prime factor | `Fugue.sum_exp_eq`, `detector_iff`, `first_spike_is_minFac` | composing detectors is trial division; the "wall" is the √N limit and explains nothing (D11, D12, D15) |
| `landscape.py` | 36 | π(10⁶) = 78498; Dickman (1930); Erdős–Kac (1940) | — | the Gaussian uses the sample σ, not √(log log N); a picture, not a test (D19) |
| `dna_algebra.py` | 37 | Möbius inversion; ζ² = d; Jacobi's two-square count; log is a derivation, log·μ = Λ | `Fugue.zeta_mul_moebius`, `zeta_mul_zeta_eq_sigma_zero`, `Fugue.card_sq_add_sq`, `Fugue.Epstein.epstein_square`, `Fugue.pmul_log_mul` | σ₃ ≠ ζ^{*4} (C8); δ is not an identity in the Cauchy ring (C3); figure titles still say "DNA" (C4–C6, C9) |
| `lfunctions.py` | 39 | Dirichlet's class number formula; ζ_K = ζ · L(s, χ_D) for quadratic K | `Fugue.LFunction_chi4_one`, `Fugue.card_hexagonal`, `Fugue.Epstein.epstein_hexagonal`, `Fugue.multiplicativeSubmonoid` | only L(1, χ₋₄) is proved in Lean, the rest verified to ~1e−7; nothing about zeros (C7, C11, D6, D7, D8) |
| `bsd.py` | 40 | Hasse bound; modularity (Wiles et al.); Kolyvagin: L(E, 1) ≠ 0 ⇒ rank 0; Gross–Zagier | `Fugue.multiplicativeSubmonoid` (ring closure only; modularity not formalised) | BSD itself: no evidence for the leading term, ranks ≥ 2, or Ш; w = −1 zeros say nothing about rank; "0.306" does not exist (C16, D16, D17, D18) |
| `hsln_two_rings.py` | 43 | the Cauchy and Dirichlet products on ℕ → R are distinct, non-isomorphic rings; multiplicative functions are a submonoid | `Fugue.cauchy_ne_dirichlet`, `not_ringEquiv_powerSeries`, `delta_is_X`, `pmul_log_mul`, `multiplicativeSubmonoid`, `zeta_add_zeta_not_multiplicative` | not "the same operation in two realisations" (C1, C2, C3, C9, C11, C17) |
| `hsln_hecke_exercise.py` | — | Mordell (1917), Hecke (1937): an eigenform's coefficients are determined by their prime values; τ ≡ σ₁₁ (mod 691) | `Fugue.Tau691.tau_congr_691`, `τ_int`, `Fugue.RamanujanL.τ_two`, `τ_three` | a determination, not an inversion; Mordell multiplicativity not formalised (H14) |
