# Spectral and Fourier analysis

The family that the June exploration began with. One idea
runs through it: a modular form is a Fourier series, its coefficient vector (the Fourier
coefficients, which the June files called the DNA) is the object the network trains, and the
same vector carries arithmetic — divisor sums, Hecke eigenvalues, the primes. Each figure below
opens with a picture and a physical reading before its theorem. The figures keep their June
numbers. Two scripts were listed as unnumbered; in fact the archive had numbered them:
`hecke_compression.py` wrote 41, 42 and 43 (its third figure is folded into 42 here) and
`hsln_e8_divisor_bridge.py` wrote 56, so those are the numbers used. Every grade cites the
audit's evidence ledger (`rmorleysmith/fugue`, `Fugue/02-threads/evidence-ledger.md`) by row.

## 34 — the zeros as a spectrum (`spectral.py`)

The picture: |ζ(½ + it)| touching zero at each γₙ, then the histogram of gaps between
consecutive zeros, then the histogram of gaps between consecutive eigenvalues of a random
Hermitian matrix. The two histograms are the same curve. The physical reading is Wigner's:
energy levels of a heavy nucleus repel one another, and so do the zeros — small gaps are rare in
a way that independent points (Poisson, the dashed curve) never show.

The precise statement is Montgomery–Odlyzko: the unfolded spacings of the zeros follow the GUE
law. Here it is a measurement — KS distance 0.048 to GUE against 0.328 to Poisson, on 479
spacings — and the ledger grades it NUMERICAL (D5). The functional equation that puts the
critical line at Re s = ½ is PROVEN (D2, Mathlib `completedRiemannZeta_one_sub`). What the picture
*suggests* is Hilbert–Pólya: a self-adjoint operator whose eigenvalues are the γₙ, which would
make them real and prove RH. That is a question. The only formal statement in that direction is
the trivial half — a symmetric operator has real eigenvalues (Mathlib
`LinearMap.IsSymmetric.conj_eigenvalue_eq_self`), graded H7 as "PROVEN as an implication;
hypothesis open".

## 35 — the zeros rebuild the primes (`explicit.py`)

The picture: Chebyshev's staircase ψ(x) = Σ_{pᵏ≤x} log p, and under it the smooth curve x minus
one oscillation per zero. One zero gives a bend; ten give waves; five hundred give the staircase,
steps at 2, 3, 4, 5, 7, 8, 9, … with the right heights. The physical reading is Fourier
synthesis: the primes are a signal and the zeros are its frequencies.

The author's own route to this was an intuition: before knowing the
residue theorem, a residue was imagined as "what the primes leave behind as a function unfolds
along the number line", a running log of position. The precise statement is that
−ζ′/ζ = Σ Λ(n) n^{−s} (Mathlib `LSeries_vonMangoldt_eq_deriv_riemannZeta_div`) is exactly the series
of those logs; its residue at s = 1 is 1 (`riemannZeta_residue_one`) and gives the main term x;
each zero ρ is a further residue, the oscillation −x^ρ/ρ; the explicit formula is that residue
calculus (von Mangoldt 1895). The quantitative shadow — zero-free region ⟺ error term in ψ — is
`Ingham.zeroFreeRight_iff_psiError` in FormalPNT. Ledger D4: PROVEN classically, NUMERICAL here
(mean error 0.58 → 0.04 from 1 to 500 zeros).

## 38 — one generator, two bases (`lie_derivative.py`)

The picture has two halves. Left: a particle sliding down the landscape |F|² into a zero, and
the measured dE/dt lying exactly on −|∇E|². Right: the coefficients of q d/dq E₄ as bars, with the
coefficients of (E₂E₄ − E₆)/3 as circles on top of them.

The June question was whether the network had "a Lie derivative of its own". The left half
answers with the chain rule: along a gradient flow, dE/dt = −|∇E|², nothing more. The right half
is the substantive one. The generator of the dilation q ↦ λq is D = q d/dq, which on a q-series
is aₙ ↦ n·aₙ; Ramanujan's system says the ring C[E₂, E₄, E₆] is closed under it, and Serre's
correction ϑ_k = D − (k/12)E₂ sends weight k to weight k + 2 (both in Mathlib:
`normalizedDerivOfComplex_E₄`, `serreDerivative_E₄`; ledger C10). Read in the Dirichlet basis
n^{−s}, the same dilation generator is aₙ ↦ aₙ log n — a derivation of the Dirichlet ring
(`Fugue.pmul_log_mul`, C9) — and applied to ζ and normalised by μ it is the von Mangoldt function
(`log_mul_moebius_eq_vonMangoldt`). That is the bridge back to figure 35: the operator that
climbs the weight ladder on the modular side is the operator that extracts the primes on the
Dirichlet side. The June suptitle called the flow and the grading "two faces of one generator";
they are not, and the script now says so.

## 63 — the generator on a divisor series (`hsln_lie_landscape.py`)

The picture: the divisor series D_N = Σ_{n|N} qⁿ for N = 83·97, its spikes at 1, 83, 97
reweighted by n and n² under θ and θ²; the sums Σθᵏ D_N standing exactly on σ_k(N); and a flat
window for a larger N where the only divisor in range is 1.

The true content is Σ_n θᵏ D_N = σ_k(N) — and for k = 3 those are the coefficients of
E₄ = Θ_{E8} (A2) — together with ϑ₄E₄ = −E₆/3 (C10). The honest content is the flat panel:
θ is diagonal in the grading basis, so it multiplies a coefficient that is absent by n and gets
0. The June script hoped the graded symmetry would expose structure the coefficient vector did
not show; its own conclusion, kept here, was "structure, not search". Its last panel had used a
249-bit number from an external source; it is replaced by an ordinary composite.

## 41, 42 — the primes determine the form (`hecke_compression.py`)

The picture (41): the values a_p = p + 1 − #E(F_p) of the curve y² + y = x³ − x, fenced by
Hasse's ±2√p and, once the angle arccos(a_p/2√p) is histogrammed, filling Sato–Tate's sin² law.
Then (42): each column a_{p^r} unrolling from its single a_p by a two-term recurrence, the
reconstructed a_n for n ≤ 200, and that reconstruction standing exactly on LMFDB's listed
q-expansion of newform 37.2.a.a.

The guess was that an L-function's coefficient vector is a compression scheme with a decoder of
our own. The precise statement is Hecke's (1937): a normalised eigenform's coefficients are its
Hecke eigenvalues, multiplicative, with a_{p^{r+2}} = a_p a_{p^{r+1}} − p^{k−1} a_{p^r} at each
prime, so the prime values determine everything; the decoder is that recursion. For a curve the
identification a_p = p + 1 − #E(F_p) is modularity (Wiles 1995 for 37a1). Ledger A5 (E₄ is an
eigenform, numerically) and D16 (37a1 point counts; L(37a1, 1) vanishes numerically). Mathlib has
no Hecke operators at this commit, so none of this has a Lean name; the nearest formal object is
`Fugue.HeckeE4.heckeE4_functional_equation`, which is Hecke's converse direction for E₄. The
ledger's H14 refutes the June hope that the theorem runs backwards — a_p ↦ p is not even
injective in weight 2 — and the June panel that captioned a partial Euler product at s = 1 as
showing L(E, 1) = 0 is dropped, since partial Euler products at the centre do not converge to
L(E, 1).

## 56 — the divisor function is the bridge (`hsln_e8_divisor_bridge.py`)

The picture: four rows of the Dirichlet ring each built from ζ (d = ζ*ζ, σ₃ = Id₃*ζ,
Λ = log*μ, μ = ζ⁻¹); 240σ₃(n) standing on the counted E8 vectors of squared norm 2n; the
coefficients of E₄² = Θ_{E8⊕E8}, 480 at n = 1; and Σσ₃(n)n^{−s} lying on ζ(s)ζ(s−3).

This is the family's closing bridge: the element σ₃ of figure 37's ring is the coefficient
vector of figure 38's E₄, which is the theta series of the lattice the whole June network was
built on. Every arrow is a theorem with a Lean name: `Fugue.sigma_three_eq` (and the refutation
of the June guess σ₃ = ζ⁴, C8), `Fugue.thetaE8_eq_E₄` and `reprCount_e8` (A2),
`theta_e8_sum_e8` and `E₄sq_eq` with `sigma_seven_eq` (C13, C1×C8), `thetaD16_eq_thetaE8_sq`
(Witt's isospectral pair), `HeckeE4.LSeriesHasSum_sigma_three` and `Epstein.epstein_E8`.

## The numerical companion (`hsln_spectral.py`)

No figure. A stdlib Jacobi eigensolver checks the elementary facts the family leans on: a
symmetric matrix has a real spectrum, scaling the matrix scales the spectrum, the spectral zeta
Σλₙ^{−s} of the spectrum {1, 2, 3, …} is ζ (ζ_T(2) = π²/6), and a GOE matrix shows level
repulsion. The June version announced "all theorems of HSLNSpectral.lean verified"; that file is
not in fugue-formal and the facts are elementary, so the script now says only what it checks.

## The bridges

- **One generator, two bases.** D = q d/dq on the q-side is aₙ ↦ n aₙ (figures 38, 63);
  −d/ds on the Dirichlet side is aₙ ↦ aₙ log n (figure 38's corner note). Both are the Euler
  operators of the two inequivalent rings of ledger C1 (`Fugue.TwoRings`); the first closes on
  Ramanujan's ring, the second turns ζ into the primes.
- **The explicit formula is the residue calculus of −ζ′/ζ.** The series that the Dirichlet-side
  generator produces from ζ (figure 38) is the one whose residues at 1, 0, the trivial zeros and
  the ρ give the four terms of figure 35. The author's residue intuition is right in substance.
- **Hilbert–Pólya is a question, not a result.** Figure 34 shows statistics consistent with an
  operator; nothing in the family exhibits one, and H7's implication "operator ⇒ RH" has an open
  hypothesis. The family's one formal spectral fact is that symmetric operators have real
  eigenvalues.
- **Fourier coefficients carry arithmetic.** σ₃ (56), Hecke eigenvalues (41, 42) and the divisor
  sums under θ (63) are three readings of the same coefficient vector; the eigenform property is
  what makes the prime values sufficient, and it is the one ingredient without a Lean name.

## Table

| script | figure | theorem | Lean name | what is not claimed |
|---|---|---|---|---|
| `spectral.py` | 34 | Montgomery–Odlyzko GUE statistics (D5, NUMERICAL); functional equation (D2, PROVEN) | Mathlib `completedRiemannZeta_one_sub`; `LinearMap.IsSymmetric.conj_eigenvalue_eq_self` (H7, implication only) | that a Hilbert–Pólya operator exists, or that the network is one |
| `explicit.py` | 35 | von Mangoldt's explicit formula (D4, PROVEN classically / NUMERICAL here) | Mathlib `LSeries_vonMangoldt_eq_deriv_riemannZeta_div`, `riemannZeta_residue_one`; FormalPNT `Ingham.zeroFreeRight_iff_psiError` | anything about the location of the zeros; "interpolation" as a proof technique |
| `lie_derivative.py` | 38 | Ramanujan's system and Serre's derivative (C10, PROVEN); log is a derivation, log*μ = Λ (C9, PROVEN); zeros are minima of \|F\| (A9) | Mathlib `normalizedDerivOfComplex_E₂/E₄/E₆`, `serreDerivative_E₄/E₆`, `log_mul_moebius_eq_vonMangoldt`; `Fugue.pmul_log_mul`, `Fugue.isLocalMin_norm_of_eq_zero` | that the flow's dE/dt and the grading's D are one operator; that a network "knows its derivative" |
| `hsln_lie_landscape.py` | 63 | Σ_n nᵏ[n\|N] = σ_k(N) (definition); 240σ₃ are E₄'s coefficients (A2, PROVEN); ϑ₄E₄ = −E₆/3 (C10) | Mathlib `ArithmeticFunction.sigma`, `serreDerivative_E₄`; `Fugue.thetaE8_eq_E₄`, `Fugue.reprCount_e8` | that θ exposes anything beyond a reweighting; the June external-number panel |
| `hecke_compression.py` | 41, 42 | Hecke 1937 (eigenvalues = coefficients, recursion, multiplicativity); Hasse 1933; Sato–Tate (BLGHT 2011); modularity (Wiles 1995); A5 (NUMERICAL), D16, H14 (REFUTED backwards) | none for Hecke operators (not in Mathlib); nearest: `Fugue.HeckeE4.heckeE4_functional_equation` | that the determination theorem inverts; the June partial-Euler-product-at-s=1 panel |
| `hsln_e8_divisor_bridge.py` | 56 | σ₃ = Id₃*ζ (C8 corrects ζ⁴); Θ_{E8} = E₄ (A2); Θ_{L⊕M} = Θ_LΘ_M (C13); E₄² = E₈, σ₇ identity (C1×C8); Witt 1941; Σσ₃n^{−s} = ζ(s)ζ(s−3) | `Fugue.sigma_three_eq`, `thetaE8_eq_E₄`, `reprCount_e8`, `theta_e8_sum_e8`, `E₄sq_eq`, `sigma_seven_eq`, `thetaD16_eq_thetaE8_sq`, `HeckeE4.LSeriesHasSum_sigma_three`, `Epstein.epstein_E8` (all PROVEN) | that the chain is new or carries information back from E8 to the primes; any claim about string theory |
| `hsln_spectral.py` | — | symmetric ⇒ real spectrum; spectral zeta of {n} is ζ; GOE level repulsion (H7 implication; D5) | Mathlib `LinearMap.IsSymmetric.conj_eigenvalue_eq_self` | that any of it bears on where the zeros are; "theorems of HSLNSpectral.lean verified" |
