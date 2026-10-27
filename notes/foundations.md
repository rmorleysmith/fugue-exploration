# Foundations: figures 01–13

The June exploration started from one picture: a point moving in the upper half-plane,
pulled downhill by a modular form, coming to rest where the form vanishes. Everything else
in the repository is built on the objects in these thirteen figures, all drawn by
[`scripts/render.py`](../scripts/render.py) with the shared modules `modular`, `e8`, `siegel`,
`geom`, `coloring`, `dna` and `style`. This note reads them in order, names the four bridges
between them, and ends with the table of what each figure rests on and what it does not claim.
Grades are those of the audit's evidence ledger
(`rmorleysmith/fugue`, `Fugue/02-threads/evidence-ledger.md`), cited by row.

Vocabulary: the Fourier coefficients of a form, which the June files called the "DNA", are
the *coefficient vector*; the zeros of a form, which June called "neurons", are *zeros*; the
off-diagonal entry of a degree-2 Siegel period, which June called "gravity", is the
*coupling* z (or κ once normalised); the "genus" of a theta series is its *degree*.

## 1. The form (figures 01, 02, 13)

**Figure 01** is the phase portrait of E₄ on the upper half-plane: hue is the argument,
brightness bands are level curves of |E₄|, and the fundamental domain of SL(2,ℤ) is outlined.
Every hue meets at one kind of point, τ ≡ ρ = e^{2πi/3}, and at no other: E₄ has exactly one
zero per fundamental domain, simple, at the corner of the domain. This is the valence
formula (Serre, *A Course in Arithmetic*, VII.3; ledger A3, PROVEN; Lean `Fugue.E₄_rho`, and
`Fugue.eq_zero_at_rho` for every level-one form of weight not divisible by 6). The picture
also shows what the half-plane does near the real axis: the same zero recurs at every
SL(2,ℤ)-image of ρ, and the images accumulate towards Im τ = 0. In the q-disk these images
sit at |q| = exp(−π√3/(c² − cd + d²)) with (c, d) coprime — countably many points, piling up
at the rim (ledger A6, PROVEN). "The zero set of E₄ is a single point" is wrong as stated;
"one per fundamental domain" is right (A10).

**Figure 02** is the same function after the Cayley transform τ ↦ (τ − i)/(τ + i), which
sends the half-plane to the unit disk and the real axis to the boundary circle. The cusps
ℚ ∪ {∞} are now visible all round the rim and the fractal accumulation of the modular tiling
is the edge of the picture. **Figure 13** is this disk with no axes or title, a cover image.

## 2. The bridge from a lattice to the form (figure 06)

**Figure 06** shows where E₄ comes from. The left panel is the 240 roots of the E8 lattice
projected to the Coxeter plane, the eigenplane of the Coxeter element for the eigenvalue
e^{2πi/30}: eight concentric 30-gons, Petrie's projection (Bourbaki, *Lie Groups and Lie
Algebras*, ch. VI). The right panel is the number of lattice vectors of each squared norm:
240, 2160, 6720, 17520, … = 240 σ₃(n), so the theta series Σ_v q^{‖v‖²/2} of E8 is
1 + 240 Σ σ₃(n) qⁿ = E₄.

*Bridge 1: Θ_{E8} = E₄.* The theta series of an even unimodular lattice of rank 8 is a level-one
modular form of weight 4, and that space is one-dimensional (Serre VII; ledger A1–A2,
PROVEN; Lean `Fugue.thetaE8_eq_E₄`, `Fugue.reprCount_e8`, `Fugue.e8Form_even`,
`Fugue.e8Form_pos`). So the coefficient vector of the E8 landscape is a classical divisor sum,
and the lattice itself enters the exploration only through this function: the state space is
the half-plane, not E8 (system-spec §4.3.3).

## 3. The energy and its minima (figures 03, 04, 08)

**Figure 03** draws log₁₀|F|² as a surface. For E₄ (left) there is one well per fundamental
domain, at ρ. For the right panel we took the weight-4 level-11 newform 11.4.a.a, truncated
to n ≤ 5, and found four wells.

*Bridge 2: a holomorphic energy whose minima are zeros.* If F is holomorphic and not locally
constant, |F| has no local minimum except where F = 0: the minimum-modulus principle, a
corollary of the maximum-modulus principle applied to 1/F (ledger A9, PROVEN; Lean
`Fugue.eq_zero_of_isLocalMin_norm`, `Fugue.isLocalMin_norm_of_eq_zero`). Gradient descent on
|F|² therefore has the zeros, and only the zeros, as attractors; every other critical point is
a saddle. This is the sharpest observation of the June work and the one everything else
leans on.

Two things the right panel of figure 03 does not show. The four wells are zeros of a degree-5
polynomial in q, not of the newform: a truncation of degree N has N roots whatever the form,
and only the limit points of those roots, as N grows, are zeros of the form (Hurwitz;
ledger A7, NUMERICAL, and A8, REFUTED: the Sturm bound governs uniqueness of a form, not its
number of zeros). And 11.4.a.a is not "E₄ truncated to the same order"; the precise relation
is a congruence, a(n) ≡ σ₃(n) − σ₃(n/11) modulo the prime (61, √3 − 8) of ℤ[√3], the
weight-4 level-11 twin of τ(n) ≡ σ₁₁(n) (mod 691) (ledger C16b).

**Figure 04** is the flow. For the rest of the family the landscapes are *designed*: the
coefficient vector of ∏(q − qᵢ) has its zeros at the prescribed qᵢ by construction
(`dna.dna_from_neurons`). The left panel follows the direction of −∇|F|² from a grid of seeds
at unit speed and colours each path by the zero it reaches: three basins, separated by
ridges through the saddles. The right panel integrates the ODE τ̇ = −α∇|F|² itself from three
seeds and plots |F(τ(t))| together with the speed. For holomorphic F the Euclidean gradient
of |F|² has modulus exactly 2|F||F′| (ledger A11), so the speed is large where |F| is large
(near the real axis), tiny high in the strip where F is nearly constant, and zero at the
zero: motion freezes on arrival, and rescaling the coefficient vector by c multiplies every
rate by |c|² at once, one clock for the whole landscape. The June version of this panel
showed a "driven spike train"; with the normalisation used, the drive was 10⁷ times the
landscape gradient and the threshold was the 88th percentile of the very trace it was applied
to, so the spikes were not at zeros (ledger A12, REFUTED). That panel is gone.

**Figure 08** is the left panel of figure 04 at scale: three zeros and then ten, the same
vectorised integration, no field evaluated on a grid. It is a picture of basins of
polynomials. It is not evidence of a "controllable number of neurons": the ledger refutes the
capacity claims that were once hung on it (A7, A8).

## 4. Products union zero sets (figures 05, 11, 12)

*Bridge 3: multiplying forms unions zero sets.* A product vanishes exactly where a factor
does, in any integral domain; for q-series, multiplying the forms is the Cauchy product of
the coefficient vectors (ledger B1, B2, PROVEN; Mathlib `mul_eq_zero`).

**Figure 05** shows F_A with two prescribed zeros, F_B with two others, and F_A·F_B with all
four. The script checks that the product's roots are the union of the parents' to 1e−16.
**Figure 11** puts the two ways of combining side by side: the sum F_A + F_B has new zeros and
none of the parents' (ledger B4), the product keeps every one. Addition reshapes the field;
multiplication stacks the attractors.

**Figure 12** is the same law with a lattice-theoretic headline. A direct sum of lattices
multiplies theta series, Θ_{L⊕M} = Θ_L·Θ_M, at every degree, because the Gram matrix is
block-diagonal (ledger C13, PROVEN; Lean `Fugue.theta_directSum`, `Fugue.theta_e8_sum_e8`), so
Θ_{E8⊕E8} = E₄² = E₈ (Lean `Fugue.thetaE8_sq_eq_E₈`). The panels illustrate it
with two designed polynomials, a fixed factor and a second factor: the second factor
reshapes the field of the first and its zeros are added to the set. June called the second
factor the "gravity dial" of an E8×E8 heterotic model.

What these three figures are not: compositions of modular forms. M₈(SL(2,ℤ)) is spanned by
E₄², which has one double zero at ρ, so no weight-8 level-one form has the four distinct
zeros of figure 05 (ledger B5, REFUTED as a statement about modular forms). The pictures are
true for polynomials, and the zero-union law needs no modularity at all (B6). Nothing here is
trained, and the "zero-shot merge" of two trained models was never run (F9).

## 5. Degree 2 (figures 07, 09, 10)

*Bridge 4: the degree-2 Siegel space as the coupled pair.* The Siegel upper half-space ℍ₂ is
the set of symmetric 2×2 complex matrices Z = [[τ₁, z], [z, τ₂]] with positive-definite
imaginary part: two copies of the half-plane and a coupling z between them. The degree-2
theta series Θ^{(2)}_{E8}(Z) = Σ_{v₁,v₂} exp(πi tr(G(v₁,v₂) Z)) is a Siegel modular form of
weight 4 for Sp(4,ℤ) (Freitag, *Siegelsche Modulfunktionen*, 1983; ledger A2), computed here
from exact representation numbers r₂(T) by enumerating pairs of lattice vectors (`siegel.py`).
When z = 0 the Gram matrices split and Θ^{(2)} = E₄(τ₁)E₄(τ₂); the script checks this to 1e−15.
The normalised coupling κ = (Im z)²/(y₁y₂) of the two flows is a genuine Siegel metric
coefficient, positive definite exactly when |κ| < 1 (ledger E2, PROVEN; Lean
`Fugue.couplingMatrix_posDef_iff`), and a common zero of the two factors stays an equilibrium
for every coupling (E3; Lean `Fugue.equilibrium_preserved`).

**Figure 07** shows three complex-line slices of ℍ₂: the diagonal diag(τ, τ), where the
portrait is E₄(τ)² (double zeros at ρ); a coupled slice with z = 0.35, where the double
zeros split and move; and the coupling plane itself, Z = [[1.15i, z], [z, 1.15i]] with z
varying, cut off where Im Z stops being positive definite.

**Figure 09** follows the zeros along the one-parameter family of slices
Z = [[τ, z], [z, τ + w]]. On each complex slice the zero set is a finite set of points (red);
swept over w it traces a curve. Decoupled (z = 0) the curve is the two straight lines
Im τ = √3/2, as E₄(τ)E₄(τ + w) = 0 requires (the script checks this to 1e−8; any bend in the
left panel is the 3-D projection). Coupled (z = 0.35) the branch genuinely bends and
reconnects. **Figure 10** repeats the coupled picture for z = 0.10, 0.25, 0.40, 0.55.

What the headline of the June figure got wrong: the zero set of one holomorphic equation on
the complex 3-fold ℍ₂ is a *divisor* of complex dimension 2, not a curve (ledger A13,
REFUTED). The curve in figures 09–10 is its trace on a two-real-parameter family of slices.
The right generalisation of "how many zeros" is not a count but a divisor class, 4λ₁ on the
moduli space of principally polarised abelian surfaces, which at degree 1 is the valence
formula again. The figures also keep only the roots with Im τ in a window (2 of the 8 roots
per slice), the region where the norm ≤ 8 truncation is accurate; the caption says so.
Tabulating r₂ costs about 38 s at norm cutoff 8, so degree 2 is priced, not blocked, and it
is nothing like a training loop.

## 6. What survives

The four bridges are theorems: Θ_{E8} = E₄; zeros are the only minima of |F|²; products union
zero sets and coefficient vectors convolve; direct sums multiply theta series and the degree-2
coupling has a precise positivity condition. The arithmetic of ρ's orbit in the q-disk is
exact and elegant. What did not survive is the bookkeeping that turned these facts into a
network with a controllable number of zeros: the figures that seemed to show it were about
polynomials, and the one that seemed to show spiking was a driven free particle.

## Table

| Script | Figure | Theorem | Lean name (`rmorleysmith/fugue-formal`) | What is not claimed |
|---|---|---|---|---|
| `render.py e4_half` | 01 `01_e4_halfplane.png` | E₄ has one simple zero per fundamental domain, at ρ (valence formula; A3 PROVEN); its q-images lie at exp(−π√3/(c²−cd+d²)) (A6 PROVEN) | `Fugue.E₄_rho`, `Fugue.eq_zero_at_rho` | that the zero set is a single point (A10 REFUTED): it is one orbit, infinite in the q-disk |
| `render.py e4_disk` | 02 `02_e4_disk.png` | Cayley transform ℍ → 𝔻 is a biholomorphism; same zero as 01 | — | nothing beyond a change of coordinates |
| `render.py energy` | 03 `03_energy_zeros.png` | minimum-modulus principle: local minima of \|F\| are the zeros (A9 PROVEN); 11.4.a.a ≡ (E₄(τ)−E₄(11τ))/240 mod (61, √3−8) (C16b, numerically certified) | `Fugue.eq_zero_of_isLocalMin_norm`, `Fugue.isLocalMin_norm_of_eq_zero` | that the four wells are zeros of the newform (A7 NUMERICAL: roots of a truncation), or that their number is set by the Sturm bound (A8 REFUTED) |
| `render.py dynamics` | 04 `04_dynamics.png` | zeros are the only attractors of τ̇ = −α∇\|F\|² (A9); \|τ̇\| = 2α\|F\|\|F′\| exactly, so motion freezes at zeros (A11, SCOPE-LIMITED: Euclidean, not hyperbolic, gradient) | `Fugue.eq_zero_of_isLocalMin_norm` | spiking, liquid time constants, per-zero tunable clocks (A12 REFUTED; thread architecture §4); the landscapes are polynomials (B5) |
| `render.py merge` | 05 `05_merge.png` | zeros(FG) = zeros(F) ∪ zeros(G) (B1 PROVEN); product of forms = Cauchy product of coefficient vectors (B2 PROVEN) | `mul_eq_zero` (Mathlib) | that F_A, F_B, F_A F_B are modular forms (B5 REFUTED), or that any trained model was merged (B3 NUMERICAL, F9 untested) |
| `render.py roots` | 06 `06_e8_roots_coefficients.png` | E8 even unimodular of rank 8 (A1 PROVEN); #{‖v‖² = 2n} = 240σ₃(n), Θ_{E8} = E₄ (A2 PROVEN); Petrie projection has 8 rings of 30 | `Fugue.e8Form_even`, `Fugue.e8Form_pos`, `Fugue.reprCount_e8`, `Fugue.thetaE8_eq_E₄` | that the lattice is the state space (spec §4.3.3): it enters only through E₄ |
| `render.py siegel` | 07 `07_siegel_degree2.png` | Θ^{(2)}_{E8} is a Siegel form of weight 4 for Sp(4,ℤ) (Freitag; A2); on the diagonal it is E₄(τ₁)E₄(τ₂) (C13 PROVEN); Im Z ≻ 0 ⟺ \|κ\| < 1 (E2 PROVEN) | `Fugue.theta_directSum`, `Fugue.couplingMatrix_posDef_iff` | Siegel forms are not in Mathlib; the truncation is at norm ≤ 6 |
| `render.py basins` | 08 `08_basins_paths.png` | basins of a gradient flow are separated by stable manifolds of saddles (A9) | — | any capacity or "neuron count" statement (A7 NUMERICAL, A8 REFUTED); the ten zeros are prescribed, not learned |
| `render.py divisor` | 09 `09_degree2_divisor.png` | the zero set of Θ^{(2)} is a divisor of complex dimension 2 (A13); on z = 0 it is Im τ = √3/2 for all w (C13) | `Fugue.theta_directSum` | that the zero set is a curve (A13 REFUTED); 6 of 8 roots per slice are discarded outside the accurate window |
| `render.py divisor_sweep` | 10 `10_divisor_sweep.png` | as 09, for four couplings | — | as 09; no dynamics on ℍ₂ is shown |
| `render.py compose` | 11 `11_compose_add_vs_mul.png` | product unions zeros (B1 PROVEN); the sum does not (B4 PROVEN) | `mul_eq_zero` | that the operands are modular forms (B5 REFUTED) |
| `render.py direct_sum` | 12 `12_heterotic_e8xe8.png` | Θ_{L⊕M} = Θ_L Θ_M (C13 PROVEN); Θ_{E8⊕E8} = E₄² = E₈ | `Fugue.theta_directSum`, `Fugue.theta_e8_sum_e8`, `Fugue.thetaE8_sq_eq_E₈` | that the panels are theta series (they are designed polynomials); that "direct sum" and "tensor product" differ as operations on coefficient vectors (B9 REFUTED: both are Cauchy convolution) |
| `render.py disk_clean` | 13 `13_e4_disk_clean.png` | as 02 | — | nothing; a cover image |
