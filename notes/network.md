# The network: proof-of-concept and composition experiments

Figures 14–30, 44 and 45. Scripts `poc`, `poc_sine`, `poc_xor`, `poc_layer`, `hsln`, `hsln_dyn`,
`hsln_genus2`, `coupling_sweep`, `cm_alphabet`, `cm_task`, `cm_classifier`, `cm_tensor`,
`timeseries`, `scaling`, `skill_compose`, `grok`, `hsln_compose_two`, `hsln_skill_forge`, plus the
figureless `bench` and `animate`. Ledger rows are those of the audit
(`rmorleysmith/fugue`, `Fugue/02-threads/evidence-ledger.md`); Lean names are in
`rmorleysmith/fugue-formal`.

## The picture first

Take a holomorphic function F on the upper half-plane and call E = |F|² its energy. A ball placed
anywhere and allowed to roll downhill ends in a dent, and the dents are exactly the zeros of F:
a non-constant holomorphic function has no interior minimum of |F| except where it vanishes
(`Fugue.isLocalMin_norm_of_eq_zero`, `Fugue.eq_zero_of_isLocalMin_norm`; ledger A9). That is the
whole architectural seed, and it needs nothing modular. The June exploration asked four
questions of this landscape, and the figures of this family are the answers in the order they
were asked:

1. Can the landscape be trained — can gradient descent on the coefficients of F move its zeros
   so that a driven ball does a task? (figures 14–17)
2. Can two trained landscapes be merged without retraining? (18, 19, 20, 22, 26, 44, 45)
3. Can the landscape say something discrete? (23, 24, 25)
4. What else does the picture suggest — spectra, cost, stacking, and the circle? (27–30)

## The five bridges

**Gradient flow on |F|² as a reservoir.** Figures 14–16 are the smallest trainable version:
a degree-5 polynomial in q = e^{2πiτ}, a drive B·u, Euler steps, and backpropagation through
the rollout. It trains (ledger F1), but the honest reading of figure 14 is that the loss is a
minimum over the path and the targets are not attractors (F2); figure 15 is a 15-parameter
recurrent model fitted by backpropagation through time; figure 16 contains no time at all —
it is a static four-point discriminant in which training suppresses two spikes the untrained
field already had (F3). Figure 17 shows that "layering by tensor product" is a
reparameterisation of a monic quartic, not added capacity (F4); the direct quartic was added
to the figure so that the comparison is visible.

**The coefficient vector is a finite list, by Sturm — but these are not modular forms.** The
spec's reason for a finite parameter list is the Sturm bound (system-spec §2.2; THE-STORY §1a),
which makes a modular form a finite object. The scripts, however, train free polynomials whose
roots are not the zeros of any modular form (THE-STORY §3). So the finite list is real, the
modular justification is not used, and every header in this family says so. The one place the
vector really is a modular object is `dna.py` in the Foundations family, whose level-11 newform
coefficients are only displayed here.

**Merge by multiplication.** zeros(F·G) = zeros(F) ∪ zeros(G) — a product vanishes where a
factor does (`mul_eq_zero`; ledger B1) — and in coefficients multiplication is the Cauchy
product, one convolution. Figure 18 trains two detectors apart and multiplies them; figure 45
does it for three and then divides one out again by exact polynomial division; figure 26 does
it for four single-zero models; figure 44 trains two factors jointly with the loss on the
product only. In all four the zero union is verified to 1e−9 or better (B3: 1.76e−16 in the
audit's run). What is **not** a theorem is that the merged readout still does both jobs: the
spike map normalises by max|F| and each factor modulates the other's 1/|F| profile. Figure 18
and figure 45 measure that it does, for these detectors. Figure 19 shows the other side: the
gradient of the product is a weighted sum of the two flows, not their union, so basins
reshape and both inputs are captured by the wrong attractor in the crossing geometry.

**Direct sum of degree-2 forms for independent coordinates.** Figure 20 is the repair: put
each task on its own coordinate of the 2×2 period matrix Z = diag(τ₁, τ₂). On the diagonal
slice the energy factorises and each coordinate follows its parent's flow, exactly by
construction. The algebra behind the picture is Θ_{L⊕M} = Θ_L·Θ_M at every degree
(`Fugue.reprCount_directSum`, `Fugue.theta_directSum`; C13). In standard
vocabulary the June "genus 2" is Siegel **degree** 2, and a direct sum of lattices does not
raise it — choosing the degree-2 state space does. Figure 22 then turns on the off-diagonal
coupling κ (the June "gravity"): the coupling matrix K is the Siegel metric's block (E2), it is
positive definite iff |κ| < 1 (`Fugue.couplingMatrix_posDef_iff`; E2/E5), a common zero stays
an equilibrium for every K (`Fugue.equilibrium_preserved`; E3), routing error stays below
0.006 but is not zero (E6) — and the June claim that κ "unlocks chaos" is **refuted**: the
separation of nearby trajectories is bounded, non-monotone and collapses below 1e−9 on 1368
of 12 000 steps (E4). The figure's third panel now says so.

**CM points as a symbol alphabet.** Figure 23 is the codebook: the nine Heegner points, whose
j-values are rational integers and perfect cubes (classical; G1). Figures 24 and 25 use it,
and the honest finding is that only two symbols — j = 1728 at i and j = 0 at ρ — are reached by
the flow, so the "alphabet" is a bit code (G3), and the classifier of figure 25 is reservoir
computing on a hand-fixed landscape with a six-parameter linear encoder (F8). Figure 26 shows
the nearest-zero readout of a composed model is the nearest-centroid classifier.

## The tail: spectra, cost, stacking, the circle

Figure 27 drives a one-zero well with a single tone and reads the spectrum of the motion: weak
harmonics (about 1% of the fundamental) appear, the ordinary signature of an anharmonic well;
the June panel's rectangular window had leakage larger than the harmonic it pointed at, and the
June title's "in the E8 modular form" was wrong — the well is |q − q₀|². Figure 28 measures
cost and corrects the June caption: the degree-2 theta is a table built once, after which
evaluation is comparable to a degree-1 series (F5). Figure 29 stacks two skills; the audit
showed the answer is the identity Σ_present ω^p = −ω^missing taken in numpy before the flow,
and the well merely transports it (F7). Figure 30 compares a 12-parameter circular model with
an MLP on (a+b) mod 11: the circular model generalises at once and the phases form a regular
11-gon to within a degree, but the control never groks within the budget, so nothing is
"shortcut" (F6).

Two June scripts with no figure numbers are given 44 and 45. `hsln_compose_two` was written to
look for the factors of a large integer in zero positions; that use is gone (ledger appendix A:
the only composition of divisibility detectors that factors is trial division,
`Fugue.first_spike_is_minFac`) and what remains is the joint-training experiment. `bench` prints
timings and finds, on the way, that recovering 100 prescribed zeros from the coefficients by
`np.roots` is ill-conditioned (error 0.23). `animate` needs ffmpeg, absent on the machine that
produced this repository; it says so and exits 0.

## What this family establishes

The zero-union merge law, implemented and measured, is the corpus's best empirical result
(THE-STORY §7, item 2). The dynamical merge does not follow from it and in general fails
(figure 19); separating coordinates restores it (figure 20); the coupling bends paths and keeps
targets without producing chaos (figure 22). Everything called "learning" here is either a
small recurrent model trained by backpropagation (14–16), a reparameterisation (17), reservoir
computing on a fixed landscape (25), or an exact identity carried by a flow (29). None of the
scripts trains a modular form.

## Table

| script | figure | theorem behind it | Lean name | what is not claimed (ledger) |
|---|---|---|---|---|
| `poc.py` | 14 | zeros of holomorphic F are the minima of \|F\|; implicit function theorem moves simple zeros | `Fugue.isLocalMin_norm_of_eq_zero`, `Fugue.eq_zero_of_isLocalMin_norm` | targets are not attractors: the loss is a minimum over the path (F2); loss numbers are seed-dependent (F1) |
| `poc_sine.py` | 15 | gradient flow dissipates energy (dE/dt = −\|∇E\|²); reservoir computing with a trained readout | — | no modular property used; 15-parameter recurrent model, a sanity check (E1 "tautological") |
| `poc_xor.py` | 16 | a degree-4 polynomial has any four prescribed roots | — | no ODE, no time: a static four-point discriminant; training suppresses two existing spikes (F3) |
| `poc_layer.py` | 17 | every monic quartic is a product of two monic quadratics; zero union | `mul_eq_zero` (B1) | layering adds no capacity: a reparameterisation (F4); factors trained jointly, not the merge test (F9) |
| `hsln.py` | 18 | zeros(FG) = zeros(F) ∪ zeros(G); Cauchy product of coefficients | `mul_eq_zero` (B1) | that the merged readout does both tasks is measured, not proved; not modular forms (THE-STORY §3) |
| `hsln_dyn.py` | 19 | ∇\|F_AF_B\|² = \|F_B\|²∇\|F_A\|² + \|F_A\|²∇\|F_B\|² (product rule) | `mul_eq_zero` (B1), `Fugue.isLocalMin_norm_of_eq_zero` (A9) | the merge law does not transfer input→output behaviour; cross-talk measured in one geometry |
| `hsln_genus2.py` | 20 | Θ_{L⊕M} = Θ_L·Θ_M at every degree; energy factorises on the diagonal slice | `Fugue.theta_directSum`, `Fugue.reprCount_directSum` (C13), `Fugue.e8e8_det`, `Fugue.equilibrium_preserved` | "genus 2" is degree 2 and the direct sum does not raise it; the decoupling is exact by construction |
| `coupling_sweep.py` | 22 | K(κ) is the Siegel metric block, PSD iff \|κ\| ≤ 1, PD iff \|κ\| < 1; −K·0 = 0; Poincaré–Bendixson for κ = 0 | `Fugue.couplingMatrix_posSemidef_iff`, `Fugue.couplingMatrix_posDef_iff`, `Fugue.equilibrium_preserved`, `Fugue.descends_iff` (E2, E3, E5) | chaos: REFUTED, no positive Lyapunov exponent (E4); error "≈0" is 0.001–0.006 (E6); common zero only to \|F\| ≈ 2e−4 (E5) |
| `cm_alphabet.py` | 23 | nine Heegner numbers; j of a class-number-one CM point is a rational integer and a cube | — (classical; verified at 80 digits) | that the network uses nine symbols: only two are reached (G1 proved, G3) |
| `cm_task.py` | 24 | j is a bijection ℍ/SL(2,ℤ) → ℂ; j(i) = 1728, j(ρ) = 0 | `Fugue.E₆_I`, `Fugue.E₄_rho` (A3), G2 | a two-symbol code spelling a bit-string; nothing is "asked" (G3) |
| `cm_classifier.py` | 25 | zeros are attractors; reservoir computing | `Fugue.isLocalMin_norm_of_eq_zero` (A9) | the landscape is fixed by hand; only a 6-parameter encoder is trained; 95% held-out is the encoder's (F8) |
| `cm_tensor.py` | 26 | zero union for four factors; nearest-centroid classification | `mul_eq_zero` (B1, B3) | composition contributes nothing to the classification beyond bookkeeping; not the behavioural merge test (F9) |
| `timeseries.py` | 27 | a driven anharmonic first-order system responds at the drive tone and its harmonics | — | the landscape is \|q − q₀\|², not E₄; harmonics are weak (~1%); the June rectangular-window panel showed leakage, not harmonics |
| `scaling.py` | 28 | convolution O(mn); representation table O(N_vec²) | — | "25k/s", "genus is the cost driver", "GPU-trivial": not measured or wrong (F5); timings are printed, not asserted |
| `skill_compose.py` | 29 | Σ of the n-th roots of unity is 0 | `Fugue.sum_exp_eq` (D12, general form) | the reservoir computes nothing: the sum is taken before the flow and the result is drive-invariant (F7) |
| `grok.py` | 30 | addition mod n is rotation on the roots of unity; 12 parameters cannot fit a random 121-table | — | no grokking is shortcut, because the control never groks (F6); nothing about the ℍ-state network |
| `hsln_compose_two.py` | 44 | zero union; degree bounds the root count | `mul_eq_zero` (B1) | joint training, not the separate-training merge test (F9); the June integer-factoring use is dropped (appendix A, `Fugue.first_spike_is_minFac`) |
| `hsln_skill_forge.py` | 45 | zero union; exact division in ℂ[q] | `mul_eq_zero` (B1) | "machine-checked by neurons_mul" and "no standard network can do either" removed; the merged readout's three spikes are measured |
| `bench.py` | — | ∏(q − q_i) has the prescribed roots; root-finding O(d³) | `Fugue.reprCount_e8` (A1/A2) for the 240·126 check | nothing about GPUs or "neuron count"; recovery by `np.roots` is ill-conditioned at d ≥ 100 (printed) |
| `animate.py` | — | Θ_{L⊕M} = Θ_L·Θ_M; zero union; continuity of roots in the coefficients | `Fugue.theta_directSum` (C13), `mul_eq_zero` (B1) | illustration only; roots inside the q-disk are truncation roots, not neurons of a modular form (THE-STORY §3); needs ffmpeg |
