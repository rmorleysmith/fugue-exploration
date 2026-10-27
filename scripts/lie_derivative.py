"""The Euler operator D = q d/dq: the ring C[E₂, E₄, E₆] is closed under it (Ramanujan), and
the same operator read in the Dirichlet basis is a_n ↦ a_n log n.

WHAT IT COMPUTES
    (1) A gradient flow Ż = −∇E on the energy E = |F|² for the polynomial F(z) = (z−z₁)(z−z₂),
        integrated by RK4 from z₀ = −1.15 + 1.55i. The derivative of E along the flow is, by the
        chain rule, dE/dt = ∇E · Ż = −|∇E|² ≤ 0. Check: the numerical dE/dt matches −|∇E|² to
        relative error < 5e−3 along the trajectory and E decreases monotonically into a zero of F.
    (2) The q-expansions of E₂ = 1 − 24Σσ₁(n)qⁿ, E₄ = 1 + 240Σσ₃(n)qⁿ, E₆ = 1 − 504Σσ₅(n)qⁿ to
        order 12, the operator D(Σ aₙqⁿ) = Σ n aₙ qⁿ, and Ramanujan's system
            D E₂ = (E₂² − E₄)/12,  D E₄ = (E₂E₄ − E₆)/3,  D E₆ = (E₂E₆ − E₄²)/2,
        plus the Serre derivative ϑ_k = D − (k/12) E₂·, which maps weight k to weight k + 2:
            ϑ₄ E₄ = −E₆/3,  ϑ₆ E₆ = −E₄²/2.
        Check: all five identities hold coefficientwise to 1e−6 on n ≤ 12 (they are exact in
        integers; the residuals print as 0).

WHAT THE FIGURE SHOWS
    A. The energy landscape log|F|² with the trajectory descending into a zero of F.
    B. dE/dt measured along the trajectory (gold) on top of the predicted −|∇E|² (dashed); inset:
       E(t) decreasing.
    C. The coefficients of D E₄ (bars) and of (E₂E₄ − E₆)/3 (circles): the same vector.
    D. The Serre derivatives ϑ₄E₄ and ϑ₆E₆ against −E₆/3 and −E₄²/2, each rescaled to unit max.
       The corner note records the Dirichlet-basis reading of D.

WHAT IS PROVED / KNOWN
    Ramanujan's differential system (Ramanujan 1916, "On certain arithmetical functions", Trans.
    Camb. Phil. Soc. 22) is in Mathlib: `normalizedDerivOfComplex_E₂`, `normalizedDerivOfComplex_E₄`,
    `normalizedDerivOfComplex_E₆`; the Serre derivatives are `serreDerivative_E₄` and
    `serreDerivative_E₆` (Mathlib, NumberTheory/ModularForms/RamanujanFormula.lean). Evidence
    ledger C10. In the Dirichlet ring, pointwise multiplication by log n is a derivation
    (`Fugue.pmul_log_mul` in fugue-formal's TwoRings.lean, ledger C9) and
    `log * μ = Λ` is Mathlib's `log_mul_moebius_eq_vonMangoldt`, i.e. −ζ′/ζ = Σ Λ(n) n^{−s}.
    That zeros of a holomorphic F are the local minima of |F| is `Fugue.isLocalMin_norm_of_eq_zero`
    (ledger A9). Part (1) is the chain rule.

WHAT IS NOT CLAIMED
    That the two halves are one operator. The flow's derivative in (1) is along the vector field
    −∇E; the operator D in (2) is along the dilation q ↦ λq. They are different generators, and
    the June title "two faces of one generator" conflated them. The genuine bridge is within (2):
    the dilation generator of the additive grading (n aₙ on qⁿ) and of the multiplicative grading
    (aₙ log n on n^{−s}) are the Euler operators of the two rings of ledger C1. Nor does closure
    under D say anything about a trained network "knowing its own derivative": it is a property
    of the ring C[E₂, E₄, E₆], established by Ramanujan.

HOW THIS WAS FOUND
    The June question was whether the network had a Lie derivative of its own and could be
    taught it. We found two candidates and guessed they were the same thing: the dissipation
    −|∇E|² along the gradient flow, and the grading operator q d/dq on the coefficient vector
    (the Fourier coefficients, which the June files called the DNA). The precise statements are
    that the first is the chain rule, that the second closes on Ramanujan's ring, and that the
    second, carried to the Dirichlet basis, becomes multiplication by log n — whose action on ζ,
    normalised by μ, is the von Mangoldt function. One generator in two bases is the true
    content; the flow is a separate story.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# --------------------------------------------------------------------- (1) the flow
# F(z) = (z − z1)(z − z2): a polynomial with two zeros. E = |F|², gradient flow Ż = −∇E.
Z1, Z2 = -0.60 + 0.85j, 0.70 + 0.55j


def _F(z):
    return (z - Z1) * (z - Z2)


def _energy(z):
    return np.abs(_F(z)) ** 2


def _grad(z):
    """∇E in (x, y) for E = |F|², F holomorphic: ∇E = (2 Re w, −2 Im w), w = conj(F)·F′."""
    Fp = (z - Z1) + (z - Z2)
    w = np.conj(_F(z)) * Fp
    return np.array([2 * w.real, -2 * w.imag])


def flow(z0, dt=2.0e-4, steps=6000):
    z = z0
    zs, Es, dissip = [], [], []
    for _ in range(steps):
        g = _grad(z)
        zs.append(z); Es.append(_energy(z)); dissip.append(-(g @ g))

        def f(p):
            return -_grad(p[0] + 1j * p[1])
        p = np.array([z.real, z.imag])
        k1 = f(p); k2 = f(p + dt / 2 * k1); k3 = f(p + dt / 2 * k2); k4 = f(p + dt * k3)
        p = p + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        z = p[0] + 1j * p[1]
    return np.array(zs), np.array(Es), np.array(dissip), dt


# --------------------------------------------------------------------- (2) the Euler operator
N = 12  # truncation; integer coefficients stay exact in float64 at this order


def sigma_k(n, k):
    return sum(d ** k for d in range(1, n + 1) if n % d == 0)


def eisenstein(k, N=N):
    """Coefficients a_0..a_N of E_k for k = 2, 4, 6 (normalised to constant term 1)."""
    const, power = {2: (-24, 1), 4: (240, 3), 6: (-504, 5)}[k]
    a = np.zeros(N + 1); a[0] = 1.0
    for n in range(1, N + 1):
        a[n] = const * sigma_k(n, power)
    return a


def D(a):
    """q d/dq on a q-series: aₙ ↦ n·aₙ."""
    return a * np.arange(len(a))


def qmul(a, b, N=None):
    """Product of two q-series, truncated to the shorter length (or to N + 1)."""
    if N is None:
        N = min(len(a), len(b)) - 1
    return np.convolve(a, b)[:N + 1]


def run(make_fig=True):
    # ---- (1) the flow ----
    zs, Es, dissip, dt = flow(-1.15 + 1.55j)
    dEdt = np.gradient(Es, dt)
    interior = slice(2, len(Es) - 2)
    rel = np.abs(dEdt[interior] - dissip[interior]) / (np.abs(dissip[interior]) + 1e-9)
    monotone = bool(np.all(np.diff(Es) <= 1e-9))
    print(f"(1) flow: dE/dt vs −|∇E|² max rel. error {rel.max():.2e}; "
          f"energy {Es[0]:.3f} → {Es[-1]:.2e}, monotone={monotone}")
    assert rel.max() < 5e-3 and monotone

    # ---- (2) Ramanujan's system ----
    E2, E4, E6 = eisenstein(2), eisenstein(4), eisenstein(6)
    res2 = np.abs(D(E2) - (qmul(E2, E2) - E4) / 12).max()
    res4 = np.abs(D(E4) - (qmul(E2, E4) - E6) / 3).max()
    res6 = np.abs(D(E6) - (qmul(E2, E6) - qmul(E4, E4)) / 2).max()
    print(f"(2) Ramanujan: |DE₂−(E₂²−E₄)/12|={res2:.1e}  |DE₄−(E₂E₄−E₆)/3|={res4:.1e}  "
          f"|DE₆−(E₂E₆−E₄²)/2|={res6:.1e}  (n ≤ {N})")

    # ---- Serre derivative raises the weight ----
    theta4 = D(E4) - (4 / 12) * qmul(E2, E4)
    theta6 = D(E6) - (6 / 12) * qmul(E2, E6)
    sres4 = np.abs(theta4 - (-E6 / 3)).max()
    sres6 = np.abs(theta6 - (-qmul(E4, E4) / 2)).max()
    print(f"    Serre: |ϑ₄E₄−(−E₆/3)|={sres4:.1e} (weight 4→6)  "
          f"|ϑ₆E₆−(−E₄²/2)|={sres6:.1e} (weight 6→8)")
    assert max(res2, res4, res6, sres4, sres6) < 1e-6

    if make_fig:
        _figure(zs, Es, dissip, dEdt, dt, (E2, E4, E6, theta4, theta6))
    return True


def _figure(zs, Es, dissip, dEdt, dt, modular):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    INK, PAPER = "#0a0a12", "#e8e6f0"
    TEAL, GOLD, RED, GREEN, VIO = "#46f0d0", "#ffd166", "#ff3b6b", "#7CFC8A", "#c0a3ff"
    E2, E4, E6, theta4, theta6 = modular
    t = np.arange(len(Es)) * dt

    fig = plt.figure(figsize=(16.5, 9.2))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24)
    A = fig.add_subplot(gs[0, 0]); B = fig.add_subplot(gs[0, 1])
    C = fig.add_subplot(gs[1, 0]); Dax = fig.add_subplot(gs[1, 1])

    # --- A: gradient flow on |F|² descending to a zero ---
    xs = np.linspace(-1.6, 1.4, 400); ys = np.linspace(0.1, 1.8, 400)
    X, Y = np.meshgrid(xs, ys)
    EE = _energy(X + 1j * Y)
    A.contourf(X, Y, np.log10(EE + 1e-6), levels=40, cmap="magma")
    A.plot(zs.real, zs.imag, color=TEAL, lw=2.2, label="gradient-flow trajectory  Ż = −∇E")
    A.plot([zs.real[0]], [zs.imag[0]], "o", color=PAPER, ms=7)
    for z in (Z1, Z2):
        A.plot([z.real], [z.imag], "*", color=GREEN, ms=16,
               label="zero of F (equilibrium)" if z is Z1 else None)
    A.set_xlabel("Re z"); A.set_ylabel("Im z")
    A.set_title("the flow on the energy  E = |F|²\n(gradient descent into a zero)", fontsize=12)
    A.legend(loc="upper right", fontsize=8.2, framealpha=0.35)

    # --- B: dE/dt = −|∇E|² along the flow ---
    B.plot(t, dEdt, color=GOLD, lw=3.4, alpha=0.85, label=r"$dE/dt$ (measured along the flow)")
    B.plot(t, dissip, color=RED, lw=1.3, ls="--", label=r"$-|\nabla E|^2$ (chain rule)")
    B.set_xlabel("time t"); B.set_ylabel(r"$dE/dt$")
    B.set_title("the derivative of E along the flow is the dissipation\n"
                r"$dE/dt = \nabla E\cdot\dot{Z} = -|\nabla E|^2 \leq 0$", fontsize=12)
    B.legend(loc="lower right", fontsize=9, framealpha=0.35)
    bin = B.inset_axes([0.55, 0.55, 0.42, 0.4])
    bin.plot(t, Es, color=TEAL, lw=2)
    bin.set_title("E(t) ↓", fontsize=8, color=PAPER); bin.tick_params(labelsize=6)
    bin.set_facecolor(INK)

    # --- C: Ramanujan's system (E4 shown) ---
    ns = np.arange(len(E4))
    lhs = D(E4); rhs = (qmul(E2, E4) - E6) / 3
    C.vlines(ns, 0, lhs, color=VIO, lw=2.4, label=r"$D E_4 = q\,dE_4/dq$")
    C.plot(ns, rhs, "o", color=GOLD, ms=8, mfc="none", mew=1.8,
           label=r"$(E_2 E_4 - E_6)/3$  (back in the ring)")
    C.set_xlabel("q-power n"); C.set_ylabel("coefficient")
    C.set_title("the ring $\\mathbb{C}[E_2,E_4,E_6]$ is closed under $D = q\\,d/dq$\n"
                r"Ramanujan (1916): $D E_4 = (E_2 E_4 - E_6)/3$", fontsize=12)
    C.legend(loc="upper left", fontsize=8.6, framealpha=0.35)

    # --- D: the Serre derivative raises the weight ---
    E4sq = qmul(E4, E4)
    Dax.vlines(ns - 0.12, 0, theta4 / (np.abs(theta4).max()), color=TEAL, lw=2.2,
               label=r"$\vartheta_4 E_4$  (= $-E_6/3$, weight 4→6)")
    Dax.plot(ns - 0.12, (-E6 / 3) / (np.abs(theta4).max()), "o", color=TEAL, ms=5, mfc="none")
    Dax.vlines(ns + 0.12, 0, theta6 / (np.abs(theta6).max()), color=RED, lw=2.2,
               label=r"$\vartheta_6 E_6$  (= $-E_4^2/2$, weight 6→8)")
    Dax.plot(ns + 0.12, (-E4sq / 2) / (np.abs(theta6).max()), "s", color=RED, ms=5, mfc="none")
    Dax.axhline(0, color="#555", lw=0.7)
    Dax.set_xlabel("q-power n"); Dax.set_ylabel("coefficient (scaled)")
    Dax.set_title("the Serre derivative raises the weight\n"
                  r"$\vartheta_k = D - \frac{k}{12}E_2$ :  $M_k \to M_{k+2}$", fontsize=12)
    Dax.legend(loc="upper right", fontsize=8.4, framealpha=0.35)
    Dax.text(0.02, 0.03,
             "the same operator in the Dirichlet basis:\n"
             r"$a_n\mapsto a_n\log n$;  on $\zeta$, times $\mu$:  $-\zeta'/\zeta=\sum\Lambda(n)n^{-s}$",
             transform=Dax.transAxes, color="#9a96b0", fontsize=8.0, va="bottom")

    fig.suptitle("Along the flow, dE/dt = −|∇E|² (chain rule).  Along the grading, D = q d/dq "
                 "closes on C[E₂,E₄,E₆] (Ramanujan) and reads as aₙ log n in the Dirichlet basis",
                 fontsize=12.5, y=0.99)
    out = Path(__file__).parent.parent / "figures" / "38_lie_derivative.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
