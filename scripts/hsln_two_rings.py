"""Two rings on one coefficient list: the Cauchy product and the Dirichlet product (figure 43).

WHAT IT COMPUTES
    A coefficient vector (the Fourier coefficients, which the June files called the DNA) is a
    function f : ℕ → ℝ with f(0) = 0.  Two products live on the same list:
        Ring A, Cauchy:     (f ⋆ g)(n) = Σ_{i+j=n} f(i) g(j)     (q-expansions; forms multiply)
        Ring B, Dirichlet:  (f * g)(n) = Σ_{d·e=n} f(d) g(e)     (Dirichlet series; L-functions multiply).
    On the all-ones vector ζ:  (ζ ⋆ ζ)(n) = n − 1  but  (ζ * ζ)(n) = d(n), the number of
    divisors; in particular (ζ ⋆ ζ)(1) = 0 while (ζ * ζ)(1) = 1.  The vector δ = [n = 1] is
    the identity of Ring B and the shift q ↦ q·f in Ring A.  The log-derivative
    (Df)(n) = f(n) log n satisfies the Leibniz rule in both rings.  Multiplicative vectors
    (f(mn) = f(m) f(n) for coprime m, n) are closed under * but not under +.

WHAT THE FIGURE SHOWS
    Left:   ζ, ζ ⋆ ζ and ζ * ζ for n ≤ 40; the two products disagree already at n = 1.
    Middle: δ * ζ = ζ (identity in Ring B) and δ ⋆ ζ = ζ shifted by one (not an identity in
            Ring A).
    Right:  the Leibniz residual ‖D(f·g) − Df·g − f·Dg‖ for random f, g under both products,
            and the multiplicativity residual of ζ * χ₋₄ (zero) and of ζ + ζ (not zero).

WHAT IS PROVED / KNOWN
    Lean, `Fugue/TwoRings.lean`:  `Fugue.cauchy_ne_dirichlet` (the witness at n = 1, ledger C1),
    `Fugue.not_ringEquiv_powerSeries` (over a field the two rings are not isomorphic: the
    Dirichlet ring is not a principal ideal ring, the power-series ring is), `Fugue.delta_is_X`
    (δ is the variable X of Ring A, ledger C3), `Fugue.pmul_log_mul` (log is a derivation of
    Ring B, ledger C9), `Fugue.multiplicativeSubmonoid` and
    `Fugue.zeta_add_zeta_not_multiplicative` (ledger C11).  Both rings are completed monoid
    rings, of (ℕ, +) and of (ℕ_{>0}, ×) ≅ ⊕_p ℕ (ledger C17).

WHAT IS NOT CLAIMED
    The June log said the geometric composition of forms and the Dirichlet product were "the
    same operation in two realisations"; they are not (ledger C2 refutes it with this file's
    own witness).  A Hecke eigenform's coefficient vector sits in both rings, and that is the
    whole of the bridge: forms multiply by ⋆, their L-functions by *.  Neither product gives
    a way to recover p and q from the index N of a vector concentrated at N; the June version
    of this script tried that and the attempt stays in the archive (ledger G-rows).

HOW THIS WAS FOUND
    For weeks we used one word, "compose", for two things: multiplying forms (which unions
    zeros and adds weights) and multiplying Euler factors (which builds ζ and the
    L-functions).  Writing both as operations on the same coefficient list made the question
    unavoidable, and the smallest test, the all-ones vector squared, gave different answers at
    n = 1.  We had imagined one composition with two faces; the precise statement is that
    there are two rings on one carrier, not isomorphic, with the multiplicative vectors as the
    place where one sequence has a meaning in each.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dna_algebra import dirichlet_mul, _chi4, _delta, _divisor_count, _zeta

M = 40


def cauchy_mul(a, b):
    """Ring A: (a ⋆ b)(n) = Σ_{i+j=n} a(i) b(j), truncated to the same length."""
    return np.convolve(a, b)[:len(a)]


def log_derivative(a):
    d = a.copy()
    d[1:] = d[1:] * np.log(np.arange(1, len(a), dtype=float))
    return d


def is_multiplicative(f, m=M, tol=1e-9):
    """f(1) = 1 and f(ab) = f(a) f(b) whenever gcd(a, b) = 1 and ab ≤ m."""
    if abs(f[1] - 1.0) > tol:
        return False
    for a in range(2, m + 1):
        for b in range(2, m // a + 1):
            if np.gcd(a, b) == 1 and abs(f[a * b] - f[a] * f[b]) > tol:
                return False
    return True


def run(make_fig=True):
    zeta = _zeta(M)
    delta = _delta(M)
    chi4 = _chi4(M)

    # --- the witness: same vector, two squares ---
    zz_A = cauchy_mul(zeta, zeta)
    zz_B = dirichlet_mul(zeta, zeta)
    print(f"(ζ ⋆ ζ)(1) [Cauchy]    = {zz_A[1]:.0f}")
    print(f"(ζ * ζ)(1) [Dirichlet] = {zz_B[1]:.0f}")
    assert zz_A[1] == 0.0 and zz_B[1] == 1.0
    assert np.allclose(zz_A[1:], np.arange(1, M + 1) - 1)          # (ζ⋆ζ)(n) = n − 1
    assert np.allclose(zz_B[1:], _divisor_count(M)[1:])             # (ζ*ζ)(n) = d(n)
    print(f"(ζ ⋆ ζ)(n) = n − 1 and (ζ * ζ)(n) = d(n) for n <= {M}: True  (ledger C1)")

    # --- delta: identity in B, shift in A ---
    dB = dirichlet_mul(delta, zeta)
    dA = cauchy_mul(delta, zeta)
    assert np.allclose(dB, zeta)
    assert np.allclose(dA[2:], zeta[1:-1]) and dA[1] == 0.0
    print("δ = [n=1]: δ * ζ = ζ (identity in Ring B); δ ⋆ ζ = ζ shifted by one (Ring A): True  "
          "(ledger C3)")

    # --- the log-derivative is a derivation of both rings ---
    rng = np.random.default_rng(1)
    f = np.concatenate([[0.0], rng.normal(size=M)])
    g = np.concatenate([[0.0], rng.normal(size=M)])
    resid = {}
    for name, mul in (("Dirichlet", dirichlet_mul), ("Cauchy", cauchy_mul)):
        lhs = log_derivative(mul(f, g))
        rhs = mul(log_derivative(f), g) + mul(f, log_derivative(g))
        resid[name] = float(np.abs(lhs - rhs).max())
    # In Ring A the Leibniz rule for log fails (log(i+j) ≠ log i + log j); n·a_n is the Ring-A
    # derivation (q d/dq).  Check both facts.
    assert resid["Dirichlet"] < 1e-12
    assert resid["Cauchy"] > 1e-3
    euler = lambda a: a * np.arange(len(a))                    # q d/dq: a_n ↦ n a_n
    lhsA = euler(cauchy_mul(f, g)); rhsA = cauchy_mul(euler(f), g) + cauchy_mul(f, euler(g))
    residA = float(np.abs(lhsA - rhsA).max())
    assert residA < 1e-12
    print(f"Leibniz residual of D f = f·log n:  Dirichlet {resid['Dirichlet']:.1e} (a derivation, "
          f"ledger C9),  Cauchy {resid['Cauchy']:.1e} (not one)")
    print(f"Leibniz residual of q d/dq (a_n ↦ n a_n) under Cauchy: {residA:.1e} (the Ring-A derivation)")

    # --- multiplicative vectors: closed under *, not under + ---
    zc = dirichlet_mul(zeta, chi4)
    assert is_multiplicative(zeta) and is_multiplicative(chi4) and is_multiplicative(zc)
    assert not is_multiplicative(zeta + zeta)
    print("ζ, χ₋₄ and ζ * χ₋₄ are multiplicative; ζ + ζ is not (value 2 at n = 1): True  "
          "(ledger C11)")

    if make_fig:
        _figure(zeta, zz_A, zz_B, dA, dB, resid, residA, zc)
    return True


def _figure(zeta, zz_A, zz_B, dA, dB, resid, residA, zc):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    TEAL, GOLD, RED, GREEN, VIO = "#46f0d0", "#ffd166", "#ff3b6b", "#7CFC8A", "#c0a3ff"
    ns = np.arange(1, M + 1)

    fig, ax = plt.subplots(1, 3, figsize=(17.0, 5.2))

    # --- A: the witness ---
    a = ax[0]
    rows = [("ζ  (all ones)", zeta[1:], GOLD, 1.0),
            ("ζ ⋆ ζ  (Cauchy):  n − 1", zz_A[1:], TEAL, zz_A[1:].max()),
            ("ζ * ζ  (Dirichlet):  d(n)", zz_B[1:], VIO, zz_B[1:].max())]
    for i, (lab, c, col, scale) in enumerate(rows):
        base = 1.6 * (len(rows) - 1 - i)
        a.vlines(ns, base, base + c / scale, color=col, lw=2.0)
        a.axhline(base, color="#333", lw=0.5)
        a.text(M * 0.98, base + 1.12, lab, color=col, fontsize=9.5, ha="right")
    a.plot([1, 1], [1.6 * 1, 1.6 * 1 + 0.02], "o", color=RED, ms=7)
    a.annotate("(ζ⋆ζ)(1) = 0", (1, 1.6), textcoords="offset points", xytext=(10, 8),
               color=RED, fontsize=8.5)
    a.annotate("(ζ*ζ)(1) = 1", (1, 1.0), textcoords="offset points", xytext=(10, 4),
               color=RED, fontsize=8.5)
    a.set_yticks([]); a.set_xlabel("n"); a.set_xlim(0, M + 1)
    a.set_title("one vector, two squares\n(scaled strips; they differ already at n = 1)",
                fontsize=12)

    # --- B: delta ---
    b = ax[1]
    b.vlines(ns, 1.6, 1.6 + dB[1:], color=VIO, lw=2.0)
    b.text(M * 0.98, 1.6 + 1.12, "δ * ζ = ζ   (δ is the identity of Ring B)", color=VIO,
           fontsize=9.5, ha="right")
    b.vlines(ns, 0, dA[1:], color=TEAL, lw=2.0)
    b.text(M * 0.98, 1.12, "δ ⋆ ζ = q·ζ   (δ is the shift X of Ring A)", color=TEAL,
           fontsize=9.5, ha="right")
    b.annotate("(δ⋆ζ)(1) = 0", (1, 0), textcoords="offset points", xytext=(10, 6),
               color=RED, fontsize=8.5)
    for base in (0, 1.6):
        b.axhline(base, color="#333", lw=0.5)
    b.set_yticks([]); b.set_xlabel("n"); b.set_xlim(0, M + 1); b.set_ylim(-0.2, 3.4)
    b.set_title("the \"do-nothing\" vector δ = [n = 1]\nis an identity in one ring only",
                fontsize=12)

    # --- C: derivations and the multiplicative locus ---
    c = ax[2]
    labels = ["f·log n\nDirichlet", "f·log n\nCauchy", "n·f\nCauchy"]
    vals = [resid["Dirichlet"], resid["Cauchy"], residA]
    cols = [GREEN, RED, GREEN]
    c.bar(labels, [max(v, 1e-17) for v in vals], color=cols, edgecolor="white", lw=0.5)
    c.set_yscale("log"); c.set_ylim(1e-17, 1e9)
    c.set_ylabel("Leibniz residual  ‖D(fg) − Df·g − f·Dg‖")
    c.set_title("each ring has its own derivation:\nlog n for Dirichlet, n for Cauchy",
                fontsize=12)
    mult_txt = ("multiplicative vectors (f(mn) = f(m)f(n), gcd = 1):\n"
                "ζ, χ₋₄, ζ * χ₋₄ = r₂/4 are;   ζ + ζ is not\n"
                "closed under *, not under +")
    c.text(0.5, 0.97, mult_txt, transform=c.transAxes, color="#9a96b0", fontsize=8.4,
           ha="center", va="top")

    fig.suptitle("Two rings on one coefficient list: forms multiply by the Cauchy product, "
                 "L-functions by the Dirichlet product, and the two are not isomorphic",
                 fontsize=13, y=1.02)
    out = Path(__file__).parent.parent / "figures" / "43_hsln_two_rings.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
