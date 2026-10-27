"""Dirichlet L-functions: a family of generators in the Dirichlet ring (figure 39).

WHAT IT COMPUTES
    Four real Dirichlet characters as coefficient vectors (the Fourier coefficients, which the
    June files called the DNA):  χ₋₄ (mod 4), χ₋₃ (mod 3), χ₋₇ (mod 7, the Legendre symbol),
    χ₈ (mod 8).  For each:
      (1) the Euler product ∏_p (1 − χ(p) p^{−s})^{−1}, multiplied out in the Dirichlet ring by
          `dna_algebra.compose_L`, is checked to return the vector χ(n) itself for n ≤ 96
          (complete multiplicativity);
      (2) the direct series Σ_{n≤K} χ(n)/n with K = 4·10⁶ is compared with the class number
          formula  L(1, χ_D) = 2πh/(w√|D|)  (imaginary quadratic)  or  2hR/√D  (real):
              L(1,χ₋₄) = π/4,   L(1,χ₋₃) = π/(3√3),   L(1,χ₋₇) = π/√7,
              L(1,χ₈) = log(1+√2)/√2,   and L(2,χ₋₄) = G (Catalan's constant);
      (3) the coefficient identity  6·(ζ * χ₋₃)(n) = #{(a,b) ∈ ℤ² : a² − ab + b² = n},
          the Eisenstein-integer count, for n ≤ 96, against brute-force enumeration.

WHAT THE FIGURE SHOWS
    Top left:     the four character vectors.
    Top right:    partial sums Σ_{n≤K} χ(n)/n against their class-number-formula values.
    Bottom left:  lattice points of the hexagonal norm form on the circles a² − ab + b² = n,
                  labelled with 6·(ζ*χ₋₃)(n).
    Bottom right: partial sums of Σ χ₋₄(n)/n² against Catalan's constant.

WHAT IS PROVED / KNOWN
    Dirichlet (1839), class number formula (Davenport, "Multiplicative Number Theory", ch. 6).
    Lean: `Fugue.LFunction_chi4_one` in `Fugue/LeibnizL.lean` proves L(1, χ₋₄) = π/4
    (ledger D6); `Fugue.card_hexagonal` in `Fugue/HexagonalCount.lean` proves the count
    #{a² − ab + b² = n} = 6 Σ_{d|n} χ₋₃(d) (ledger C7, D8), and `Fugue.Epstein.epstein_hexagonal`
    / `epstein_square` lift it to  Z_hex(s) = 6 ζ(s) L(s, χ₋₃)  and  Z_{ℤ²}(s) = 4 ζ(s) L(s, χ₋₄)
    for Re s > 1.  L(2, χ₋₄) = G is the definition of Catalan's constant (ledger D7).  That
    multiplicative vectors are closed under the Dirichlet product is Mathlib's
    `IsMultiplicative.mul`, exposed as `Fugue.multiplicativeSubmonoid` (ledger C11).

WHAT IS NOT CLAIMED
    The special values are verified numerically to about 1e−7 (the alternating series to
    4·10⁶ terms); only L(1, χ₋₄) has a Lean proof.  "A family of generators" means no more
    than: the characters are multiplicative vectors in the same ring as ζ, so the same
    product, inverse and log-derivative apply to them.  Nothing here concerns the zeros of
    L(s, χ) or the generalised Riemann hypothesis.

HOW THIS WAS FOUND
    Figure 37 had composed ζ with the mod-4 character and counted sums of two squares.  We
    asked what else the same move gives if the all-ones vector of ζ is replaced by other
    periodic sign patterns, and noticed that "χ₋₄ to degree 2" (the s = 2 value) tends to
    Catalan's constant.  The lofty guess was that composition with ζ unlocks a number field
    per character; the precise statement is Dedekind's factorisation ζ_K = ζ · L(s, χ_D) for
    quadratic K, whose coefficients count ideals of each norm, and for ℚ(√−3) those are the
    hexagonal lattice points divided by the six units.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dna_algebra import compose_L, dirichlet_mul

M = 96  # number of Dirichlet coefficients for the ring checks and the figure

# (period, [χ(1), ..., χ(period)]) for the four real characters.
CHARS = {
    "χ₋₄": (4, [1, 0, -1, 0]),
    "χ₋₃": (3, [1, -1, 0]),
    "χ₋₇": (7, [1, 1, -1, 1, -1, -1, 0]),
    "χ₈":  (8, [1, 0, -1, 0, -1, 0, 1, 0]),
}

CATALAN = 0.91596559417721901505
TARGETS_L1 = {
    "χ₋₄": np.pi / 4,                              # ℚ(i):     h=1, w=4, |D|=4
    "χ₋₃": np.pi / (3 * np.sqrt(3)),               # ℚ(√−3):   h=1, w=6, |D|=3
    "χ₋₇": np.pi / np.sqrt(7),                     # ℚ(√−7):   h=1, w=2, |D|=7
    "χ₈":  np.log(1 + np.sqrt(2)) / np.sqrt(2),    # ℚ(√2):    h=1, R=log(1+√2), D=8
}


def chi_vector(name, m=M):
    """χ(n) for n = 0..m (index 0 unused and set to 0)."""
    p, vals = CHARS[name]
    c = np.zeros(m + 1)
    c[1:] = np.array(vals, float)[(np.arange(1, m + 1) - 1) % p]
    return c


def partial_sums(name, K, s):
    """Cumulative sums of χ(n)/n^s for n = 1..K."""
    p, vals = CHARS[name]
    ns = np.arange(1, K + 1, dtype=float)
    cn = np.array(vals, float)[(np.arange(1, K + 1) - 1) % p]
    return np.cumsum(cn / ns ** s)


def hex_norm_counts(m=M):
    """#{(a, b) ∈ ℤ² : a² − ab + b² = n} by enumeration."""
    r = np.zeros(m + 1)
    lim = int(np.sqrt(m * 4 / 3)) + 2
    for a in range(-lim, lim + 1):
        for b in range(-lim, lim + 1):
            n = a * a - a * b + b * b
            if 1 <= n <= m:
                r[n] += 1.0
    return r


def run(make_fig=True):
    # ---------- (1) Euler product in the ring returns chi(n) ----------
    for name in CHARS:
        chi = chi_vector(name)
        L = compose_L(chi, M)
        ok = np.allclose(L.c, chi, atol=1e-10)
        print(f"  prod_p (1 - {name}(p) p^-s)^-1 multiplied out in the ring == {name}(n), "
              f"n <= {M}: {ok}")
        assert ok

    # ---------- (2) special values by direct summation ----------
    K = 4_000_000
    print(f"\n  direct series to K = {K:,} terms vs the class number formula:")
    checks = {}
    for name, want in TARGETS_L1.items():
        got = partial_sums(name, K, 1.0)[-1]
        checks[f"L(1,{name})"] = (got, want)
    checks["L(2,χ₋₄)"] = (partial_sums("χ₋₄", K, 2.0)[-1], CATALAN)
    for label, (got, want) in checks.items():
        err = abs(got - want)
        print(f"  {label:<9} numerical {got:.8f}   target {want:.8f}   |err| = {err:.1e}")
        assert err < 1e-5, label
    print("  all five within 1e-5: True")

    # ---------- (3) 6 (zeta * chi_-3)(n) = hexagonal lattice count ----------
    zeta_arr = np.ones(M + 1); zeta_arr[0] = 0.0
    zeta_chi3 = dirichlet_mul(zeta_arr, chi_vector("χ₋₃"))
    rhex = hex_norm_counts(M)
    hex_ok = np.allclose(6.0 * zeta_chi3[1:], rhex[1:])
    print(f"\n  6 (zeta * χ₋₃)(n) == #{{a² − ab + b² = n}} for n <= {M}: {hex_ok}")
    assert hex_ok

    if make_fig:
        _figure(zeta_chi3)
    return True


def _figure(zeta_chi3):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    TEAL, GOLD, RED, GREEN, VIO = "#46f0d0", "#ffd166", "#ff3b6b", "#7CFC8A", "#c0a3ff"
    PAPER_DIM = "#9a96b0"

    fig = plt.figure(figsize=(17.0, 9.4))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.22)
    A = fig.add_subplot(gs[0, 0]); B = fig.add_subplot(gs[0, 1])
    C = fig.add_subplot(gs[1, 0]); Dax = fig.add_subplot(gs[1, 1])
    ns = np.arange(1, M + 1)

    # -- A: the character vectors --
    rows = [("χ₋₄  (mod 4 — ℚ(i))", "χ₋₄", GOLD),
            ("χ₋₃  (mod 3 — ℚ(√−3))", "χ₋₃", RED),
            ("χ₋₇  (mod 7 — ℚ(√−7))", "χ₋₇", GREEN),
            ("χ₈   (mod 8 — ℚ(√2), real)", "χ₈", TEAL)]
    for i, (lab, name, col) in enumerate(rows):
        base = 3.0 * (len(rows) - 1 - i)
        cc = chi_vector(name)[1:M + 1]
        A.vlines(ns, base, base + cc, color=col, lw=1.6)
        A.axhline(base, color="#333", lw=0.5)
        A.text(M * 0.5, base + 1.35, lab, color=col, fontsize=9.5, ha="center")
    A.set_yticks([]); A.set_xlabel("n")
    A.set_title("four Dirichlet characters as coefficient vectors\n"
                "(each is multiplicative, so ∏ₚ(1 − χ(p)p⁻ˢ)⁻¹ returns it)", fontsize=12)

    # -- B: partial sums at s = 1 --
    K = 4000
    for lab, name, col in [("χ₋₄ → π/4", "χ₋₄", GOLD), ("χ₋₃ → π/(3√3)", "χ₋₃", RED),
                           ("χ₋₇ → π/√7", "χ₋₇", GREEN), ("χ₈ → log(1+√2)/√2", "χ₈", TEAL)]:
        tgt = TARGETS_L1[name]
        B.plot(np.arange(1, K + 1), partial_sums(name, K, 1.0), color=col, lw=1.4, alpha=0.9)
        B.axhline(tgt, color=col, ls="--", lw=1.0, alpha=0.75)
        below = name == "χ₋₃"
        B.text(K * 0.98, tgt + (-0.02 if below else 0.02), lab, color=col, fontsize=8.3,
               ha="right", va="top" if below else "bottom")
    B.set_xscale("log"); B.set_xlabel("partial sum length K")
    B.set_ylabel(r"$\sum_{n\leq K} \chi(n)/n$")
    B.set_title("L(1, χ_D) by direct series against the class number formula\n"
                r"$L(1,\chi_D)=\frac{2\pi h}{w\sqrt{|D|}}$ (imaginary) /  $\frac{2hR}{\sqrt{D}}$ (real)",
                fontsize=12)

    # -- C: hexagonal lattice points --
    circle_ns = [1, 3, 7, 13, 21, 31]
    cols = [GOLD, RED, TEAL, GREEN, VIO, "#ff7b6b"]
    th = np.linspace(0, 2 * np.pi, 400)

    def to_xy(a, b):          # basis 1, ω = (−1/2, √3/2)
        return a - 0.5 * b, (np.sqrt(3) / 2) * b
    for n, col in zip(circle_ns, cols):
        rad = np.sqrt(n)
        C.plot(rad * np.cos(th), rad * np.sin(th), color=col, lw=0.9, alpha=0.55)
        pts = [(a, b) for a in range(-7, 8) for b in range(-7, 8) if a * a - a * b + b * b == n]
        xs, ys = zip(*(to_xy(a, b) for (a, b) in pts))
        C.plot(xs, ys, "o", color=col, ms=6)
        ang = 0.30 * np.pi
        C.text((rad + 0.35) * np.cos(ang), (rad + 0.35) * np.sin(ang),
               f" {n}: {len(pts)} points = 6·{int(round(zeta_chi3[n]))}",
               color=col, fontsize=7.6, ha="left", va="bottom")
    C.set_aspect("equal"); C.set_xlim(-6.5, 6.5); C.set_ylim(-6.5, 6.5)
    C.axhline(0, color="#333", lw=0.6); C.axvline(0, color="#333", lw=0.6)
    C.set_xlabel("x"); C.set_ylabel("y")
    C.set_title(r"$\zeta\cdot L(\cdot,\chi_{-3}) = \zeta_{\mathbb{Q}(\sqrt{-3})}$:  "
                "6·(ζ*χ₋₃)(n) = #{a² − ab + b² = n}\n(Eisenstein integers; six units)", fontsize=12)

    # -- D: Catalan --
    K2 = 4000
    Dax.plot(np.arange(1, K2 + 1), partial_sums("χ₋₄", K2, 2.0), color=GOLD, lw=1.6,
             label=r"$\sum_{n\leq K}\chi_{-4}(n)/n^2$")
    Dax.axhline(CATALAN, color=RED, ls="--", lw=1.2, label=f"G = Catalan ≈ {CATALAN:.7f}")
    Dax.set_xscale("log"); Dax.set_xlabel("K")
    Dax.set_ylabel(r"$L(2, \chi_{-4})$")
    Dax.set_title("L(2, χ₋₄) is Catalan's constant (by definition)", fontsize=12)
    Dax.legend(loc="lower right", fontsize=9, framealpha=0.3)
    Dax.text(0.02, 0.04,
             "the same ring operations (product, inverse, log-derivative)\n"
             "apply to every multiplicative coefficient vector, ζ or χ alike.",
             transform=Dax.transAxes, color=PAPER_DIM, fontsize=8.4, va="bottom")

    fig.suptitle("Dirichlet L-functions: characters are multiplicative coefficient vectors in "
                 "the same ring as ζ; their s = 1 values are the class number formula",
                 fontsize=13, y=0.99)
    out = Path(__file__).parent.parent / "figures" / "39_lfunctions.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
