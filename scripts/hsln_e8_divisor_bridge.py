"""The divisor function σ₃ is the bridge: Dirichlet ring → Eisenstein series E₄ → E8 lattice →
E8 ⊕ E8, and its L-function is ζ(s)ζ(s − 3).

WHAT IT COMPUTES
    In the Dirichlet ring of arithmetic functions (the coefficient vectors, which the June files
    called the DNA; `dna_algebra`): d = ζ * ζ, σ₃ = Id₃ * ζ, Λ = log * μ, μ = ζ⁻¹, on n ≤ 30.
    The E8 lattice vectors of squared norm 2, 4, 6 enumerated by `e8.shells`, counted against
    240σ₃(n) (n = 1, 2, 3). The Cauchy square of the E₄ coefficient vector against
    E₈ = 1 + 480Σσ₇(n)qⁿ on n ≤ 30. The partial sums Σ_{n<4000} σ₃(n) n^{−s} against
    ζ(s)ζ(s − 3) at s = 5, 6 (mpmath). Printed checks: σ₃ = Id₃ * ζ exactly; the three E8 shell
    counts are 240, 2160, 6720; (E₄²)(n) = 480σ₇(n) for 1 ≤ n ≤ 30; |partial sum − ζ(s)ζ(s−3)|
    below 1e−3 at s = 5 and 1e−6 at s = 6.

WHAT THE FIGURE SHOWS
    (a) Four rows of the Dirichlet ring, each obtained from ζ by ring operations: d, σ₃, Λ, μ.
    (b) 240σ₃(n) beside the counted number of E8 vectors of squared norm 2n: equal.
    (c) The coefficients of E₄² = Θ_{E8⊕E8}: 480 vectors of squared norm 2, i.e. 480σ₇(1).
    (d) Σσ₃(n)n^{−s} (dashed) on ζ(s)ζ(s − 3) (solid) for 4.2 ≤ s ≤ 8.

WHAT IS PROVED / KNOWN
    σ₃ = Id₃ * ζ is the definition of σ₃ as a Dirichlet convolution (`Fugue.sigma_three_eq`;
    ledger C8 refutes the June guess σ₃ = ζ⁴). Θ_{E8} = E₄ with #{v ∈ E8 : |v|² = 2n} = 240σ₃(n):
    `Fugue.thetaE8_eq_E₄`, `Fugue.reprCount_e8` (ledger A2; Serre, A Course in Arithmetic,
    ch. VII). Θ_{L⊕M} = Θ_L Θ_M: `Fugue.theta_directSum`, `theta_e8_sum_e8` (ledger C13).
    E₄² = E₈ and σ₇ = σ₃ + 120 σ₃⋆σ₃: `Fugue.E₄sq_eq`, `Fugue.sigma_seven_eq` (Ramanujan 1916).
    Θ_{D16⁺} = Θ_{E8⊕E8} (Witt 1941): `Fugue.thetaD16_eq_thetaE8_sq`. Σσ₃(n)n^{−s} = ζ(s)ζ(s−3)
    for Re s > 4: `Fugue.HeckeE4.LSeriesHasSum_sigma_three`; the Epstein form Z_{E8} = 240ζ(s)ζ(s−3)
    is `Fugue.Epstein.epstein_E8`. Ledger rows C8, A2, C13, C1×C8, D16's neighbour D6–D8.

WHAT IS NOT CLAIMED
    That any of this is new, or that the chain carries information back from E8 to the primes:
    every arrow is a classical identity, and the lattice is a picture of σ₃, not a tool for
    computing it. "E8 ⊕ E8 and D16⁺ are the lattices of the two heterotic strings" is a true
    sentence of physics (Gross–Harvey–Martinec–Rohm 1985) about the same theta series; nothing
    here is a statement about string theory.

HOW THIS WAS FOUND
    We had the Dirichlet ring (figure 37), its derivation (figure 38) and the L-functions
    (figure 39), and noticed that one element, σ₃, was already in our hands twice: as a ring
    product Id₃ * ζ and as the coefficient vector of Θ_{E8}, the form the whole June network
    was built on. We guessed that the divisor function was the on-ramp from arithmetic to
    geometry, and first guessed its ring form wrongly (ζ⁴). The precise statements are the
    identities above, each a theorem with a Lean name; the one genuine surprise for us was that
    the composition of two E8 networks (the product E₄²) counts the vectors of E8 ⊕ E8 — and
    that a different lattice, D16⁺, counts them identically.
"""
from __future__ import annotations

from pathlib import Path

import mpmath as mp
import numpy as np

import dna_algebra as da
import e8
import style  # noqa: F401

NMAX = 30


def main():
    ns = np.arange(1, NMAX + 1)

    # --- the Dirichlet ring: everything from ζ ---
    zeta = da.ZETA.c[:NMAX + 1]
    id3 = np.concatenate([[0.0], ns.astype(float) ** 3])
    d = da.dirichlet_mul(zeta, zeta)[1:]
    sig3 = da.dirichlet_mul(id3, zeta)[1:]
    lam = da.LAMBDA.c[1:NMAX + 1]
    mu = da.MU.c[1:NMAX + 1]
    sig3_direct = np.array([e8.sigma(3, n) for n in ns], dtype=float)
    assert np.array_equal(sig3, sig3_direct)
    print(f"σ₃ = Id₃ * ζ in the Dirichlet ring: exact for n ≤ {NMAX}  (ledger C8: not ζ⁴)")

    # --- E8: #vectors of squared norm 2n = 240 σ₃(n) ---
    shells = e8.shells(6.0)
    counts = {n: int(shells[2 * n].shape[0]) for n in (1, 2, 3)}
    print("E8 shell counts vs 240σ₃(n):")
    for n in (1, 2, 3):
        pred = 240 * e8.sigma(3, n)
        print(f"    n={n}: counted {counts[n]:5d}   240σ₃({n}) = {pred:5d}   match={counts[n] == pred}")
        assert counts[n] == pred

    # --- E₄² = Θ_{E8⊕E8} = E₈ = 1 + 480 Σ σ₇(n) qⁿ ---
    e4 = e8.theta1_coeffs(NMAX)
    e4sq = np.convolve(e4, e4)[:NMAX + 1]
    sig7 = np.array([480 * e8.sigma(7, n) for n in ns], dtype=float)
    assert np.array_equal(e4sq[1:], sig7)
    print(f"(E₄²)(n) = 480σ₇(n) for n ≤ {NMAX}: exact  (E₄² = E₈; Θ(E8⊕E8) has {int(e4sq[1])} "
          f"vectors of squared norm 2)")

    # --- L-function: Σ σ₃(n) n^{-s} = ζ(s) ζ(s−3) ---
    mp.mp.dps = 25
    print("Σ σ₃(n) n^{−s} = ζ(s)ζ(s−3):")
    for s, tol in ((5, 1e-3), (6, 1e-6)):
        lhs = sum(mp.mpf(e8.sigma(3, n)) / mp.mpf(n) ** s for n in range(1, 4000))
        rhs = mp.zeta(s) * mp.zeta(s - 3)
        delta = abs(lhs - rhs)
        print(f"    s={s}: partial sum {mp.nstr(lhs, 12)}   ζ(s)ζ(s−3) = {mp.nstr(rhs, 12)}   "
              f"Δ = {mp.nstr(delta, 3)} (< {tol})")
        assert delta < tol

    _figure(ns, d, sig3, lam, mu, counts, e4sq)


def _figure(ns, d, sig3, lam, mu, counts, e4sq):
    import matplotlib.pyplot as plt

    TEAL, GOLD, RED, GREEN, VIO = "#46f0d0", "#ffd166", "#ff3b6b", "#7CFC8A", "#c0a3ff"
    fig = plt.figure(figsize=(13.5, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.26)

    # (a) everything from ζ
    ax = fig.add_subplot(gs[0, 0])
    for row, (lbl, arr, col) in enumerate([
            ("d = ζ*ζ", d / d.max(), TEAL),
            ("σ₃ = Id₃*ζ", sig3 / sig3.max(), GREEN),
            ("Λ = log*μ", lam / max(lam.max(), 1e-9), RED),
            ("μ = ζ⁻¹", (mu + 1) / 2, VIO)]):
        ax.vlines(ns, row, row + arr * 0.85, color=col, lw=2)
        ax.text(-1.5, row + 0.3, lbl, ha="right", fontsize=9)
    ax.set_xlim(-8, NMAX + 1); ax.set_ylim(-0.2, 4); ax.set_yticks([])
    ax.set_xlabel("n")
    ax.set_title("(a) the Dirichlet ring: every row is ζ with ring operations", fontsize=12)

    # (b) σ₃ = E₄ = Θ(E8)
    ax = fig.add_subplot(gs[0, 1])
    nn = np.arange(1, 6)
    ax.bar(nn - 0.2, [240 * e8.sigma(3, n) for n in nn], 0.4, label="240·σ₃(n)", color=GREEN)
    ax.bar(nn + 0.2, [counts.get(n, 0) for n in nn], 0.4,
           label="counted E8 vectors of |v|² = 2n (n ≤ 3)", color=GOLD, alpha=0.9)
    ax.set_yscale("log"); ax.set_xlabel("n"); ax.legend(fontsize=9, framealpha=0.3)
    ax.set_title("(b) 240σ₃(n) counts the E8 vectors of squared norm 2n:  E₄ = Θ(E8)", fontsize=12)

    # (c) composition = direct sum: E₄² = Θ(E8⊕E8)
    ax = fig.add_subplot(gs[1, 0])
    kk = np.arange(0, 8)
    ax.bar(kk, e4sq[:8], color=VIO)
    for k in (1, 2):
        ax.annotate(f"{int(e4sq[k])}", (k, e4sq[k]), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("n"); ax.set_ylabel("coefficient of E₄² = 480σ₇(n)")
    ax.set_title("(c) Θ(E8)·Θ(E8) = E₄² = Θ(E8⊕E8): product of forms = direct sum\n"
                 "(D16⁺ has the same theta series, Witt 1941)", fontsize=11)

    # (d) the L-function of σ₃ is ζ(s)ζ(s−3)
    ax = fig.add_subplot(gs[1, 1])
    svals = np.linspace(4.2, 8, 120)
    lhs = [float(sum(mp.mpf(e8.sigma(3, n)) / mp.mpf(n) ** s for n in range(1, 1500))) for s in svals]
    rhs = [float(mp.zeta(s) * mp.zeta(s - 3)) for s in svals]
    ax.plot(svals, rhs, lw=3, color=TEAL, label="ζ(s)·ζ(s−3)")
    ax.plot(svals, lhs, "--", lw=1.5, color=RED, label="Σ σ₃(n) n⁻ˢ (1500 terms)")
    ax.set_xlabel("s"); ax.set_yscale("log"); ax.legend(fontsize=9, framealpha=0.3)
    ax.set_title("(d) the L-function of σ₃ factors: ζ(s)ζ(s−3)  (Re s > 4)", fontsize=12)

    fig.suptitle("The divisor function σ₃ is the bridge:  Dirichlet ring → E₄ → E8 → E8⊕E8",
                 fontsize=14, y=0.97)
    out = Path(__file__).parent.parent / "figures" / "56_hsln_e8_divisor_bridge.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
