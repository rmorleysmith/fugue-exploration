"""The explicit formula: x minus a sum over the zeros rebuilds the prime-power staircase ψ(x).

WHAT IT COMPUTES
    Chebyshev's ψ(x) = Σ_{p^k ≤ x} log p (the staircase; equivalently log lcm(1, …, ⌊x⌋)) on
    1.6 ≤ x ≤ 30, and its truncations by the von Mangoldt explicit formula

        ψ(x) = x − log 2π − ½ log(1 − x⁻²) − Σ_ρ x^ρ/ρ ,

    where ρ runs over the nontrivial zeros. Pairing ρ = 1/2 + iγ with its conjugate, each zero
    contributes the oscillation 2√x · Re[x^{iγ}/(1/2 + iγ)]; ψ_K keeps the first K zeros
    (K = 1, 10, 50, 200, 500, from the 500 cached by `zeros.py`). The four terms are the residues
    of −ζ′(s)/ζ(s) · x^s/s: at s = 1 (residue 1, the main term x), at s = 0 (−log 2π), at the
    trivial zeros (the ½ log term), and at each ρ. Printed checks: ψ(30) = log lcm(1..30), and
    the mean |ψ_K − ψ| on the window decreases with K and is below 0.1 at K = 500.

WHAT THE FIGURE SHOWS
    A. ψ_K for K = 1, 10, 50 over the true staircase: each zero adds one oscillation.
    B. ψ_500 (green) on the staircase, with the prime powers 2, 3, 4, 5, 7, 8, 9, … marked.

WHAT IS PROVED / KNOWN
    The explicit formula is von Mangoldt's theorem (1895; Davenport, Multiplicative Number
    Theory, ch. 17), with the sum over zeros conditionally convergent, taken symmetrically. The
    Dirichlet series −ζ′/ζ = Σ Λ(n) n^{−s} is Mathlib's
    `ArithmeticFunction.LSeries_vonMangoldt_eq_deriv_riemannZeta_div`; the residue 1 at s = 1 is
    `riemannZeta_residue_one`. The quantitative shadow of the formula — zero-free region ⟺
    error term in ψ(x) = x + O(x^θ) — is `Ingham.zeroFreeRight_iff_psiError` in FormalPNT.
    Evidence ledger D4 (PROVEN classically; the reconstruction here is NUMERICAL: mean error
    0.58 → 0.08 from 1 to 500 zeros).

WHAT IS NOT CLAIMED
    Nothing about where the zeros are: the formula holds whether or not RH does, and using
    Re ρ = 1/2 for the first 500 zeros uses a computation, not a theorem. "The zeros interpolate
    the primes" is a reading of the picture, not a proof technique.

HOW THIS WAS FOUND
    Before the June exploration knew the residue theorem, we imagined a residue as what the
    primes leave behind as a function unfolds along the number line — a running log of position.
    The precise statement is that −ζ′/ζ = Σ Λ(n) n^{−s} is the series whose coefficients are
    exactly those logs, that its residue at s = 1 is the main term x of ψ, and that each zero
    is a further residue, an oscillation; the explicit formula is that residue calculus. We then
    guessed the zeros "interpolate" the primes; the picture bears the guess out, and the ledger
    records it as a classical theorem checked numerically, not as something new.
"""
from __future__ import annotations

from math import lcm, log
from pathlib import Path

import numpy as np

from zeros import get_zeros


def true_psi(xs):
    """Chebyshev's ψ on the grid, and the list of (prime power, log p) jumps."""
    M = int(xs.max()) + 1
    primes = [p for p in range(2, M + 1) if all(p % d for d in range(2, int(p ** 0.5) + 1))]
    jumps = []
    for p in primes:
        pk = p
        while pk <= M:
            jumps.append((pk, np.log(p)))
            pk *= p
    psi = np.zeros_like(xs)
    for loc, lp in jumps:
        psi += lp * (xs >= loc)
    return psi, sorted(jumps)


def psi_K(xs, g, K):
    """The explicit formula truncated to the first K zeros (each paired with its conjugate)."""
    out = xs - np.log(2 * np.pi) - 0.5 * np.log(1.0 - xs ** -2.0)
    lnx = np.log(xs)
    for k in range(K):
        gk = g[k]
        out = out - 2 * np.sqrt(xs) * np.real(np.exp(1j * gk * lnx) / (0.5 + 1j * gk))
    return out


def run(make_fig=True):
    g = get_zeros(500)
    xs = np.linspace(1.6, 30.0, 4000)
    psi_true, jumps = true_psi(xs)

    # ψ(x) = log lcm(1..⌊x⌋): an independent definition of the staircase
    psi30 = sum(lp for loc, lp in jumps if loc <= 30)
    print(f"ψ(30) = Σ_(p^k ≤ 30) log p = {psi30:.6f};  log lcm(1..30) = {log(lcm(*range(1, 31))):.6f}")
    assert abs(psi30 - log(lcm(*range(1, 31)))) < 1e-9

    print(f"reconstructing ψ(x) on [1.6, 30] from up to {len(g)} zeros:")
    errs = []
    for K in (1, 10, 50, 200, 500):
        err = np.abs(psi_K(xs, g, K) - psi_true).mean()
        errs.append(err)
        print(f"  K={K:4d} zeros: mean |ψ_K − ψ| = {err:.4f}")
    assert all(a > b for a, b in zip(errs, errs[1:])) and errs[-1] < 0.1
    print("mean error decreases with K and is below 0.1 at K = 500: OK")

    if make_fig:
        _figure(xs, psi_true, jumps, g)
    return True


def _figure(xs, psi_true, jumps, g):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    TEAL, GOLD, RED, GREEN = "#46f0d0", "#ffd166", "#ff3b6b", "#7CFC8A"

    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.6))

    # A: progressive reconstruction
    a = ax[0]
    a.plot(xs, psi_true, color="white", lw=2.2, label="ψ(x) (prime-power staircase)")
    for K, col in [(1, RED), (10, GOLD), (50, TEAL)]:
        a.plot(xs, psi_K(xs, g, K), color=col, lw=1.3, alpha=0.9, label=f"{K} zeros")
    a.set_xlabel("x"); a.set_ylabel("ψ(x)")
    a.set_title("each zero adds an oscillation;\nmore zeros → the staircase emerges", fontsize=12)
    a.legend(loc="upper left", fontsize=8.5, framealpha=0.3)

    # B: K=500 tracks the staircase; prime powers marked
    b = ax[1]
    b.plot(xs, psi_true, color="white", lw=2.4, label="ψ(x)")
    b.plot(xs, psi_K(xs, g, 500), color=GREEN, lw=1.4, label="500 zeros")
    for loc, lp in jumps:
        if loc <= 30:
            b.axvline(loc, color="#445", lw=0.6, alpha=0.6)
    b.set_xlabel("x"); b.set_ylabel("ψ(x)")
    b.set_title("500 zeros reconstruct the prime-power staircase\n(vertical lines = prime powers)",
                fontsize=12)
    b.legend(loc="upper left", fontsize=9, framealpha=0.3)

    fig.suptitle("The explicit formula: x minus a sum over the zeros rebuilds ψ(x) — "
                 "the residue calculus of −ζ′/ζ = Σ Λ(n) n⁻ˢ", fontsize=13, y=1.0)
    out = Path(__file__).parent.parent / "figures" / "35_explicit.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
