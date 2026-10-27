"""The Euler product, coefficient by coefficient: unique factorisation builds ζ (figure 32).

WHAT IT COMPUTES
    For each prime p the Euler factor  (1 − p^{−s})^{−1} = Σ_{k≥0} p^{−ks},  as a Dirichlet
    coefficient vector: 1 at n = 1, p, p², … and 0 elsewhere.  The factors are multiplied in
    the Dirichlet ring,  (a * b)(n) = Σ_{de=n} a(d) b(e),  one prime at a time.  After the
    primes ≤ M = 48 are in, the coefficient at every n ≤ 48 is 1: the window of ζ(s) = Σ n^{−s}.
    Numerically, at s = 2, the partial Euler product and the partial series both approach
    ζ(2) = π²/6.

WHAT THE FIGURE SHOWS
    Top left:  the Euler-factor vectors of 2, 3, 5, 7 (spikes at the prime powers).
    Top right: the partial product over primes and the partial series at s = 2, both against
               the dashed line π²/6.
    Bottom:    a grid, one row per prime added, one column per n: a cell lights when the
               coefficient at n has become 1.  The top row is all ones.

WHAT IS PROVED / KNOWN
    Euler (1737): ∏_p (1 − p^{−s})^{−1} = Σ_n n^{−s} for Re s > 1, which is the fundamental
    theorem of arithmetic written as a generating function.  Lean, in `Fugue/EulerCompose.lean`:
    `Fugue.EulerCompose.prod_eulerFactor` (∏_{p∈S} eulerFactor p is the indicator of the
    S-smooth numbers), `prod_eulerFactor_apply_of_le` (the all-ones row for n ≤ N),
    `smooth_tendsto_zeta` (the coefficientwise limit is ζ), and Mathlib's analytic product
    `riemannZeta_eulerProduct` via `zeta_eulerProduct`.  Ledger D1.

WHAT IS NOT CLAIMED
    Nothing about the analytic continuation of ζ, its zeros, or the Riemann hypothesis; the
    product only converges for Re s > 1, and the zeros live beyond it.  "Composing per-prime
    models" is not a mechanism of ours: it is the Euler product, and the grid is a picture of
    unique factorisation, not evidence for anything beyond it.

HOW THIS WAS FOUND
    After the sieve (figure 31) we had a dual picture: there we multiplied "drop the multiples
    of p" and the primes survived; here we multiply "keep the powers of p" and every integer is
    built exactly once.  We imagined each prime as a small model that knows only its own
    powers, and asked what merging the models would give.  The lofty guess was that this
    merging was the same operation as our geometric composition of forms; it is not (see
    figure 43 and ledger C1, C2).  The precise statement is Euler's: in the Dirichlet ring the
    product of the Euler factors is the smooth-number indicator, and once all primes ≤ N are in,
    the coefficients ≤ N are all 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dna_algebra import dirichlet_mul

M = 48


def primes_upto(n):
    return [k for k in range(2, n + 1) if all(k % d for d in range(2, int(k ** 0.5) + 1))]


def euler_factor(p, M=M):
    """Dirichlet coefficients of (1 − p^{−s})^{−1}: 1 at 1, p, p², …, else 0."""
    c = np.zeros(M + 1)
    k = 1
    while k <= M:
        c[k] = 1.0
        k *= p
    return c


def compose_primes(primes, M=M):
    """Multiply the Euler factors in the Dirichlet ring, keeping a snapshot after each prime."""
    c = np.zeros(M + 1); c[1] = 1.0
    snaps = []
    for p in primes:
        c = dirichlet_mul(c, euler_factor(p, M))
        snaps.append(c.copy())
    return c, snaps


def is_smooth(n, primes):
    """True iff every prime factor of n lies in `primes`."""
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1


def run(make_fig=True):
    ps = primes_upto(M)
    composite, snaps = compose_primes(ps)
    coeffs = composite[1:M + 1]
    all_ones = np.allclose(coeffs, 1.0)
    print(f"multiplied {len(ps)} Euler factors (primes up to {M}) in the Dirichlet ring.")
    print(f"coefficient a(n) for n = 1..{M}: {'all ones' if all_ones else 'NOT all ones'}  "
          f"(the window of zeta(s) = sum n^-s)")
    assert all_ones

    # The general statement (prod_eulerFactor): after the first j primes the coefficient at n
    # is 1 iff n is smooth over those primes, 0 otherwise.
    for j, snap in enumerate(snaps):
        want = np.array([1.0 if is_smooth(n, ps[:j + 1]) else 0.0 for n in range(1, M + 1)])
        assert np.allclose(snap[1:], want), f"snapshot after {ps[:j + 1]} is not the smooth indicator"
    print(f"after each prime the vector is the indicator of the smooth numbers: True")

    # Numerics at s = 2: Euler product vs series vs pi^2/6.
    target = np.pi ** 2 / 6
    prod_by_P = np.cumprod([1.0 / (1.0 - p ** (-2.0)) for p in ps])
    series_by_N = np.cumsum(1.0 / np.arange(1, M + 1) ** 2.0)
    print(f"Euler product over primes <= {M} at s=2: {prod_by_P[-1]:.6f}")
    print(f"series sum_(n <= {M}) 1/n^2:            {series_by_N[-1]:.6f}")
    print(f"zeta(2) = pi^2/6:                       {target:.6f}")
    # Both are below the limit and within the tail bounds 1/M (series) and ~1/(M log M) (product).
    assert 0 < target - series_by_N[-1] < 1.0 / M
    assert 0 < target - prod_by_P[-1] < 1.0 / M
    print(f"both partial values lie below pi^2/6 and within 1/{M} of it: True")

    if make_fig:
        _figure(ps, snaps, prod_by_P, series_by_N, target)
    return all_ones


def _figure(ps, snaps, prod_by_P, series_by_N, target):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import style  # noqa: F401

    TEAL, GOLD, RED, GREEN = "#46f0d0", "#ffd166", "#ff3b6b", "#7CFC8A"
    COLP = ["#46f0d0", "#ffd166", "#ff7b6b", "#c0a3ff"]

    fig = plt.figure(figsize=(15.5, 8.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.34)
    topo = gs[0].subgridspec(1, 2, width_ratios=[1.15, 1])
    a = fig.add_subplot(topo[0]); c = fig.add_subplot(topo[1])
    b = fig.add_subplot(gs[1])

    # --- A: the Euler-factor vectors --- #
    ns = np.arange(1, M + 1)
    for i, p in enumerate([2, 3, 5, 7]):
        ef = euler_factor(p)[1:]
        base = 1.4 * i
        a.vlines(ns[ef > 0], base, base + 1.0, color=COLP[i], lw=2)
        a.plot(ns[ef > 0], base + np.ones((ef > 0).sum()), "o", color=COLP[i], ms=5)
        a.text(M * 0.62, base + 0.35, f"p = {p}: n = 1, {p}, {p}², …",
               color=COLP[i], fontsize=8.5)
    a.set_yticks([]); a.set_xlabel("n"); a.set_ylim(-0.3, 1.4 * 4); a.set_xlim(0, M + 1)
    a.set_title("Euler factors $(1-p^{-s})^{-1}$ as coefficient vectors\n"
                "(1 at the powers of p, 0 elsewhere)", fontsize=12)

    # --- C: numerics at s = 2 --- #
    c.plot(np.arange(1, len(prod_by_P) + 1), prod_by_P, "o-", color=GOLD, lw=2, ms=5,
           label="Euler product (vs number of primes)")
    c.plot(np.arange(1, M + 1), series_by_N, color=TEAL, lw=1.8,
           label="series Σ 1/n² (vs number of terms)")
    c.axhline(target, color=RED, ls="--", lw=1.2, label=r"$\zeta(2)=\pi^2/6$")
    c.set_xlabel("number of primes (gold) / terms (teal) included")
    c.set_ylabel(r"value at $s=2$")
    c.set_title("the same identity numerically at s = 2:\nproduct and series both approach ζ(2)",
                fontsize=12)
    c.legend(loc="lower right", fontsize=8.5, framealpha=0.3)

    # --- B: one row per prime added --- #
    grid = np.array(snaps)[:, 1:M + 1]
    grid = (grid > 0).astype(float)
    b.imshow(grid, origin="lower", aspect="auto", cmap=ListedColormap(["#11111a", GREEN]),
             extent=[0.5, M + 0.5, -0.5, len(ps) - 0.5], interpolation="nearest")
    b.set_yticks(range(len(ps)))
    b.set_yticklabels([f"≤{p}" for p in ps], fontsize=7)
    b.set_xlabel("integer n  (lit = coefficient is 1 = n is smooth over the primes so far)")
    b.set_ylabel("primes included up to")
    b.set_title("each row is the indicator of the smooth numbers; once all primes ≤ 48 are in, "
                "every n ≤ 48 is built exactly once (unique factorisation)", fontsize=12)

    fig.suptitle("The Euler product coefficient by coefficient: multiplying the per-prime factors "
                 "in the Dirichlet ring gives the all-ones vector of ζ(s) = Σ n⁻ˢ (Re s > 1)",
                 fontsize=13, y=0.99)
    out = Path(__file__).parent.parent / "figures" / "32_zeta_compose.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
