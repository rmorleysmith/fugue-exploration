"""The zeta zeros have the spacing statistics of a random Hermitian matrix (Montgomery–Odlyzko).

WHAT IT COMPUTES
    The imaginary parts γ_n of the first 500 nontrivial zeros ρ_n = 1/2 + iγ_n of ζ (cached by
    `zeros.py` from mpmath.zetazero); |ζ(1/2 + it)| on 0 < t < 50; the unfolded nearest-neighbour
    spacings s_n = N(γ_{n+1}) − N(γ_n) with N(T) = (T/2π)(log(T/2π) − 1) + 7/8, so that the mean
    spacing is 1; the GUE Wigner surmise p(s) = (32/π²) s² exp(−4s²/π); the Poisson law exp(−s);
    and the bulk eigenvalue spacings of one 1500 × 1500 random Hermitian matrix H = (A + A*)/2.
    Printed checks: ζ vanishes at the first five γ_n (to 1e−6), the unfolded mean spacing is 1,
    the Kolmogorov–Smirnov distance of the zero spacings to GUE is below that to Poisson, and the
    small-gap probability P(s < 0.2) shows level repulsion for both the zeros and the matrix.

WHAT THE FIGURE SHOWS
    A. |ζ(1/2 + it)| on the critical line, touching zero at each γ_n (dotted lines).
    B. The histogram of unfolded zero spacings against the GUE surmise (gold) and the Poisson law
       (dashed): the spacings repel, as GUE does and Poisson does not.
    C. The bulk spacings of an actual random Hermitian matrix, against the same GUE curve.

WHAT IS PROVED / KNOWN
    Montgomery's pair-correlation conjecture (Montgomery 1973, "The pair correlation of zeros of
    the zeta function", Proc. Symp. Pure Math. 24) and Odlyzko's computations (Odlyzko 1987, Math.
    Comp. 48) are the classical statements; the pair correlation is a theorem only for test
    functions with restricted Fourier support (Montgomery 1973). The Wigner surmise is the exact
    2 × 2 GUE spacing law and an approximation to the large-N law (Mehta, Random Matrices, 3rd ed.,
    ch. 6–7). Evidence ledger rows D2 (functional equation, `completedRiemannZeta_one_sub` in
    Mathlib) and D5 (GUE repulsion, NUMERICAL: KS 0.046 vs GUE, 0.326 vs Poisson). Lean: the
    only formal statement in this direction is Mathlib's
    `LinearMap.IsSymmetric.conj_eigenvalue_eq_self` (a symmetric operator has real eigenvalues),
    which is the implication half of Hilbert–Pólya (ledger H7).

WHAT IS NOT CLAIMED
    That a self-adjoint operator with spectrum {γ_n} exists. Hilbert–Pólya is a question, and
    this script cannot see an operator: it sees statistics, and the statistics are verified
    numerically on 480 spacings, not proved. Nor is the June network such an operator; nothing
    here involves the network at all.

HOW THIS WAS FOUND
    We wanted to know what "a model exhibiting the Riemann hypothesis" could honestly mean, and
    the matrix-shaped answer was Hilbert–Pólya: if the γ_n were the eigenvalues of a self-adjoint
    operator they would be real, and all zeros would sit on the line. We imagined the network's
    coupling matrix as a candidate; the precise statement available is weaker and classical —
    the zeros have the spacing statistics of a random Hermitian matrix — and that is what this
    figure checks, with real zeros rather than a model of them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import erf
from scipy.stats import kstest

from zeros import get_zeros


def unfold(g):
    """Smooth counting function N(T) ~ (T/2π)(log(T/2π) − 1) + 7/8; gives mean spacing 1."""
    T = g / (2 * np.pi)
    return T * (np.log(T) - 1) + 7 / 8


def wigner_gue(s):
    """GUE Wigner surmise density p(s) = (32/π²) s² exp(−4s²/π)."""
    return (32 / np.pi ** 2) * s ** 2 * np.exp(-4 * s ** 2 / np.pi)


def wigner_gue_cdf(s):
    """Its distribution function: erf(2s/√π) − (4s/π) exp(−4s²/π)."""
    return erf(2 * s / np.sqrt(np.pi)) - (4 * s / np.pi) * np.exp(-4 * s ** 2 / np.pi)


def run(make_fig=True):
    g = get_zeros(500)
    print(f"loaded {len(g)} zeta zeros; first: 1/2 + {g[0]:.4f}i")

    # A: |ζ(1/2 + it)| on the line, and the check that it vanishes at the cached zeros
    import mpmath as mp
    mp.mp.dps = 15
    ts = np.linspace(0.1, 50, 1200)
    zmag = np.array([abs(complex(mp.zeta(0.5 + 1j * t))) for t in ts])
    at_zeros = max(abs(complex(mp.zeta(0.5 + 1j * gi))) for gi in g[:5])
    print(f"|ζ(1/2 + iγ_n)| at the first five zeros: max {at_zeros:.1e}  (target < 1e-6)")
    assert at_zeros < 1e-6

    # B: unfolded spacings (skip the small-γ transient where N(T) is a poor approximation)
    u = unfold(g[20:])
    s = np.diff(u)
    ks_gue = kstest(s, wigner_gue_cdf).statistic
    ks_poisson = kstest(s, lambda x: 1 - np.exp(-x)).statistic
    small_zeros = float((s < 0.2).mean())
    small_poisson = 1 - np.exp(-0.2)
    print(f"unfolded zero spacings: n = {len(s)}, mean {s.mean():.3f} (target 1.0)")
    print(f"KS distance to GUE {ks_gue:.4f}, to Poisson {ks_poisson:.4f}  (GUE must be closer)")
    assert abs(s.mean() - 1.0) < 0.05 and ks_gue < ks_poisson and ks_gue < 0.1

    # C: bulk spacings of a random Hermitian (GUE) matrix
    rng = np.random.default_rng(0)
    Nm = 1500
    A = rng.standard_normal((Nm, Nm)) + 1j * rng.standard_normal((Nm, Nm))
    H = (A + A.conj().T) / 2
    ev = np.sort(np.linalg.eigvalsh(H))
    mid = ev[Nm // 4: 3 * Nm // 4]
    sm = np.diff(mid)
    sm = sm / sm.mean()
    small_gue = float((sm < 0.2).mean())
    print(f"P(s < 0.2): zeros {small_zeros:.4f}, Hermitian matrix {small_gue:.4f}, "
          f"Poisson {small_poisson:.4f}  (level repulsion: the first two are small)")
    assert small_zeros < 0.05 and small_gue < 0.05

    if make_fig:
        _figure(ts, zmag, g, s, sm)
    return s, sm


def _figure(ts, zmag, g, s, sm):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    TEAL, GOLD, RED = "#46f0d0", "#ffd166", "#ff3b6b"
    ss = np.linspace(0, 3.2, 200)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # A: zeta on the critical line
    a = ax[0]
    a.plot(ts, zmag, color=TEAL, lw=1.4)
    for gi in g[g < 50]:
        a.axvline(gi, color=RED, ls=":", lw=0.8, alpha=0.7)
    a.set_xlabel("t"); a.set_ylabel(r"$|\zeta(\frac{1}{2} + it)|$")
    a.set_title("zeros on the critical line\n(|ζ| → 0 at each γₙ)", fontsize=12)

    # B: zero spacings vs GUE & Poisson
    b = ax[1]
    b.hist(s, bins=28, density=True, color="#33445a", edgecolor="#667",
           label="zeta-zero spacings (unfolded)")
    b.plot(ss, wigner_gue(ss), color=GOLD, lw=2.4, label="GUE Wigner surmise")
    b.plot(ss, np.exp(-ss), color=RED, lw=2, ls="--", label="Poisson (independent)")
    b.set_xlabel("normalised spacing s"); b.set_ylabel("density")
    b.set_title("zero spacings follow GUE, not Poisson\n(Montgomery–Odlyzko; numerical)", fontsize=12)
    b.legend(loc="upper right", fontsize=8.5, framealpha=0.3)

    # C: GUE matrix eigenvalue spacings
    c = ax[2]
    c.hist(sm, bins=28, density=True, color="#3a5a45", edgecolor="#576",
           label="random Hermitian eigenvalue spacings")
    c.plot(ss, wigner_gue(ss), color=GOLD, lw=2.4, label="GUE Wigner surmise")
    c.set_xlabel("normalised spacing s"); c.set_ylabel("density")
    c.set_title("a random Hermitian matrix's spectrum:\nthe same spacing law", fontsize=12)
    c.legend(loc="upper right", fontsize=8.5, framealpha=0.3)

    fig.suptitle("The zeta zeros have the spacing statistics of a random Hermitian matrix "
                 "(Montgomery–Odlyzko) — Hilbert–Pólya remains a question", fontsize=13, y=1.0)
    out = Path(__file__).parent.parent / "figures" / "34_spectral.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
