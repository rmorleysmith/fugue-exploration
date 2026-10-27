"""The nine Heegner CM points as a discrete alphabet inside the continuous state space ℍ.

WHAT IT COMPUTES — At a CM point (τ imaginary quadratic) the j-invariant is an algebraic
    integer. For the nine Heegner numbers d ∈ {1, 2, 3, 7, 11, 19, 43, 67, 163} the class
    number of ℚ(√−d) is 1, so j(τ_d) is a rational integer, and in fact a perfect cube:
        d=1: 1728 = 12³,  d=2: 8000 = 20³,  d=3: 0,  d=7: −3375 = (−15)³,  d=11: (−32)³,
        d=19: (−96)³,  d=43: (−960)³,  d=67: (−5280)³,  d=163: (−640320)³,
    with τ_d = i√d for d = 1, 2 and τ_d = (1 + i√d)/2 otherwise. Each value is recomputed
    with mpmath.kleinj at 80 digits (classical j = 1728·kleinj) and checked to 1e−20.
WHAT THE FIGURE SHOWS — Left: the nine CM points in the upper half-plane with their integer
    j. Right: the cube roots r with j = r³, on a symlog axis.
WHAT IS PROVED / KNOWN — The class-number-one discriminants are exactly these nine
    (Heegner 1952, Stark 1967, Baker 1966); j of a CM point is an algebraic integer of degree
    the class number (Weber; Cox, "Primes of the form x² + ny²", Thm 11.1); the cube property
    is γ₂ = j^{1/3} being a rational integer for d ≡ 3 mod 8 (Weber), and e^{π√163} ≈
    640320³ + 744 follows from the q-expansion j = 1/q + 744 + …. Ledger G1 (PROVEN, classical).
WHAT IS NOT CLAIMED — That the network uses this alphabet. In the scripts that follow
    (cm_task.py, cm_classifier.py) only two of the nine symbols, j = 1728 at i and j = 0 at
    ρ, are reached by the flow, so the alphabet is in practice a two-symbol code (ledger G3).
    Nothing about the flow is computed here; this script is the codebook.
HOW THIS WAS FOUND — We wanted a way for a continuous reservoir to say something discrete.
    The picture was a landscape with a few named landmarks: land on one, read its name. CM
    points are the named landmarks of ℍ — their j-values are integers you can look up — so
    we imagined a continuous flow spelling words by visiting them. The precise statement is
    the classical one above; what the imagined machine can actually spell is figure 24.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# Heegner numbers, their CM point tau_d, integer j, and the cube root r (j = r^3).
HEEGNER = [
    # d,   re,  im=sqrt(d)/2 (or sqrt(d) for d=1,2),  j,                     r
    (1, 0.0, 1.0, 1728, 12),
    (2, 0.0, float(np.sqrt(2)), 8000, 20),
    (3, 0.5, float(np.sqrt(3) / 2), 0, 0),
    (7, 0.5, float(np.sqrt(7) / 2), -3375, -15),
    (11, 0.5, float(np.sqrt(11) / 2), -32768, -32),
    (19, 0.5, float(np.sqrt(19) / 2), -884736, -96),
    (43, 0.5, float(np.sqrt(43) / 2), -884736000, -960),
    (67, 0.5, float(np.sqrt(67) / 2), -147197952000, -5280),
    (163, 0.5, float(np.sqrt(163) / 2), -262537412640768000, -640320),
]


def verify():
    import mpmath as mp
    mp.mp.dps = 80
    print("classical j = 1728*kleinj at the nine Heegner CM points (80 digits):")
    for d, _re, _im, j, r in HEEGNER:
        im = mp.sqrt(d) if d in (1, 2) else mp.sqrt(d) / 2   # high-precision Im(tau)
        tau = mp.mpc(mp.mpf(0) if d in (1, 2) else mp.mpf(1) / 2, im)
        jc = 1728 * mp.kleinj(tau)
        ok = abs(jc.real - j) < mp.mpf(10) ** -20 and abs(jc.imag) < mp.mpf(10) ** -20 and r ** 3 == j
        print(f"  d={d:3d}  j={j:>21}  = ({r})^3   {'OK' if ok else 'MISMATCH'}")
        assert ok, f"j at the Heegner point d={d} did not match"
    e = mp.exp(mp.pi * mp.sqrt(163))
    gap = e - (640320 ** 3 + 744)
    print(f"  e^(pi sqrt 163) - (640320^3 + 744) = {mp.nstr(gap, 6)}  (the q-expansion tail)")
    assert abs(gap) < 1e-12


def _jlabel(j, r):
    if j == 0:
        return "j = 0"
    if abs(j) < 1e5:
        return f"j = {j}"
    return f"j = ({r})³"


def figure():
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    GOLD, TEAL, NEURON = "#ffd166", "#46f0d0", "#ff3b6b"

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.4), gridspec_kw={"width_ratios": [1.05, 1]})

    # --- A: the CM ladder in the upper half-plane --- #
    a = ax[0]
    a.set_facecolor("#0b0b14")
    th = np.linspace(np.pi / 3, 2 * np.pi / 3, 100)
    a.plot(np.cos(th), np.sin(th), color="#3a3a4a", lw=1)
    a.plot([-0.5, -0.5], [np.sin(np.pi / 3), 6.7], color="#3a3a4a", lw=1)
    a.plot([0.5, 0.5], [np.sin(np.pi / 3), 6.7], color="#3a3a4a", lw=1)

    offs = {1: (-14, 0, "right"), 2: (-14, 0, "right"),
            3: (14, -12, "left"), 7: (14, 2, "left"), 11: (14, 6, "left")}
    names = {1: "  ($\\tau=i$)", 3: "  ($\\rho$)"}
    for d, re, im, j, r in HEEGNER:
        big = abs(j) > 1e6
        col = GOLD if d in (1, 3) else (NEURON if big else TEAL)
        dx, dy, ha = offs.get(d, (14, -3, "left"))
        a.plot(re, im, "o", color=col, ms=11, mec="white", mew=0.7, zorder=5)
        a.annotate(f"d={d}   {_jlabel(j, r)}{names.get(d, '')}", (re, im),
                   textcoords="offset points", xytext=(dx, dy),
                   color=col, fontsize=9.5, va="center", ha=ha)
    a.text(-0.85, 6.3, "toward the cusp\n($\\mathrm{Im}\\,\\tau\\to\\infty$)",
           color="#8888aa", fontsize=8.5)
    a.set_xlim(-0.95, 1.25)
    a.set_ylim(0.55, 6.7)
    a.set_xlabel(r"$\mathrm{Re}\,\tau$")
    a.set_ylabel(r"$\mathrm{Im}\,\tau$")
    a.set_title("the 9 Heegner CM points: a discrete alphabet in $\\mathbb{H}$\n"
                "(gold: the two the flow reaches in figures 24–25)", fontsize=12)

    # --- B: every Heegner j is a perfect cube --- #
    b = ax[1]
    ds = [h[0] for h in HEEGNER]
    x = np.arange(len(ds))
    b.axhline(0, color="#444", lw=0.8)
    for xi, (d, _, _, j, r) in zip(x, HEEGNER):
        big = abs(j) > 1e6
        col = GOLD if d in (1, 3) else (NEURON if big else TEAL)
        b.plot(xi, r, "o", color=col, ms=10, mec="white", mew=0.6, zorder=5)
        lab = "0" if r == 0 else f"{r}"
        b.annotate(lab, (xi, r), textcoords="offset points",
                   xytext=(0, 11 if r >= 0 else -16), color=col, fontsize=9, ha="center")
    b.set_yscale("symlog", linthresh=10)
    b.set_xticks(x)
    b.set_xticklabels([str(d) for d in ds])
    b.set_xlabel("Heegner number $d$")
    b.set_ylabel(r"cube root $r$, where $j=r^3$  (symlog)")
    b.set_title("each symbol's name $j(\\tau)$ is a perfect cube (Weber)", fontsize=12.5)

    fig.suptitle("CM points as an intrinsic symbol set: a point of $\\mathbb{H}$ with an integer name "
                 "$j(\\tau)$ (classical: Heegner–Stark–Baker)", fontsize=13, y=1.0)
    out = Path(__file__).parent.parent / "figures" / "23_cm_alphabet.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    verify()
    figure()
