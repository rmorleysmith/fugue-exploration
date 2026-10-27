"""Composing per-prime divisibility detectors is trial division (figure 33).

WHAT IT COMPUTES
    For each prime p ≤ 100 the roots-of-unity detector
        D_p(n) = (1/p) Σ_{k=0}^{p−1} exp(2πi k n / p),
    which equals 1 when p | n (the p phasors align) and 0 otherwise (they cancel).  The
    detectors are applied in order; whenever one fires, p is divided out and the detector is
    applied again.  The result is the prime-power decomposition of n restricted to the
    primes ≤ 100, plus a residual cofactor with no prime factor ≤ 100.  The decomposition is
    checked against ordinary trial division on a small test set.

WHAT THE FIGURE SHOWS
    Top left:  the phasor sums for n = 12: p = 3 aligned (sum 3), p = 5 cancelling (sum 0).
    Top right: which of the 25 detectors fire on n = 2310 = 2·3·5·7·11.
    Bottom:    for n ≤ 520, the largest prime factor against n; green below the line
               lpf ≤ 100 (the detector set finds every factor), red above it (a residual
               remains).  The line is the reach of the detector set, nothing more.

WHAT IS PROVED / KNOWN
    Σ_{k<p} e^{2πikn/p} = p·[p | n] is the geometric series (ledger D12); Lean
    `Fugue.sum_exp_eq` and `Fugue.detector_iff` in `Fugue/Factoring.lean`.  Scanning the
    detectors in increasing order and stopping at the first that fires finds the least prime
    factor: `Fugue.first_spike_is_minFac`.  That is the definition of trial division.

WHAT IS NOT CLAIMED
    This script is kept for one reason: to show that composing per-prime divisibility
    detectors IS trial division.  Each detector costs p operations to answer "does p divide
    n", a question one integer remainder answers; composing them is the division ladder of
    Eratosthenes with a Fourier transform in front of each rung.  A set of detectors can only
    find the factors it contains, so a number whose prime factors all exceed 100 (such as
    101 · 103) is left whole; the June text called that line "the factoring wall" and read it
    as an explanation of why the problem is hard.  It is not (ledger D11): it restates the
    √N limit of trial division and says nothing about the existence or absence of faster
    methods.  The June script also used the primes below 10⁴ while saying "≤ 100", and ran
    trial division to √n on a 39-digit test number, so it did not terminate (ledger D15);
    both are fixed here.

HOW THIS WAS FOUND
    Figure 32 built the integers from per-prime Euler factors, and we asked for the inverse:
    take an integer apart by per-prime pieces.  The roots-of-unity comb was our native
    representation (the q-expansion basis), so each prime became a resonance detector, and
    "apply, divide out, repeat" looked like a composition of small models.  The precise
    statement is that the detector is the indicator of p | n and the composed procedure is
    trial division; the figure is an honest picture of that, and of how far a finite detector
    set reaches.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PRIMES = [p for p in range(2, 101) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def detect(p, n):
    """Roots-of-unity divisibility detector: True iff p | n (phasor sum equals p)."""
    k = np.arange(p)
    s = np.exp(2j * np.pi * k * (n % p) / p).sum() / p
    return abs(s - 1.0) < 1e-6


def factorise(n, primes=PRIMES):
    """Apply each detector; divide out while it fires. Returns ({p: e}, residual)."""
    facs, m = {}, n
    for p in primes:
        while detect(p, m):
            facs[p] = facs.get(p, 0) + 1
            m //= p
    return facs, m


def trial_division(n):
    facs, m, d = {}, n, 2
    while d * d <= m:
        while m % d == 0:
            facs[d] = facs.get(d, 0) + 1; m //= d
        d += 1
    if m > 1:
        facs[m] = facs.get(m, 0) + 1
    return facs


def fmt(facs, residual):
    parts = [f"{p}" + (f"^{e}" if e > 1 else "") for p, e in sorted(facs.items())]
    if residual > 1:
        parts.append(f"[{residual}]")
    return " · ".join(parts) if parts else "1"


def run(make_fig=True):
    # The detector is the indicator of p | n (Fugue.detector_iff), checked on a window.
    for p in PRIMES[:6]:
        assert all(detect(p, n) == (n % p == 0) for n in range(1, 400)), f"detector p={p}"
    print(f"D_p(n) = 1 iff p | n, checked for p in {PRIMES[:6]} and n < 400: True")

    tests = [2310, 9999, 10403, 1234567, 65536, 97 * 97 * 2, 101 * 101]
    print(f"\n{len(PRIMES)} detectors (primes <= 100), applied in order:")
    for n in tests:
        facs, res = factorise(n)
        tf = trial_division(n)
        small_ok = all(facs.get(p, 0) == e for p, e in tf.items() if p <= 100) \
            and all(p <= 100 for p in facs)
        res_ok = res == int(np.prod([p ** e for p, e in tf.items() if p > 100], dtype=object))
        assert small_ok and res_ok, f"{n}: detectors {facs},{res} vs trial division {tf}"
        tag = "complete" if res == 1 else f"residual {res}: every prime factor > 100"
        print(f"  {n:>8} = {fmt(facs, res):<28} ({tag})  agrees with trial division: True")
    print("\n  10403 = 101·103 and 1234567 = 127·9721 have no prime factor <= 100, so the "
          "detector set finds nothing:\n  a finite detector set reaches exactly the factors "
          "it contains, which is the sqrt(N) limit of trial division.")

    if make_fig:
        _figure()
    return True


def _figure():
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    GOLD, RED, GREEN = "#ffd166", "#ff3b6b", "#7CFC8A"

    fig = plt.figure(figsize=(15.5, 8.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.36)
    topo = gs[0].subgridspec(1, 2, width_ratios=[1, 1.25])
    a = fig.add_subplot(topo[0]); b = fig.add_subplot(topo[1])
    c = fig.add_subplot(gs[1])

    # --- A: divisibility as a phasor sum --- #
    n_demo = 12
    for p, col, lab, cx in [(3, GREEN, "p = 3 | 12: aligned, Σ = p", 0.0),
                            (5, RED, "p = 5 ∤ 12: cancels, Σ = 0", 2.6)]:
        k = np.arange(p)
        ph = np.exp(2j * np.pi * k * (n_demo % p) / p)
        for z in ph:
            a.annotate("", xy=(cx + z.real, z.imag), xytext=(cx, 0),
                       arrowprops=dict(arrowstyle="->", color=col, lw=1.6, alpha=0.85))
        S = ph.sum()
        a.annotate("", xy=(cx + S.real, S.imag), xytext=(cx, 0),
                   arrowprops=dict(arrowstyle="-|>", color="white", lw=2.4))
        a.text(cx, -1.7, lab, color=col, fontsize=8.5, ha="center")
    a.set_xlim(-1.5, 4.5); a.set_ylim(-2.1, 1.6); a.set_aspect("equal"); a.axis("off")
    a.set_title("the detector: Σ_k exp(2πikn/p) = p·[p | n]\n(n = 12)", fontsize=12)

    # --- B: which detectors fire on one n --- #
    n = 2310
    facs, res = factorise(n)
    resp = [1 if detect(p, n) else 0 for p in PRIMES]
    b.bar(range(len(PRIMES)), resp, color=[GOLD if r else "#33334a" for r in resp])
    b.set_xticks(range(len(PRIMES))); b.set_xticklabels(PRIMES, fontsize=6, rotation=90)
    b.set_yticks([])
    b.set_xlabel("detector p")
    b.set_title(f"detectors that fire on n = {n}\n{n} = {fmt(facs, res)}", fontsize=12)

    # --- C: the reach of the detector set --- #
    Ns = np.arange(2, 521)
    lpf = np.array([max(trial_division(int(k)).keys()) for k in Ns])
    full = lpf <= 100
    c.scatter(Ns[full], lpf[full], s=10, color=GREEN,
              label="every prime factor ≤ 100: detectors find all of them")
    c.scatter(Ns[~full], lpf[~full], s=14, color=RED,
              label="a prime factor > 100: a residual is left")
    c.axhline(100, color=GOLD, ls="--", lw=1.2)
    c.text(8, 108, "largest prime in the detector set", color=GOLD, fontsize=9)
    c.set_yscale("log"); c.set_xlabel("n"); c.set_ylabel("largest prime factor (log)")
    c.set_title("a detector set reaches exactly the primes it contains: "
                "the √N limit of trial division, drawn", fontsize=12)
    c.legend(loc="lower right", fontsize=8.5, framealpha=0.8)

    fig.suptitle("Composing per-prime divisibility detectors is trial division: each detector "
                 "is the indicator of p | n, and applying them in order is the division ladder",
                 fontsize=13, y=0.99)
    out = Path(__file__).parent.parent / "figures" / "33_factorize.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
