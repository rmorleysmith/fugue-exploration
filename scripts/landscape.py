"""The largest prime factor and ω(n) for every n ≤ 10⁶, sieved in one pass (figure 36).

WHAT IT COMPUTES
    By a sieve over the primes p ≤ N (for each p, mark every multiple), two arithmetic
    functions on 2 ≤ n ≤ N = 10⁶:  lpf(n), the largest prime factor, and ω(n), the number
    of distinct prime factors.  From them: the prime count π(N), the proportion of
    100-smooth integers (lpf ≤ 100), the smoothness exponent α(n) = log lpf(n) / log n, and
    the empirical distribution of ω(n) against a Gaussian with its sample mean and variance.

WHAT THE FIGURE SHOWS
    Left:   a 2-D histogram of (log₁₀ n, log₁₀ lpf(n)); the diagonal is the primes, the
            curves below it are n = k·lpf for k = 2, 3, ….
    Middle: the density of α(n); the spike at α = 1 is the primes, the rest is the
            Dickman-type smoothness distribution.
    Right:  P(ω(n) = k) with a Gaussian of the same sample mean and variance (Erdős–Kac).

WHAT IS PROVED / KNOWN
    π(10⁶) = 78498 (Legendre's table; checked here by the sieve).  Dickman (1930): the
    proportion of n ≤ N with lpf(n) ≤ N^{1/u} tends to ρ(u).  Erdős–Kac (1940): the
    distribution of (ω(n) − log log n)/√(log log n) over n ≤ N tends to the standard normal.
    No Lean declaration covers these; ledger D19.

WHAT IS NOT CLAIMED
    The Gaussian in the right panel uses the sample σ ≈ 1.05, not the theoretical
    √(log log N) ≈ 1.62 (the convergence in Erdős–Kac is very slow), so the panel is a
    picture of the theorem, not a test of it.  The smoothness panel is a measurement, not a
    fit of Dickman's ρ.  The June text read the picture as "which integers are easy to take
    apart"; it is not about that (ledger D11): it is a census of the integers.

HOW THIS WAS FOUND
    After figures 31–33 we wanted to see the terrain those scripts walk on: for every
    integer, where its largest prime sits, and how many primes it has.  We expected the
    largest-prime panel to look like geology (strata of n = k·p), and it does.  The precise
    statements are Dickman's and Erdős–Kac's, both classical, and nothing in the picture
    is new.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


def sieve(N):
    isp = np.ones(N + 1, dtype=bool)
    isp[:2] = False
    for p in range(2, int(N ** 0.5) + 1):
        if isp[p]:
            isp[p * p::p] = False
    primes = np.nonzero(isp)[0]
    lpf = np.zeros(N + 1, dtype=np.int64)
    omega = np.zeros(N + 1, dtype=np.int16)
    for p in primes:                 # p increasing => the largest prime factor wins
        lpf[p::p] = p
        omega[p::p] += 1
    return lpf, omega


PI = {10 ** 4: 1229, 10 ** 5: 9592, 10 ** 6: 78498, 10 ** 7: 664579}


def run(N=1_000_000, make_fig=True):
    t0 = time.perf_counter()
    lpf, omega = sieve(N)
    print(f"sieved lpf(n) and omega(n) for n <= {N:,} in {time.perf_counter() - t0:.2f}s")
    n = np.arange(2, N + 1)
    lp = lpf[2:]
    om = omega[2:]
    nprimes = int((lp == n).sum())
    print(f"  primes <= {N:,}: {nprimes:,}  ({100 * nprimes / len(n):.2f}% of integers)")
    if N in PI:
        assert nprimes == PI[N], f"pi({N}) = {nprimes} != {PI[N]}"
        print(f"  pi({N:,}) = {PI[N]:,}: True")
    smooth100 = int((lp <= 100).sum())
    print(f"  100-smooth (lpf <= 100): {100 * smooth100 / len(n):.2f}%")

    # Spot checks of the sieve against direct computation.
    for k in (360, 1001, 65537, 999983, 2 ** 19 * 3):
        if k <= N:
            m, big, cnt, d = k, 1, 0, 2
            while d * d <= m:
                if m % d == 0:
                    cnt += 1; big = d
                    while m % d == 0:
                        m //= d
                d += 1
            if m > 1:
                cnt += 1; big = m
            assert lpf[k] == big and omega[k] == cnt, (k, lpf[k], big, omega[k], cnt)
    print("  lpf and omega agree with direct trial division on spot checks: True")

    mu, sd = om.mean(), om.std()
    th = np.sqrt(np.log(np.log(N)))
    print(f"  omega(n): sample mean {mu:.3f}, sample sigma {sd:.3f};  "
          f"log log N = {np.log(np.log(N)):.3f}, sqrt(log log N) = {th:.3f}")
    assert abs(mu - np.log(np.log(N))) < 1.0

    if make_fig:
        _figure(N, n, lp, om, mu, sd, th)
    return lpf, omega


def _figure(N, n, lp, om, mu, sd, th):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    fig, ax = plt.subplots(1, 3, figsize=(17.0, 5.4))

    # A: (n, largest prime factor)
    a = ax[0]
    a.hist2d(np.log10(n), np.log10(lp), bins=320, cmap="magma", norm="log", cmin=1)
    a.plot([0, np.log10(N)], [0, np.log10(N)], color="#46f0d0", lw=0.8, ls="--", alpha=0.7)
    a.text(np.log10(N) - 1.7, np.log10(N) - 0.45, "primes (lpf = n)", color="#46f0d0",
           fontsize=8, rotation=33)
    a.set_xlabel(r"$\log_{10} n$"); a.set_ylabel(r"$\log_{10}$ (largest prime factor)")
    a.set_title(f"largest prime factor of every n ≤ {N:,}\nprime diagonal and the curves "
                "n = k·lpf", fontsize=12)

    # B: smoothness exponent
    b = ax[1]
    alpha = np.log(lp) / np.log(n)
    b.hist(alpha, bins=200, color="#ffd166", density=True)
    b.axvline(1.0, color="#ff3b6b", lw=1.2)
    b.text(0.86, b.get_ylim()[1] * 0.6, "α = 1: primes", color="#ff3b6b", fontsize=8,
           rotation=90)
    b.set_xlabel(r"smoothness exponent  $\alpha=\log(\mathrm{lpf})/\log n$")
    b.set_ylabel("density")
    b.set_title("how smooth the integers are\n(Dickman-type distribution; the spike is the primes)",
                fontsize=12)

    # C: Erdős–Kac
    c = ax[2]
    kmax = int(om.max())
    ks = np.arange(1, kmax + 1)
    probs = np.array([(om == k).mean() for k in ks])
    c.bar(ks, probs, width=0.8, color="#46f0d0", label="P(ω(n) = k), n ≤ N")
    xx = np.linspace(0.5, kmax + 0.5, 300)
    c.plot(xx, np.exp(-(xx - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)),
           color="#ff3b6b", lw=2.4, label=f"Gaussian, sample μ={mu:.2f}, σ={sd:.2f}")
    c.text(0.98, 0.55, f"Erdős–Kac: σ → √(log log N) = {th:.2f}\n(slowly; the fit uses the "
           "sample σ)", transform=c.transAxes, color="#9a96b0", fontsize=8.2, ha="right")
    c.set_xlabel("ω(n) = number of distinct prime factors")
    c.set_ylabel("probability")
    c.set_title("ω(n) is bell-shaped\n(Erdős–Kac: asymptotically Gaussian)", fontsize=12)
    c.legend(loc="upper right", fontsize=8.5, framealpha=0.3)

    fig.suptitle(f"A census of the integers to {N:,}: largest prime factor, smoothness, and the "
                 "number of distinct prime factors", fontsize=13, y=1.0)
    out = Path(__file__).parent.parent / "figures" / "36_landscape.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    run(N)
