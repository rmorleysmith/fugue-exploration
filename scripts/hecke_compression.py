"""Hecke's determination theorem in pictures: the prime values a_p of a weight-2 newform determine
every coefficient a_n, by a two-term recursion at each prime and multiplicativity across primes.

WHAT IT COMPUTES
    For the elliptic curve E: y² + y = x³ − x (Cremona 37a1, LMFDB 37.a1; conductor 37, rank 1)
    the coefficients a_p = p + 1 − #E(F_p) for every prime p ≤ 5000, by counting points with
    Legendre symbols of the discriminant 4(x³ − x) + 1. At p = 37 the same count on the nodal
    reduction gives a_37 = −1 (nonsplit multiplicative, LMFDB). From the a_p alone:
        a_{p^{r+2}} = a_p · a_{p^{r+1}} − p · a_{p^r}   (p ≠ 37),   a_{37^r} = a_37^r,
        a_{mn} = a_m · a_n   for gcd(m, n) = 1,
    which is the Hecke recursion in weight 2; the Sato–Tate angles θ_p = arccos(a_p / 2√p);
    and the fraction of n ≤ 10⁴ whose a_n is determined by the primes ≤ X.
    Printed checks: Hasse's bound |a_p| ≤ 2√p for all p ≤ 5000; a_37 = −1; for p ≤ 23 the
    recursion's a_{p²} = a_p² − p agrees with direct point counting over F_{p²}, through
    #E(F_{p²}) = p² + 1 − (a_{p²} − p) (the Frobenius trace over F_{p²} is α² + β² = a_p² − 2p,
    which differs from the Hecke coefficient a_{p²} by exactly p); the reconstructed a_n for
    n ≤ 18 equal the q-expansion of newform 37.2.a.a as listed by LMFDB; and the first two
    Sato–Tate moments (mean cos θ_p = 0, mean cos² θ_p = 1/4) hold within sampling error.

WHAT THE FIGURE SHOWS
    Figure 41 (41_hecke_compression.png):
    A. a_p for p ≤ 200 inside the Hasse envelope ±2√p.
    B. a_p for p ≤ 5000: the envelope is filled.
    C. The histogram of θ_p against the Sato–Tate density (2/π) sin² θ.
    D. a_p / 2√p on a log axis of p: bounded by ±1.
    Figure 42 (42_hecke_compression_recursion.png):
    A. a_{p^r} / ((r + 1) p^{r/2}) for p ≤ 200, r ≤ 12: every column generated from its r = 1 entry
       by the recursion, and bounded by 1 (Deligne).
    B. a_{p^r} / p^{r/2} for nine small primes, inside the envelope ±(r + 1).
    C. The reconstructed a_n for n ≤ 200.
    D. The reconstructed a_n for n ≤ 18 against LMFDB's listed values.
    E. π(N) against N and N/log N: the data a_p is logarithmically sparser than the data a_n.
    F. The fraction of a_n, n ≤ 10⁴, determined by the primes p ≤ X.

WHAT IS PROVED / KNOWN
    Hecke's theory (Hecke 1937, Math. Ann. 114, 1–28): a normalised eigenform's coefficients are
    its Hecke eigenvalues, multiplicative, with the two-term recursion at each prime; the
    Fourier coefficients of a Hecke eigenform are therefore determined by the prime-indexed ones.
    Modularity of 37a1 (Wiles 1995 for semistable curves) identifies a_p with p + 1 − #E(F_p).
    Hasse's bound (Hasse 1933). Sato–Tate for non-CM elliptic curves over Q (Barnet-Lamb,
    Geraghty, Harris, Taylor 2011, Publ. RIMS 47). Evidence ledger A5 (E₄ is an eigenform with
    eigenvalue σ₃(p); numerically) and D16 (37a1 point counts respect Hasse; L(37a1, 1) vanishes
    numerically). Lean: Mathlib at this commit has no Hecke operators (CORPUS-MAP, A5), so none
    of the eigenform statements has a Lean name; the one formalised Hecke-type statement is
    `Fugue.HeckeE4.heckeE4_functional_equation` (modularity of E₄ ⇒ the functional equation of
    ζ(s)ζ(s − 3)), which is about E₄, not about this form.

WHAT IS NOT CLAIMED
    That the determination theorem can be run backwards. Ledger H14 refutes the June hope that
    "primes determine everything" inverts into a way of recovering unknown prime factors from
    a_N: the map a_p ↦ p is not even injective in weight 2. The June script's third figure, a
    "big-ish prime" demonstration, and its partial Euler product at s = 1 are dropped: the latter
    was captioned as showing L(E, 1) = 0, but partial Euler products at the centre do not
    converge to L(E, 1) (Goldfeld 1982), so the panel showed nothing of the kind. The June
    "multiplicativity check" compared the construction with itself and is replaced by the
    independent checks above.

HOW THIS WAS FOUND
    We noticed that an L-function's coefficient vector (the Fourier coefficients, which the June
    files called the DNA) is enormously redundant: the values at primes fix the rest. We guessed
    this was a compression scheme with a decoder of our own; the precise statement is Hecke's
    theorem, and the decoder is the recursion he wrote down in 1937. The pictures are the
    theorem watched happening on one curve: a column of a_{p^r} unrolling from a single a_p,
    and the angle histogram filling Sato–Tate's sin² law as the primes accumulate.
"""
from __future__ import annotations

from math import gcd, log, sqrt
from pathlib import Path

import numpy as np

CONDUCTOR = 37
WEIGHT = 2
BOUND = 5000

# LMFDB newform 37.2.a.a, q-expansion through q^18 (terms with zero coefficient are not
# printed on the page; a_8 = a_17 = 0).  https://www.lmfdb.org/ModularForm/GL2/Q/holomorphic/37/2/a/a/
LMFDB_37_2_A_A = [1, -2, -3, 2, -2, 6, -1, 0, 6, 4, -5, -6, -2, 2, 6, -4, 0, -12]


# ── point counting ──────────────────────────────────────────────────────────────

def count_points_fp(p: int) -> int:
    """#E(F_p) for E: y² + y = x³ − x, p odd. Multiply by 4: (2y+1)² = 4(x³−x) + 1."""
    if p == 2:
        return 5
    count = 1  # point at infinity
    for x in range(p):
        disc = (4 * (x * x * x - x) + 1) % p
        if disc == 0:
            count += 1
        elif pow(disc, (p - 1) // 2, p) == 1:
            count += 2
    return count


def a_p_curve(p: int) -> int:
    """a_p = p + 1 − #E(F_p). At p = 37 the nodal count gives the multiplicative-reduction a_p."""
    return p + 1 - count_points_fp(p)


def count_points_fp2(p: int) -> int:
    """#E(F_{p²}) by direct enumeration, F_{p²} = F_p[t]/(t² − r) with r a non-residue."""
    r = next(v for v in range(2, p) if pow(v, (p - 1) // 2, p) == p - 1)

    def mul(u, v):
        return ((u[0] * v[0] + u[1] * v[1] * r) % p, (u[0] * v[1] + u[1] * v[0]) % p)

    def power(u, e):
        out, base = (1, 0), u
        while e:
            if e & 1:
                out = mul(out, base)
            base = mul(base, base)
            e >>= 1
        return out

    half = (p * p - 1) // 2
    count = 1
    for a in range(p):
        for b in range(p):
            x = (a, b)
            x3 = mul(mul(x, x), x)
            c = ((x3[0] - a) % p, (x3[1] - b) % p)
            disc = ((4 * c[0] + 1) % p, (4 * c[1]) % p)
            if disc == (0, 0):
                count += 1
            elif power(disc, half) == (1, 0):
                count += 2
    return count


def primes_up_to(N: int) -> list[int]:
    sieve = bytearray([1]) * (N + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(N ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(range(i * i, N + 1, i)))
    return [i for i in range(N + 1) if sieve[i]]


# ── Hecke recursion and multiplicativity ────────────────────────────────────────

def a_prime_power(p: int, r: int, ap: int) -> int:
    """a_{p^r} from a_p: the weight-2 Hecke recursion (a_{37^r} = a_37^r at the bad prime)."""
    if r == 0:
        return 1
    if p == CONDUCTOR:
        return ap ** r
    pk = p ** (WEIGHT - 1)
    prev_prev, prev = 1, ap
    for _ in range(2, r + 1):
        prev_prev, prev = prev, ap * prev - pk * prev_prev
    return prev


def a_n_from_primes(n: int, a_p: dict[int, int]) -> int | None:
    """a_n by multiplicativity and the recursion; None if a prime factor of n is not in a_p."""
    result, temp, p = 1, n, 2
    while p * p <= temp:
        if temp % p == 0:
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            if p not in a_p:
                return None
            result *= a_prime_power(p, e, a_p[p])
        p += 1
    if temp > 1:
        if temp not in a_p:
            return None
        result *= a_p[temp]
    return result


# ── figures ──────────────────────────────────────────────────────────────────────

def _fig41(a_p, primes, out):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    TEAL, GOLD, RED, GREEN, VIO = "#46f0d0", "#ffd166", "#ff3b6b", "#7CFC8A", "#c0a3ff"
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    sub = [p for p in primes if p <= 200]
    aps = [a_p[p] for p in sub]
    ax.vlines(sub, 0, aps, linewidth=0.9, color=TEAL, alpha=0.85)
    ax.scatter(sub, aps, s=14, color=TEAL, zorder=3)
    ax.axhline(0, color="#888", linewidth=0.6)
    pr = np.linspace(2, 200, 400)
    ax.plot(pr, 2 * np.sqrt(pr), "--", color=RED, alpha=0.6, label=r"Hasse bound $\pm 2\sqrt{p}$")
    ax.plot(pr, -2 * np.sqrt(pr), "--", color=RED, alpha=0.6)
    ax.set_xlabel("prime p"); ax.set_ylabel(r"$a_p$")
    ax.set_title(r"A. $a_p = p + 1 - |E(\mathbb{F}_p)|$ for $p \leq 200$"
                 "\n(the data that determines every $a_n$)", fontsize=12)
    ax.legend(loc="upper right"); ax.grid(alpha=0.2)

    ax = axes[0, 1]
    sub2 = [p for p in primes if p <= 5000]
    ax.scatter(sub2, [a_p[p] for p in sub2], s=4, color=TEAL, alpha=0.55)
    pr = np.linspace(2, 5000, 1000)
    ax.plot(pr, 2 * np.sqrt(pr), "--", color=RED, alpha=0.6, label=r"Hasse bound $\pm 2\sqrt{p}$")
    ax.plot(pr, -2 * np.sqrt(pr), "--", color=RED, alpha=0.6)
    ax.axhline(0, color="#888", linewidth=0.5)
    ax.set_xlabel("prime p"); ax.set_ylabel(r"$a_p$")
    ax.set_title(f"B. $a_p$ for $p \\leq 5000$ ({len(sub2)} primes)\nthe Hasse envelope is filled",
                 fontsize=12)
    ax.legend(loc="upper right"); ax.grid(alpha=0.2)

    ax = axes[1, 0]
    angles = [np.arccos(np.clip(a_p[p] / (2 * sqrt(p)), -1, 1)) for p in primes if p != CONDUCTOR]
    ax.hist(angles, bins=42, density=True, alpha=0.75, color="#3a5a45", edgecolor="#576")
    th = np.linspace(0, np.pi, 200)
    ax.plot(th, (2 / np.pi) * np.sin(th) ** 2, color=GOLD, linewidth=2.5,
            label=r"Sato–Tate: $\frac{2}{\pi}\sin^2\theta$")
    ax.set_xlabel(r"$\theta_p = \arccos(a_p / 2\sqrt{p})$"); ax.set_ylabel("density")
    ax.set_title(f"C. Sato–Tate angles over {len(angles)} primes\n"
                 "(Barnet-Lamb–Geraghty–Harris–Taylor 2011)", fontsize=12)
    ax.legend(loc="upper center"); ax.grid(alpha=0.2)

    ax = axes[1, 1]
    xs = [p for p in primes if p != CONDUCTOR]
    ax.scatter(xs, [a_p[p] / (2 * sqrt(p)) for p in xs], s=4, alpha=0.5, color=VIO)
    ax.axhline(1, color=RED, linestyle="--", alpha=0.5); ax.axhline(-1, color=RED, linestyle="--", alpha=0.5)
    ax.axhline(0, color="#888", linewidth=0.5)
    ax.set_xscale("log"); ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("prime p (log scale)"); ax.set_ylabel(r"$a_p / 2\sqrt{p} \in [-1, 1]$")
    ax.set_title("D. normalised prime values\n(bounded by 1; distributed by Sato–Tate)", fontsize=12)
    ax.grid(alpha=0.2, which="both")

    fig.suptitle("Hecke's determination theorem, I: the prime values $a_p$ of the newform 37.2.a.a "
                 "(curve 37a1), inside Hasse's bound, distributed by Sato–Tate", fontsize=13, y=1.0)
    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _fig42(a_p, primes, ans200, pct, Xs2, out):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    TEAL, GOLD, RED, GREEN, VIO = "#46f0d0", "#ffd166", "#ff3b6b", "#7CFC8A", "#c0a3ff"
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # A: heatmap of a_{p^r} normalised by Deligne's bound
    ax = axes[0, 0]
    heat_primes = [p for p in primes if p <= 200 and p != CONDUCTOR]
    R_MAX = 12
    heat = np.zeros((R_MAX + 1, len(heat_primes)))
    for j, p in enumerate(heat_primes):
        for r in range(R_MAX + 1):
            heat[r, j] = a_prime_power(p, r, a_p[p]) / ((r + 1) * p ** (r / 2))
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, origin="lower",
                   extent=[heat_primes[0] - 0.5, heat_primes[-1] + 0.5, -0.5, R_MAX + 0.5])
    ax.set_xlabel("prime p"); ax.set_ylabel("exponent r")
    ax.set_title(r"A. $a_{p^r}/[(r+1)\,p^{r/2}]$" + "\n(every column generated from row r = 1)",
                 fontsize=12)
    plt.colorbar(im, ax=ax, label="normalised value")

    # B: a_{p^r} for a few primes vs r
    ax = axes[0, 1]
    SHOW = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    cols = plt.cm.viridis(np.linspace(0.2, 0.95, len(SHOW)))
    rs = np.arange(0, R_MAX + 1)
    for p, c in zip(SHOW, cols):
        vals = [a_prime_power(p, r, a_p[p]) / p ** (r / 2) for r in rs]
        ax.plot(rs, vals, "-o", color=c, label=f"p = {p}, $a_p$ = {a_p[p]}", markersize=4, lw=1.4)
    ax.fill_between(rs, -(rs + 1), rs + 1, alpha=0.1, color=RED, label=r"envelope $\pm(r+1)$")
    ax.axhline(0, color="#888", linewidth=0.5)
    ax.set_xlabel("r"); ax.set_ylabel(r"$a_{p^r} / p^{r/2}$")
    ax.set_title("B. the recursion unrolled per prime\n(a two-term linear recurrence at each p)",
                 fontsize=12)
    ax.legend(loc="lower left", ncols=2, fontsize=7.5, framealpha=0.3); ax.grid(alpha=0.2)

    # C: reconstructed a_n for n ≤ 200
    ax = axes[0, 2]
    ns = np.arange(1, len(ans200) + 1)
    ax.bar(ns, ans200, color=TEAL, width=0.8)
    ax.axhline(0, color="#888", linewidth=0.5)
    ax.set_xlabel("n"); ax.set_ylabel(r"$a_n$")
    ax.set_title(f"C. $a_n$ for $n \\leq {len(ans200)}$\n(every value reconstructed from the $a_p$)",
                 fontsize=12)
    ax.grid(alpha=0.2)

    # D: reconstructed a_n vs LMFDB for n ≤ 18
    ax = axes[1, 0]
    n18 = np.arange(1, len(LMFDB_37_2_A_A) + 1)
    ax.bar(n18 - 0.2, ans200[:len(n18)], 0.4, color=TEAL, label="reconstructed from $a_p$")
    ax.bar(n18 + 0.2, LMFDB_37_2_A_A, 0.4, color=GOLD, label="LMFDB 37.2.a.a")
    ax.axhline(0, color="#888", linewidth=0.5)
    ax.set_xticks(n18); ax.set_xlabel("n"); ax.set_ylabel(r"$a_n$")
    ax.set_title("D. reconstruction against the listed q-expansion\n(n ≤ 18: all equal)", fontsize=12)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.3); ax.grid(alpha=0.2)

    # E: π(N) vs N
    ax = axes[1, 1]
    Ns = np.unique(np.logspace(1, 6, 200).astype(int))
    big_primes = np.array(primes_up_to(10 ** 6))
    pi_counts = [int((big_primes <= N).sum()) for N in Ns]
    ax.loglog(Ns, Ns, "--", color="#aaa", linewidth=1.4, label="integers n ≤ N (all $a_n$)")
    ax.loglog(Ns, pi_counts, color=TEAL, linewidth=2.4, label="primes p ≤ N (the $a_p$)")
    ax.loglog(Ns, [N / log(N) for N in Ns], ":", color=RED, linewidth=1.6, label=r"$N / \log N$ (PNT)")
    ax.set_xlabel("N"); ax.set_ylabel("count")
    ax.set_title("E. how much is determined by how little\n(π(N) against N)", fontsize=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.3); ax.grid(alpha=0.2, which="both")

    # F: reconstruction efficiency
    ax = axes[1, 2]
    ax.semilogx(Xs2, pct, "o-", color=VIO, markersize=7, linewidth=2)
    ax.fill_between(Xs2, 0, pct, alpha=0.15, color=VIO)
    ax.set_xlabel("prime cutoff X (log scale)"); ax.set_ylabel(r"% of $a_n$ determined, $n \leq 10^4$")
    ax.set_title("F. fraction of $a_n$, $n \\leq 10^4$, determined by $a_p$ for $p \\leq X$", fontsize=12)
    ax.set_ylim(0, 105); ax.grid(alpha=0.2, which="both")

    fig.suptitle("Hecke's determination theorem, II: the recursion at each prime and multiplicativity "
                 "rebuild every $a_n$ of 37.2.a.a from the $a_p$", fontsize=13, y=1.0)
    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    print("Hecke's determination theorem on E: y² + y = x³ − x (37a1 / newform 37.2.a.a)")
    primes = primes_up_to(BOUND)
    a_p = {p: a_p_curve(p) for p in primes}
    print(f"computed a_p by point counting for {len(primes)} primes ≤ {BOUND}")

    # Hasse's bound
    worst = max(abs(a_p[p]) / (2 * sqrt(p)) for p in primes)
    print(f"Hasse: max |a_p| / 2√p = {worst:.4f}  (≤ 1)")
    assert worst <= 1.0

    # the bad prime
    print(f"a_37 = {a_p[CONDUCTOR]}  (LMFDB: nonsplit multiplicative, so −1)")
    assert a_p[CONDUCTOR] == -1

    # a_{p²} from the recursion vs direct counting over F_{p²}
    for p in (3, 5, 7, 11, 13, 17, 19, 23):
        ap2 = a_prime_power(p, 2, a_p[p])
        direct = p * p + 1 - count_points_fp2(p)        # = α² + β² = a_p² − 2p
        assert ap2 - p == direct, (p, ap2, direct)
    print("a_{p²} = a_p² − p from the recursion agrees with #E(F_{p²}) = p² + 1 − (a_{p²} − p) "
          "by direct counting, p ≤ 23")

    # reconstruction vs LMFDB
    ans200 = [a_n_from_primes(n, a_p) for n in range(1, 201)]
    assert all(v is not None for v in ans200)
    assert ans200[:len(LMFDB_37_2_A_A)] == LMFDB_37_2_A_A, ans200[:18]
    print(f"reconstructed a_n, n ≤ 18: {ans200[:18]} = LMFDB 37.2.a.a")

    # Sato–Tate moments
    cosines = np.array([a_p[p] / (2 * sqrt(p)) for p in primes if p != CONDUCTOR])
    m1, m2 = cosines.mean(), (cosines ** 2).mean()
    print(f"Sato–Tate moments over {len(cosines)} primes: mean cos θ = {m1:+.4f} (0), "
          f"mean cos² θ = {m2:.4f} (0.25)")
    assert abs(m1) < 0.08 and abs(m2 - 0.25) < 0.05

    # reconstruction efficiency
    Xs2 = [3, 5, 7, 11, 17, 31, 53, 89, 151, 257, 433, 743, 1259, 2153, 3677]
    N_eval = 10_000
    pct = []
    for X in Xs2:
        restricted = {p: v for p, v in a_p.items() if p <= X}
        good = sum(1 for n in range(1, N_eval + 1) if a_n_from_primes(n, restricted) is not None)
        pct.append(100 * good / N_eval)
    print(f"fraction of a_n (n ≤ {N_eval}) determined by primes ≤ 3677: {pct[-1]:.1f}%")

    figs = Path(__file__).parent.parent / "figures"
    _fig41(a_p, primes, figs / "41_hecke_compression.png")
    _fig42(a_p, primes, ans200, pct, Xs2, figs / "42_hecke_compression_recursion.png")


if __name__ == "__main__":
    main()
