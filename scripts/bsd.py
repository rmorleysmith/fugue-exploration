"""L(E, 1) for five elliptic curves, from point counts composed in the Dirichlet ring (figure 40).

WHAT IT COMPUTES
    For E/ℚ with conductor N and root number w, the coefficients a_p = p + 1 − #E(𝔽_p) by
    counting points on the minimal Weierstrass model, the local Euler factors
        good p:  (1 − a_p p^{−s} + p^{1−2s})^{−1},  so  a_{p^{k}} = a_p a_{p^{k−1}} − p a_{p^{k−2}};
        bad p:   (1 − a_p p^{−s})^{−1},            so  a_{p^{k}} = a_p^{k},
    multiplied in the Dirichlet ring (`dna_algebra.dirichlet_mul`) to give (a_n)_{n ≤ NN}.
    Then the smoothed sum that follows from the functional equation Λ(2 − s) = w Λ(s):
        L(E, 1) = (1 + w) Σ_n (a_n / n) e^{−2πn/√N}.
    Curves (Cremona labels): 11a1 (rank 0), 14a1 (rank 0), 37a1 (rank 1), 389a1 (rank 2),
    5077a1 (rank 3).  Checks: a_p of 11a1 equals the LMFDB coefficients of newform 11.2.a.a
    at the first fifteen primes; L(11a1, 1) = 0.253841860856 and L(14a1, 1) = 0.330223659344
    (LMFDB) to 1e−5; L(389a1, 1) vanishes to 1e−12 although w = +1.

WHAT THE FIGURE SHOWS
    Top left:     a_p / (2√p) for p < 60, one curve per colour, inside the Hasse bound ±1.
    Top right:    the smoothed partial sums against K; rank-0 curves settle at a positive value,
                  w = −1 curves are identically 0, 389a1 cancels to 0 with w = +1.
    Bottom left:  L(E, 1) per curve, labelled by rank.
    Bottom right: a_p from point counting on 11a1 against the LMFDB newform 11.2.a.a.

WHAT IS PROVED / KNOWN
    Hasse (1933): |a_p| ≤ 2√p.  Modularity (Wiles 1995, Taylor–Wiles, Breuil–Conrad–Diamond–
    Taylor 2001): (a_n) is the coefficient vector of a weight-2 newform on Γ₀(N), so L(E, s)
    is entire with Λ(2 − s) = w Λ(s), which is what makes the smoothed sum exact.  Kolyvagin
    (1989) with Gross–Zagier (1986): if L(E, 1) ≠ 0 then E(ℚ) is finite, and if
    ord_{s=1} L(E, s) = 1 then the rank is 1.  Ledger D16 (L-values), D17 (389a1 is a genuine
    cancellation, not forced by the sign), C16 (η²(τ)η²(11τ) is the form of 11a1).  No Lean
    declaration covers modularity; the Dirichlet-ring closure used to multiply the factors is
    `Fugue.multiplicativeSubmonoid` (ledger C11).

WHAT IS NOT CLAIMED
    Nothing about the Birch–Swinnerton-Dyer conjecture is tested here.  The script shows that
    L(E, 1) ≠ 0 for the two rank-0 curves and L(E, 1) = 0 for the three others, which is one
    direction of BSD for rank 0 (a theorem, Kolyvagin) and, for ranks 1–3, is consistent with
    the conjecture but is not evidence for its leading-term formula, for ranks ≥ 2, or for the
    finiteness of Ш.  For 37a1 and 5077a1 the vanishing is forced by w = −1 and says nothing
    about the rank.  The June text's "the BSD pattern falls right out" is withdrawn.  The
    value "0.306" that a June figure caption attached to 389a1 does not exist (ledger D18): it
    is L′(37a1, 1), the rank-1 curve's derivative, which no script computes.  The modularity
    check is fifteen integers against a table, not a proof of anything.

HOW THIS WAS FOUND
    Having multiplied Euler factors for ζ (figure 32) and for characters (figure 39), we
    wanted the next family of coefficient vectors that the same ring product builds, and
    elliptic curves were the natural one: their a_p come from counting points, and modularity
    says the result is a weight-2 form, so the curve's L-function lives in the same ring.  The
    lofty guess was that a few L-values would show the rank.  The precise statement is that
    they show L(E, 1) ≠ 0 exactly for the rank-0 curves, which is Kolyvagin's theorem seen
    numerically; everything above rank 1 is conjecture, and the only surprising number here
    is that 389a1 cancels to 1e−16 without the sign making it do so.

References: LMFDB (https://www.lmfdb.org), Silverman "The Arithmetic of Elliptic Curves",
Dokchitser "Computing special values of motivic L-functions" (arXiv:math/0207280).
"""
from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np

from dna_algebra import dirichlet_mul, _delta

# Weierstrass coefficients [a1, a2, a3, a4, a6] (Cremona minimal models).
CURVES = {
    "11a1":   {"E": [0, -1, 1, -10, -20], "N": 11,   "w": +1, "rank": 0},
    "14a1":   {"E": [1,  0, 1,   4,  -6], "N": 14,   "w": +1, "rank": 0},
    "37a1":   {"E": [0,  0, 1,  -1,   0], "N": 37,   "w": -1, "rank": 1},
    "389a1":  {"E": [0,  1, 1,  -2,   0], "N": 389,  "w": +1, "rank": 2},
    "5077a1": {"E": [0,  0, 1,  -7,   6], "N": 5077, "w": -1, "rank": 3},
}

# LMFDB values of L(E, 1) for the rank-0 curves.
LMFDB_L1 = {"11a1": 0.25384186085591068, "14a1": 0.3302236593444805}

# LMFDB newform 11.2.a.a: a_p at the first fifteen primes.
LMFDB_11_2_a_a = {2: -2, 3: -1, 5: 1, 7: -2, 11: 1, 13: 4, 17: -2,
                  19: 0, 23: -1, 29: 0, 31: 7, 37: 3, 41: -8, 43: -6, 47: 8}


def primes_upto(n):
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool); sieve[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return [int(p) for p in np.nonzero(sieve)[0]]


def point_count(E, p):
    """#E(𝔽_p) including the point at infinity, for y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆."""
    a1, a2, a3, a4, a6 = [a % p for a in E]
    if p == 2:
        cnt = 1
        for x in (0, 1):
            for y in (0, 1):
                lhs = (y * y + a1 * x * y + a3 * y) % 2
                rhs = (x * x * x + a2 * x * x + a4 * x + a6) % 2
                if lhs == rhs:
                    cnt += 1
        return cnt
    cnt = 1
    for x in range(p):
        A = (a1 * x + a3) % p
        B = (x * x * x + a2 * x * x + a4 * x + a6) % p
        D = (A * A + 4 * B) % p          # (2y + A)² = A² + 4B
        if D == 0:
            cnt += 1
        elif pow(D, (p - 1) // 2, p) == 1:
            cnt += 2
    return cnt


def a_p(E, p):
    return p + 1 - point_count(E, p)


def local_factor_coeffs(ap, p, NN, divides_N):
    """Dirichlet coefficients of the local Euler factor at p, truncated to length NN + 1."""
    c = np.zeros(NN + 1); c[1] = 1.0
    if p > NN:
        return c
    seq = [1.0, float(ap)]
    pk, k = p, 1
    while pk <= NN:
        c[pk] = seq[k]
        nxt_pk = pk * p
        if nxt_pk > NN:
            break
        seq.append(seq[-1] * ap if divides_N else ap * seq[-1] - p * seq[-2])
        pk = nxt_pk
        k += 1
    return c


def build_an(curve, NN):
    """(a_n)_{n ≤ NN} by multiplying the local Euler factors in the Dirichlet ring."""
    E, N = curve["E"], curve["N"]
    factors = [local_factor_coeffs(a_p(E, p), p, NN, N % p == 0) for p in primes_upto(NN)]
    return reduce(dirichlet_mul, factors, _delta(NN))


def L_E_1_partial(an, w, N, K_grid):
    """Smoothed partial sums (1 + w) Σ_{n ≤ K} a_n/n · exp(−2πn/√N) at each K in K_grid."""
    ns = np.arange(1, len(an), dtype=float)
    contrib = an[1:] * np.exp(-2 * np.pi * ns / np.sqrt(N)) / ns
    return (1 + w) * np.cumsum(contrib)[np.asarray(K_grid) - 1]


def L_E_1(curve, NN):
    an = build_an(curve, NN)
    return float(L_E_1_partial(an, curve["w"], curve["N"], [NN])[0]), an


def run(make_fig=True):
    # ---------- modularity check: a_p of 11a1 against LMFDB 11.2.a.a ----------
    E11 = CURVES["11a1"]["E"]
    ap_match = {p: (a_p(E11, p), LMFDB_11_2_a_a[p]) for p in LMFDB_11_2_a_a}
    all_ok = all(my == lm for (my, lm) in ap_match.values())
    print("a_p of 11a1 by point counting vs LMFDB newform 11.2.a.a:")
    for p, (my, lm) in ap_match.items():
        print(f"  p={p:>3}   a_p = {my:>3}   LMFDB = {lm:>3}   {'ok' if my == lm else 'MISMATCH'}")
    assert all_ok
    print(f"  all {len(ap_match)} agree: True")

    # ---------- Hasse bound on every curve ----------
    for name, c in CURVES.items():
        assert all(abs(a_p(c["E"], p)) <= 2 * np.sqrt(p) for p in primes_upto(200)), name
    print("Hasse bound |a_p| <= 2 sqrt(p) holds for p < 200 on all five curves: True")

    # ---------- L(E, 1) ----------
    print("\nL(E, 1) = (1+w) sum a_n/n exp(-2 pi n / sqrt N):")
    out_vals, out_an = {}, {}
    for name, c in CURVES.items():
        NN = max(200, int(40 * np.sqrt(c["N"])))
        L1, an = L_E_1(c, NN)
        out_vals[name], out_an[name] = L1, an
        print(f"  {name:>6}  N={c['N']:>4}  w={c['w']:+d}  rank={c['rank']}  "
              f"L(E,1) = {L1:+.9f}   ({'nonzero' if abs(L1) > 1e-4 else 'zero'})")
    for name, want in LMFDB_L1.items():
        err = abs(out_vals[name] - want)
        print(f"  {name}: LMFDB {want:.12f}, |err| = {err:.1e}")
        assert err < 1e-5
    for name in ("37a1", "5077a1"):
        assert out_vals[name] == 0.0          # (1 + w) = 0: forced by the sign
    assert abs(out_vals["389a1"]) < 1e-12     # w = +1: genuine cancellation (ledger D17)
    S389 = float(np.sum(out_an["389a1"][1:] * np.exp(-2 * np.pi * np.arange(1, len(out_an["389a1"]))
                                                    / np.sqrt(389)) / np.arange(1, len(out_an["389a1"]))))
    print(f"  389a1: w = +1, so nothing forces a zero; the unsigned sum S = {S389:+.2e} cancels on its own")
    print("  rank-0 values match LMFDB to 1e-5; w = -1 values are 0; 389a1 cancels to 1e-12: True")

    if make_fig:
        _figure(out_vals, out_an, ap_match)
    return True


def _figure(L_vals, an_dict, ap_match):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    TEAL, GOLD, RED, GREEN = "#46f0d0", "#ffd166", "#ff3b6b", "#7CFC8A"
    PAPER_DIM = "#9a96b0"
    RANK_COLOR = {0: GREEN, 1: GOLD, 2: TEAL, 3: RED}

    fig = plt.figure(figsize=(17.0, 9.6))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.22)
    A = fig.add_subplot(gs[0, 0]); B = fig.add_subplot(gs[0, 1])
    C = fig.add_subplot(gs[1, 0]); Dax = fig.add_subplot(gs[1, 1])

    # -- A: a_p across primes --
    P_show = primes_upto(60)
    for name, curve in CURVES.items():
        col = RANK_COLOR[curve["rank"]]
        norm = [a_p(curve["E"], p) / (2 * np.sqrt(p)) for p in P_show]
        A.plot(P_show, norm, "o-", color=col, lw=1.3, ms=4.5, alpha=0.85,
               label=f"{name}  (rank {curve['rank']})")
    A.axhline(1, color="#666", ls=":", lw=0.7); A.axhline(-1, color="#666", ls=":", lw=0.7)
    A.text(58, 0.94, "Hasse: |aₚ| ≤ 2√p", color=PAPER_DIM, fontsize=8.4, ha="right")
    A.set_xlabel("prime p"); A.set_ylabel(r"$a_p / (2\sqrt{p})$")
    A.set_title("the coefficient vector of each curve: aₚ = p + 1 − #E(𝔽ₚ)\n"
                "(normalised by the Hasse bound)", fontsize=12)
    A.legend(loc="lower center", fontsize=7.8, framealpha=0.6, ncol=3)

    # -- B: convergence of the smoothed sum --
    K_grid = np.unique(np.geomspace(1, 1200, 60).astype(int))
    for name, curve in CURVES.items():
        col = RANK_COLOR[curve["rank"]]
        an = an_dict[name]
        K_use = K_grid[K_grid < len(an)]
        B.plot(K_use, L_E_1_partial(an, curve["w"], curve["N"], K_use), "-", color=col,
               lw=1.5, alpha=0.92, label=f"{name}  rank {curve['rank']}  (w={curve['w']:+d})")
    B.axhline(0, color="#555", lw=0.8)
    B.set_xscale("log"); B.set_xlabel("partial sum length K")
    B.set_ylabel(r"$(1+w)\sum_{n\leq K} a_n/n \cdot e^{-2\pi n/\sqrt{N}}$")
    B.set_title("L(E, 1) by the smoothed sum\n"
                "rank 0: positive;  w = −1: identically 0;  389a1 (w = +1): cancels to 0",
                fontsize=12)
    B.legend(loc="upper right", fontsize=8.2, framealpha=0.3)

    # -- C: L(E, 1) per curve --
    names = list(CURVES.keys())
    vals = [L_vals[n] for n in names]
    ranks = [CURVES[n]["rank"] for n in names]
    bars = C.bar(names, vals, color=[RANK_COLOR[r] for r in ranks], edgecolor="white", lw=0.6)
    for b, r, v in zip(bars, ranks, vals):
        C.text(b.get_x() + b.get_width() / 2, b.get_height() + (0.012 if abs(v) > 1e-4 else 0.09),
               f"rank {r}", ha="center", color=RANK_COLOR[r], fontsize=9)
        if abs(v) < 1e-4:
            C.text(b.get_x() + b.get_width() / 2, 0.015, "L(E,1) = 0", ha="center",
                   color="white", fontsize=8.2, rotation=90, va="bottom")
    C.axhline(0, color="#666", lw=0.7)
    C.set_ylabel("L(E, 1)")
    C.set_title("L(E, 1) ≠ 0 for the rank-0 curves, 0 for the others\n"
                "(L(E,1) ≠ 0 ⇒ rank 0 is Kolyvagin's theorem; the rest of BSD is open)",
                fontsize=12)

    # -- D: modularity in coefficients --
    ps = sorted(LMFDB_11_2_a_a.keys())
    mine = [ap_match[p][0] for p in ps]
    lm = [ap_match[p][1] for p in ps]
    Dax.vlines(ps, 0, mine, color=GREEN, lw=2.5, label="aₚ by point counting on 11a1")
    Dax.plot(ps, lm, "o", color=GOLD, ms=8, mfc="none", mew=2,
             label="LMFDB newform 11.2.a.a coefficients")
    Dax.axhline(0, color="#555", lw=0.6)
    Dax.set_xlabel("prime p"); Dax.set_ylabel(r"$a_p$")
    Dax.set_title("modularity, seen in coefficients\n"
                  "the curve's aₚ equal the weight-2 newform's aₚ (Wiles et al.; fifteen primes)",
                  fontsize=12)
    Dax.legend(loc="lower left", fontsize=8.5, framealpha=0.3)

    fig.suptitle("L(E, s) from point counts: per-prime Euler factors multiplied in the Dirichlet "
                 "ring, L(E, 1) by the functional-equation smoothed sum — numerics only, BSD untouched",
                 fontsize=12.5, y=0.99)
    out = Path(__file__).parent.parent / "figures" / "40_bsd.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    run()
