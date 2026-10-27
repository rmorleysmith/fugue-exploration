"""The Euler operator θ = q d/dq on a divisor series: Σ_n θᵏ D_N = σ_k(N); and θ is diagonal, so
it reweights a coefficient vector and cannot create coefficients.

WHAT IT COMPUTES
    The divisor series D_N(q) = Σ_{n | N} qⁿ, i.e. the coefficient vector a_n = [n | N], for
    N = 8051 = 83 · 97 (all n ≤ N) and, over the window n ≤ 200 000, for the ordinary composite
    N = 1 000 003 · 1 000 033 (both factors prime, checked by trial division). The operator
    θ(Σ aₙqⁿ) = Σ n aₙ qⁿ (`lie_derivative.D`, figure 38) and θ². The sums
        Σ_n θᵏD_N = Σ_{d | N} dᵏ = σ_k(N),  k = 0, 1, 2,
    against the closed form in exact integers. The Serre derivative ϑ₄E₄ = θE₄ − (1/3)E₂E₄
    coefficientwise against E₆, for n ≤ 7. Printed checks: the three σ_k agree exactly; the
    window of D_N for the large N contains only the coefficient at n = 1, and so do θD_N and
    θ²D_N; ϑ₄E₄(n)/E₆(n) = −1/3 for 1 ≤ n ≤ 7.

WHAT THE FIGURE SHOWS
    1. D_N, θD_N and θ²D_N for N = 8051 on n ≤ 120, each normalised: the spikes at the divisors
       1, 83, 97, reweighted by n and n².
    2. Σ θᵏD_N against σ_k(N) for k = 0, 1, 2 (log scale): equal.
    3. The window n ≤ 200 000 of D_N for N = 1 000 003 · 1 000 033, with θD_N and θ²D_N: flat
       beyond n = 1, because the next divisor is outside the window and θ is diagonal.
    4. The gains 1, n, n² of the identity, θ and θ² on the grade n: a fixed reweighting.
    5. ϑ₄E₄(n)/E₆(n) = −1/3: the Serre derivative keeps E₄ inside the ring (weight 4 → 6).
    6. The printed checks.

WHAT IS PROVED / KNOWN
    Σ_n nᵏ [n | N] = σ_k(N) is the definition of σ_k (Mathlib `ArithmeticFunction.sigma`). The
    numbers 240σ₃(n) are the Fourier coefficients of E₄ = Θ_{E8} (`Fugue.thetaE8_eq_E₄`,
    `Fugue.reprCount_e8`; ledger A2). The Serre derivative identity ϑ₄E₄ = −E₆/3 is Mathlib's
    `serreDerivative_E₄` (ledger C10). That θ is diagonal in the basis {qⁿ} is immediate from
    its definition.

WHAT IS NOT CLAIMED
    That θ or θ² exposes anything the coefficient vector does not already contain. The June
    script applied them to "the complete landscape of a trained model" hoping the graded
    symmetry would reveal hidden structure; its own conclusion was that a diagonal operator
    gives structure, not search, and that conclusion is kept. The June panel used a 249-bit
    number from an external source; it is replaced here by an ordinary composite, and nothing
    in this script is about finding divisors. The June verdict panel's claims about cohomology
    and about what had been machine-checked are dropped.

HOW THIS WAS FOUND
    We were working in a graded ring whose generator we knew from figure 38, and we guessed that
    applying θ and θ² to the complete coefficient vector of a network — here the divisor series
    D_N — would make the grading tell us something new. What it tells us is exact and pretty:
    the sums Σθᵏ D_N are the divisor power sums σ_k(N), the very numbers that are the Fourier
    coefficients of E₄ when k = 3. But θ multiplies the n-th coefficient by n, and n · 0 = 0,
    so a coefficient that is absent stays absent. The precise statement is the identity above;
    the lofty guess was that a symmetry could do the work of a search, and it cannot.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import style
from lie_derivative import D as theta, eisenstein, qmul

FG = style.PAPER
ACCENT = style.ACCENT
ACCENT2 = "#4cc9f0"
COMP = "#9b8cff"
GRID = "#555566"

WINDOW = 200_000
SMALL_P, SMALL_Q = 83, 97
BIG_P, BIG_Q = 1_000_003, 1_000_033


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def divisor_landscape(N: int, nmax: int) -> np.ndarray:
    """Coefficients a_0..a_nmax of D_N(q) = Σ_{n | N} qⁿ (a_0 := 0)."""
    a = np.zeros(nmax + 1)
    for n in range(1, nmax + 1):
        if N % n == 0:
            a[n] = 1.0
    return a


def sigma(N: int, power: int) -> int:
    """σ_power(N) = Σ_{d | N} d^power, exact."""
    s, d = 0, 1
    while d * d <= N:
        if N % d == 0:
            s += d ** power
            if d != N // d:
                s += (N // d) ** power
        d += 1
    return s


def _grid(ax):
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    ax.tick_params(colors=FG)
    for s in ax.spines.values():
        s.set_color(GRID)


def main():
    small = SMALL_P * SMALL_Q
    big = BIG_P * BIG_Q
    assert is_prime(SMALL_P) and is_prime(SMALL_Q) and is_prime(BIG_P) and is_prime(BIG_Q)

    # --- Σ θᵏ D_N = σ_k(N) on the full series of the small composite ---
    F_full = divisor_landscape(small, small)
    lie = [F_full.sum(), theta(F_full).sum(), theta(theta(F_full)).sum()]
    sig = [sigma(small, k) for k in range(3)]
    print(f"N = {small} = {SMALL_P}·{SMALL_Q}, divisors {{1, {SMALL_P}, {SMALL_Q}, {small}}}")
    for k in range(3):
        print(f"  Σ θ^{k} D_N = {int(lie[k])}   σ_{k}(N) = {sig[k]}   match={int(lie[k]) == sig[k]}")
    assert all(int(lie[k]) == sig[k] for k in range(3))

    # --- the window of a larger composite: θ is diagonal, so nothing appears ---
    F_c = divisor_landscape(big, WINDOW)
    thF_c = theta(F_c)
    th2F_c = theta(thF_c)
    nz = [int((np.abs(v) > 0).sum()) for v in (F_c, thF_c, th2F_c)]
    print(f"N = {BIG_P}·{BIG_Q}, window n ≤ {WINDOW:,}: nonzero coefficients in "
          f"D, θD, θ²D = {nz}  (only n = 1; θ cannot create a coefficient)")
    assert nz == [1, 1, 1]

    # --- Serre derivative keeps E₄ in the ring: ϑ₄E₄ = −E₆/3 ---
    n_max = 7
    E2, E4, E6 = eisenstein(2, n_max), eisenstein(4, n_max), eisenstein(6, n_max)
    serre_E4 = theta(E4) - (4 / 12.0) * qmul(E2, E4)
    ratio = serre_E4[1:] / E6[1:]
    print(f"ϑ₄E₄(n)/E₆(n) for n = 1..{n_max}: {np.round(ratio, 6)}  (target −1/3)")
    assert np.allclose(ratio, -1 / 3, atol=1e-9)

    # --- the small-composite window for panel 1 ---
    nmax_s = 120
    F_s = divisor_landscape(small, nmax_s)
    thF_s = theta(F_s)
    th2F_s = theta(thF_s)

    _figure(F_s, thF_s, th2F_s, nmax_s, lie, sig, F_c, thF_c, th2F_c, ratio, nz)


def _figure(F_s, thF_s, th2F_s, nmax_s, lie, sig, F_c, thF_c, th2F_c, ratio, nz):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(17, 10))
    fig.suptitle("The Euler operator θ = q d/dq on a divisor series: Σ θᵏD_N = σ_k(N), and θ is "
                 "diagonal (structure, not search)", color=FG, fontsize=15, y=0.98)

    # 1 · D, θD, θ²D on the small composite
    ax = fig.add_subplot(2, 3, 1)
    for arr, lab, col in [(F_s, "D_N", ACCENT2),
                          (thF_s / max(thF_s.max(), 1), "θD_N  (×n)", ACCENT),
                          (th2F_s / max(th2F_s.max(), 1), "θ²D_N (×n²)", COMP)]:
        idx = np.where(arr > 0)[0]
        ax.scatter(idx, arr[idx], s=30, c=col, label=lab, zorder=4)
        ax.vlines(idx, 0, arr[idx], color=col, lw=1.6, alpha=0.8)
    for d in (SMALL_P, SMALL_Q):
        ax.annotate(str(d), (d, 1.04), color=FG, fontsize=9, ha="center")
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("grade n", color=FG); ax.set_ylabel("coefficient (each normalised)", color=FG)
    ax.legend(fontsize=8, labelcolor=FG, framealpha=0.2, loc="center right")
    ax.set_title(f"1 · N = {SMALL_P}·{SMALL_Q}: θ, θ² reweight the divisor spikes by n, n²",
                 color=FG, fontsize=11)
    _grid(ax)

    # 2 · Σ θᵏ D_N = σ_k(N)
    ax = fig.add_subplot(2, 3, 2)
    cats = ["Σθ⁰D = σ₀", "Σθ¹D = σ₁", "Σθ²D = σ₂"]
    xp = np.arange(3)
    ax.bar(xp - 0.2, np.log10(np.maximum(lie, 1)), 0.4, color=ACCENT, edgecolor=FG,
           label="from θ")
    ax.bar(xp + 0.2, np.log10(np.maximum(sig, 1)), 0.4, color=ACCENT2, edgecolor=FG,
           label="closed-form σ_k")
    for i, sv in enumerate(sig):
        ax.annotate(f"{int(sv)}", (i, np.log10(max(sv, 1)) + 0.1), color=FG, fontsize=8,
                    ha="center")
    ax.set_xticks(xp); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("log₁₀ value", color=FG)
    ax.legend(fontsize=8, labelcolor=FG, framealpha=0.2)
    ax.set_title("2 · Σθᵏ D_N = σ_k(N)   (k = 3 gives the coefficients of E₄)",
                 color=FG, fontsize=11)
    _grid(ax)

    # 3 · a larger composite over a window: flat beyond n = 1
    ax = fig.add_subplot(2, 3, 3)
    xs = np.arange(WINDOW + 1)
    ax.plot(xs, F_c, color=ACCENT2, lw=1.2, label="D_N")
    ax.plot(xs, thF_c, color=ACCENT, lw=1.2, ls="--", label="θD_N")
    ax.plot(xs, th2F_c, color=COMP, lw=1.2, ls=":", label="θ²D_N")
    ax.set_ylim(-0.1, 1.25)
    ax.text(0.5, 0.55, "one coefficient, at n = 1; flat beyond it\nn·0 = n²·0 = 0: θ reweights, "
            "it cannot create", transform=ax.transAxes, color=ACCENT2, fontsize=11, ha="center")
    ax.set_xlabel(f"grade n (to {WINDOW:,})", color=FG); ax.set_ylabel("coefficient", color=FG)
    ax.legend(fontsize=8, labelcolor=FG, framealpha=0.2, loc="upper right")
    ax.set_title(f"3 · N = {BIG_P}·{BIG_Q}: no divisor in the window", color=FG, fontsize=11)
    _grid(ax)

    # 4 · θ, θ² are diagonal: a fixed reweighting of the grade
    ax = fig.add_subplot(2, 3, 4)
    nn = np.arange(1, 60)
    ax.plot(nn, nn ** 2, color=COMP, lw=2, label="θ² gain = n²")
    ax.plot(nn, nn, color=ACCENT, lw=2, label="θ gain = n")
    ax.plot(nn, np.ones_like(nn), color=ACCENT2, lw=2, label="identity")
    ax.set_xlabel("grade n", color=FG); ax.set_ylabel("gain on aₙ", color=FG)
    ax.set_yscale("log")
    ax.legend(fontsize=8, labelcolor=FG, framealpha=0.2, loc="upper left")
    ax.set_title("4 · θ, θ² are diagonal in the basis {qⁿ}: a fixed reweighting",
                 color=FG, fontsize=11)
    _grid(ax)

    # 5 · Serre derivative: ϑ₄E₄ = −E₆/3
    ax = fig.add_subplot(2, 3, 5)
    n6 = np.arange(1, len(ratio) + 1)
    ax.plot(n6, ratio, "o-", color=ACCENT, lw=1.6, label="ϑ₄E₄(n) / E₆(n)")
    ax.axhline(-1 / 3, color=ACCENT2, ls="--", lw=1.2, label="−1/3 (Ramanujan; Mathlib serreDerivative_E₄)")
    ax.set_ylim(-0.6, 0.0)
    ax.set_xlabel("coefficient index n", color=FG)
    ax.set_ylabel("ϑ₄E₄(n) / E₆(n)", color=FG)
    ax.legend(fontsize=8, labelcolor=FG, framealpha=0.2)
    ax.set_title("5 · Serre derivative ϑ₄ = θ − (4/12)E₂·  keeps E₄ in the ring (weight 4 → 6)",
                 color=FG, fontsize=11)
    _grid(ax)

    # 6 · the checks
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    lines = [
        ("CHECKS (all assertions in the script)", ACCENT, 12),
        ("", FG, 5),
        (f"N = {SMALL_P}·{SMALL_Q}:", FG, 10),
        (f"  Σθ⁰D_N = {int(lie[0])} = σ₀(N)", FG, 10),
        (f"  Σθ¹D_N = {int(lie[1])} = σ₁(N)", FG, 10),
        (f"  Σθ²D_N = {int(lie[2])} = σ₂(N)", FG, 10),
        ("", FG, 5),
        (f"N = {BIG_P}·{BIG_Q}, n ≤ {WINDOW:,}:", FG, 10),
        (f"  nonzero in D, θD, θ²D = {nz}  (n = 1 only)", FG, 10),
        ("", FG, 5),
        ("ϑ₄E₄(n)/E₆(n) = −1/3 for n = 1..7", FG, 10),
        ("", FG, 5),
        ("θ is diagonal: it multiplies aₙ by n.", ACCENT2, 10),
        ("It reweights what is present; it does not", ACCENT2, 10),
        ("create what is absent. Structure, not search.", ACCENT2, 10),
    ]
    yy = 0.99
    for t, c, sz in lines:
        ax.text(0.0, yy, t, color=c, fontsize=sz, va="top", transform=ax.transAxes)
        yy -= 0.058 if t else 0.026

    fig.subplots_adjust(left=0.05, right=0.97, top=0.91, bottom=0.07, hspace=0.34, wspace=0.24)
    out = Path(__file__).parent.parent / "figures" / "63_hsln_lie_landscape.png"
    fig.savefig(out, facecolor=style.INK)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
