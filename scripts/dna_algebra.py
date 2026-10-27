"""Arithmetic functions under Dirichlet convolution: ζ, μ, Λ, χ₋₄ and four classical identities.

WHAT IT COMPUTES — coefficient vectors (the June files called them DNAs) f : ℕ → ℝ with f(0) = 0,
    held as arrays of length M + 1 = 81, with pointwise addition, Dirichlet convolution
    (f * g)(n) = Σ_{d|n} f(d) g(n/d), the Dirichlet inverse by the usual recurrence, and the
    derivation D f(n) = f(n) log n. Named vectors: δ = [n = 1], ζ = 1, μ (Möbius), Λ (von Mangoldt),
    χ = χ₋₄ (the character mod 4), log. Checks: ζ * μ = δ; ζ * ζ = d(n); 4 (ζ * χ) = r₂(n) against a
    brute-force count of a² + b² = n; L(1, χ₋₄) ≈ π/4 by the alternating series to 2·10⁶ terms;
    D(ζ) = log and log * μ = Λ; and the Euler product of L(s, χ₋₄) over the primes ≤ M rebuilt
    coefficient by coefficient.
WHAT IT PROVIDES — `dirichlet_mul`, `dirichlet_inv`, the `DNA` dataclass (`*`, `+`, `inv`, `D`,
    `equals`; the class keeps its June name because other scripts import it), the vectors `DELTA`,
    `ZETA`, `MU`, `LAMBDA`, `CHI4`, `LOG`, `char_euler_factor`, `compose_L`, and `run()`, which
    asserts the identities and draws figure 37 (read in the composition family's note).
WHAT IS PROVED / KNOWN — arithmetic functions form a commutative ring under (+, *) (Apostol,
    *Introduction to Analytic Number Theory*, ch. 2; Mathlib `ArithmeticFunction`). ζ * μ = δ
    (Mathlib `zeta_mul_moebius`; ledger C4). ζ * ζ = σ₀ (Lean `Fugue.zeta_mul_zeta_eq_sigma_zero`;
    C5). r₂(n) = 4 Σ_{d|n} χ₋₄(d), Jacobi (Lean `Fugue.card_sq_add_sq`; C6). L(1, χ₋₄) = π/4 (Lean
    `Fugue.LFunction_chi4_one`; D6). f ↦ f·log is a derivation and log * μ = Λ (Lean
    `Fugue.pmul_log_mul`, Mathlib `log_mul_moebius_eq_vonMangoldt`; C9). The checks here are
    numerical, to 1e−9 on n ≤ 80.
WHAT IS NOT CLAIMED — this Dirichlet ring is not the ring in which the q-series figures multiply:
    the same arrays carry two non-isomorphic ring structures (Lean `Fugue.not_ringEquiv_powerSeries`;
    ledger C1), and δ = [1, 0, 0, …] is the identity only here — in the q-series ring it is the form
    q (ledger C3). "Composition is ring algebra" is true of this ring and says nothing about a
    trained model.
HOW THIS WAS FOUND — we had been multiplying q-series to union zero sets and asked what the "same"
    multiplication does to Dirichlet coefficients; we imagined one algebra of DNAs with ζ as a
    generator, μ as its inverse, and the primes extracted by a "Lie derivative". The precise
    statements are the four classical identities above, and the fact that the two multiplications
    are different rings on one carrier.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np

M = 80  # number of Dirichlet coefficients (index 1..M; index 0 is unused padding)
PAPER_DIM = "#9a96b0"


# ----------------------------------------------------------------------------- ring ops
def dirichlet_mul(a, b):
    """(a * b)[n] = Σ_{d|n} a[d] b[n/d] — multiplication of Dirichlet series (the ring product)."""
    m = len(a) - 1
    c = np.zeros(m + 1)
    for d in range(1, m + 1):
        if a[d] == 0:
            continue
        for e in range(1, m // d + 1):
            c[d * e] += a[d] * b[e]
    return c


def dirichlet_inv(a):
    """The Dirichlet inverse g with a * g = δ (needs a[1] ≠ 0). Newton recurrence."""
    m = len(a) - 1
    g = np.zeros(m + 1)
    g[1] = 1.0 / a[1]
    for n in range(2, m + 1):
        s = sum(a[d] * g[n // d] for d in range(2, n + 1) if n % d == 0)
        g[n] = -s / a[1]
    return g


@dataclass
class DNA:
    name: str
    c: np.ndarray

    def __mul__(self, o):       # the ring product (Dirichlet convolution)
        return DNA(f"{self.name}*{o.name}", dirichlet_mul(self.c, o.c))

    def __add__(self, o):
        return DNA(f"{self.name}+{o.name}", self.c + o.c)

    def inv(self):              # the Dirichlet inverse
        return DNA(f"{self.name}⁻¹", dirichlet_inv(self.c))

    def D(self):                # the derivation a_n ↦ a_n · log n (June called it the Lie derivative)
        d = self.c.copy()
        n = np.arange(len(d), dtype=float)
        d[1:] = d[1:] * np.log(n[1:])
        return DNA(f"D({self.name})", d)

    def equals(self, o):
        return np.allclose(self.c, o.c, atol=1e-9)


# ----------------------------------------------------------------------------- the alphabet
def primes_upto(n):
    return [k for k in range(2, n + 1) if all(k % d for d in range(2, int(k ** 0.5) + 1))]


def _delta(m=M):
    c = np.zeros(m + 1); c[1] = 1.0; return c


def _zeta(m=M):
    c = np.ones(m + 1); c[0] = 0.0; return c


def _moebius(m=M):
    mu = np.ones(m + 1); mu[0] = 0.0
    for p in primes_upto(m):
        mu[p::p] *= -1.0
        mu[p * p::p * p] = 0.0
    return mu


def _von_mangoldt(m=M):
    c = np.zeros(m + 1)
    for p in primes_upto(m):
        pk = p
        while pk <= m:
            c[pk] = np.log(p)
            pk *= p
    return c


def _chi4(m=M):
    c = np.zeros(m + 1)
    for n in range(1, m + 1):
        c[n] = {1: 1.0, 3: -1.0}.get(n % 4, 0.0)
    return c


def _divisor_count(m=M):
    d = np.zeros(m + 1)
    for k in range(1, m + 1):
        d[k::k] += 1.0
    return d


def _r2(m=M):
    """r₂(n) = #{(a,b) ∈ ℤ²: a²+b²=n} by brute force (the ground truth for fact 3)."""
    r = np.zeros(m + 1)
    lim = int(m ** 0.5) + 1
    for a in range(-lim, lim + 1):
        for b in range(-lim, lim + 1):
            n = a * a + b * b
            if 1 <= n <= m:
                r[n] += 1.0
    return r


DELTA = DNA("δ", _delta())
ZETA = DNA("ζ", _zeta())
MU = DNA("μ", _moebius())
LAMBDA = DNA("Λ", _von_mangoldt())
CHI4 = DNA("χ", _chi4())
LOG = DNA("log", np.concatenate([[0.0], np.log(np.arange(1, M + 1, dtype=float))]))


# ----------------------------------------------------------------------------- compose χ as Euler product
def char_euler_factor(p, chi, m=M):
    """Dirichlet coeffs of (1 − χ(p) p^{-s})^{-1} = Σ χ(p)^k p^{-ks}: χ(p)^k at n = p^k."""
    c = np.zeros(m + 1)
    k, val = 1, 1.0
    while k <= m:
        c[k] = val
        val *= chi[p] if p <= m else 0.0
        k *= p
    return c


def compose_L(chi, m=M):
    """Build L(s,χ) = ∏_p (1 − χ(p)p^{-s})^{-1} by composing per-prime character-twisted factors.
    Its Dirichlet coefficients come out equal to χ(n) (χ completely multiplicative)."""
    factors = [char_euler_factor(p, chi, m) for p in primes_upto(m)]
    c = reduce(dirichlet_mul, factors, _delta(m))
    return DNA("∏ₚ(1−χ(p)p⁻ˢ)⁻¹", c)


def run(make_fig=True):
    # ---- fact 1: Möbius inversion  ζ * μ = δ,   and  ζ⁻¹ = μ
    inv_ok = ZETA.inv().equals(MU)
    delta_ok = (ZETA * MU).equals(DELTA)
    print(f"[1] group inverse:  ζ⁻¹ == μ : {inv_ok};   ζ * μ == δ : {delta_ok}")

    # ---- fact 2: ζ * ζ = number-of-divisors d(n)
    div_ok = np.allclose((ZETA * ZETA).c[1:], _divisor_count()[1:])
    print(f"[2] ζ * ζ == d(n) (divisor count) : {div_ok}")

    # ---- fact 3: ζ * χ₋₄  →  sums of two squares (Jacobi)
    L_composed = compose_L(CHI4.c)
    L_is_chi = L_composed.equals(CHI4)            # Euler product rebuilds χ(n)
    zeta_chi = ZETA * CHI4
    r2 = _r2()
    two_sq_ok = np.allclose(4.0 * zeta_chi.c[1:], r2[1:])   # Jacobi: r₂(n) = 4 Σ_{d|n} χ₄(d)
    leibniz = float(np.sum(CHI4.c[1:M + 1] / np.arange(1, M + 1)))
    # slow alternating series; evaluate L(1,χ₄) with many terms to show → π/4
    NN = 2_000_000
    ks = np.arange(1, NN + 1)
    chival = np.where(ks % 4 == 1, 1.0, np.where(ks % 4 == 3, -1.0, 0.0))
    leibniz_big = float(np.sum(chival / ks))
    print(f"[3] L(s,χ₄) = ∏ₚ Euler == χ(n) : {L_is_chi};   ζ * χ == r₂/4 (sum of two squares) : "
          f"{two_sq_ok}")
    print(f"    L(1,χ₄) = 1 − 1/3 + 1/5 − …  ≈ {leibniz_big:.6f}   (π/4 = {np.pi/4:.6f})")

    # ---- fact 4: the derivation f·log, followed by μ, gives Λ
    Dzeta_is_log = ZETA.D().equals(LOG)
    lam = LOG * MU                                # log * μ = Λ  (= −ζ'/ζ)
    lam_ok = lam.equals(LAMBDA)
    print(f"[4] Lie derivative:  D(ζ) == log : {Dzeta_is_log};   log * μ == Λ (the primes) : "
          f"{lam_ok}")

    assert all([inv_ok, delta_ok, div_ok, L_is_chi, two_sq_ok, Dzeta_is_log, lam_ok])
    print("All Dirichlet-ring identities hold to machine precision (asserted).")

    if make_fig:
        _figure(zeta_chi, r2, lam, leibniz_big)
    return True


def _figure(zeta_chi, r2, lam, leibniz_big):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    TEAL, GOLD, NEURON, GREEN, VIO = "#46f0d0", "#ffd166", "#ff3b6b", "#7CFC8A", "#c0a3ff"
    fig = plt.figure(figsize=(16.5, 9.2))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.22)
    A = fig.add_subplot(gs[0, 0]); B = fig.add_subplot(gs[0, 1])
    C = fig.add_subplot(gs[1, 0]); D = fig.add_subplot(gs[1, 1])
    ns = np.arange(1, M + 1)

    # --- A: the alphabet of named coefficient vectors (strips) -------------------
    rows = [("ζ  (generator)", ZETA.c, GOLD), ("μ  (Möbius = ζ⁻¹)", MU.c, VIO),
            ("Λ  (von Mangoldt / primes)", LAMBDA.c, NEURON), ("χ  (modular arithmetic, mod 4)", CHI4.c, TEAL)]
    for i, (lab, c, col) in enumerate(rows):
        base = 3.0 * (len(rows) - 1 - i)
        cc = c[1:M + 1]
        sc = cc / (np.abs(cc).max() + 1e-9)
        A.vlines(ns, base, base + sc, color=col, lw=1.6)
        A.axhline(base, color="#333", lw=0.6)
        A.text(M * 0.5, base + 1.35, lab, color=col, fontsize=9.5, ha="center")
    A.set_yticks([]); A.set_xlabel("n")
    A.set_title("the alphabet: named coefficient vectors as Dirichlet coefficients\n(reason about them, then compose)",
                fontsize=12)

    # --- B: the group — ζ*μ collapses to δ; ζ*ζ = divisor count -----------------
    conv = (ZETA * MU).c[1:M + 1]
    dd = (ZETA * ZETA).c[1:M + 1]
    B.vlines(ns, 0, conv, color=GREEN, lw=2.2)
    B.plot(ns[conv > 0.5], conv[conv > 0.5], "o", color=GREEN, ms=7,
           label="ζ * μ = δ  (single spike at n=1 → the inverse)")
    B.vlines(ns + 0.0, 0, -dd / dd.max() * 1.0, color=GOLD, lw=1.4, alpha=0.85)
    B.plot(ns, -dd / dd.max() * 1.0, ".", color=GOLD, ms=3,
           label="ζ * ζ = d(n)  (divisor count, scaled, below axis)")
    B.axhline(0, color="#555", lw=0.7)
    B.set_xlabel("n"); B.set_yticks([])
    B.set_title("the group + closure\nζ⁻¹=μ (so ζ*μ=δ);  ζ²=number of divisors", fontsize=12)
    B.legend(loc="upper right", fontsize=8.2, framealpha=0.3)

    # --- C: ζ * χ₋₄ = sums of two squares (Gaussian integers) --------------------
    circle_ns = [5, 25, 65, 50, 45]
    cols = [TEAL, GOLD, NEURON, GREEN, VIO]
    th = np.linspace(0, 2 * np.pi, 400)
    for n, col in zip(circle_ns, cols):
        rad = np.sqrt(n)
        C.plot(rad * np.cos(th), rad * np.sin(th), color=col, lw=0.9, alpha=0.55)
        pts = [(a, b) for a in range(-9, 10) for b in range(-9, 10) if a * a + b * b == n]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        C.plot(xs, ys, "o", color=col, ms=6)
        ang = 0.32 * np.pi
        C.text(rad * np.cos(ang), rad * np.sin(ang),
               f" {n}: r₂={len(pts)}=4·{int(zeta_chi.c[n])}", color=col, fontsize=7.6,
               ha="left", va="bottom")
    C.set_aspect("equal"); C.set_xlim(-9.5, 9.5); C.set_ylim(-9.5, 9.5)
    C.axhline(0, color="#333", lw=0.6); C.axvline(0, color="#333", lw=0.6)
    C.set_xlabel("a"); C.set_ylabel("b")
    C.set_title("modular arithmetic ⊗ ζ  =  sum-of-two-squares\n"
                r"$r_2(n)=4\sum_{d|n}\chi_4(d)$  →  Gaussian integers $\mathbb{Z}[i]$", fontsize=12)
    C.text(0.98, 0.02, f"L(1,χ₄)=1−⅓+⅕−…=π/4≈{leibniz_big:.4f}", transform=C.transAxes,
           color=PAPER_DIM, fontsize=8.5, ha="right", va="bottom")

    # --- D: the derivation → the primes -----------------------------------------
    D.plot(ns, LOG.c[1:M + 1], color=VIO, lw=1.8, label="D(ζ)(n) = log n  (Lie derivative of ζ)")
    lamc = lam.c[1:M + 1]
    D.vlines(ns, 0, lamc, color=NEURON, lw=2.0)
    D.plot(ns[lamc > 1e-6], lamc[lamc > 1e-6], "o", color=NEURON, ms=5,
           label="log * μ = Λ  (von Mangoldt → prime powers)")
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
        if p <= M:
            D.text(p, lam.c[p] + 0.12, str(p), color=NEURON, fontsize=7.5, ha="center")
    D.set_xlabel("n"); D.set_ylabel("value")
    D.set_title("the Lie derivative extracts the primes\n"
                r"differentiate the generator, normalise by $\zeta^{-1}=\mu$:  $-\zeta'/\zeta=\Lambda$",
                fontsize=12)
    D.legend(loc="upper left", fontsize=8.2, framealpha=0.3)

    fig.suptitle("The Dirichlet ring of coefficient vectors: name the operands, and ζ⁻¹ = μ, "
                 "ζ·χ₋₄ = r₂/4, −ζ′/ζ = Λ follow (checked here to machine precision; Mathlib/Fugue declarations in the header)",
                 fontsize=13, y=0.99)
    out = Path(__file__).parent.parent / "figures" / "37_dna_algebra.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
