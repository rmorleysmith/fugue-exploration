"""Eta quotients: multiplying forms adds exponent vectors, and admissibility is linear (figure 21).

WHAT IT COMPUTES
    Eta quotients on Γ₀(N),  f_r(τ) = ∏_{d|N} η(dτ)^{r_d},  with η(τ) = q^{1/24} ∏_{n≥1} (1 − qⁿ),
    q = e^{2πiτ}.  Three things:
      (1) f_r · f_s = f_{r+s} exactly, with weight k = ½ Σ r_d adding; checked at 60 random τ
          for a random pair r, s at level 12.
      (2) At level 11, the exponent pairs (r₁, r₁₁) in a box that satisfy the two congruences
          Σ d·r_d ≡ 0 and Σ (N/d)·r_d ≡ 0 (mod 24) (Newman), and among them those whose
          Ligozat cusp orders are all ≥ 0 with integer weight ≥ 1 (holomorphic forms).  Both
          conditions are linear in r, so the admissible set is closed under addition:
          (2, 2) + (2, 2) = (4, 4) is checked.
      (3) η(τ)²η(11τ)² has no zero in the upper half-plane; E₄ has one at ρ = e^{2πi/3}.

WHAT THE FIGURE SHOWS
    Left:   log₁₀ of the relative residual |f_r f_s − f_{r+s}| / |f_{r+s}|, at machine precision.
    Middle: the level-11 exponent lattice; grey = the congruences hold, gold = holomorphic;
            green arrows: (2,2), the weight-2 form of 11a1, added to itself gives (4,4).
    Right:  along a vertical line, |η(τ)²η(11τ)²| / |q| stays of order one (no zero), while
            |E₄| on the line Re τ = −½ dips to zero at Im τ = √3/2.

WHAT IS PROVED / KNOWN
    (1) is immediate from the definition since η ≠ 0 on ℍ (ledger C14).  (2) is the
    Gordon–Hughes–Newman / Ligozat criterion (Ono, "The Web of Modularity", Thms 1.64–1.65):
    with the two congruences and integer weight, f_r is a weakly holomorphic form on Γ₀(N) with
    a quadratic character, and the order at the cusp c/N is (N/24) Σ_d gcd(c,d)² r_d /
    (gcd(c, N/c) c d).  Linearity of both conditions in r is what makes the admissible set a
    monoid.  (3): Lean `Fugue.etaQuotient_ne_zero` in `Fugue/SieveEta.lean`, from Mathlib's
    `ModularForm.eta_ne_zero` (ledger C15).  That η(τ)²η(11τ)² is the newform of X₀(11),
    i.e. its coefficients are p + 1 − #E(𝔽_p) for the curve 11a1, is ledger C16 (checked for
    p ≤ 31 there and in figure 40 here).

WHAT IS NOT CLAIMED
    Eta quotients cannot place a zero in the interior of ℍ, so they cannot reproduce E₄ or any
    form with an interior zero (ledger C15 refutes "a zero anywhere").  The additive monoid is
    a bookkeeping device for exponents, not a composition rule for general modular forms; the
    script does not check the square condition for a trivial character, so the grey points may
    carry a quadratic character.  No statement about the trained network is made here.

HOW THIS WAS FOUND
    We wanted a family of forms in which composing two valid pieces gives a valid piece with no
    retraining, and noticed that for eta quotients multiplication is addition of exponent
    vectors (the exponents play the role the June files gave to a "DNA", i.e. the coefficient
    vector).  The lofty guess was that this would be the composition rule for the whole
    project.  The precise statement is narrower: the eta-quotient monoid is closed under
    addition because Newman's and Ligozat's conditions are linear, but every zero of an eta
    quotient is at a cusp, so the family misses exactly the interior zeros we cared about.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
#  Dedekind eta and eta quotients (q-expansion; accurate for Im tau >~ 0.1)    #
# --------------------------------------------------------------------------- #
def eta(tau, terms: int = 300):
    tau = np.asarray(tau, complex)
    q = np.exp(2j * np.pi * tau)
    pref = np.exp(1j * np.pi * tau / 12.0)        # q^{1/24}
    prod = np.ones_like(tau)
    qn = np.ones_like(tau)
    for _ in range(terms):
        qn = qn * q
        prod = prod * (1.0 - qn)
    return pref * prod


def eta_quotient(tau, divisors, r):
    tau = np.asarray(tau, complex)
    out = np.ones_like(tau)
    for d, rd in zip(divisors, r):
        out = out * eta(d * tau) ** rd
    return out


# --------------------------------------------------------------------------- #
#  Newman congruences + Ligozat cusp orders                                   #
# --------------------------------------------------------------------------- #
def divisors(N):
    return [d for d in range(1, N + 1) if N % d == 0]


def newman_residues(N, r):
    """The two Newman sums mod 24; both must vanish for a form on Gamma0(N)."""
    ds = divisors(N)
    s1 = sum(d * rd for d, rd in zip(ds, r))
    s2 = sum((N // d) * rd for d, rd in zip(ds, r))
    return s1 % 24, s2 % 24


def cusp_orders(N, r):
    """Ligozat order of vanishing at each cusp 1/c, c | N  (>= 0 everywhere == holomorphic)."""
    ds = divisors(N)
    orders = {}
    for c in ds:
        s = 0.0
        for d, rd in zip(ds, r):
            s += (gcd(c, d) ** 2 * rd) / (gcd(c, N // c) * c * d)
        orders[c] = (N / 24.0) * s
    return orders


def weight(r):
    return 0.5 * sum(r)


def is_admissible(N, r):
    """Holomorphic form on Gamma0(N) (quadratic character allowed): congruences + cone + weight."""
    r1, r2 = newman_residues(N, r)
    w = weight(r)
    cong = (r1 == 0 and r2 == 0)
    holo = all(o >= -1e-9 for o in cusp_orders(N, r).values())
    intwt = abs(w - round(w)) < 1e-9 and w >= 1 - 1e-9
    return cong and holo and intwt


# --------------------------------------------------------------------------- #
#  E4 (for the interior-zero contrast)                                        #
# --------------------------------------------------------------------------- #
def _sigma3(n):
    return sum(d ** 3 for d in range(1, n + 1) if n % d == 0)


def e4(tau, terms: int = 60):
    tau = np.asarray(tau, complex)
    q = np.exp(2j * np.pi * tau)
    out = np.ones_like(tau)
    for n in range(1, terms + 1):
        out = out + 240.0 * _sigma3(n) * q ** n
    return out


# --------------------------------------------------------------------------- #
def run(make_fig: bool = True):
    rng = np.random.default_rng(0)

    # ---- (1) multiplication adds exponents: f_r * f_s == f_{r+s} ----------- #
    N_lin = 12
    ds = divisors(N_lin)                          # [1,2,3,4,6,12]
    r = rng.integers(-3, 4, size=len(ds))
    s = rng.integers(-3, 4, size=len(ds))
    taus = (rng.uniform(-0.5, 0.5, 60) + 1j * rng.uniform(0.15, 1.5, 60))
    fr = eta_quotient(taus, ds, r)
    fs = eta_quotient(taus, ds, s)
    frs = eta_quotient(taus, ds, r + s)
    rel = np.abs(fr * fs - frs) / np.abs(frs)
    print(f"[1] level {N_lin}:  r={[int(v) for v in r]}  s={[int(v) for v in s]}")
    print(f"    weight(r)={weight(r)}  weight(s)={weight(s)}  weight(r+s)={weight(r + s)}  "
          f"(= {weight(r) + weight(s)})")
    print(f"    max relative residual |f_r f_s - f_(r+s)| / |f_(r+s)| = {rel.max():.2e}")
    assert rel.max() < 1e-10
    assert abs(weight(r + s) - weight(r) - weight(s)) < 1e-12

    # ---- (2) admissibility lattice + additive closure (N = 11) ------------ #
    N = 11                                        # divisors [1, 11]
    box = range(-12, 13)
    cong, adm = [], []
    for a in box:
        for b in box:
            r1, r2 = newman_residues(N, (a, b))
            if r1 == 0 and r2 == 0:
                cong.append((a, b))
                if is_admissible(N, (a, b)):
                    adm.append((a, b))
    base = (2, 2)                                 # eta(tau)^2 eta(11 tau)^2, weight 2
    summ = (4, 4)
    print(f"[2] level 11: {len(cong)} exponent pairs satisfy the congruences in the box, "
          f"{len(adm)} are holomorphic forms")
    print(f"    (2,2): admissible={is_admissible(N, base)} weight={weight(base)}  cusp orders="
          f"{ {k: round(v, 3) for k, v in cusp_orders(N, base).items()} }")
    print(f"    (4,4) = (2,2)+(2,2): admissible={is_admissible(N, summ)} weight={weight(summ)}")
    assert is_admissible(N, base) and is_admissible(N, summ)
    # closure of the admissible set under addition, inside the box
    adm_set = set(adm)
    closed = all((x1 + x2, y1 + y2) in adm_set for (x1, y1) in adm for (x2, y2) in adm
                 if abs(x1 + x2) <= 12 and abs(y1 + y2) <= 12)
    print(f"    admissible set closed under addition (within the box): {closed}")
    assert closed

    # ---- (3) no interior zero for the eta quotient; E4 vanishes at rho ---- #
    y = np.linspace(0.30, 2.3, 400)
    f_nf = eta_quotient(0.13 + 1j * y, [1, 11], [2, 2])
    f_norm = np.abs(f_nf) * np.exp(2 * np.pi * y)          # divide out |q| = e^{-2 pi y}
    e4_line = np.abs(e4(-0.5 + 1j * y))                     # zero at rho (y = sqrt(3)/2)
    print(f"[3] min |eta(t)^2 eta(11t)^2| / |q| on the slice = {f_norm.min():.3f} (no zero)")
    print(f"    min |E4| on the slice Re t = -1/2 = {e4_line.min():.2e}, at Im t = "
          f"{y[e4_line.argmin()]:.3f} (sqrt(3)/2 = {np.sqrt(3) / 2:.3f})")
    assert f_norm.min() > 0.1
    assert e4_line.min() < 1e-2 and abs(y[e4_line.argmin()] - np.sqrt(3) / 2) < 0.01

    if make_fig:
        _figure(rel, cong, adm, base, summ, N, y, f_norm, e4_line)
    return rel.max()


def _figure(rel, cong, adm, base, summ, N, y, f_norm, e4_line):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    GOLD, TEAL, GREEN, RED = "#ffd166", "#46f0d0", "#7CFC8A", "#ff3b6b"

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # --- A: residual --- #
    a = ax[0]
    a.plot(np.arange(len(rel)), np.log10(rel + 1e-18), "o", color=TEAL, ms=4)
    a.axhline(-15, color=GOLD, ls="--", lw=1, alpha=0.7, label="machine precision")
    a.set_ylim(-18, 0)
    a.set_xlabel("sample point  $\\tau$")
    a.set_ylabel(r"$\log_{10}\,|f_r f_s - f_{r+s}|/|f_{r+s}|$")
    a.set_title("multiply forms = add exponent vectors\n(exact; η has no zeros)", fontsize=12)
    a.legend(loc="upper right", fontsize=9, framealpha=0.3)

    # --- B: lattice + closure --- #
    b = ax[1]
    cong = np.array(cong); adm = np.array(adm)
    b.scatter(cong[:, 0], cong[:, 1], s=22, color="#556", label="Newman congruences hold")
    if len(adm):
        b.scatter(adm[:, 0], adm[:, 1], s=42, color=GOLD, edgecolor="white",
                  lw=0.4, label="holomorphic (Ligozat orders ≥ 0)", zorder=4)
    b.annotate("", xy=summ, xytext=base, arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
    b.annotate("", xy=base, xytext=(0, 0),
               arrowprops=dict(arrowstyle="->", color=GREEN, lw=2, alpha=0.6))
    for p, lab in [(base, f"$r$={base}, k={weight(base):.0f}  (11a1)"),
                   (summ, f"$r+r$={summ}, k={weight(summ):.0f}")]:
        b.plot(*p, "o", color=GREEN, ms=9, zorder=5)
        b.annotate(lab, p, textcoords="offset points", xytext=(8, 6), color=GREEN, fontsize=9)
    b.axhline(0, color="#444", lw=0.6); b.axvline(0, color="#444", lw=0.6)
    b.set_xlabel(r"$r_1$  (exponent of $\eta(\tau)$)")
    b.set_ylabel(r"$r_{11}$  (exponent of $\eta(11\tau)$)")
    b.set_title(f"level {N}: both conditions are linear in r,\nso the admissible set is "
                "closed under addition", fontsize=12)
    b.legend(loc="upper left", fontsize=8, framealpha=0.3)
    b.set_xlim(-12.5, 12.5); b.set_ylim(-12.5, 12.5)

    # --- C: no interior zero vs E4's zero at rho --- #
    c = ax[2]
    c.plot(y, np.log10(f_norm), color=GOLD, lw=2.2,
           label=r"$|\eta^2(\tau)\eta^2(11\tau)|\,/\,|q|$  on Re τ = 0.13")
    c.plot(y, np.log10(e4_line), color=RED, lw=2.2, label=r"$|E_4|$  on Re τ = −½")
    c.axvline(np.sqrt(3) / 2, color=RED, ls=":", lw=1.2, alpha=0.8)
    c.annotate("zero of $E_4$ at $\\rho$", (np.sqrt(3) / 2 + 0.05, -2.3),
               color=RED, fontsize=9, ha="left")
    c.set_xlabel(r"$\mathrm{Im}\,\tau$")
    c.set_ylabel(r"$\log_{10}$")
    c.set_title("eta quotients have no zero in ℍ (only at cusps);\n$E_4$ has one at ρ",
                fontsize=12)
    c.legend(loc="lower right", fontsize=8.5, framealpha=0.3)

    fig.suptitle("Eta quotients: composition is addition of exponent vectors, admissibility is "
                 "linear, and every zero is at a cusp", fontsize=13, y=1.02)
    out = Path(__file__).parent.parent / "figures" / "21_eta_compose.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
