"""Elementary spectral facts checked with a stdlib eigensolver: a symmetric matrix has a real
spectrum, scaling the matrix scales the spectrum, the spectral zeta of {1, 2, 3, …} is ζ, and
a random symmetric matrix shows level repulsion.

WHAT IT COMPUTES
    No figure. A cyclic-Jacobi eigensolver for real symmetric matrices (stdlib only), and with it:
    (a) 300 random symmetric 5 × 5 matrices: the computed spectrum is real by construction and
        satisfies Σλ = tr A and Σλ² = ‖A‖_F² to 1e−9 (the solver's own consistency checks);
    (b) spec(cT) = c · spec(T) over 300 trials, to 1e−5;
    (c) with μ the eigenvalue of largest modulus, T/μ has the eigenvalue 1;
    (d) a symmetric T with a kernel keeps its zero eigenvalue under T ↦ cT, c ≠ 0;
    (e) the spectral zeta ζ_T(s) = Σ λₙ^{−s} of the spectrum {1, 2, 3, …}: ζ_T(2) → π²/6 and
        ζ_T(4) → π⁴/90 (partial sums to 10⁵, with the tail bound 1/((s−1)N^{s−1}));
    (f) six 50 × 50 GOE matrices: the fraction of bulk nearest-neighbour spacings below 0.1 of
        the mean is under 0.04, where independent (Poisson) spacings would give 1 − e^{−0.1} ≈ 0.095.

WHAT THE FIGURE SHOWS
    Nothing is drawn; this is the numerical companion of figure 34.

WHAT IS PROVED / KNOWN
    (a) is the spectral theorem for symmetric operators; the "real eigenvalues" half is Mathlib's
    `LinearMap.IsSymmetric.conj_eigenvalue_eq_self`. (b)–(d) are one-line consequences of
    Tv = λv. (e) is the definition of a spectral zeta function (Minakshisundaram–Pleijel 1949,
    Canad. J. Math. 1) applied to the simplest spectrum, with Euler's ζ(2) = π²/6. (f) is
    Wigner's level repulsion for the GOE (Mehta, Random Matrices, ch. 3). Evidence ledger H7:
    "self-adjoint ⇒ real spectrum ⇒ Hilbert–Pólya ⇒ RH" is PROVEN as an implication with its
    hypothesis open; D5 for the GUE statistics of the actual zeros.

WHAT IS NOT CLAIMED
    That any of this bears on where the zeros of ζ are. The June script announced "all theorems
    of HSLNSpectral.lean verified numerically" and dressed (b)–(d) as a reactor-criticality
    story; that Lean file is not in fugue-formal, and (b)–(d) are elementary. Hilbert–Pólya is
    a question: no operator with spectrum {γₙ} is exhibited here or anywhere.

HOW THIS WAS FOUND
    We had three stories in which ζ appears as an eigenvalue — the spectral zeta of an operator,
    the Hilbert–Pólya reading of the zeros, and the rescaling of an operator until its dominant
    mode is 1 — and guessed they were one story. The precise statement that unifies them is
    only that a self-adjoint operator has real spectrum; everything else here is a consequence
    of linearity, and the one non-trivial empirical fact (level repulsion) is the fingerprint,
    not the operator.
"""
from __future__ import annotations

import math
import random

TOL = 1e-7


# ── tiny linear algebra (stdlib only) ───────────────────────────────────────────

def scale(A, c):
    return [[c * A[i][j] for j in range(len(A))] for i in range(len(A))]


def sym_random(n, lo=-2.0, hi=2.0):
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            A[i][j] = A[j][i] = round(random.uniform(lo, hi), 4)
    return A


def jacobi_eigen(A, sweeps=100):
    """Eigenvalues of a real symmetric matrix via cyclic Jacobi rotations."""
    n = len(A)
    a = [row[:] for row in A]
    for _ in range(sweeps):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-20:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-18:
                    continue
                theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1))
                c = 1.0 / math.sqrt(t * t + 1)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
    return sorted(a[i][i] for i in range(n))


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert ok, f"FAILED: {name}"


# ── checks ───────────────────────────────────────────────────────────────────────

def main():
    random.seed(163)
    print("Elementary spectral facts, checked numerically (stdlib Jacobi eigensolver)\n")

    # (a) symmetric ⇒ real spectrum; solver consistency: Σλ = tr A, Σλ² = ‖A‖_F²
    print("(a) symmetric ⇒ real spectrum (Mathlib IsSymmetric.conj_eigenvalue_eq_self):")
    worst = 0.0
    for _ in range(300):
        A = sym_random(5)
        ev = jacobi_eigen(A)
        tr = sum(A[i][i] for i in range(5))
        fro = sum(A[i][j] ** 2 for i in range(5) for j in range(5))
        worst = max(worst, abs(sum(ev) - tr), abs(sum(x * x for x in ev) - fro))
    check(f"300 symmetric 5×5: real spectra with Σλ = tr A and Σλ² = ‖A‖_F² (max dev {worst:.1e})",
          worst < 1e-9)

    # (b) spec(cT) = c·spec(T)
    print("\n(b) spec(cT) = c·spec(T):")
    fails = 0
    for _ in range(300):
        A = sym_random(4)
        c = round(random.uniform(-3, 3), 4)
        expect = sorted(c * x for x in jacobi_eigen(A))
        got = jacobi_eigen(scale(A, c))
        if any(abs(u - v) > 1e-5 for u, v in zip(got, expect)):
            fails += 1
    check("spec(c·T) = c·spec(T) over 300 trials", fails == 0)

    # (c) T/μ has the eigenvalue 1
    print("\n(c) with μ the dominant eigenvalue, T/μ has the eigenvalue 1:")
    fails = 0
    for _ in range(300):
        A = sym_random(4)
        mu = max(jacobi_eigen(A), key=abs)
        if abs(mu) < 1e-3:
            continue
        if not any(abs(x - 1.0) < 1e-5 for x in jacobi_eigen(scale(A, 1.0 / mu))):
            fails += 1
    check("T/μ always has an eigenvalue equal to 1", fails == 0)

    # (d) a zero eigenvalue is preserved by rescaling
    print("\n(d) rescaling by c ≠ 0 keeps a zero eigenvalue:")
    fails = 0
    for _ in range(200):
        n = 4
        B = sym_random(n)
        for i in range(n):
            B[0][i] = B[i][0] = 0.0   # e0 is in the kernel
        c = round(random.uniform(0.2, 3.0), 4) * random.choice([-1, 1])
        has0_before = any(abs(x) < 1e-6 for x in jacobi_eigen(B))
        has0_after = any(abs(x) < 1e-6 for x in jacobi_eigen(scale(B, c)))
        if not (has0_before and has0_after):
            fails += 1
    check("λ = 0 preserved under T ↦ c·T over 200 trials", fails == 0)

    # (e) spectral zeta of the spectrum {1, 2, 3, …} is ζ
    print("\n(e) spectral zeta ζ_T(s) = Σ λₙ^{−s} for the spectrum {1, 2, 3, …}:")
    Nt = 100_000
    for s, target, name in ((2.0, math.pi ** 2 / 6, "π²/6"), (4.0, math.pi ** 4 / 90, "π⁴/90")):
        partial = sum(n ** (-s) for n in range(1, Nt + 1))
        tail = 1.0 / ((s - 1) * Nt ** (s - 1))
        check(f"ζ_T({s:.0f}) = {partial:.8f} + tail ≤ {tail:.1e}  vs {name} = {target:.8f}",
              abs(partial - target) <= tail + 1e-12)

    # (f) level repulsion in GOE spectra
    print("\n(f) GOE level repulsion (Wigner): fraction of bulk spacings below 0.1 of the mean:")
    spac = []
    for _ in range(6):
        N = 50
        G = [[0.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(i, N):
                g = random.gauss(0, 1)
                G[i][j] = G[j][i] = g if i != j else g * math.sqrt(2)
        ev = jacobi_eigen(G, sweeps=40)
        bulk = ev[N // 4: 3 * N // 4]
        gaps = [bulk[i + 1] - bulk[i] for i in range(len(bulk) - 1)]
        mean_gap = sum(gaps) / len(gaps)
        spac.extend(g / mean_gap for g in gaps)
    tiny = sum(1 for x in spac if x < 0.1) / len(spac)
    check(f"P(s < 0.1) = {tiny:.3f} over {len(spac)} spacings  (Poisson would give 0.095)",
          tiny < 0.04)

    print("\nall checks passed")


if __name__ == "__main__":
    main()
