"""The degree-2 Siegel theta series of E8 on the Siegel upper half-space H_2.

WHAT IT COMPUTES —
    Theta^(2)_{E8}(Z) = sum_{v1,v2 in E8} exp( pi i * tr( G(v1,v2) Z ) ),
    G = [[v1.v1, v1.v2],[v1.v2, v2.v2]],  Z = [[Z11,Z12],[Z12,Z22]] in H_2,
    by its Fourier expansion sum_T r2(T) exp(pi i tr(T Z)), tr(T Z) = a*Z11 + 2b*Z12 + c*Z22
    for T = [[a,b],[b,c]]. The representation numbers r2(T) = #{(v1,v2): G(v1,v2) = T} are exact:
    every pair of lattice vectors with both norms <= maxn is enumerated and its Gram matrix
    histogrammed, so the truncation is exact for every T in that box.
WHAT IT PROVIDES — `representation_numbers(maxn)` and `theta2(dct, Z11, Z12, Z22)`. The
    `__main__` block asserts r2([[2,0],[0,2]]) = 240*126 (each root is orthogonal to 126 others),
    r2([[2,1],[1,2]]) = 240*56 (the pairs at 60 degrees), and that on the diagonal
    Z = diag(tau1, tau2) the truncated series equals the product of truncated E4's to 1e-7.
WHAT IS PROVED / KNOWN — for an even unimodular lattice of rank 2k, Theta^(g) is a Siegel modular
    form of weight k and degree g for Sp(2g,Z) (Freitag, *Siegelsche Modulfunktionen*, Springer
    1983); here k = 4, g = 2. On the diagonal Theta^(2) = E4(tau1) E4(tau2), because the Gram
    matrix of the pair splits. At degree 1 the direct-sum factorisation Theta_{L+M} = Theta_L
    Theta_M is Lean `Fugue.theta_directSum`. Audit ledger rows A2, C13. Siegel modular forms
    themselves are not in Mathlib.
WHAT IS NOT CLAIMED — the zero set of Theta^(2) in H_2 (complex dimension 3) is a divisor of
    complex dimension 2, not a curve and not a finite set (ledger A13); the figures drawn with
    this module show one complex slice at a time. Tabulating r2 costs O(|Lambda_{<=M}|^2):
    about 5 s at maxn = 6 and 38 s at maxn = 8, so nothing here is a training loop.
HOW THIS WAS FOUND — we imagined two coupled copies of the degree-1 picture with the
    off-diagonal entry Z12 as a dial between them (the June files called it "gravity"); the
    precise statement is that Z12 = 0 decouples the form into E4 (x) E4, and Z12 != 0 is the
    genuine degree-2 period, with the positivity condition |kappa| < 1 on the normalised
    coupling (Lean `Fugue.couplingMatrix_posDef_iff`).
"""
from __future__ import annotations

import numpy as np

import e8


def representation_numbers(maxn: int = 6, chunk: int = 512) -> dict[tuple[int, int, int], int]:
    """r2(T) for all T = [[a,b],[b,c]] with a, c <= maxn (a, c even; |b| <= maxn).

    Exact: built from Gram matrices of all E8 vector pairs with both norms <= maxn.
    """
    V = e8.e8_vectors(float(maxn))  # includes the origin
    norms = np.rint((V ** 2).sum(axis=1)).astype(int)
    ai_all = norms // 2
    nA = maxn // 2 + 1
    B0 = maxn
    counts = np.zeros((nA, nA, 2 * B0 + 1), dtype=np.int64)
    VT = V.T
    N = V.shape[0]
    for i0 in range(0, N, chunk):
        blk = V[i0:i0 + chunk]
        ai = ai_all[i0:i0 + chunk]
        B = np.rint(blk @ VT).astype(int)
        B = np.clip(B, -B0, B0)
        AI = np.repeat(ai, N)
        CJ = np.tile(ai_all, blk.shape[0])
        BB = B.ravel() + B0
        np.add.at(counts, (AI, CJ, BB), 1)

    out: dict[tuple[int, int, int], int] = {}
    for ia in range(nA):
        for ic in range(nA):
            for ib in range(2 * B0 + 1):
                r = int(counts[ia, ic, ib])
                if r:
                    out[(2 * ia, ib - B0, 2 * ic)] = r
    return out


def theta2(dct, Z11, Z12, Z22):
    """Evaluate the truncated Theta^(2)_{E8} on (broadcastable) arrays of Z entries."""
    Z11 = np.asarray(Z11, dtype=np.complex128)
    Z12 = np.asarray(Z12, dtype=np.complex128)
    Z22 = np.asarray(Z22, dtype=np.complex128)
    shape = np.broadcast_shapes(Z11.shape, Z12.shape, Z22.shape)
    acc = np.zeros(shape, dtype=np.complex128)
    for (a, b, c), r in dct.items():
        acc = acc + r * np.exp(1j * np.pi * (a * Z11 + 2 * b * Z12 + c * Z22))
    return acc


if __name__ == "__main__":
    import modular

    dct = representation_numbers(maxn=6)
    print(f"{len(dct)} Fourier terms up to norm 6")

    # Source-verified: each E8 root is orthogonal to exactly 126 others.
    r_220 = dct.get((2, 0, 2), 0)
    print(f"r2([[2,0],[0,2]]) = {r_220}  (expect 240*126 = {240*126})")
    assert r_220 == 240 * 126

    # Derived (root pairs at 60 degrees): 240*56.
    r_212 = dct.get((2, 1, 2), 0)
    print(f"r2([[2,1],[1,2]]) = {r_212}  (expect 240*56 = {240*56})")
    assert r_212 == 240 * 56

    # On the diagonal, Theta^(2) factorises as E4 (x) E4.
    # With Fourier cutoff norm<=6 this equals the order-3 truncation of E4 in each slot.
    r1 = {a: sum(dct.get((a, b, c0), 0) for b in range(-6, 7))
          for a in (0, 2, 4, 6) for c0 in (0,)}  # not used directly; sanity below
    tau1, tau2 = 0.2 + 1.1j, -0.1 + 1.3j
    lhs = theta2(dct, tau1, 0.0, tau2)
    # truncated E4 (orders 0..3) in each variable
    a_e4 = np.array([1.0, 240.0, 2160.0, 6720.0])
    q1 = np.exp(2j * np.pi * tau1)
    q2 = np.exp(2j * np.pi * tau2)
    e4t1 = sum(a_e4[k] * q1 ** k for k in range(4))
    e4t2 = sum(a_e4[k] * q2 ** k for k in range(4))
    rhs = e4t1 * e4t2
    print(f"diagonal factorization residual: {abs(lhs - rhs):.3e}")
    assert abs(lhs - rhs) < 1e-7
    print("siegel module OK")
