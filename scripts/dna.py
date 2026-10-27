"""Truncated coefficient vectors: E4, the level-11 newform 11.4.a.a, and designed q-polynomials.

WHAT IT COMPUTES — the Fourier coefficients, which the June files called the DNA, of three kinds
    of truncated form F(tau) = sum_{n<=N} a_n q^n: E4 to order 5 (`E4_TRUNC`); the weight-4
    level-11 newform 11.4.a.a to order 5 (`LEVEL11_4`, LMFDB, embedding beta = +sqrt(3)); and
    `dna_from_neurons(taus)`, the polynomial prod (q - q_i) with q_i = exp(2 pi i tau_i), whose
    zeros are the prescribed tau_i by construction. Also the q <-> tau conversions.
WHAT IT PROVIDES — `LEVEL11_4`, `E4_TRUNC`, `q_of_tau`, `tau_of_q`, `dna_from_neurons` (the
    function keeps its June name because every script imports it). The `__main__` block lists
    the roots of the two truncations in the q-disk.
WHAT IS PROVED / KNOWN — 11.4.a.a is the newform of S_4(Gamma_0(11)) with coefficient field
    Q(sqrt 3) (LMFDB, https://www.lmfdb.org/ModularForm/GL2/Q/holomorphic/11/4/a/a/). Its
    coefficients are congruent to those of (E4(tau) - E4(11 tau))/240 in M_4(Gamma_0(11))
    modulo the prime p = (61, sqrt(3) - 8) of Z[sqrt 3]: a(n) = sigma_3(n) - sigma_3(n/11)
    (mod p) for all n, an Eisenstein congruence with 61 | 11^4 - 1, checked to n = 200 against a
    Sturm bound of 4 (audit ledger row C16b). The roots of prod (q - q_i) are the q_i: the
    fundamental theorem of algebra.
WHAT IS NOT CLAIMED — a designed coefficient vector is a polynomial in q, not a modular form
    (ledger B5): no level-1 form of weight 8 has four distinct zeros, so the merged pictures built
    from these vectors illustrate the zero-union law for polynomials. The truncation of 11.4.a.a
    has 4 roots in the q-disk; the polynomial degree and the Gamma_0(11) valence count agree on
    that number by coincidence (audit thread architecture.md, 3.6), and no count of zeros is
    fixed by the Sturm bound (ledger A8). "Tweak a coefficient, move a zero" is the implicit
    function theorem at a simple zero, nothing more.
HOW THIS WAS FOUND — we needed a landscape with several wells, and E4 has only one per
    fundamental domain; the smallest level with a weight-4 cusp form is 11, so its newform became
    the proof-of-concept target. We noticed its coefficients looked "like E4 truncated to the
    same order"; the precise statement is the congruence above, the weight-4 level-11 twin of
    Ramanujan's tau(n) = sigma_11(n) (mod 691).
"""
from __future__ import annotations

import numpy as np

SQRT3 = np.sqrt(3.0)

# LMFDB newform 11.4.a.a, embedding beta = +sqrt(3). Cusp form, so a0 = 0.
# Coefficients a0..a5. The Sturm bound for M_4(Gamma_0(11)) is floor(4*12/12) = 4, so a1..a4
# already determine the form; a5 is carried for the pictures.
# https://www.lmfdb.org/ModularForm/GL2/Q/holomorphic/11/4/a/a/
LEVEL11_4 = np.array(
    [0.0, 1.0, SQRT3 + 1.0, -4.0 * SQRT3 - 1.0, 2.0 * SQRT3 - 4.0, 8.0 * SQRT3 + 1.0],
    dtype=float,
)

# E4 truncated to the same order (1 + 240 q + 2160 q^2 + ...). The relation to 11.4.a.a is a
# congruence, not an identity: 11.4.a.a is congruent to (E4(tau) - E4(11 tau))/240 modulo the
# prime (61, sqrt(3) - 8) of Z[sqrt 3], i.e. a(n) = sigma_3(n) - sigma_3(n/11) (mod p) for all n
# (audit ledger row C16b; checked to n = 200, Sturm bound 4).
E4_TRUNC = np.array([1.0, 240.0, 2160.0, 6720.0, 17520.0, 30240.0], dtype=float)


def q_of_tau(tau):
    return np.exp(2j * np.pi * np.asarray(tau, dtype=np.complex128))


def tau_of_q(q):
    q = np.asarray(q, dtype=np.complex128)
    tau = np.log(q) / (2j * np.pi)
    return (tau.real % 1.0) + 1j * tau.imag


def dna_from_neurons(taus, lead=1.0):
    """Coefficients (ascending) of the polynomial prod (q - q_i) with roots at the given taus.

    This is a polynomial in q, not a modular form. Normalised to unit max-magnitude: scaling
    F by a constant moves no zeros and only rescales the flow speed, but it keeps high-degree
    coefficients in range.
    """
    qs = q_of_tau(np.asarray(taus, dtype=np.complex128))
    poly_desc = np.poly(qs)  # prod (q - q_i), highest power first
    c = (poly_desc[::-1] * lead).astype(np.complex128)
    return c / np.max(np.abs(c))


if __name__ == "__main__":
    import modular

    for name, c in (("E4 (trunc)", E4_TRUNC), ("11.4.a.a", LEVEL11_4)):
        z = modular.qpoly_zeros(c)
        print(f"{name}: {len(z)} zero(s) of the truncation in the q-disk")
        for t in sorted(z, key=lambda w: -w.imag):
            print(f"    tau = {t.real:+.4f} {t.imag:+.4f}i   |q| = {abs(q_of_tau(t)):.4f}")
