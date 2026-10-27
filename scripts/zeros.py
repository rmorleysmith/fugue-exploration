"""Cached ordinates gamma_n of the nontrivial zeros rho_n = 1/2 + i gamma_n of the zeta function.

WHAT IT COMPUTES — the first N ordinates with `mpmath.zetazero` at `dps` digits, saved to
    `zeta_zeros_<N>.npy` beside this file and loaded from there afterwards.
WHAT IT PROVIDES — `get_zeros(N=500, dps=25)` -> float array of gamma_1 .. gamma_N.
WHAT IS PROVED / KNOWN — `zetazero` locates the n-th zero by Gram-point separation with a certified
    count (Titchmarsh, *The Theory of the Riemann Zeta-Function*, ch. IX); the first 10^13 zeros
    are known to lie on Re s = 1/2 (Gourdon 2004).
WHAT IS NOT CLAIMED — nothing about zeros beyond the computed range, and nothing about the
    Riemann hypothesis.
HOW THIS WAS FOUND — the spectral and explicit-formula figures each needed the same list, so it
    was computed once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def get_zeros(N: int = 500, dps: int = 25) -> np.ndarray:
    path = Path(__file__).parent / f"zeta_zeros_{N}.npy"
    if path.exists():
        return np.load(path)
    import mpmath as mp
    mp.mp.dps = dps
    g = np.array([float(mp.zetazero(n).imag) for n in range(1, N + 1)])
    np.save(path, g)
    return g
