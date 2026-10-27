"""Cost benchmark: designing a coefficient vector from chosen zeros, recovering them, and the
degree-2 table build.

WHAT IT COMPUTES — Wall-clock timings on this CPU for
      1. building a coefficient vector with d prescribed zeros (numpy.poly) and recovering
         them by root-finding (numpy.roots, O(d³)), for d = 10, 100, 1000;
      2. evaluating E₄ on a 1500² grid by fundamental-domain reduction and domain-colouring
         the result (the expensive rendering step the figure scripts avoid);
      3. the degree-2 representation table r₂(T) of E8 for norms ≤ 6 and ≤ 8 (siegel.py),
         the one quadratic cost, O(N_vec²).
    Structural facts are asserted (the 10 prescribed zeros are recovered to 1e−8; the table
    has 76 and 161 terms, including r₂([[2,0],[0,2]]) = 240·126); timings are printed, not
    asserted. For d = 100 and 1000 the recovery error is printed: root-finding from
    coefficients is ill-conditioned at that degree (Wilkinson 1963), which is a fact about
    the coefficient representation, not a timing.
WHAT THE FIGURE SHOWS — No figure: this script prints a table. (The June file had none.)
WHAT IS PROVED / KNOWN — A monic polynomial of degree d with d prescribed roots is
    ∏(q − q_i) (fundamental theorem of algebra); companion-matrix root-finding costs O(d³);
    each E8 root is orthogonal to exactly 126 others, so r₂([[2,0],[0,2]]) = 240·126
    (`Fugue.reprCount_e8` gives the 240 and the norm counts, ledger A1/A2).
WHAT IS NOT CLAIMED — Any statement about GPUs or about the June "neuron count vs cost"
    framing: the roots of a truncated polynomial are not the zeros of a modular form
    (THE-STORY §3), so "d zeros" here means d prescribed roots of a degree-d polynomial. The
    June section timing the path-integration renderer is omitted because that renderer is
    not part of this family.
HOW THIS WAS FOUND — We wanted to know what was cheap before drawing more pictures. The
    picture was that the exotic step (degree 2) would dominate; the measurement says it is a
    table built once.
"""
from __future__ import annotations

import time

import numpy as np

import coloring
import dna
import e8
import geom
import modular
import siegel


def _timeit(f, *a, n=1, **k):
    f(*a, **k)  # warm up
    t0 = time.perf_counter()
    for _ in range(n):
        r = f(*a, **k)
    return (time.perf_counter() - t0) / n, r


def zero_set(d):
    """d distinct zeros inside the q-disk (Im in [0.2, 0.7])."""
    rng = np.random.default_rng(d)
    re = rng.uniform(0.02, 0.98, d)
    im = rng.uniform(0.20, 0.70, d)
    return re + 1j * im


def run():
    print("=" * 64)
    print("1) coefficient vector with d prescribed zeros, then recovering them (np.roots ~ O(d^3)):")
    for d in (10, 100, 1000):
        taus = zero_set(d)
        c = dna.dna_from_neurons(taus)
        dt_find, z = _timeit(modular.qpoly_zeros, c, n=3)
        dt_make, _ = _timeit(dna.dna_from_neurons, taus, n=3)
        err = max(np.min(np.abs(z - t)) for t in taus) if len(z) else np.inf
        print(f"   d={d:5d}:  build vector {dt_make*1e3:7.2f} ms   recover {len(z):5d} zeros {dt_find*1e3:8.2f} ms"
              f"   max recovery error {err:.1e}")
        if d == 10:
            assert len(z) == d and err < 1e-8, "the 10 prescribed zeros were not recovered"
    print("   (recovery by np.roots is ill-conditioned for large d: the error above is real, not a bug)")

    print("\n2) the expensive rendering step the figure scripts avoid (dense grid + colouring):")
    TAU, _ = geom.half_plane_grid((-2, 2), (0.05, 2.3), 1500)
    dt_e4, F = _timeit(modular.e4, TAU, n=1)
    dt_col, img = _timeit(coloring.domain_color, F, n=1)
    assert img.shape == (1500, 1500, 3)
    print(f"   E4 on 1500^2 grid: {dt_e4*1e3:6.0f} ms   +  domain_color: {dt_col*1e3:5.0f} ms")

    print("\n3) degree-2 representation table of E8 (the one quadratic cost, O(N_vec^2)):")
    expect_terms = {6: 76, 8: 161}
    for m in (6, 8):
        nvec = e8.e8_vectors(float(m)).shape[0]
        dt_rep, dct = _timeit(siegel.representation_numbers, m, n=1)
        print(f"   norm<= {m}:  {nvec:6d} E8 vectors  ->  {len(dct):4d} Fourier terms  in {dt_rep:6.2f} s")
        assert len(dct) == expect_terms[m]
        assert dct[(2, 0, 2)] == 240 * 126, "r2([[2,0],[0,2]]) must be 240*126"
    print("(one-off table; evaluating the degree-2 sum on a grid afterwards is an exponential sum)")
    print("=" * 64)


if __name__ == "__main__":
    run()
