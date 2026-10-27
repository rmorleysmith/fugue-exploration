"""Cost of the operations: batched flows, long coefficient vectors, convolution, and the
degree-2 theta table.

WHAT IT COMPUTES — Wall-clock CPU timings (this machine, no GPU; noisy run to run) for
      A. a batched 120-step unit-speed descent on |F|² (jax.vmap + jit) versus batch size;
      B. the same for 256 inputs versus the length of the coefficient vector;
      C. composing K coefficient vectors of length 3 by iterated convolution;
      D. the degree-2 theta data of E8: building the representation-number table
         r₂(T) for norms ≤ 4, 6, 8 (a one-off cost), and the per-point cost of evaluating
         the resulting Fourier sum versus a degree-1 q-series of comparable length.
    Structural facts are asserted (output sizes, monotone table growth); timings are printed,
    not asserted, because they are not reproducible to better than a factor of a few.
WHAT THE FIGURE SHOWS — A: throughput versus batch size. B: time versus vector length, with
    the convolution cost alongside. C: the one-off table build for the three norm bounds,
    beside the per-point evaluation costs of the degree-1 series and the degree-2 sum.
WHAT IS PROVED / KNOWN — Convolution of vectors of length m and n costs O(mn) and the result
    has length m + n − 1; the representation table costs O(N_vec²) in the number of lattice
    vectors enumerated (siegel.representation_numbers); the Siegel Fourier sum, once tabled,
    is a sum of exponentials like the degree-1 series.
WHAT IS NOT CLAIMED — The June caption's "25k forward passes/sec" and "genus is the cost
    driver" (ledger F5). The audit measured a non-monotone peak near 7k/s with 2.6–7× noise
    between runs, and found that the degree-2 cost is the ONE-OFF table build, while per point
    the 161-term degree-2 sum was faster than a 199-term degree-1 series. The "GPU-trivial"
    extrapolation is not measured here and is not claimed.
HOW THIS WAS FOUND — We asked whether the machinery was affordable before building more of
    it. The picture was that the exotic part (Siegel degree 2) would be the expensive part;
    the measurement says the expense is a table built once, after which evaluation is an
    ordinary exponential sum.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, lax, vmap, jit

import siegel

jax.config.update("jax_enable_x64", True)


def _energy(coeffs, xy):
    q = jnp.exp(2j * jnp.pi * (xy[0] + 1j * xy[1]))
    f = jnp.sum(coeffs * q ** jnp.arange(coeffs.shape[0]))
    return jnp.real(f * jnp.conj(f))


_g = grad(_energy, argnums=1)


def _rollout(coeffs, x0, T=120, dt=0.02):
    def step(xy, _):
        g = _g(coeffs, xy)
        gn = g / (jnp.sqrt(g[0] ** 2 + g[1] ** 2) + 1e-4)
        n = jnp.clip(xy - dt * gn, 0.02, 0.98)
        return n, None
    xy, _ = lax.scan(step, x0, jnp.arange(T))
    return xy


def _timeit(fn, reps=3):
    fn()  # warm / compile
    t0 = time.perf_counter()
    for _ in range(reps):
        r = fn()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    return (time.perf_counter() - t0) / reps


def bench_batch(coeffs, Ns=(1, 16, 128, 1024, 4096)):
    rng = np.random.default_rng(0)
    f = jit(vmap(lambda x0: _rollout(coeffs, x0)))
    out = []
    for N in Ns:
        starts = jnp.asarray(rng.uniform([0.1, 0.2], [0.9, 0.8], size=(N, 2)))
        dt = _timeit(lambda: f(starts))
        out.append((N, dt, N / dt))
        print(f"  batch {N:5d}: {dt*1e3:8.2f} ms   -> {N/dt:10.0f} inputs/sec")
    return out


def bench_len(Ls=(4, 16, 64, 256, 1024)):
    rng = np.random.default_rng(1)
    starts = jnp.asarray(rng.uniform([0.1, 0.2], [0.9, 0.8], size=(256, 2)))
    out = []
    for L in Ls:
        coeffs = jnp.asarray((rng.standard_normal(L) + 1j * rng.standard_normal(L)) / L)
        f = jit(vmap(lambda x0: _rollout(coeffs, x0)))
        dt = _timeit(lambda: f(starts))
        out.append((L, dt))
        print(f"  vector length {L:5d}: {dt*1e3:8.2f} ms / 256 inputs")
    return out


def bench_compose(Ks=(2, 4, 8, 16, 32, 64)):
    rng = np.random.default_rng(2)
    out = []
    for K in Ks:
        vecs = [rng.standard_normal(3) + 1j * rng.standard_normal(3) for _ in range(K)]
        t0 = time.perf_counter()
        comp = vecs[0]
        for d in vecs[1:]:
            comp = np.convolve(comp, d)
        dt = (time.perf_counter() - t0)
        assert len(comp) == 2 * K + 1, "convolution length is m + n - 1 at each step"
        out.append((K, dt, len(comp)))
        print(f"  compose {K:3d} vectors: {dt*1e6:8.1f} us   (compound length {len(comp)})")
    return out


def bench_degree():
    """Degree-1 q-series versus the degree-2 Siegel sum: one-off table build and per-point cost."""
    tables = {}
    for maxn in (4, 6, 8):
        t0 = time.perf_counter()
        d = siegel.representation_numbers(maxn)
        dt = time.perf_counter() - t0
        tables[maxn] = (dt, len(d), d)
        print(f"  degree-2 table r2(T), norms <= {maxn}: {dt*1e3:8.1f} ms one-off ({len(d)} Fourier terms)")
    nterms = [tables[m][1] for m in (4, 6, 8)]
    assert nterms[0] < nterms[1] < nterms[2], "the table must grow with the norm bound"

    npts = 256
    rng = np.random.default_rng(5)
    Z11 = rng.uniform(-0.5, 0.5, npts) + 1j * rng.uniform(0.9, 1.4, npts)
    Z22 = rng.uniform(-0.5, 0.5, npts) + 1j * rng.uniform(0.9, 1.4, npts)
    Z12 = 0.2 * rng.uniform(-1, 1, npts) + 1j * 0.3 * rng.uniform(0, 1, npts)
    dct6 = tables[6][2]
    t2 = _timeit(lambda: siegel.theta2(dct6, Z11, Z12, Z22), reps=5) / npts
    n = np.arange(1, len(dct6) + 1)
    q = np.exp(2j * np.pi * Z11)
    t1 = _timeit(lambda: (q[:, None] ** n).sum(axis=1), reps=5) / npts
    print(f"  per point: degree-1 series with {len(n)} terms {t1*1e6:7.2f} us;  "
          f"degree-2 sum with {len(dct6)} terms {t2*1e6:7.2f} us  (comparable)")
    return tables, t1, t2


def run(make_fig=True):
    rng = np.random.default_rng(3)
    coeffs = jnp.asarray((rng.standard_normal(8) + 1j * rng.standard_normal(8)) / 8)
    print("A. batched forward pass (degree-1 flow, T=120):")
    batch = bench_batch(coeffs)
    print("B. forward pass versus coefficient-vector length:")
    lens = bench_len()
    print("C. composition cost versus number of vectors:")
    comp = bench_compose()
    print("D. degree-1 versus degree-2 evaluation:")
    tables, t1, t2 = bench_degree()

    peak = max(r[2] for r in batch)
    print(f"\nmeasured peak CPU throughput this run: {peak:,.0f} forward passes/sec (noisy, not asserted);")
    print("composition is microseconds; the degree-2 cost is a one-off table build, and per-point "
          "evaluation is comparable to a degree-1 series (ledger F5)")
    assert peak > 0 and all(np.isfinite(r[1]) for r in lens)

    if make_fig:
        _figure(batch, lens, comp, tables, t1, t2)
    return batch


def _figure(batch, lens, comp, tables, t1, t2):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    TEAL, GOLD, NEURON, GREEN = "#46f0d0", "#ffd166", "#ff3b6b", "#7CFC8A"

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    a = ax[0]
    Ns = [r[0] for r in batch]; tput = [r[2] for r in batch]
    a.loglog(Ns, tput, "o-", color=TEAL, lw=2, ms=7)
    a.set_xlabel("batch size (parallel inputs)")
    a.set_ylabel("throughput (forward passes / sec)")
    a.set_title("batched forward pass (CPU, jax.vmap)\none run; noisy by a factor of a few", fontsize=12)
    a.grid(True, which="both", alpha=0.15)

    b = ax[1]
    Ls = [r[0] for r in lens]; ts = [r[1] * 1e3 for r in lens]
    b.loglog(Ls, ts, "o-", color=GOLD, lw=2, ms=7, label="forward pass / 256 inputs")
    Ks = [r[0] for r in comp]; cts = [r[1] * 1e3 for r in comp]
    b.loglog(Ks, cts, "s--", color=GREEN, lw=2, ms=6, label="compose K vectors (convolve)")
    b.set_xlabel("coefficient-vector length  /  number of vectors composed")
    b.set_ylabel("time (ms)")
    b.set_title("cost versus length; composition is microseconds", fontsize=12)
    b.legend(loc="upper left", fontsize=8.5, framealpha=0.3)
    b.grid(True, which="both", alpha=0.15)

    c = ax[2]
    labels = [f"table build\nnorm≤{m}" for m in (4, 6, 8)] + ["degree-1\nper point", "degree-2\nper point"]
    vals = [tables[m][0] * 1e3 for m in (4, 6, 8)] + [t1 * 1e3, t2 * 1e3]
    cols = [NEURON] * 3 + [TEAL, GOLD]
    c.bar(range(len(vals)), vals, color=cols)
    c.set_yscale("log")
    c.set_xticks(range(len(vals))); c.set_xticklabels(labels, fontsize=9)
    c.set_ylabel("time (ms, log)")
    c.set_title("degree 2: a one-off table build;\nper-point evaluation is comparable to degree 1", fontsize=12)

    fig.suptitle("Cost on one CPU: batched flows and composition are cheap; the degree-2 theta data is a table "
                 "built once, after which evaluation is an ordinary exponential sum (timings, not claims)",
                 fontsize=12.5, y=1.0)
    out = Path(__file__).parent.parent / "figures" / "28_scaling.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
