"""Layering by multiplication is a reparameterisation of a monic quartic, not added capacity.

WHAT IT COMPUTES — Four target points in the τ-strip. Three parameterisations of a truncated
    form are trained by Adam to put zeros on them (loss Σ_targets |F(τ_k)|²):
      (i)  one monic quadratic in q (4 real parameters, ≤ 2 zeros),
      (ii) the product of two monic quadratics (8 real parameters, ≤ 4 zeros; the June
           "two sectors tensored"),
      (iii) one monic quartic directly (8 real parameters, ≤ 4 zeros).
    The residual |F(τ_k)| at each target measures whether a zero was placed there.
WHAT THE FIGURE SHOWS — Left: the single quadratic reaches at most two of the four targets.
    Middle: the product of two quadratics reaches all four. Right: residuals at the four
    targets for all three parameterisations; the direct quartic does as well as the product.
WHAT IS PROVED / KNOWN — Over ℂ every monic quartic is a product of two monic quadratics
    (fundamental theorem of algebra), so (ii) and (iii) parameterise the same set of forms;
    zeros(F·G) = zeros(F) ∪ zeros(G) (`mul_eq_zero`, ledger B1). Hence the product has up to
    four zeros because it has degree four, not because it was "layered" (ledger F4).
WHAT IS NOT CLAIMED — The June caption read this as "layering doubles the neurons" and as
    capacity gained by composition. It is not: the map (quadratic, quadratic) → quartic is
    onto, so training the product is training a quartic with a different coordinate system
    (ledger F4; THE-STORY §3 on truncation roots). The two factors are trained JOINTLY, so this
    is not the separate-training merge test of system-spec §4.4(6) either (ledger F9).
HOW THIS WAS FOUND — We imagined stacking two small models as one stacks layers, and saw the
    zero count double. The precise statement is only that degree adds under multiplication;
    the honest comparison, added here, is against a single form of the same degree, which
    does the same job.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad

import modular

jax.config.update("jax_enable_x64", True)

TARGETS = np.array([0.18 + 0.12j, 0.42 + 0.17j, 0.62 + 0.14j, 0.83 + 0.10j])
_TX = jnp.array([[t.real, t.imag] for t in TARGETS])


def sector(free):
    """Monic degree-2 form from 4 reals: q^2 + a1 q + a0  (<= 2 zeros)."""
    a0 = free[0] + 1j * free[2]
    a1 = free[1] + 1j * free[3]
    return jnp.array([a0, a1, 1.0 + 0j])


def coeffs_single(params):
    return sector(params)


def coeffs_layered(params):
    return jnp.convolve(sector(params[:4]), sector(params[4:]))  # product -> degree 4


def coeffs_quartic(params):
    """Monic degree-4 form from 8 reals: the same family as the product, parameterised directly."""
    a = params[:4] + 1j * params[4:]
    return jnp.concatenate([a, jnp.array([1.0 + 0j])])


def Fmag2(coeffs, xy):
    q = jnp.exp(2j * jnp.pi * (xy[0] + 1j * xy[1]))
    f = jnp.sum(coeffs * q ** jnp.arange(coeffs.shape[0]))
    return jnp.real(f * jnp.conj(f))


def make_loss(coeff_fn):
    def loss(params):
        c = coeff_fn(params)
        return jnp.sum(jax.vmap(lambda t: Fmag2(c, t))(_TX)) + 1e-4 * jnp.sum(params ** 2)
    return loss


def train(coeff_fn, nparams, steps=800, lr=0.02, seed=0):
    rng = np.random.default_rng(seed)
    p = jnp.array(0.3 * rng.standard_normal(nparams))
    lg = jax.jit(value_and_grad(make_loss(coeff_fn)))
    m = jnp.zeros_like(p); v = jnp.zeros_like(p)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for t in range(1, steps + 1):
        L, g = lg(p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        p = p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
        hist.append(float(L))
    return p, hist


def residuals(coeff_fn, p):
    c = np.array(coeff_fn(p))
    return np.array([float(Fmag2(jnp.array(c), t)) ** 0.5 for t in _TX])


def run(make_fig=True, seed=0):
    ps, _ = train(coeffs_single, 4, seed=seed)
    pl, _ = train(coeffs_layered, 8, seed=seed)
    pq, _ = train(coeffs_quartic, 8, seed=seed)
    rs = residuals(coeffs_single, ps)
    rl = residuals(coeffs_layered, pl)
    rq = residuals(coeffs_quartic, pq)
    tol = 0.05
    print(f"one quadratic (4 params)      residuals {np.array2string(rs, precision=2)}  covers {(rs < tol).sum()}/4")
    print(f"product of two quadratics (8) residuals {np.array2string(rl, precision=2)}  covers {(rl < tol).sum()}/4")
    print(f"one quartic directly (8)      residuals {np.array2string(rq, precision=2)}  covers {(rq < tol).sum()}/4")
    assert (rs < tol).sum() <= 2, "a quadratic cannot have more than two zeros"
    assert (rl < tol).sum() == 4, "the product of two quadratics did not reach all four targets"
    assert (rq < tol).sum() == 4, "the direct quartic did not reach all four targets"
    print("the product gains zeros by degree, not by layering: the direct quartic does the same "
          "(ledger F4)")

    if make_fig:
        _figure(ps, pl, rs, rl, rq)
    return rs, rl, rq


def _figure(ps, pl, rs, rl, rq):
    import matplotlib.pyplot as plt
    import style

    zs = modular.qpoly_zeros(np.array(coeffs_single(ps)))
    zl = modular.qpoly_zeros(np.array(coeffs_layered(pl)))

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.0))
    for a, (z, title) in zip(ax[:2], [(zs, "one quadratic  (≤ 2 zeros)"),
                                      (zl, "product of two quadratics  (≤ 4 zeros)")]):
        a.set_facecolor("#0b0b14")
        a.plot(TARGETS.real, TARGETS.imag, "*", color=style.ACCENT, ms=22, mec="white",
               mew=0.9, zorder=4, label="targets")
        a.plot(z.real, z.imag, "o", color="#46f0d0", ms=12, mfc="none", mew=2.2,
               zorder=5, label="learned zeros")
        a.set_xlim(0, 1); a.set_ylim(0.02, 0.35)
        a.set_xlabel(r"Re $\tau$"); a.set_title(f"{title}: {len(z)} zeros", fontsize=12)
        a.legend(loc="upper right", fontsize=8, framealpha=0.3)
    ax[0].set_ylabel(r"Im $\tau$")

    x = np.arange(4)
    ax[2].bar(x - 0.26, rs, 0.24, color="#ff3b6b", label="one quadratic (4 params)")
    ax[2].bar(x, rl, 0.24, color="#46f0d0", label="two quadratics multiplied (8)")
    ax[2].bar(x + 0.26, rq, 0.24, color="#c0a3ff", label="one quartic directly (8)")
    ax[2].set_xticks(x); ax[2].set_xticklabels([f"T{i+1}" for i in range(4)])
    ax[2].set_yscale("log"); ax[2].set_ylim(1e-6, 1.0)
    ax[2].set_ylabel(r"$|F|$ at target  (log; small = zero placed)")
    ax[2].set_title("coverage of the 4 targets:\nthe product and the direct quartic agree", fontsize=12)
    ax[2].legend(fontsize=8.5, framealpha=0.3)

    fig.suptitle("Multiplying two quadratics gives four zeros because the degree is four: a direct quartic with "
                 "the same 8 parameters\ncovers the targets equally well (zeros union, coefficient vectors "
                 "convolve; no capacity is gained by layering)", fontsize=12, y=1.06)
    out = Path(__file__).parent.parent / "figures" / "17_poc_layering.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
