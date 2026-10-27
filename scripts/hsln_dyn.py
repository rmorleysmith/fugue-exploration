"""Composition in the dynamical regime: the zeros union, but the basins do not.

WHAT IT COMPUTES — Two coefficient vectors a_A, a_B ∈ ℂ^4 (the Fourier coefficients, which
    the June files called the DNA) are trained separately so that the unit-speed descent
        τ̇ = −∇|F|² / |∇|F|²|,   220 steps of 0.02,
    routes one input to one target: A sends (0.40, 0.58) → (0.65, 0.28) rightward, B sends
    (0.60, 0.58) → (0.35, 0.28) leftward, a crossing geometry. The compound is the product
    F_A·F_B (Cauchy product of the vectors); both inputs are then released on the compound
    and their endpoints compared with the parents' targets.
WHAT THE FIGURE SHOWS — Left and middle: each parent alone routes its input to its target.
    Right: on the product field the zeros of both parents are present, but the paths are
    reshaped by the other factor and an input can be captured by the wrong attractor.
WHAT IS PROVED / KNOWN — zeros(F_A F_B) = zeros(F_A) ∪ zeros(F_B) (`mul_eq_zero`, ledger B1),
    and the gradient of the product energy is
        ∇|F_A F_B|² = |F_B|² ∇|F_A|² + |F_A|² ∇|F_B|²,
    a weighted sum of the parent flows (product rule), not their union. Zeros remain the
    minima (`Fugue.isLocalMin_norm_of_eq_zero`, ledger A9); the basins of attraction are not
    preserved by multiplication, and no theorem says they should be.
WHAT IS NOT CLAIMED — That the merge law transfers input→output behaviour. It unions the
    attractor sets (THE-STORY §1(c)); whether a given input still reaches the parent's target
    is empirical, and in this geometry it fails for at least one input. The figure is one
    seed; the extent of cross-talk depends on the geometry chosen.
HOW THIS WAS FOUND — After the spike readout merged cleanly (figure 18) we asked whether the
    flow merges too. The physical picture made us doubt it: multiplying two landscapes
    multiplies their heights, so a deep valley of A flattens B's slopes nearby. We wrote the
    product rule down and it says exactly that. The precise statement is the gradient
    identity above; the remedy is to give each task its own coordinate (figure 20).
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, lax, value_and_grad

import modular

jax.config.update("jax_enable_x64", True)

Y0 = 0.58            # inputs start here
YT = 0.28            # targets sit here
T, DT = 220, 0.02


def energy(coeffs, xy):
    q = jnp.exp(2j * jnp.pi * (xy[0] + 1j * xy[1]))
    f = jnp.sum(coeffs * q ** jnp.arange(coeffs.shape[0]))
    return jnp.real(f * jnp.conj(f))


_genergy = grad(energy, argnums=1)


def rollout(coeffs, start, alpha=1.0):
    def step(xy, _):
        g = _genergy(coeffs, xy)
        gn = g / (jnp.sqrt(g[0] ** 2 + g[1] ** 2) + 1e-4)
        nx = jnp.clip(xy[0] - DT * alpha * gn[0], 0.02, 0.98)
        ny = jnp.clip(xy[1] - DT * alpha * gn[1], 0.05, 0.95)
        new = jnp.array([nx, ny])
        return new, new
    _, traj = lax.scan(step, jnp.asarray(start), jnp.arange(T))
    return traj


def _central_init(seed, dg):
    rng = np.random.default_rng(seed)
    zs = np.array([0.5 + (YT + 0.05 * k) * 1j for k in range(dg - 1)])
    c = np.poly(np.exp(2j * np.pi * zs))[::-1]
    c = c / np.max(np.abs(c))
    return jnp.array(np.concatenate([c.real, c.imag]) + 0.04 * rng.standard_normal(2 * dg))


def train_routing(starts, targets, dg=4, steps=1400, lr=0.01, seed=0):
    """Train a coefficient vector so each start settles at its target (mean of the last 30 steps)."""
    starts = jnp.asarray(starts, float)
    targets = jnp.asarray(targets, float)

    def to_coeffs(p):
        return p[:dg] + 1j * p[dg:]

    def loss(p):
        c = to_coeffs(p)
        L = 0.0
        for i in range(starts.shape[0]):
            tr = rollout(c, starts[i])
            end = jnp.mean(tr[-30:], axis=0)           # where it settles
            L = L + jnp.sum((end - targets[i]) ** 2)
        return L + 1e-4 * jnp.sum(p ** 2)

    lg = jax.jit(value_and_grad(loss))
    p = _central_init(seed, dg)
    m = jnp.zeros_like(p); v = jnp.zeros_like(p)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for t in range(1, steps + 1):
        L, g = lg(p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        p = p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
        hist.append(float(L))
    return np.array(to_coeffs(p)), hist


def endpoint(coeffs, start):
    return np.array(jnp.mean(rollout(jnp.asarray(coeffs), start)[-30:], axis=0))


# crossing geometry: A sends 0.40 -> 0.65 (rightward); B sends 0.60 -> 0.35 (leftward).
A_START, A_TARGET = (0.40, Y0), (0.65, YT)
B_START, B_TARGET = (0.60, Y0), (0.35, YT)


def near(e, t):
    return float(np.hypot(e[0] - t[0], e[1] - t[1]))


def run(make_fig=True):
    cA, hA = train_routing([A_START], [A_TARGET], seed=1)
    cB, hB = train_routing([B_START], [B_TARGET], seed=2)
    cAB = np.convolve(cA, cB)

    eAA = endpoint(cA, A_START)
    eBB = endpoint(cB, B_START)
    eABa = endpoint(cAB, A_START)
    eABb = endpoint(cAB, B_START)

    print(f"parent A: input {A_START[0]} -> {np.round(eAA,3)}  (target {A_TARGET})  err {near(eAA,A_TARGET):.3f}")
    print(f"parent B: input {B_START[0]} -> {np.round(eBB,3)}  (target {B_TARGET})  err {near(eBB,B_TARGET):.3f}")
    assert near(eAA, A_TARGET) < 0.05 and near(eBB, B_TARGET) < 0.05, "a parent failed to route"

    # the zero sets union exactly (the algebra), whatever the flow does
    zAB = modular.qpoly_zeros(cAB)
    parents = np.concatenate([modular.qpoly_zeros(cA), modular.qpoly_zeros(cB)])
    res = max(np.min(np.abs(zAB - z)) for z in parents)
    print(f"zeros of the product = union of the parents' zeros to {res:.1e}")
    assert res < 1e-9

    print("--- compound F_A · F_B ---")
    print(f"  A-input {A_START[0]} -> {np.round(eABa,3)}   wants {A_TARGET}  "
          f"(to A:{near(eABa,A_TARGET):.2f}  to B:{near(eABa,B_TARGET):.2f})")
    print(f"  B-input {B_START[0]} -> {np.round(eABb,3)}   wants {B_TARGET}  "
          f"(to A:{near(eABb,A_TARGET):.2f}  to B:{near(eABb,B_TARGET):.2f})")
    okA = near(eABa, A_TARGET) < near(eABa, B_TARGET)
    okB = near(eABb, B_TARGET) < near(eABb, A_TARGET)
    print(f"compound routes A-input correctly: {okA};  B-input correctly: {okB}")
    assert not (okA and okB), "expected cross-talk in the crossing geometry; the figure's claim would be wrong"
    print("cross-talk observed: the product of the flows is not the union of the flows (measured)")

    if make_fig:
        _figure(cA, cB, cAB, okA, okB)
    return cA, cB, cAB


def _figure(cA, cB, cAB, okA, okB):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    colA, colB = "#46f0d0", "#ffd166"

    def attractors(c):
        z = modular.qpoly_zeros(np.asarray(c))
        return z[z.imag < 0.7]  # the zeros inside the plotted window

    verdict = {True: "✓", False: "✗"}
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
    panels = [
        (cA, [(A_START, A_TARGET, colA)], "A alone:  0.40 → 0.65  ✓"),
        (cB, [(B_START, B_TARGET, colB)], "B alone:  0.60 → 0.35  ✓"),
        (cAB, [(A_START, A_TARGET, colA), (B_START, B_TARGET, colB)],
         f"F_A·F_B:  A {verdict[okA]}   B {verdict[okB]}   (✗ = caught by the other basin)"),
    ]
    for a, (c, runs, title) in zip(ax, panels):
        a.set_facecolor("#0b0b14")
        za = attractors(c)
        a.plot(za.real, za.imag, "o", color="#888", ms=9, mfc="none", mew=2,
               label="attractors (zeros)", zorder=4)
        for start, target, col in runs:
            tr = np.array(rollout(jnp.asarray(c), start))
            a.plot(tr[:, 0], tr[:, 1], color=col, lw=2.0, zorder=5)
            a.plot(start[0], start[1], "o", color=col, mec="white", mew=0.8, ms=8, zorder=6)
            a.plot(target[0], target[1], "*", color=col, ms=20, mec="white", mew=0.9, zorder=6)
        a.set_xlim(0, 1); a.set_ylim(0.0, 0.7); a.set_xlabel(r"Re $\tau$")
        a.set_title(title, fontsize=12)
        a.legend(loc="upper right", fontsize=8, framealpha=0.3)
    ax[0].set_ylabel(r"Im $\tau$")
    ax[2].text(0.03, 0.03, "● input   ★ trained target", color="white", fontsize=8)

    fig.suptitle("Composition in the dynamical regime: the product keeps every zero but reshapes the basins\n"
                 r"$\nabla|F_AF_B|^2 = |F_B|^2\nabla|F_A|^2 + |F_A|^2\nabla|F_B|^2$ is a weighted sum of the flows, "
                 "not their union — routing can cross-talk (measured)", fontsize=12.5, y=1.03)
    out = Path(__file__).parent.parent / "figures" / "19_dynamical_composition.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
