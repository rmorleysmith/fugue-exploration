"""Give each task its own coordinate: the block-diagonal degree-2 state removes the cross-talk.

WHAT IT COMPUTES — The same two routing coefficient vectors as hsln_dyn.py (figure 19),
    trained separately. Instead of multiplying them on one coordinate τ, the state is the
    2×2 period matrix on the diagonal slice of the Siegel upper half-space of degree 2,
        Z = diag(τ₁, τ₂) ∈ ℍ₂,   F(Z) = F_A(τ₁)·F_B(τ₂),
    so the energy factorises, |F|² = |F_A(τ₁)|²·|F_B(τ₂)|², and the gradient in each
    coordinate is a positive multiple of the parent's gradient. With the per-coordinate
    unit-speed step used throughout, the two coordinates follow the parents' trajectories
    exactly. We compare the routing error of this with the shared-coordinate product.
WHAT THE FIGURE SHOWS — Left: the joint trajectory (Re τ₁, Re τ₂) runs from the joint input
    to the joint target, while the shared-coordinate product (cross) lands elsewhere. Right:
    routing error for each task, shared coordinate versus separate coordinates.
WHAT IS PROVED / KNOWN — Theta series of a direct sum of lattices multiply at every degree:
    Θ_{L⊕M} = Θ_L·Θ_M (`Fugue.reprCount_directSum`, `Fugue.theta_directSum`, ledger C13), and
    a block-diagonal Gram matrix is what "direct sum" means (`Fugue.e8e8_det`, `e8e8_symm`
    in DirectSumGenus.lean). On the diagonal slice a product of functions of separate
    variables has gradient (|F_B|²∇₁|F_A|², |F_A|²∇₂|F_B|²) (product rule), so the directions
    are the parents' and a zero of either factor is an equilibrium (`Fugue.equilibrium_preserved`).
WHAT IS NOT CLAIMED — The June title called this "genus 2". In standard vocabulary
    the 2×2 period matrix is Siegel DEGREE 2, and a direct sum of lattices does not raise it;
    what raises the dimension of the state is choosing the degree-2 state space. Nor is this
    a theta series of a lattice: F_A and F_B are free polynomials. "Decoupled" is a statement
    about the diagonal slice only; the off-diagonal entry couples the coordinates
    (coupling_sweep.py, figure 22). The decoupled result is exact by construction and the
    figure is a consistency check, not a discovery.
HOW THIS WAS FOUND — Figure 19 showed two tasks fighting over one coordinate. The physical
    picture was two balls in two separate bowls instead of one ball in a product bowl; the
    algebraic picture was the direct sum of lattices, whose theta series is the product of
    the two theta series evaluated on a block-diagonal period matrix. The precise statement
    is C13 plus the product rule; the open part is what the off-diagonal does.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

import hsln_dyn as g1

A_TARGET = g1.A_TARGET   # (0.65, 0.28), reached in tau1
B_TARGET = g1.B_TARGET   # (0.35, 0.28), reached in tau2
A_IN = g1.A_START        # (0.40, 0.58)
B_IN = g1.B_START        # (0.60, 0.58)


def _err(e, t):
    return float(np.hypot(e[0] - t[0], e[1] - t[1]))


def run(make_fig=True):
    # the same separately-trained vectors that cross-talked on one coordinate
    cA, _ = g1.train_routing([A_IN], [A_TARGET], seed=1)
    cB, _ = g1.train_routing([B_IN], [B_TARGET], seed=2)

    # shared coordinate: the product on one τ -- the cross-talking baseline
    cAB = np.convolve(cA, cB)
    g1_A = g1.endpoint(cAB, A_IN)
    g1_B = g1.endpoint(cAB, B_IN)

    # degree-2 diagonal slice: τ₁ under A, τ₂ under B, each a positive rescaling of its parent flow
    tau1 = np.array(g1.rollout(jnp.asarray(cA), jnp.asarray(A_IN)))   # task A on coordinate 1
    tau2 = np.array(g1.rollout(jnp.asarray(cB), jnp.asarray(B_IN)))   # task B on coordinate 2
    g2_A = tau1[-30:].mean(axis=0)
    g2_B = tau2[-30:].mean(axis=0)

    e1 = (_err(g1_A, A_TARGET), _err(g1_B, B_TARGET))
    e2 = (_err(g2_A, A_TARGET), _err(g2_B, B_TARGET))
    print("                                      task A err   task B err")
    print(f"shared coordinate, product F_A F_B :   {e1[0]:.3f}        {e1[1]:.3f}")
    print(f"diag(τ₁, τ₂), F_A(τ₁) F_B(τ₂)      :   {e2[0]:.3f}        {e2[1]:.3f}")
    assert max(e2) < 0.05, "separate coordinates did not route both tasks"
    assert max(e1) > max(e2), "expected the shared-coordinate product to do worse"
    # the energy factorises on the diagonal slice: check it at the joint endpoint
    z = (g2_A[0] + 1j * g2_A[1], g2_B[0] + 1j * g2_B[1])
    import modular
    prod = abs(modular.qpoly(cA, z[0])) ** 2 * abs(modular.qpoly(cB, z[1])) ** 2
    joint = abs(modular.qpoly(cA, z[0]) * modular.qpoly(cB, z[1])) ** 2
    assert abs(prod - joint) < 1e-12 * (1 + joint)
    print("|F_A(τ₁)F_B(τ₂)|² = |F_A(τ₁)|²|F_B(τ₂)|² at the endpoint (exact factorisation)")

    if make_fig:
        _figure(tau1, tau2, g1_A, g1_B, g2_A, g2_B)
    return cA, cB


def _figure(tau1, tau2, g1_A, g1_B, g2_A, g2_B):
    import matplotlib.pyplot as plt
    import style

    cA, cB, good = "#46f0d0", "#ffd166", "#7CFC8A"

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})

    # --- joint state-space: x = task A coordinate (Re tau1), y = task B coordinate (Re tau2)
    a = ax[0]
    a.set_facecolor("#0b0b14")
    a.plot(tau1[:, 0], tau2[:, 0], color=good, lw=2.4, zorder=5, label="trajectory on diag(τ₁, τ₂)")
    a.plot(A_IN[0], B_IN[0], "o", color="white", ms=10, zorder=6, label="input (0.40, 0.60)")
    a.plot(A_TARGET[0], B_TARGET[0], "*", color=good, ms=24, mec="white", mew=1.0,
           zorder=6, label="joint target (0.65, 0.35)")
    a.plot(g1_A[0], g1_B[0], "X", color=style.NEURON, ms=14, mew=2, zorder=6,
           label="product on one shared τ (cross-talk)")
    a.axvline(A_TARGET[0], color=cA, ls=":", lw=1, alpha=0.6)
    a.axhline(B_TARGET[0], color=cB, ls=":", lw=1, alpha=0.6)
    a.set_xlim(0.1, 0.9); a.set_ylim(0.1, 0.9)
    a.set_xlabel(r"task A coordinate  (Re $\tau_1$)")
    a.set_ylabel(r"task B coordinate  (Re $\tau_2$)")
    a.set_title(r"degree-2 diagonal slice: $Z=\mathrm{diag}(\tau_1,\tau_2)$" "\n"
                "each task on its own coordinate", fontsize=12)
    a.legend(loc="upper left", fontsize=8, framealpha=0.3)

    b = ax[1]
    x = np.arange(2)
    g1errs = [_err(g1_A, A_TARGET), _err(g1_B, B_TARGET)]
    g2errs = [_err(g2_A, A_TARGET), _err(g2_B, B_TARGET)]
    b.bar(x - 0.18, g1errs, 0.34, color=style.NEURON, label="product on one shared τ")
    b.bar(x + 0.18, g2errs, 0.34, color=good, label="F_A(τ₁)·F_B(τ₂), separate coordinates")
    for xi, (v1, v2) in enumerate(zip(g1errs, g2errs)):
        b.text(xi - 0.18, v1 + 0.005, f"{v1:.3f}", color="white", ha="center", fontsize=9)
        b.text(xi + 0.18, v2 + 0.005, f"{v2:.3f}", color=good, ha="center", fontsize=9)
    b.set_xticks(x); b.set_xticklabels(["task A", "task B"])
    b.set_ylabel("routing error  (distance to trained target)")
    b.set_title("same trained vectors, two ways to combine", fontsize=12)
    b.legend(fontsize=9, framealpha=0.3)

    fig.suptitle("Separate coordinates remove the cross-talk: on the diagonal slice the energy factorises, "
                 "so each coordinate follows its parent's flow (exact by construction)",
                 fontsize=12.5, y=1.0)
    out = Path(__file__).parent.parent / "figures" / "20_genus2_directsum.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
