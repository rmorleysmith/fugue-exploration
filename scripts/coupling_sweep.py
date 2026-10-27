"""The off-diagonal coupling κ of a degree-2 period matrix: it bends the paths, keeps the
targets, and does not produce chaos.

WHAT IT COMPUTES — Degree-2 state Z = [[τ₁, z],[z, τ₂]], Y = Im Z = [[y₁, w],[w, y₂]]. The
    invariant metric of ℍ₂ makes the Riemannian gradient mix the two coordinates through Y;
    with the dimensionless coupling κ defined by w² = κ y₁y₂ the velocity is
        v = −K ∇log(|F_A(τ₁)|²|F_B(τ₂)|² + ε),   K = [[y₁², κy₁y₂],[κy₁y₂, y₂²]],
    applied to the x- and y-channels, normalised to unit speed. (The June files called κ
    "gravity".) Three measurements:
      A. routing error of the two separately-trained tasks of figure 19/20 versus κ ∈ [0, 0.95];
      B. the joint trajectory (Re τ₁, Re τ₂) at κ = 0, 0.4, 0.8;
      C. the separation of two trajectories started 1e−7 apart in a periodically driven
         four-well landscape, at κ = 0 and κ = 0.85, over 12 000 steps.
WHAT THE FIGURE SHOWS — A: the routing error stays below 0.01 for every κ, small but not
    zero and not monotone. B: the paths bend with κ but end at the same joint target. C: the
    log-separation at κ = 0.85 wanders, grows to about 1e−2 and collapses again; it is
    bounded and non-monotone, not exponential. The June title said "coupling unlocks chaos";
    the panel does not show that and the audit refuted it (ledger E4).
WHAT IS PROVED / KNOWN — K is positive semidefinite iff |κ| ≤ 1 and positive definite iff
    |κ| < 1 (`Fugue.couplingMatrix_posSemidef_iff`, `Fugue.couplingMatrix_posDef_iff`; ledger
    E2/E5), and K is the inverse Siegel metric on the diagonal block, so κ = (Im z)²/(y₁y₂)
    is the normalised correlation of Im Z (ledger E2, verified numerically to 1e−15). At a
    simultaneous zero both gradients vanish and −K·0 = 0 for ANY K
    (`Fugue.equilibrium_preserved`; ledger E3): the coupling bends the path and keeps the
    destination. Along ż = −K∇E the energy descends for every gradient iff K ⪰ 0
    (`Fugue.descends_iff`). Poincaré–Bendixson forbids chaos for the κ = 0 driven flow, whose
    extended phase space is two-dimensional; κ > 0 makes it three-dimensional, which is a
    necessary condition for chaos and nothing more.
WHAT IS NOT CLAIMED — Chaos. The audit's rerun of this very rollout found no positive
    Lyapunov exponent: the separation is bounded and non-monotone, collapses below 1e−9 on
    more than a thousand of the 12 000 steps and ends near 1e−6 (ledger E4, REFUTED). The
    checks below assert exactly that. "Routing error ≈ 0 for all κ" is also too strong: it
    is 0.001–0.01 and non-monotone (ledger E6). The equilibrium statement is about a common
    zero reached only to |F| ≈ 2e−4 (ledger E5).
HOW THIS WAS FOUND — With two tasks on two coordinates (figure 20) the natural question was
    what the off-diagonal of the period matrix does; we pictured it as a gravity between two
    sectors, bending each path toward the other. The precise content is the Siegel metric's
    off-diagonal block and the positivity condition |κ| < 1. We then guessed that the extra
    dimension would make a driven flow chaotic; that guess was wrong on measurement, and the
    figure now says so.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, lax

import hsln_dyn as g1

jax.config.update("jax_enable_x64", True)

EPS = 1e-2

A_START, A_TARGET = g1.A_START, g1.A_TARGET   # (0.40,0.58) -> (0.65,0.28) on coordinate 1
B_START, B_TARGET = g1.B_START, g1.B_TARGET   # (0.60,0.58) -> (0.35,0.28) on coordinate 2


def _logE(coeffs, xy):
    return jnp.log(g1.energy(coeffs, xy) + EPS)


_glogE = grad(_logE, argnums=1)               # d log(|F|^2+eps) / d(x,y)


# --------------------------------------------------------------------------- #
#  Coupled degree-2 routing flow (unit-speed, no drive)                       #
# --------------------------------------------------------------------------- #
def coupled_rollout(cA, cB, s0, kappa, T=320, dt=0.02):
    def step(s, _):
        x1, y1, x2, y2 = s[0], s[1], s[2], s[3]
        gA = _glogE(cA, jnp.array([x1, y1]))      # [d/dx1, d/dy1]
        gB = _glogE(cB, jnp.array([x2, y2]))      # [d/dx2, d/dy2]
        off = kappa * y1 * y2
        # v = -K @ grad,  K = [[y1^2, off],[off, y2^2]],  applied to x- and y-channels
        vx1 = -(y1 ** 2 * gA[0] + off * gB[0])
        vx2 = -(off * gA[0] + y2 ** 2 * gB[0])
        vy1 = -(y1 ** 2 * gA[1] + off * gB[1])
        vy2 = -(off * gA[1] + y2 ** 2 * gB[1])
        v = jnp.array([vx1, vy1, vx2, vy2])
        v = v / (jnp.linalg.norm(v) + 1e-6)
        ns = s + dt * v
        ns = ns.at[0].set(jnp.clip(ns[0], 0.02, 0.98))
        ns = ns.at[2].set(jnp.clip(ns[2], 0.02, 0.98))
        ns = ns.at[1].set(jnp.clip(ns[1], 0.06, 0.95))
        ns = ns.at[3].set(jnp.clip(ns[3], 0.06, 0.95))
        return ns, ns
    _, tr = lax.scan(step, jnp.asarray(s0), jnp.arange(T))
    return tr


def routing_errors(cA, cB, kappa):
    s0 = jnp.array([A_START[0], A_START[1], B_START[0], B_START[1]])
    tr = coupled_rollout(cA, cB, s0, kappa)
    end = jnp.mean(tr[-30:], axis=0)
    eA = float(jnp.hypot(end[0] - A_TARGET[0], end[1] - A_TARGET[1]))
    eB = float(jnp.hypot(end[2] - B_TARGET[0], end[3] - B_TARGET[1]))
    return eA, eB, np.array(tr)


# --------------------------------------------------------------------------- #
#  Driven flow on a multi-well landscape: the sensitive-dependence test        #
#                                                                              #
#  A first-order flow needs >= 3 phase dimensions to be chaotic. Drive only x1: #
#   kappa=0  -> x1 is a forced 1-D first-order flow (2-D extended phase, x2     #
#              autonomous) => Poincare-Bendixson forbids chaos: it entrains.   #
#   kappa>0  -> x1,x2 couple, x1 forced => 3-D extended phase => chaos ALLOWED. #
#  Allowed is not observed: the measurement below finds none.                  #
# --------------------------------------------------------------------------- #
def multiwell_coeffs(xs=(0.2, 0.4, 0.6, 0.8), y0=0.42):
    zs = np.array([x + 1j * y0 for x in xs])
    c = np.poly(np.exp(2j * np.pi * zs))[::-1]          # |F| vanishes at each x
    xg = np.linspace(0, 1, 400)
    qg = np.exp(2j * np.pi * (xg + 1j * y0))
    Fg = np.array([np.sum(c * qq ** np.arange(len(c))) for qq in qg])
    c = c / np.sqrt(np.mean(np.abs(Fg) ** 2))              # normalise so |F| ~ O(1)
    return jnp.asarray(c.astype(complex))


def _gx(coeffs, x, y0):
    return _glogE(coeffs, jnp.array([x, y0]))[0]           # d log E / dx at fixed y


def driven_rollout(coeffs, x0, kappa, y0=0.42, amp=1.0, om=0.9,
                   T=12000, dt=0.005, alpha=0.3):
    def step(carry, k):
        x1, x2 = carry[0], carry[1]
        t = k * dt
        gA = _gx(coeffs, x1, y0)
        gB = _gx(coeffs, x2, y0)
        off = kappa * y0 * y0
        vx1 = -alpha * (y0 ** 2 * gA + off * gB) + amp * jnp.sin(om * t)
        vx2 = -alpha * (off * gA + y0 ** 2 * gB)
        vx1 = jnp.clip(vx1, -8.0, 8.0)
        vx2 = jnp.clip(vx2, -8.0, 8.0)
        nx1 = jnp.clip(x1 + dt * vx1, 0.02, 0.98)
        nx2 = jnp.clip(x2 + dt * vx2, 0.02, 0.98)
        ns = jnp.array([nx1, nx2])
        return ns, ns
    _, tr = lax.scan(step, jnp.asarray(x0), jnp.arange(T))
    return tr


def separation(coeffs, kappa, delta=1e-7):
    x0 = jnp.array([0.50, 0.50])
    a = driven_rollout(coeffs, x0, kappa)
    b = driven_rollout(coeffs, x0 + jnp.array([delta, 0.0]), kappa)
    return np.array(jnp.sqrt(jnp.sum((a - b) ** 2, axis=1)))


def separation_stats(sep):
    """(max, final, steps below 1e-9, median over the second half): the audit's criteria.

    Chaos would mean exponential growth from 1e-7 up to the attractor size, after which the
    separation stays there; a bounded separation that collapses below 1e-9 again is not that.
    """
    n = len(sep)
    return float(sep.max()), float(sep[-1]), int(np.sum(sep < 1e-9)), float(np.median(sep[n // 2:]))


# --------------------------------------------------------------------------- #
def run(make_fig=True):
    cA, _ = g1.train_routing([A_START], [A_TARGET], seed=1)
    cB, _ = g1.train_routing([B_START], [B_TARGET], seed=2)
    cA, cB = jnp.asarray(cA), jnp.asarray(cB)

    kappas = np.linspace(0.0, 0.95, 13)
    errsA, errsB, trajs = [], [], {}
    keep = {0.0, 0.4, 0.8}
    for kap in kappas:
        eA, eB, tr = routing_errors(cA, cB, float(kap))
        errsA.append(eA); errsB.append(eB)
        if any(abs(kap - k) < 0.04 for k in keep):
            trajs[round(float(kap), 2)] = tr
    print("kappa   errA    errB")
    for kap, eA, eB in zip(kappas, errsA, errsB):
        print(f"{kap:.3f}  {eA:.3f}  {eB:.3f}")
    worst = max(max(errsA), max(errsB))
    print(f"routing error stays below {worst:.3f} for every κ in [0, 0.95] (small, not zero; ledger E6)")
    assert worst < 0.05, "a coupled rollout missed its target"
    assert np.all(np.isfinite(errsA)) and np.all(np.isfinite(errsB))

    c_mw = multiwell_coeffs()
    sep0 = separation(c_mw, 0.0)
    sep1 = separation(c_mw, 0.85)
    for kap, sep in ((0.0, sep0), (0.85, sep1)):
        mx, fin, col, med = separation_stats(sep)
        print(f"[sensitive dependence] κ={kap:<4}: max sep {mx:.2e}, final {fin:.2e}, "
              f"median (2nd half) {med:.2e}, {col} steps below 1e-9")
    mx, fin, col, med = separation_stats(sep1)
    # ledger E4: bounded, non-monotone, collapsing -- no sustained exponential divergence
    assert mx < 0.1, "separation grew to the size of the strip"
    assert fin < 1e-4, "separation did not collapse again"
    assert med < 1e-2 and col > 0, "separation stayed large: the chaos claim would need revisiting"
    print("no sustained divergence, no positive Lyapunov exponent measured: the June 'coupling unlocks "
          "chaos' claim is refuted (ledger E4)")

    if make_fig:
        _figure(kappas, errsA, errsB, trajs, sep0, sep1, col)
    return errsA, errsB


def _figure(kappas, errsA, errsB, trajs, sep0, sep1, collapses):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    TEAL, GOLD, GREEN, RED = "#46f0d0", "#ffd166", "#7CFC8A", "#ff3b6b"

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # --- A: routing error vs coupling --- #
    a = ax[0]
    a.plot(kappas, errsA, "o-", color=TEAL, lw=2, label="task A (coordinate 1)")
    a.plot(kappas, errsB, "s-", color=GOLD, lw=2, label="task B (coordinate 2)")
    a.axhline(0.05, color="#888", ls="--", lw=1, alpha=0.7, label="routing tolerance")
    a.set_ylim(0, 0.2)
    a.set_xlabel(r"coupling  $\kappa$   ($(\mathrm{Im}\,z)^2 = \kappa\, y_1 y_2$)")
    a.set_ylabel("routing error (distance to target)")
    a.set_title("routing survives the coupling:\nsmall, non-monotone error for every κ", fontsize=12)
    a.legend(loc="upper left", fontsize=9, framealpha=0.3)

    # --- B: joint trajectories bending --- #
    b = ax[1]
    cols = {0.0: GREEN, 0.4: GOLD, 0.8: RED}
    for kap in sorted(trajs):
        tr = trajs[kap]
        col = cols.get(kap, "#aaa")
        b.plot(tr[:, 0], tr[:, 2], color=col, lw=2.0, label=fr"$\kappa$={kap:.1f}")
        b.plot(tr[0, 0], tr[0, 2], "o", color=col, ms=6, mec="white", mew=0.6)
    b.plot(A_TARGET[0], B_TARGET[0], "*", color="white", ms=22, mec=GREEN, mew=1.2,
           label="joint target", zorder=6)
    b.set_xlabel(r"$\mathrm{Re}\,\tau_1$  (task A coordinate)")
    b.set_ylabel(r"$\mathrm{Re}\,\tau_2$  (task B coordinate)")
    b.set_title("the coupling bends the route\n(a common zero stays an equilibrium for every K)", fontsize=12)
    b.legend(loc="lower left", fontsize=8.5, framealpha=0.3)

    # --- C: separation of nearby trajectories --- #
    c = ax[2]
    t0 = np.arange(len(sep0)) * 0.005
    c.plot(t0, np.log10(sep0 + 1e-18), color=GREEN, lw=1.6, label=r"$\kappa$=0  (decoupled)")
    c.plot(t0, np.log10(sep1 + 1e-18), color=RED, lw=1.6, label=r"$\kappa$=0.85 (coupled)")
    c.set_xlabel("time  (driven flow)")
    c.set_ylabel(r"$\log_{10}$ trajectory separation")
    c.set_title("no chaos found: the separation stays bounded and\n"
                f"collapses below 1e-9 again on {collapses} of 12 000 steps", fontsize=12)
    c.legend(loc="center left", fontsize=9, framealpha=0.3)

    fig.suptitle("The off-diagonal coupling κ of the degree-2 period matrix bends the paths and keeps the targets;\n"
                 "under a periodic drive it does NOT produce sensitive dependence (the June chaos claim is refuted, ledger E4)",
                 fontsize=12.5, y=1.06)
    out = Path(__file__).parent.parent / "figures" / "22_coupling_sweep.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
