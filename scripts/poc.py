"""A minimal trainable reservoir: backpropagate through the gradient flow on |F|².

WHAT IT COMPUTES — F(τ) = Σ_{n<6} a_n q^n, q = e^{2πiτ}, with a learnable coefficient vector
    a ∈ ℂ^6 (the Fourier coefficients, which the June files called the DNA). The state
    τ = x + iy obeys the driven, unit-speed descent
        τ̇ = −α ∇|F|² / |∇|F|²|  +  B u,     B = (0.35, 0),  u = ±1,
    integrated for 160 Euler steps of 0.02 from one shared start. The loss is the closest
    approach of each path to its class target (a minimum over the trajectory) plus a small
    weight decay; Adam updates a through the whole rollout (JAX reverse-mode autodiff).
WHAT THE FIGURE SHOWS — Left: before training, both paths and the zeros of F. Middle: after
    training, the two paths diverge under the opposite drives and each passes its class
    target; the point of closest approach is marked. Right: the training loss (log scale).
WHAT IS PROVED / KNOWN — Zeros of a holomorphic F are exactly the local minima of |F|
    (minimum-modulus principle; `Fugue.isLocalMin_norm_of_eq_zero`,
    `Fugue.eq_zero_of_isLocalMin_norm`, ledger A9), so the unforced flow has the zero set as
    its attractors (THE-STORY §0). Around a simple zero the implicit function theorem makes
    the zero a smooth function of a (system-spec §2.5), which is why gradient descent on a
    is well-defined. Backpropagation through an explicit ODE solver is standard.
WHAT IS NOT CLAIMED — The targets are not attractors of the trained field (ledger F2): the
    loss is a minimum over the path, the paths pass within a few hundredths of the targets
    and then move on, and |F| at the targets is not small. The loss figure is a measurement
    with seed-to-seed spread (ledger F1); nothing about modular forms is used.
HOW THIS WAS FOUND — We wanted the smallest thing that could be called "training the
    landscape": one holomorphic potential, one drive, two classes. The physical picture was
    a ball rolling in a bowl whose dents are the zeros, with a constant wind telling the two
    classes apart. The precise statement is only that the rollout is differentiable in a and
    the loss decreases; what the June caption over-read — that the trained zeros sit at the
    targets — is refused above.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, lax, value_and_grad

jax.config.update("jax_enable_x64", True)

DG = 6  # DG complex coefficients -> 2*DG real learnable parameters


def coeffs_of(p):
    return p[:DG] + 1j * p[DG:]


def Fval(p, x, y):
    q = jnp.exp(2j * jnp.pi * (x + 1j * y))
    return jnp.sum(coeffs_of(p) * q ** jnp.arange(DG))


def energy(p, xy):
    f = Fval(p, xy[0], xy[1])
    return jnp.real(f * jnp.conj(f))


_grad_energy = grad(energy, argnums=1)

T = 160

# --- task: a CONSTANT input drive distinguishes the classes (spec §2.3 with u = const) ---
START = jnp.array([0.50, 0.55])
B_VEC = jnp.array([0.35, 0.0])
U_A, U_B = 1.0, -1.0
TARGET_A = jnp.array([0.76, 0.30])
TARGET_B = jnp.array([0.24, 0.30])


def rollout(p, u, alpha=1.0, dt=0.02, ymin=0.05, ymax=0.95):
    def stepfn(xy, _):
        g = _grad_energy(p, xy)
        gn = g / (jnp.sqrt(g[0] ** 2 + g[1] ** 2) + 1e-4)  # unit-speed descent
        nx = jnp.clip(xy[0] - dt * alpha * gn[0] + dt * B_VEC[0] * u, 0.02, 0.98)
        ny = jnp.clip(xy[1] - dt * alpha * gn[1] + dt * B_VEC[1] * u, ymin, ymax)
        new = jnp.array([nx, ny])
        return new, new
    final, traj = lax.scan(stepfn, START, jnp.arange(T))
    return final, traj


def loss(p):
    _, ta = rollout(p, U_A)
    _, tb = rollout(p, U_B)
    dA = jnp.min(jnp.sum((ta - TARGET_A) ** 2, axis=1))   # closest approach, not endpoint
    dB = jnp.min(jnp.sum((tb - TARGET_B) ** 2, axis=1))
    return dA + dB + 1e-3 * jnp.sum(p ** 2)


_loss_and_grad = jax.jit(value_and_grad(loss))


def init_params(seed=0):
    """Seed the coefficient vector with a central column of zeros (an active field, not flat)."""
    rng = np.random.default_rng(seed)
    zs = np.array([0.50 + 0.40j, 0.50 + 0.50j, 0.50 + 0.30j])
    c = np.poly(np.exp(2j * np.pi * zs))[::-1]
    c = c / np.max(np.abs(c))
    c = np.concatenate([c, np.zeros(DG - len(c), dtype=complex)])
    p = np.concatenate([c.real, c.imag]) + 0.05 * rng.standard_normal(2 * DG)
    return jnp.array(p)


def train(steps=800, lr=0.012, seed=0):
    p = init_params(seed)
    p0 = p
    m = jnp.zeros_like(p); v = jnp.zeros_like(p)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []
    for t in range(1, steps + 1):
        L, g = _loss_and_grad(p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        p = p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
        history.append(float(L))
    return p0, p, history


def approach(p, u, target):
    """(closest distance along the path, index of that step, final distance)."""
    tr = np.array(rollout(p, u)[1])
    d = np.hypot(tr[:, 0] - float(target[0]), tr[:, 1] - float(target[1]))
    k = int(np.argmin(d))
    return float(d[k]), k, float(d[-1]), tr


def run(make_fig=True, steps=800, lr=0.012, seed=0):
    p0, p, hist = train(steps=steps, lr=lr, seed=seed)
    print(f"loss {hist[0]:.4f} -> {hist[-1]:.5f}  (closest-approach loss, seed {seed})")
    assert hist[-1] < 0.25 * hist[0], "training did not reduce the loss"

    rows = []
    for u, target, lab in [(U_A, TARGET_A, "u=+1"), (U_B, TARGET_B, "u=-1")]:
        dmin, k, dend, tr = approach(p, u, target)
        fmag = float(jnp.abs(Fval(p, target[0], target[1])))
        print(f"  {lab}: closest approach {dmin:.3f} at step {k}/{T}; final distance {dend:.3f}; "
              f"|F| at target {fmag:.2f}  -> the target is passed, not settled on")
        rows.append((dmin, k, dend, tr))
    assert all(r[0] < 0.1 for r in rows), "a path did not pass within 0.1 of its target"
    # the honest statement of ledger F2: the targets are not where the paths end
    print("  the loss is a minimum over the path; the targets are not attractors (ledger F2)")

    if make_fig:
        _figure(p0, p, hist, rows)
    return p, hist


def _figure(p0, p, hist, rows):
    import matplotlib.pyplot as plt
    import modular, style

    classes = [(U_A, TARGET_A, "#46f0d0", "u = +1"), (U_B, TARGET_B, style.ACCENT, "u = −1")]

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
    for col, (pp, title) in enumerate([(p0, "before training"), (p, "after training")]):
        zs = modular.qpoly_zeros(np.array(coeffs_of(pp)))
        ax[col].set_facecolor("#0b0b14")
        ax[col].plot(zs.real, zs.imag, "o", color="#6a6a7c", ms=5, label="attractors (zeros of F)")
        for (u, target, c, lab), row in zip(classes, rows if col == 1 else [None, None]):
            tr = np.array(rollout(pp, u)[1])
            ax[col].plot(tr[:, 0], tr[:, 1], color=c, lw=1.8, zorder=5, label=lab)
            ax[col].plot(float(target[0]), float(target[1]), "*", color=c, ms=20,
                         mec="white", mew=0.9, zorder=6)
            if row is not None:
                k = row[1]
                ax[col].plot(tr[k, 0], tr[k, 1], "D", color=c, ms=7, mec="white", mew=0.8,
                             zorder=7)
                ax[col].plot(tr[-1, 0], tr[-1, 1], "s", color=c, ms=7, mec="white", mew=0.8,
                             zorder=7)
        ax[col].plot(float(START[0]), float(START[1]), "o", mfc="none", mec="white",
                     mew=1.8, ms=11, zorder=7)
        ax[col].set_xlim(0, 1); ax[col].set_ylim(0.04, 0.85)
        ax[col].set_title(title, fontsize=13); ax[col].set_xlabel(r"Re $\tau$")
    ax[0].set_ylabel(r"Im $\tau$")
    ax[0].legend(loc="lower center", ncol=2, fontsize=9, framealpha=0.3)
    ax[0].text(0.03, 0.80, "○ shared start  ★ class target  — path under constant drive",
               color="white", fontsize=9)
    ax[1].text(0.03, 0.80, "◆ closest approach (what the loss sees)   ■ where the path ends",
               color="white", fontsize=9)

    ax[2].plot(hist, color=style.ACCENT, lw=1.6)
    ax[2].set_yscale("log"); ax[2].set_xlabel("Adam step on the coefficient vector")
    ax[2].set_ylabel("closest-approach loss (log)"); ax[2].set_title("training", fontsize=13)
    ax[2].grid(alpha=0.2)

    fig.suptitle(r"Trainable reservoir (degree 1): a constant drive $Bu$ ($u=\pm1$) from a shared "
                 "start; backprop into the coefficient vector steers each path past its class target\n"
                 "(the loss is the closest approach along the path: the targets are not attractors)",
                 fontsize=12.5, y=1.04)
    out = Path(__file__).parent.parent / "figures" / "14_poc_constant_input.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
