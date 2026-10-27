"""A driven reservoir with a linear readout learns to follow a sine wave.

WHAT IT COMPUTES — The degree-1 field of poc.py, F(τ) = Σ_{n<6} a_n q^n, now driven by a
    time-varying input u(t) = sin(2π·2.5·t/T) over T = 240 steps of 0.03:
        τ̇ = clip(−α ∇|F|², ±3) + B u(t),   B = (0.55, 0),  α = 2,
    i.e. the raw (not unit-speed) gradient, so near a zero the flow is a restoring spring and
    the system is a driven first-order oscillator. The output r(t) = a(x−½) + b(y−½) + d is a
    linear readout of the state; the 12 coefficients and the 3 readout weights are trained
    jointly by Adam on the mean-square error against the target 0.35·u(t).
WHAT THE FIGURE SHOWS — Left: input, target, and the readout before and after training.
    Right: the training MSE on a log scale.
WHAT IS PROVED / KNOWN — The flow descends |F|² between drive kicks (`dE/dt = −|∇E|²` for
    any gradient flow; ledger E1, "tautological"). Fitting a linear readout on a driven
    nonlinear state is reservoir computing (Jaeger 2001; Maass, Natschläger & Markram 2002);
    here the reservoir itself is also trained, which makes it an ordinary small recurrent
    model trained by backpropagation through time.
WHAT IS NOT CLAIMED — No modular property is used: the landscape is a free degree-5
    polynomial in q. Following a sine with 15 parameters is a sanity check that the driven
    rollout is differentiable and the loss falls, not evidence of capacity. The RMSE reached
    is a measurement for this seed.
HOW THIS WAS FOUND — After the constant-drive test (figure 14) we wanted time in the loop:
    the picture was a ball in a dented bowl pushed by a swinging wind, with a ruler reading
    off where it is. The precise statement is a driven first-order system with trained
    potential and linear readout; the spec's Liquid-Time-Constant reading (§2.3) is only an
    analogy, and is not tested here.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, value_and_grad

import poc

jax.config.update("jax_enable_x64", True)

DG = poc.DG
T = 240
DT = 0.03
CYCLES = 2.5
B_VEC = jnp.array([0.55, 0.0])
START = jnp.array([0.5, 0.35])

_ts = jnp.arange(T)
U = jnp.sin(2 * jnp.pi * CYCLES * _ts / T)              # sine input
TARGET = 0.35 * jnp.sin(2 * jnp.pi * CYCLES * _ts / T)  # target waveform to follow

NP = 2 * DG  # field parameters; then readout (a, b, d)


def _split(params):
    return params[:NP], params[NP], params[NP + 1], params[NP + 2]


def rollout(params, alpha=2.0, dt=DT, vmax=3.0, ymin=0.05, ymax=0.95):
    gp = params[:NP]

    def stepfn(xy, u):
        g = poc._grad_energy(gp, xy)
        # raw gradient (a restoring spring near a zero) -> driven first-order oscillator,
        # clipped for stability; not unit-speed (which would pin the state).
        vx = jnp.clip(-alpha * g[0], -vmax, vmax)
        vy = jnp.clip(-alpha * g[1], -vmax, vmax)
        nx = jnp.clip(xy[0] + dt * (vx + B_VEC[0] * u), 0.02, 0.98)
        ny = jnp.clip(xy[1] + dt * (vy + B_VEC[1] * u), ymin, ymax)
        new = jnp.array([nx, ny])
        return new, new
    _, traj = lax.scan(stepfn, START, U)
    return traj


def readout(params, traj):
    _, a, b, d = _split(params)
    return a * (traj[:, 0] - 0.5) + b * (traj[:, 1] - 0.5) + d


def loss(params):
    r = readout(params, rollout(params))
    return jnp.mean((r - TARGET) ** 2) + 1e-4 * jnp.sum(params[:NP] ** 2)


_loss_and_grad = jax.jit(value_and_grad(loss))


def init(seed=0):
    gp = np.array(poc.init_params(seed))
    return jnp.array(np.concatenate([gp, [1.0, 0.0, 0.0]]))


def train(steps=1200, lr=0.01, seed=0):
    p = init(seed)
    p0 = p
    m = jnp.zeros_like(p); v = jnp.zeros_like(p)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for t in range(1, steps + 1):
        L, g = _loss_and_grad(p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        p = p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
        hist.append(float(L))
    return p0, p, hist


def run(make_fig=True, steps=1200, lr=0.01, seed=0):
    p0, p, hist = train(steps=steps, lr=lr, seed=seed)
    r0 = np.array(readout(p0, rollout(p0)))
    r1 = np.array(readout(p, rollout(p)))
    tgt = np.array(TARGET)
    rmse0 = float(np.sqrt(np.mean((r0 - tgt) ** 2)))
    rmse1 = float(np.sqrt(np.mean((r1 - tgt) ** 2)))
    print(f"loss {hist[0]:.4f} -> {hist[-1]:.5f};  RMSE before {rmse0:.4f}, after {rmse1:.4f} "
          f"(target amplitude 0.35)")
    assert rmse1 < 0.5 * rmse0, "training did not halve the RMSE"
    assert rmse1 < 0.1, "readout does not follow the sine to within 0.1"

    if make_fig:
        _figure(hist, r0, r1)
    return p, hist


def _figure(hist, r0, r1):
    import matplotlib.pyplot as plt
    import style

    t = np.arange(T) * DT
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    ax[0].plot(t, np.array(U), color="#5a5a6a", lw=1.0, label="input  u(t) = sin")
    ax[0].plot(t, np.array(TARGET), color="white", lw=1.8, ls="--", label="target")
    ax[0].plot(t, r0, color=style.NEURON, lw=1.2, alpha=0.55, label="readout (before)")
    ax[0].plot(t, r1, color="#46f0d0", lw=2.2, label="readout (after)")
    ax[0].set_xlabel("time"); ax[0].set_ylabel("signal")
    ax[0].set_title("a driven reservoir with a linear readout follows the sine", fontsize=13)
    ax[0].legend(loc="upper right", fontsize=9, framealpha=0.3)
    ax[0].grid(alpha=0.15)

    ax[1].plot(hist, color=style.ACCENT, lw=1.6)
    ax[1].set_yscale("log"); ax[1].set_xlabel("Adam step"); ax[1].set_ylabel("MSE (log)")
    ax[1].set_title("training (12 field + 3 readout parameters)", fontsize=12); ax[1].grid(alpha=0.2)

    fig.suptitle("Driven flow on |F|² with a trained linear readout: a 15-parameter recurrent model "
                 "fitted by backpropagation through time", fontsize=12.5, y=1.02)
    out = Path(__file__).parent.parent / "figures" / "15_poc_sine.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
