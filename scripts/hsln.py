"""Train two detectors, multiply their forms, and read the compound: the zero-union merge.

WHAT IT COMPUTES — A model is a truncated q-series F(τ) = Σ_{n<d} a_n q^n, q = e^{2πiτ}, with
    coefficient vector a ∈ ℂ^d (the Fourier coefficients, which the June files called the DNA).
    Its zeros in the q-disk are the attractors of the gradient flow on |F|² (the June files
    called them neurons). The readout along the input lane Im τ = 0.18 is the spike map
    s(x) = exp(−(|F(x+0.18i)|/max|F|)²/2σ²), which is 1 on a zero. Two coefficient vectors
    are trained by Adam so their spike maps match one-hot targets at 12 probe positions;
    the compound is the Cauchy product a_A ⋆ a_B, i.e. F_{AB} = F_A·F_B.
WHAT THE FIGURE SHOWS — Left: the two trained spike maps along the lane. Middle: the spike map
    of the product, which spikes at the union of the two target sets. Right: the zeros of
    F_A, F_B and F_A·F_B in the τ-strip: the product's zero set is exactly the union.
WHAT IS PROVED / KNOWN — zeros(F·G) = zeros(F) ∪ zeros(G) in an integral domain (`mul_eq_zero`;
    ledger B1), and multiplying truncated series is the Cauchy product of coefficient vectors.
    Zeros are exactly the local minima of |F| for holomorphic F (minimum-modulus principle;
    `Fugue.isLocalMin_norm_of_eq_zero`, `Fugue.eq_zero_of_isLocalMin_norm`, ledger A9).
    The union of zero sets is verified here to 1e−9 on the trained pair.
WHAT IS NOT CLAIMED — That the compound "does both tasks" is an empirical statement about the
    spike readout, not a theorem: the readout divides by max|F| over the lane and the 1/|F|
    profile near A's zeros is modulated by |F_B|, so a spike can be thinned or missed. The
    gradient flow of the product is not the union of the two flows either (see hsln_dyn.py,
    figure 19). Nothing here is a modular form: the coefficient vector is a free polynomial
    (THE-STORY §3).
HOW THIS WAS FOUND — We imagined a trained model as a landscape whose valleys are its skills,
    and asked what "add two models" could mean so that no valley is lost. Multiplying the
    landscapes keeps every valley because a product vanishes wherever a factor does; in
    coefficients that is the Cauchy product, so merging is one convolution and no retraining.
    The precise statement is the zero-union law above; the open, empirical question is whether
    the merged readout is still usable, which this script measures.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad

import modular

jax.config.update("jax_enable_x64", True)

LANE_Y = 0.18                       # input is encoded as a position on this lane in ℍ
SIGMA = 0.35                        # spike sharpness (on the max-normalised |F|)
PROBES = jnp.round(jnp.linspace(0.06, 0.94, 12), 4)   # input positions sampled by a task


# --------------------------------------------------------------------- the truncated form
def field(coeffs, x, y=LANE_Y):
    """F(τ) at τ = x + i y."""
    q = jnp.exp(2j * jnp.pi * (x + 1j * y))
    return jnp.sum(coeffs * q ** jnp.arange(coeffs.shape[0]))


def _fmag(coeffs, x):
    f = field(coeffs, x)
    return jnp.sqrt(jnp.real(f * jnp.conj(f)))


def spike_map(coeffs, xs):
    """1/|F| spike readout along the input lane, in [0,1] (1 = on a zero)."""
    m = jax.vmap(lambda x: _fmag(coeffs, x))(xs)
    mn = m / (jnp.max(m) + 1e-12)
    return jnp.exp(-mn ** 2 / (2 * SIGMA ** 2))


# --------------------------------------------------------------------- the model object
@dataclass
class Network:
    degree: int              # Siegel degree of the state space (1 here: τ ∈ ℍ)
    coeffs: np.ndarray       # the coefficient vector a_0..a_{d-1}

    def zeros(self):
        """Zeros of F in the q-disk, as τ values: the attractors of the flow."""
        return modular.qpoly_zeros(np.asarray(self.coeffs))

    def spikes(self, xs=PROBES):
        return np.array(spike_map(jnp.asarray(self.coeffs), jnp.asarray(xs)))


def tensor(a: Network, b: Network) -> Network:
    """Multiply the forms = Cauchy product (convolution) of the coefficient vectors."""
    assert a.degree == b.degree
    return Network(a.degree, np.convolve(np.asarray(a.coeffs), np.asarray(b.coeffs)))


# --------------------------------------------------------------------- training a task
def _central_init(seed, dg):
    rng = np.random.default_rng(seed)
    zs = np.array([0.5 + (LANE_Y + 0.04 * k) * 1j for k in range(dg - 1)])
    c = np.poly(np.exp(2j * np.pi * zs))[::-1]
    c = c / np.max(np.abs(c))
    return jnp.array(np.concatenate([c.real, c.imag]) + 0.04 * rng.standard_normal(2 * dg))


def adam(lg, p, steps, lr):
    """Plain Adam on a flat parameter vector; returns (params, loss history)."""
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


def train_task(target, dg=5, steps=1500, lr=0.01, seed=0):
    """Train a coefficient vector so its spike map matches `target` over PROBES."""
    target = jnp.asarray(target, dtype=float)

    def to_coeffs(p):
        return p[:dg] + 1j * p[dg:]

    def loss(p):
        return jnp.sum((spike_map(to_coeffs(p), PROBES) - target) ** 2) + 1e-4 * jnp.sum(p ** 2)

    p, hist = adam(jax.jit(value_and_grad(loss)), _central_init(seed, dg), steps, lr)
    return Network(1, np.array(to_coeffs(p))), hist


# --------------------------------------------------------------------- the experiment
# Task A: a detector that spikes at two input positions; Task B: spikes at a third.
TARGET_A = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float)  # probes 0.22, 0.54
TARGET_B = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=float)  # probe  0.78
TARGET_UNION = np.maximum(TARGET_A, TARGET_B)


def union_residual(netA, netB, netAB):
    """max over parent zeros of the distance to the nearest compound zero."""
    zAB = netAB.zeros()
    parents = np.concatenate([netA.zeros(), netB.zeros()])
    return max(np.min(np.abs(zAB - z)) for z in parents)


def run(make_fig=True):
    netA, hA = train_task(TARGET_A, seed=1)
    netB, hB = train_task(TARGET_B, seed=2)
    netAB = tensor(netA, netB)

    sAB = netAB.spikes()
    print(f"task A loss {hA[0]:.3f}->{hA[-1]:.4f}   task B loss {hB[0]:.3f}->{hB[-1]:.4f}")
    print("probe      :", " ".join(f"{p:.2f}" for p in np.array(PROBES)))
    print("target A|B :", " ".join(f"{t:.0f}   " for t in TARGET_UNION))
    print("compound AB:", " ".join(f"{s:.2f}" for s in sAB))

    res = union_residual(netA, netB, netAB)
    print(f"zeros(F_A F_B) ⊇ zeros(F_A) ∪ zeros(F_B): max distance {res:.1e}  "
          f"({len(netA.zeros())} + {len(netB.zeros())} = {len(netAB.zeros())} zeros)")
    assert res < 1e-9, "zero-union law failed numerically"
    assert len(netAB.zeros()) == len(netA.zeros()) + len(netB.zeros())

    hit = bool(np.all((sAB > 0.5) == (TARGET_UNION > 0.5)))
    print(f"compound spike map matches the union of both targets (empirical): {hit}")
    assert hit, "merged readout did not reproduce the union of the two tasks"

    if make_fig:
        _figure(netA, netB, netAB)
    return netA, netB, netAB


def _figure(netA, netB, netAB):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    xs = np.linspace(0.05, 0.95, 500)
    cA, cB, cAB = "#46f0d0", "#ffd166", "#ff3b6b"
    dA = [0.22, 0.54]; dB = [0.78]

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})

    ax[0].plot(xs, np.array(spike_map(jnp.asarray(netA.coeffs), jnp.asarray(xs))), color=cA, lw=2, label="task A")
    ax[0].plot(xs, np.array(spike_map(jnp.asarray(netB.coeffs), jnp.asarray(xs))), color=cB, lw=2, label="task B")
    for d in dA: ax[0].axvline(d, color=cA, ls=":", lw=1, alpha=0.6)
    for d in dB: ax[0].axvline(d, color=cB, ls=":", lw=1, alpha=0.6)
    ax[0].set_title("two trained detectors", fontsize=12)
    ax[0].set_xlabel("input position (Re τ)"); ax[0].set_ylabel("spike  1/|F|→"); ax[0].set_ylim(0, 1.1)
    ax[0].legend(fontsize=9, framealpha=0.3)

    ax[1].plot(xs, np.array(spike_map(jnp.asarray(netAB.coeffs), jnp.asarray(xs))), color=cAB, lw=2.4, label="F_A · F_B")
    for d in dA + dB:
        ax[1].axvline(d, color="white", ls=":", lw=1, alpha=0.5)
    ax[1].set_title("product of the forms: spikes at the union (measured)", fontsize=12)
    ax[1].set_xlabel("input position (Re τ)"); ax[1].set_ylim(0, 1.1)
    ax[1].legend(fontsize=9, framealpha=0.3)

    ax[2].set_facecolor("#0b0b14")
    for net, c, lab in [(netA, cA, "A"), (netB, cB, "B")]:
        z = net.zeros()
        ax[2].plot(z.real, z.imag, "o", color=c, ms=9, mfc="none", mew=2, label=f"zeros({lab})")
    z = netAB.zeros()
    ax[2].plot(z.real, z.imag, "x", color=cAB, ms=8, mew=1.6, label="zeros(F_A·F_B)")
    ax[2].axhline(LANE_Y, color="#555", ls="--", lw=0.8)
    ax[2].set_xlim(0, 1); ax[2].set_ylim(0.05, 0.9)
    ax[2].set_xlabel("Re τ"); ax[2].set_ylabel("Im τ")
    ax[2].set_title("zeros of the product = union of the zeros\n(exact: a product vanishes where a factor does)", fontsize=11)
    ax[2].legend(fontsize=8, framealpha=0.3, loc="upper right")

    fig.suptitle("Train two detectors, multiply their forms (convolve the coefficient vectors): "
                 "the zero sets union exactly; the merged readout spikes at both (measured)",
                 fontsize=13, y=1.02)
    out = Path(__file__).parent.parent / "figures" / "18_e2e_compose.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
