"""(a+b) mod 11 with a circular-phase model versus a small MLP: the circular model generalises
at once; the control memorises and does not generalise in this budget.

WHAT IT COMPUTES — All 121 pairs (a, b) with label (a + b) mod 11, split 50/50 at random.
      A. circular-phase model: one learnable phase φ(k) per symbol and one gain κ (12
         parameters); logits for the pair are κ cos(φ(a) + φ(b) − φ(c)) over candidates c;
         cross-entropy, Adam, 3000 steps.
      B. control: a 24-dimensional embedding per symbol, one tanh layer of 96 units, a
         linear head, weight decay 1e−3; cross-entropy, Adam, 6000 steps.
    Train and held-out accuracy are recorded every 100 steps; the learned phases are
    compared with the regular 11-gon.
WHAT THE FIGURE SHOWS — Left: model A's train and held-out accuracy rise together to 1.
    Middle: model B's train accuracy reaches 1 while its held-out accuracy stays near
    chance for the whole run. Right: the learned phases φ(k) on the circle, which form a
    rotated regular 11-gon (addition of phases is rotation).
WHAT IS PROVED / KNOWN — Addition mod n is rotation on the n-th roots of unity, so the
    circular model contains the exact solution with φ(k) = 2πk·u/n for a unit u, and a model
    with 12 parameters cannot fit an arbitrary 121-entry table (the audit measured train
    accuracy 0.167 on random labels; ledger F6). That trained networks on modular addition
    end up with Fourier/circular features is the finding of the grokking literature
    (Power et al. 2022, arXiv:2201.02177; Nanda et al. 2023, arXiv:2301.05217).
WHAT IS NOT CLAIMED — That this "shortcuts grokking". Grokking is a delayed generalisation of
    the control, and the control here never generalises (held-out 0.000 at 30 000 steps in
    two optimiser regimes in the audit's rerun; ledger F6), so no transition is observed to
    be shortcut. What is shown is that a model built from the right symmetry generalises
    immediately and a generic one does not within this budget: a measurement about
    inductive bias, not a theorem, and not evidence about the network of figures 14–20,
    whose state is a point of ℍ, not a phase.
HOW THIS WAS FOUND — We noticed that the circular representation which grokking networks
    are reported to discover is the natural coordinate on the boundary of the q-disk, and
    guessed that a model which starts there would skip the memorisation phase. The precise
    content is the inductive-bias comparison above; the "skip" cannot be measured until the
    control is made to grok, which it did not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import value_and_grad, jit
from jax.tree_util import tree_map

N = 11
A_, B_ = np.meshgrid(np.arange(N), np.arange(N))
PAIRS = np.stack([A_.ravel(), B_.ravel()], 1)
LABELS = (PAIRS[:, 0] + PAIRS[:, 1]) % N
_rng = np.random.default_rng(0)
_perm = _rng.permutation(len(PAIRS))
NTR = len(PAIRS) // 2
TR, TE = jnp.asarray(_perm[:NTR]), jnp.asarray(_perm[NTR:])
PAIRS_J, LABELS_J = jnp.asarray(PAIRS), jnp.asarray(LABELS)


# --- Adam over a pytree --- #
def adam_init(p):
    return {"m": tree_map(jnp.zeros_like, p), "v": tree_map(jnp.zeros_like, p), "t": 0}


def adam_step(p, g, st, lr):
    t = st["t"] + 1
    m = tree_map(lambda m, g: 0.9 * m + 0.1 * g, st["m"], g)
    v = tree_map(lambda v, g: 0.999 * v + 0.001 * g * g, st["v"], g)
    mh = tree_map(lambda m: m / (1 - 0.9 ** t), m)
    vh = tree_map(lambda v: v / (1 - 0.999 ** t), v)
    p = tree_map(lambda p, m, v: p - lr * m / (jnp.sqrt(v) + 1e-8), p, mh, vh)
    return p, {"m": m, "v": v, "t": t}


# --- Model A: circular phases --- #
def A_logits(p, idx):
    phi = p["phi"]
    a, b = PAIRS_J[idx, 0], PAIRS_J[idx, 1]
    psi = phi[a] + phi[b]
    return p["kappa"] * jnp.cos(psi[:, None] - phi[None, :])


def A_loss(p, idx):
    z = A_logits(p, idx)
    lp = z - jax.scipy.special.logsumexp(z, 1, keepdims=True)
    return -lp[jnp.arange(idx.shape[0]), LABELS_J[idx]].mean()


# --- Model B: embedding + MLP (control) --- #
def B_logits(p, idx):
    a, b = PAIRS_J[idx, 0], PAIRS_J[idx, 1]
    h = jnp.concatenate([p["E"][a], p["E"][b]], 1)
    h = jnp.tanh(h @ p["W1"] + p["b1"])
    return h @ p["W2"] + p["b2"]


def B_loss(p, idx):
    z = B_logits(p, idx)
    lp = z - jax.scipy.special.logsumexp(z, 1, keepdims=True)
    ce = -lp[jnp.arange(idx.shape[0]), LABELS_J[idx]].mean()
    wd = 1e-3 * sum(jnp.sum(w ** 2) for w in (p["W1"], p["W2"]))
    return ce + wd


def acc(logits_fn, p, idx):
    return float((jnp.argmax(logits_fn(p, idx), 1) == LABELS_J[idx]).mean())


def train(loss_fn, logits_fn, p0, steps, lr):
    lg = jit(value_and_grad(loss_fn))
    p, st = p0, adam_init(p0)
    hist = []
    for s in range(steps + 1):
        _, g = lg(p, TR)
        p, st = adam_step(p, g, st, lr)
        if s % 100 == 0:
            hist.append((s, acc(logits_fn, p, TR), acc(logits_fn, p, TE)))
    return p, np.array(hist)


def lattice_deviation(phi):
    """Max deviation (degrees) of the sorted phase gaps from 2π/N: 0 for a regular N-gon."""
    s = np.sort(np.mod(phi, 2 * np.pi))
    gaps = np.diff(np.concatenate([s, [s[0] + 2 * np.pi]]))
    return float(np.degrees(np.max(np.abs(gaps - 2 * np.pi / N))))


def run(make_fig=True):
    rng = np.random.default_rng(1)
    pA = {"phi": jnp.asarray(rng.uniform(0, 2 * np.pi, N)), "kappa": jnp.asarray(3.0)}
    pA, hA = train(A_loss, A_logits, pA, steps=3000, lr=0.05)
    print(f"Model A (circular, 12 params): final train {hA[-1,1]:.3f}  held-out {hA[-1,2]:.3f}")
    assert hA[-1, 1] > 0.999 and hA[-1, 2] > 0.999, "the circular model did not reach 100%/100%"
    dev = lattice_deviation(np.array(pA["phi"]))
    print(f"  learned phases form a regular 11-gon to within {dev:.2f} degrees")
    assert dev < 5.0

    d, h = 24, 96
    pB = {"E": jnp.asarray(rng.standard_normal((N, d)) * 0.3),
          "W1": jnp.asarray(rng.standard_normal((2 * d, h)) * 0.1), "b1": jnp.zeros(h),
          "W2": jnp.asarray(rng.standard_normal((h, N)) * 0.1), "b2": jnp.zeros(N)}
    pB, hB = train(B_loss, B_logits, pB, steps=6000, lr=0.01)
    print(f"Model B (control MLP):         final train {hB[-1,1]:.3f}  held-out {hB[-1,2]:.3f}")
    assert hB[-1, 1] > 0.95, "the control did not memorise the training set"
    assert hB[-1, 2] < 0.5, "the control generalised: the figure's claim would need revisiting"
    print("  the control memorises and does not generalise in 6000 steps: no grokking transition "
          "is observed, so none is 'shortcut' (ledger F6)")

    if make_fig:
        _figure(hA, hB, np.array(pA["phi"]))
    return hA, hB


def _figure(hA, hB, phi):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    TEAL, GOLD, NEURON, GREEN = "#46f0d0", "#ffd166", "#ff3b6b", "#7CFC8A"

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    a = ax[0]
    a.plot(hA[:, 0], hA[:, 1], color=TEAL, lw=2, label="train")
    a.plot(hA[:, 0], hA[:, 2], color=GOLD, lw=2, ls="--", label="held-out")
    a.set_ylim(0, 1.05); a.set_xlabel("step"); a.set_ylabel("accuracy")
    a.set_title("circular-phase model (12 parameters):\ntrain and held-out rise together", fontsize=12)
    a.legend(loc="lower right", fontsize=9, framealpha=0.3)

    b = ax[1]
    b.plot(hB[:, 0], hB[:, 1], color=TEAL, lw=2, label="train")
    b.plot(hB[:, 0], hB[:, 2], color=GOLD, lw=2, ls="--", label="held-out")
    b.set_ylim(0, 1.05); b.set_xlabel("step"); b.set_ylabel("accuracy")
    b.set_title("MLP control: memorises the training set,\nheld-out stays near chance (no grokking seen)",
                fontsize=12)
    b.legend(loc="lower right", fontsize=9, framealpha=0.3)

    c = ax[2]
    c.set_facecolor("#0b0b14")
    th = np.linspace(0, 2 * np.pi, 200)
    c.plot(np.cos(th), np.sin(th), color="#333", lw=1)
    order = np.argsort(phi % (2 * np.pi))
    pts = np.stack([np.cos(phi), np.sin(phi)], 1)
    c.plot(pts[order, 0], pts[order, 1], color=GREEN, lw=1, alpha=0.5)
    for k in range(N):
        c.plot(pts[k, 0], pts[k, 1], "o", color=NEURON, ms=12, mec="white", mew=0.6)
        c.annotate(str(k), pts[k], textcoords="offset points", xytext=(7, 4),
                   color=NEURON, fontsize=9)
    c.set_aspect("equal"); c.set_xlim(-1.3, 1.3); c.set_ylim(-1.3, 1.3)
    c.set_title("the learned phases φ(k): a regular 11-gon\n(addition mod 11 = rotation)", fontsize=12)

    fig.suptitle("(a+b) mod 11: a model built on the circle generalises immediately; a generic MLP memorises "
                 "and, in this budget, never generalises\n(an inductive-bias measurement; no grokking "
                 "transition is observed, so none is shortcut)", fontsize=12.5, y=1.04)
    out = Path(__file__).parent.parent / "figures" / "30_grok.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
