"""XOR as a static four-point discriminant: a zero placed at the single-pulse position.

WHAT IT COMPUTES — F(τ) = Σ_{n<5} a_n q^n with learnable a ∈ ℂ^5. Each of the four input
    patterns (u₁,u₂) ∈ {0,1}² is mapped by a FIXED affine "landing" map to a point of the
    τ-strip, τ(u) = START + 0.4(u₁B₁ + u₂B₂), and the score is
        s(u) = exp(−|F(τ(u))|² / 2σ²),   σ = 0.14,
    which is 1 at a zero of F. The loss is Σ_u (s(u) − XOR(u))² plus weight decay; Adam on a.
    XOR is solved when F has a zero at the two single-pulse landing points and none at the
    (0,0) and (1,1) points.
WHAT THE FIGURE SHOWS — Left: the four scores before and after training against the XOR
    truth table. Middle: the score along the landing line, with the four landing positions
    marked: a single spike is carved at the single-pulse positions (the two nearly coincide),
    and the untrained spikes at (0,0) and (1,1) have been suppressed. Right: training loss.
WHAT IS PROVED / KNOWN — A truncated q-series of degree 4 can be given any four prescribed
    roots in the q-disk (fundamental theorem of algebra), so a zero can be placed wherever
    the single-pulse point lands; zeros are the minima of |F| (`Fugue.isLocalMin_norm_of_eq_zero`,
    ledger A9). The truth table reached, loss and reproducibility are measurements (ledger F3).
WHAT IS NOT CLAIMED — There is no ODE, no integration and no time in this script: the June
    name "temporal XOR" was wrong (ledger F3). The "pulses" are a fixed affine encoding and
    the classifier is a static four-point discriminant. The untrained field already spikes
    at all four points, so what training does is suppress two spikes, not "learn a fold".
    This is not the spike-train XOR of system-spec §5 step 4.
HOW THIS WAS FOUND — We wanted XOR because it is the classic test that a single linear unit
    fails. The picture was a landscape with a pit exactly where "one pulse" lands and solid
    ground where "none" or "both" land; a pit is a zero, and zeros can be put anywhere, so
    the field can carve the XOR pattern. The precise statement is just that: a polynomial
    with a chosen root. Whether the same works when the position is reached by a flow in
    time is the question the June script claimed and did not test.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad

jax.config.update("jax_enable_x64", True)

DG = 5
DRIVE_GAIN = 0.40                 # the per-channel displacement of the fixed landing map
B1 = jnp.array([0.50, 0.00])
B2 = jnp.array([0.46, 0.05])      # nearly the same direction, slightly distinct channel
START = jnp.array([0.30, 0.22])
SIGMA_F = 0.14                    # spike sharpness in |F|
PATTERNS = [(0, 0), (1, 0), (0, 1), (1, 1)]
TARGET = jnp.array([0.0, 1.0, 1.0, 0.0])  # XOR


def coeffs_of(p):
    return p[:DG] + 1j * p[DG:]


def Fmag2(p, xy):
    q = jnp.exp(2j * jnp.pi * (xy[0] + 1j * xy[1]))
    f = jnp.sum(coeffs_of(p) * q ** jnp.arange(DG))
    return jnp.real(f * jnp.conj(f))


def lane(u1, u2):
    """Fixed affine landing map: input pattern -> a point in the strip (no dynamics)."""
    dx = (u1 * B1[0] + u2 * B2[0]) * DRIVE_GAIN
    dy = (u1 * B1[1] + u2 * B2[1]) * DRIVE_GAIN
    return jnp.array([START[0] + dx, START[1] + dy])


def spike(p, u1, u2):
    return jnp.exp(-Fmag2(p, lane(u1, u2)) / (2 * SIGMA_F ** 2))  # 1 near a zero, else ~0


def loss(p):
    L = 0.0
    for (u1, u2), tgt in zip(PATTERNS, TARGET):
        L = L + (spike(p, float(u1), float(u2)) - tgt) ** 2
    return L + 1e-4 * jnp.sum(p ** 2)


_loss_and_grad = jax.jit(value_and_grad(loss))


def init_params(seed=0):
    rng = np.random.default_rng(seed)
    # seed a zero at the single-pulse landing point, others parked higher up
    single = np.array([lane(1.0, 0.0)[0], lane(1.0, 0.0)[1]])
    zs = np.array([single[0] + 1j * single[1], 0.5 + 0.6j, 0.5 + 0.85j])
    c = np.poly(np.exp(2j * np.pi * zs))[::-1]
    c = c / np.max(np.abs(c))
    c = np.concatenate([c, np.zeros(DG - len(c), dtype=complex)])
    return jnp.array(np.concatenate([c.real, c.imag]) + 0.04 * rng.standard_normal(2 * DG))


def train(steps=1200, lr=0.01, seed=0):
    p = init_params(seed); p0 = p
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


def scores(p):
    return np.array([float(spike(p, float(a), float(b))) for a, b in PATTERNS])


def run(make_fig=True, steps=1200, lr=0.01, seed=0):
    p0, p, hist = train(steps=steps, lr=lr, seed=seed)
    s0, s1 = scores(p0), scores(p)
    labels = [f"({a},{b})" for a, b in PATTERNS]
    tgt = np.array(TARGET)
    print(f"loss {hist[0]:.4f} -> {hist[-1]:.5f}")
    print("  untrained scores:", " ".join(f"{l}={s:.2f}" for l, s in zip(labels, s0)),
          " (the untrained field already spikes at every pattern)")
    for lab, a, b in zip(labels, s1, tgt):
        print(f"  trained {lab}: {a:.3f}  (target {b:.0f})")
    assert np.all((s1 > 0.5) == (tgt > 0.5)), "XOR truth table not reproduced"
    assert np.max(np.abs(s1 - tgt)) < 0.1, "a score is more than 0.1 from its target"
    assert np.all(s0 > 0.5), "expected the untrained field to spike at all four patterns"
    print("  training suppressed the (0,0) and (1,1) spikes: a static discriminant, no dynamics")

    if make_fig:
        _figure(p, hist, s0, s1, labels)
    return p, hist


def _figure(p, hist, s0, s1, labels):
    import matplotlib.pyplot as plt
    import style

    cols = ["#6a6a7c", "#46f0d0", "#46f0d0", "#ff3b6b"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.1, 1.2, 1]})

    x = np.arange(4)
    ax[0].bar(x - 0.18, s0, 0.34, color="#444", label="before")
    ax[0].bar(x + 0.18, s1, 0.34, color=cols, label="after")
    ax[0].plot(x, np.array(TARGET), "k*", ms=15, mec="white", label="target (XOR)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("spike score  1/|F|→"); ax[0].set_ylim(0, 1.1)
    ax[0].set_title("XOR truth table: before, every pattern spikes;\nafter, two spikes are suppressed",
                    fontsize=11.5)
    ax[0].legend(fontsize=8, framealpha=0.3)

    yL = float(lane(1.0, 0.0)[1])
    xs = np.linspace(0.05, 0.95, 400)
    Sline = np.array([float(jnp.exp(-Fmag2(p, jnp.array([xx, yL])) / (2 * SIGMA_F ** 2)))
                      for xx in xs])
    ax[1].plot(xs, Sline, color="#46f0d0", lw=2)
    for k, ((a, b), c) in enumerate(zip(PATTERNS, cols)):
        lx = float(lane(float(a), float(b))[0])
        ax[1].axvline(lx, color=c, ls="--", lw=1.2)
        ax[1].text(lx, 1.04 + 0.07 * (k == 2), f"({a},{b})", color=c, ha="center", fontsize=8)
    ax[1].set_xlabel(r"Re $\tau$  (landing position of the pattern)"); ax[1].set_ylabel("spike score")
    ax[1].set_ylim(0, 1.2)
    ax[1].set_title("a zero carved at the single-pulse landing point", fontsize=12)

    ax[2].plot(hist, color=style.ACCENT, lw=1.6)
    ax[2].set_yscale("log"); ax[2].set_xlabel("Adam step"); ax[2].set_ylabel("loss (log)")
    ax[2].set_title("training the coefficient vector", fontsize=12); ax[2].grid(alpha=0.2)

    fig.suptitle("XOR as a static four-point discriminant: a fixed affine map lands each pattern in the strip,\n"
                 "and the field learns a zero at the single-pulse point only (no dynamics, no time)",
                 fontsize=12.5, y=1.06)
    out = Path(__file__).parent.parent / "figures" / "16_poc_xor.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
