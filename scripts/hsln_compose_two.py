"""Train two factors jointly so that only their product is constrained: where do the zeros go?

WHAT IT COMPUTES — Two coefficient vectors a ∈ ℂ^{d_a}, b ∈ ℂ^{d_b} (the Fourier coefficients,
    which the June files called the DNA) are trained TOGETHER by Adam, with the loss placed
    only on the spike map of their Cauchy product a ⋆ b along the lane of hsln.py: the
    target is a single spike at the probe nearest 0.62. The optimiser is free to put the
    on-lane zero in either factor. Five shapes (d_a, d_b) ∈ {(2,2), (3,3), (2,4), (4,4), (3,6)}
    are run for 2500 steps each (the June run used 5000; the loss has plateaued by 2500).
    For each shape we record the loss, the zeros of a, of b and of a ⋆ b, and which factor
    holds the on-lane zero.
WHAT THE FIGURE SHOWS — Top row: the training loss for each shape. Bottom row: the zeros of
    the two factors in the τ-strip, the lane and the target position; the product's zeros
    are the union and the on-lane zero sits in one factor or the other.
WHAT IS PROVED / KNOWN — zeros(F_a F_b) = zeros(F_a) ∪ zeros(F_b) (`mul_eq_zero`, ledger B1);
    a degree-(d_a + d_b − 2) polynomial has at most that many roots. Which factor the
    optimiser chooses is a measurement, and by the symmetry of the product it is a matter of
    initialisation.
WHAT IS NOT CLAIMED — This is joint training with the loss on the product, so it is NOT the
    separate-training merge test of system-spec §4.4(6) (ledger F9); that test is figure 18
    and figure 45. The June script used this joint training to look for the factors of a
    large integer in the zero positions; that use is not reproduced here, and the audit's
    verdict on it is recorded in ledger appendix A (the only composition of divisibility
    detectors that factors is trial division, `Fugue.first_spike_is_minFac`). The figure
    number 44 is assigned here; the June file had none.
HOW THIS WAS FOUND — We asked the optimiser to make a product spike without telling it which
    factor should do the work, to see whether composition "decides" anything. The picture
    was two people sharing one job. The precise content is only the zero-union law; the
    assignment of the zero to a factor is arbitrary, which is itself the honest finding.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad

import hsln
import modular

jax.config.update("jax_enable_x64", True)

LANE_Y = hsln.LANE_Y
TARGET_POS = 0.62
CONFIGS = [(2, 2), (3, 3), (2, 4), (4, 4), (3, 6)]
STEPS = 2500


def composed_spike_map(a, b, xs):
    return hsln.spike_map(jnp.convolve(a, b), xs)


def joint_train(dg_a, dg_b, target, steps=STEPS, lr=0.01, seed=0):
    """Train a (length dg_a) and b (length dg_b) so the spike map of a ⋆ b matches target."""

    def unpack(p):
        sa = 2 * dg_a
        pa, pb = p[:sa], p[sa:]
        return pa[:dg_a] + 1j * pa[dg_a:], pb[:dg_b] + 1j * pb[dg_b:]

    def loss(p):
        a, b = unpack(p)
        spikes = composed_spike_map(a, b, hsln.PROBES)
        return jnp.sum((spikes - target) ** 2) + 1e-4 * jnp.sum(p ** 2)

    p0 = jnp.concatenate([hsln._central_init(seed, dg_a), hsln._central_init(seed + 100, dg_b)])
    p, hist = hsln.adam(jax.jit(value_and_grad(loss)), p0, steps, lr)
    a, b = unpack(p)
    return np.asarray(a), np.asarray(b), hist


def on_lane(coeffs, tol=0.04):
    z = modular.qpoly_zeros(coeffs)
    return sorted(float(t.real) for t in z if abs(t.imag - LANE_Y) < tol)


def run(make_fig=True):
    probes = np.array(hsln.PROBES)
    k = int(np.argmin(np.abs(probes - TARGET_POS)))
    target = np.zeros_like(probes); target[k] = 1.0
    print(f"target: one spike at probe {k} ({probes[k]:.4f}); {STEPS} Adam steps per shape")

    results = []
    for dg_a, dg_b in CONFIGS:
        a, b, hist = joint_train(dg_a, dg_b, jnp.asarray(target))
        c = np.convolve(a, b)
        za, zb, zc = modular.qpoly_zeros(a), modular.qpoly_zeros(b), modular.qpoly_zeros(c)
        parents = np.concatenate([za, zb])
        union_err = max(np.min(np.abs(zc - z)) for z in parents) if len(parents) else 0.0
        assert len(zc) == len(za) + len(zb) and union_err < 1e-9, "zero-union failed"
        spikes = np.array(hsln.spike_map(jnp.asarray(c), hsln.PROBES))
        hit = bool(np.all((spikes > 0.5) == (target > 0.5)))
        la, lb = on_lane(a), on_lane(b)
        holder = "a" if la and not lb else ("b" if lb and not la else ("both" if la and lb else "neither"))
        print(f"  (d_a,d_b)=({dg_a},{dg_b}): loss {hist[0]:.3f}->{hist[-1]:.4f}; zeros a={len(za)} b={len(zb)} "
              f"product={len(zc)} (union to {union_err:.1e}); product spikes only at target: {hit}; "
              f"on-lane zero held by factor {holder} (a: {np.round(la,3)}, b: {np.round(lb,3)})")
        results.append(dict(shape=(dg_a, dg_b), a=a, b=b, c=c, hist=hist, hit=hit, holder=holder,
                            za=za, zb=zb, zc=zc))
    n_hit = sum(r["hit"] for r in results)
    print(f"product reproduces the one-hot target in {n_hit}/{len(results)} shapes (measured)")
    assert n_hit >= 4, "joint training failed in more than one shape"
    holders = {r["holder"] for r in results}
    print(f"which factor holds the on-lane zero: {sorted(holders)}  (arbitrary by the product's symmetry)")

    if make_fig:
        _figure(results, probes[k])
    return results


def _figure(results, pos):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    TEAL, GOLD, NEURON = "#46f0d0", "#ffd166", "#ff3b6b"
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 7.4), gridspec_kw={"height_ratios": [1, 1.25]})
    for col, r in enumerate(results):
        dg_a, dg_b = r["shape"]
        ax = axes[0, col]
        ax.semilogy(r["hist"], color=GOLD, lw=1.2)
        ax.set_xlabel("step"); ax.set_ylabel("loss on the product's spike map")
        ax.set_title(f"d_a={dg_a}, d_b={dg_b}\nfinal loss {r['hist'][-1]:.4f}", fontsize=11)
        ax.grid(alpha=0.2, which="both")

        ax = axes[1, col]
        ax.set_facecolor("#0b0b14")
        ax.plot(r["za"].real, r["za"].imag, "o", color=TEAL, ms=9, mfc="none", mew=2, label="zeros of a")
        ax.plot(r["zb"].real, r["zb"].imag, "s", color=GOLD, ms=9, mfc="none", mew=2, label="zeros of b")
        ax.plot(r["zc"].real, r["zc"].imag, "x", color="white", ms=6, mew=1.2, label="zeros of a⋆b")
        ax.axhline(LANE_Y, color="#5aa", ls="--", lw=0.8, alpha=0.7)
        ax.axvline(pos, color=NEURON, lw=1.2, alpha=0.7, label=f"target {pos:.2f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.6)
        ax.set_xlabel("Re τ"); ax.set_ylabel("Im τ")
        where = {"a": "in factor a", "b": "in factor b", "both": "in both", "neither": "none within 0.04 of the lane"}
        ax.set_title(f"on-lane zero: {where[r['holder']]}", fontsize=11)
        if col == 0:
            ax.legend(loc="upper right", fontsize=7.5, framealpha=0.3)

    fig.suptitle("Joint training of two factors with the loss on their product's spike map: the product's zeros "
                 "are the union of the factors' zeros (exact),\nand which factor holds the on-lane zero is "
                 "arbitrary (not the separate-training merge test, which is figures 18 and 45)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    out = Path(__file__).parent.parent / "figures" / "44_hsln_compose_two.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
