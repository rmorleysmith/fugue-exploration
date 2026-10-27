"""Train three detectors separately, merge them by multiplication, remove one by division.

WHAT IT COMPUTES — Three coefficient vectors of length 4 (the Fourier coefficients, which the
    June files called the DNA) are trained independently with hsln.train_task, each to spike
    at one lane position (0.22, 0.50, 0.78). MERGE: F = F_A·F_B·F_C by iterated Cauchy
    product; the zeros of F are the union of the three zero sets, with no joint training.
    ERASE: F / F_B by exact polynomial division (numpy.polydiv); the quotient equals
    F_A·F_C and skill B's zero is gone while A's and C's are untouched. The spike maps of
    the atoms, the merged form and the erased form are plotted along the lane.
WHAT THE FIGURE SHOWS — Row 1: the three atoms' spike maps. Row 2: the merged form spikes
    at all three positions (measured). Row 3: after dividing out B, the spike at 0.50 is
    gone and the other two remain. Row 4: the zeros of the merged and erased forms on the
    strip, filled where on the lane.
WHAT IS PROVED / KNOWN — zeros(FG) = zeros(F) ∪ zeros(G) (`mul_eq_zero`, ledger B1); in ℂ[q]
    division by a factor is exact, (F_A F_B F_C)/F_B = F_A F_C with zero remainder, verified
    here to 1e−10 in the coefficients. This is the separate-training merge that system-spec
    §4.4(6) asked for, at the level of the zero sets (THE-STORY §7, item 2).
WHAT IS NOT CLAIMED — The June header said this was "machine-checked by neurons_mul" and
    that "no standard network can do either"; the first named no Lean declaration in the
    formal repository (the fact is `mul_eq_zero`), and the second is an unsupported
    comparison, so both are removed. The merged SPIKE MAP showing all three spikes is a
    measurement, not a consequence of the zero union: the readout normalises by max|F| over
    the lane and the factors modulate each other's 1/|F| profile (as in figure 18). The June
    seeds used Python's string hash, randomised per process; they are now fixed. The figure
    number 45 is assigned here; the June file had none.
HOW THIS WAS FOUND — Multiplication merges (figure 18); we asked what the inverse does. The
    picture was editing a model with arithmetic: × to add a skill, ÷ to take one away, with
    nothing retrained. The precise statement is polynomial division in ℂ[q]; what remains
    empirical is whether the spike readout of the edited form is still usable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import hsln
import modular

LANE_Y = hsln.LANE_Y
SKILLS = {"A": 0.22, "B": 0.50, "C": 0.78}
SEEDS = {"A": 11, "B": 22, "C": 33}
DG, STEPS = 4, 2500


def one_hot_at(pos):
    probes = np.array(hsln.PROBES)
    idx = int(np.argmin(np.abs(probes - pos)))
    t = np.zeros_like(probes, dtype=float)
    t[idx] = 1.0
    return t, float(probes[idx])


def deconvolve(c, b):
    """Recover a from c = convolve(a, b) by exact polynomial division (ascending powers in, out)."""
    q, r = np.polydiv(np.asarray(c)[::-1], np.asarray(b)[::-1])
    return q[::-1], r[::-1]


def on_lane_zeros(coeffs, tol=0.05):
    """Re τ of the zeros sitting on the input lane (Im τ ≈ LANE_Y)."""
    return sorted(z.real % 1.0 for z in modular.qpoly_zeros(np.asarray(coeffs))
                  if abs(z.imag - LANE_Y) < tol)


def spikes_on(coeffs, xs):
    s = np.asarray(hsln.spike_map(np.asarray(coeffs), xs))
    return s / s.max()


def run(make_fig=True):
    print("train three detectors independently")
    nets, snapped = {}, {}
    for name, pos in SKILLS.items():
        target, probe = one_hot_at(pos)
        net, hist = hsln.train_task(target, dg=DG, steps=STEPS, lr=0.01, seed=SEEDS[name])
        nets[name] = net
        snapped[name] = probe
        lane = on_lane_zeros(net.coeffs)
        print(f"  skill {name}: target {pos:.2f} -> probe {probe:.3f}, loss {hist[0]:.3f} -> {hist[-1]:.4f}, "
              f"on-lane zero(s) {np.round(lane, 3)}")
        assert len(lane) == 1 and abs(lane[0] - probe) < 0.03, f"skill {name} did not place its zero"

    print("merge: F = F_A · F_B · F_C (three convolutions, no joint training)")
    merged = hsln.tensor(hsln.tensor(nets["A"], nets["B"]), nets["C"])
    lane_m = on_lane_zeros(merged.coeffs)
    expect = sorted(snapped.values())
    print(f"  on-lane zeros of the merged form: {np.round(lane_m, 3)}   expected {np.round(expect, 3)}")
    assert len(lane_m) == 3 and max(abs(a - b) for a, b in zip(lane_m, expect)) < 0.03
    xs = np.array(hsln.PROBES)
    sm = spikes_on(merged.coeffs, xs)
    idx = [int(np.argmin(np.abs(xs - p))) for p in expect]
    print(f"  merged spike map at the three probes: {np.round(sm[idx], 2)} (measured; union of zeros is exact)")
    assert np.all(sm[idx] > 0.5), "the merged readout lost a spike"

    print("erase: F / F_B by exact polynomial division")
    erased, remainder = deconvolve(np.asarray(merged.coeffs), np.asarray(nets["B"].coeffs))
    direct_AC = np.convolve(np.asarray(nets["A"].coeffs), np.asarray(nets["C"].coeffs))
    err = float(np.max(np.abs(erased - direct_AC)))
    lane_e = on_lane_zeros(erased)
    print(f"  remainder norm {np.linalg.norm(remainder):.1e};  max|(F/F_B) - F_A F_C| = {err:.1e}")
    print(f"  on-lane zeros after erasing B: {np.round(lane_e, 3)}   expected {np.round([snapped['A'], snapped['C']], 3)}")
    assert np.linalg.norm(remainder) < 1e-10 and err < 1e-10
    assert len(lane_e) == 2 and abs(lane_e[0] - snapped["A"]) < 0.03 and abs(lane_e[1] - snapped["C"]) < 0.03
    print("edited by arithmetic: × added skills, ÷ removed one; zeros bookkept exactly, readout measured")

    if make_fig:
        _figure(nets, snapped, merged, erased)
    return nets, merged, erased


def _figure(nets, snapped, merged, erased):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    TEAL, GREEN, RED = "#46f0d0", "#7CFC8A", "#ff3b6b"
    xs = np.linspace(0, 1, 1200)

    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.1], hspace=0.6, wspace=0.3)

    for j, name in enumerate("ABC"):
        ax = fig.add_subplot(gs[0, j])
        ax.fill_between(xs, 0, spikes_on(nets[name].coeffs, xs), color=TEAL, alpha=0.7)
        ax.axvline(snapped[name], color="white", ls=":", lw=1)
        ax.set_title(f"atom {name}  @ {snapped[name]:.2f}  (trained alone)", fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.1); ax.set_yticks([])
        ax.set_xlabel("lane position")

    ax = fig.add_subplot(gs[1, :])
    ax.fill_between(xs, 0, spikes_on(merged.coeffs, xs), color=GREEN, alpha=0.8)
    for name in "ABC":
        ax.axvline(snapped[name], color="white", ls=":", lw=1)
    ax.set_title("MERGE  F = F_A · F_B · F_C   (×): zeros union exactly; the three spikes in the readout are measured",
                 fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.1); ax.set_yticks([]); ax.set_xlabel("lane position")

    ax = fig.add_subplot(gs[2, :])
    ax.fill_between(xs, 0, spikes_on(erased, xs), color=RED, alpha=0.8)
    ax.axvline(snapped["A"], color="white", ls=":", lw=1)
    ax.axvline(snapped["C"], color="white", ls=":", lw=1)
    ax.axvline(snapped["B"], color="gray", ls="--", lw=1.6)
    ax.annotate("skill B removed", (snapped["B"], 0.85), color="gray", ha="center", fontsize=10)
    ax.set_title("ERASE  F / F_B   (÷, exact polynomial division): B's zero gone, A and C untouched", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.1); ax.set_yticks([]); ax.set_xlabel("lane position")

    ax = fig.add_subplot(gs[3, :])
    ax.set_facecolor("#0b0b14")

    def scatter_zeros(coeffs, y, color, label):
        for z in modular.qpoly_zeros(np.asarray(coeffs)):
            on = abs(z.imag - LANE_Y) < 0.05
            ax.scatter(z.real % 1.0, y, s=120 if on else 40, color=color if on else "#777",
                       edgecolor="white", linewidth=0.5, zorder=3 if on else 2)
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=10)
    scatter_zeros(merged.coeffs, 1, GREEN, "merged")
    scatter_zeros(erased, 0, RED, "erased")
    ax.axhline(1, color="#333", lw=0.5); ax.axhline(0, color="#333", lw=0.5)
    for name in "ABC":
        ax.axvline(snapped[name], color="white", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlim(-0.08, 1); ax.set_ylim(-0.6, 1.6); ax.set_yticks([])
    ax.set_xlabel("Re τ  (filled = on-lane zero, grey = off-lane zero)")
    ax.set_title("zero bookkeeping: merged carries A, B, C; erased carries A, C", fontsize=11)

    fig.suptitle("Edit a model with arithmetic: multiply to merge separately trained detectors, divide to remove one\n"
                 "(zero sets: exact; spike readout: measured)", fontsize=13, y=0.995)
    out = Path(__file__).parent.parent / "figures" / "45_hsln_skill_forge.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
