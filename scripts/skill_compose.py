"""Stacking two skills, find-missing then count: the answer is the roots-of-unity identity,
and the driven well only carries it.

WHAT IT COMPUTES — Six letters are encoded as the sixth roots of unity, ω^p = (cos 2πp/6,
    sin 2πp/6). Skill A: the "bag" of present letters is summed to a drive vector D =
    Σ_present ω^p, and a particle confined by a single-zero well |q − q_c|² at (0.5, 0.5) is
    pushed by D for 600 steps; its displacement from the centre, negated and normalised,
    is the recovered direction of the missing letter. Skill B: a softmax regression from
    unit vectors to letter index, trained separately on noisy copies of the six directions.
    The pipeline B(A(bag)) is run for each of the six possible missing letters.
WHAT THE FIGURE SHOWS — Left: one problem; the particle settles toward the antipode of the
    missing letter and the negated displacement points at it. Right: the pipeline's answer
    against the truth for all six problems (6/6).
WHAT IS PROVED / KNOWN — Σ_{p=0}^{n−1} ω^p = 0 for ω a primitive n-th root of unity
    (`Fugue.sum_exp_eq`, ledger D12, is the general k·[k | N] form), so Σ_present ω^p = −ω^m
    exactly, verified here to 1e−15. Function composition of two separately trained maps is
    just function composition; nothing here is the multiplicative merge of figure 18.
WHAT IS NOT CLAIMED — That the reservoir computes anything (ledger F7, reservoir role
    REFUTED). The sum over the present letters is taken in numpy before the flow; the flow
    is a confined particle pushed by a constant vector, whose displacement direction is that
    vector's (the audit found the result invariant under a 100× change of drive). Skill A is
    the identity Σ ω^p = 0 with a decorative transport step, and the 6/6 is that identity's.
    The June remark that this circular encoding is what grokking networks discover is a
    pointer to figure 30, not a result here.
HOW THIS WAS FOUND — We wanted two skills trained apart and used in sequence, with a state
    representation native to the strip: angles. The picture was a bag of pulls that cancels
    when complete, so the missing pull shows as the net force. That picture is the identity,
    and once written down the reservoir has nothing left to do; we keep the script because
    the identity is the honest content of the June figure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import modular

N = 6
LETTERS = "ABCDEF"
CENTER = np.array([0.5, 0.5])
_qc = np.exp(2j * np.pi * (CENTER[0] + 1j * CENTER[1]))
COEFFS = np.array([-_qc, 1.0 + 0j])              # confining zero of F at CENTER


def roots(n):
    th = 2 * np.pi * np.arange(n) / n
    return np.stack([np.cos(th), np.sin(th)], axis=1)


DIR = roots(N)                                   # letter -> root-of-unity direction


# --------------------------------------------------------------------------- #
#  Skill A: a confined, driven particle whose displacement carries the bag sum   #
# --------------------------------------------------------------------------- #
def _grad_E(xy):
    F, Fp = modular.qpoly_grad(COEFFS, np.array([xy[0] + 1j * xy[1]]))
    F, Fp = F[0], Fp[0]
    cF = np.conj(F)
    return np.array([2 * np.real(cF * Fp), -2 * np.imag(cF * Fp)])


def find_missing(present, dirs=DIR, steps=600, dt=0.01, alpha=0.5, Bd=0.05):
    """Sum the present letters' directions (the actual computation), push the confined
    particle with that sum, and read the negated displacement as the missing direction."""
    D = dirs[present].sum(axis=0)                # the bag sum, taken here, outside the flow
    xy = CENTER.copy()
    traj = [xy.copy()]
    for _ in range(steps):
        xy = xy - dt * alpha * _grad_E(xy) + dt * Bd * D
        xy = np.clip(xy, [0.12, 0.22], [0.88, 0.78])
        traj.append(xy.copy())
    disp = xy - CENTER
    missing_dir = -disp / (np.linalg.norm(disp) + 1e-12)
    return xy, missing_dir, np.array(traj)


# --------------------------------------------------------------------------- #
#  Skill B: a separately trained counting readout (direction -> number)         #
# --------------------------------------------------------------------------- #
def train_count(positions, dirs=DIR, seed=0, steps=500, lr=0.4):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for p in positions:
        for _ in range(80):
            v = dirs[p] + 0.10 * rng.standard_normal(2)
            Xs.append(v / (np.linalg.norm(v) + 1e-9)); ys.append(p)
    Xs, ys = np.array(Xs), np.array(ys)
    W = 0.1 * rng.standard_normal((N, 2)); b = np.zeros(N)
    for _ in range(steps):
        Z = Xs @ W.T + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = P.copy(); G[np.arange(len(ys)), ys] -= 1
        W -= lr * (G.T @ Xs) / len(ys)
        b -= lr * G.mean(0)
    return W, b


def count(W, b, v):
    return int((W @ v + b).argmax())


# --------------------------------------------------------------------------- #
def run(make_fig=True):
    W, b = train_count(range(N))

    rows, trajs = [], {}
    correct = 0
    worst_angle = 0.0
    for m in range(N):
        present = [p for p in range(N) if p != m]
        # the identity that does the work: sum of present roots of unity = -(missing root)
        ident = np.linalg.norm(DIR[present].sum(axis=0) + DIR[m])
        assert ident < 1e-12, "roots-of-unity identity failed"
        xy, mdir, tr = find_missing(present)
        angle = np.degrees(np.arccos(np.clip(mdir @ DIR[m], -1, 1)))
        worst_angle = max(worst_angle, angle)
        num = count(W, b, mdir)            # stack: B(A(present))
        ok = num == m
        correct += ok
        rows.append((m, present, xy, mdir, num, ok))
        trajs[m] = (tr, mdir)
        seq = ",".join(LETTERS[p] for p in present)
        print(f"present [{seq}]  ->  missing {LETTERS[m]}  ->  pipeline says number {num + 1}  "
              f"{'OK' if ok else 'WRONG'}   (|Σ_present ω^p + ω^m| = {ident:.1e}, direction off by {angle:.1f}°)")
    print(f"\ncomposed (find-missing then count) accuracy: {correct}/{N}")
    assert correct == N
    assert worst_angle < 15.0, "the transported direction drifted from the bag sum"
    print("the answer is the identity Σ_present ω^p = −ω^m, taken before the flow; the well only "
          "transports it (ledger F7)")

    if make_fig:
        _figure(rows, trajs, W, b)
    return correct


def _figure(rows, trajs, W, b):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    TEAL, GOLD, NEURON, GREEN = "#46f0d0", "#ffd166", "#ff3b6b", "#7CFC8A"

    fig, ax = plt.subplots(1, 2, figsize=(13.8, 6.2))

    a = ax[0]
    a.set_facecolor("#0b0b14")
    m = 2                                            # missing letter C (position 2)
    present = [p for p in range(N) if p != m]
    th = 2 * np.pi * np.arange(361) / 360
    a.plot(CENTER[0] + 0.28 * np.cos(th), CENTER[1] + 0.28 * np.sin(th), color="#333", lw=1)
    for p in range(N):
        pt = CENTER + 0.28 * DIR[p]
        col = "#555" if p == m else TEAL
        a.plot(*pt, "o", color=col, ms=14, mec="white", mew=0.6, zorder=4)
        a.annotate(LETTERS[p] + ("  (missing)" if p == m else ""), pt,
                   textcoords="offset points", xytext=(10, 6), color=col, fontsize=10)
    tr, mdir = trajs[m]
    a.plot(tr[:, 0], tr[:, 1], color=GOLD, lw=1.6, zorder=3, alpha=0.9)
    a.annotate("", xy=tr[-1], xytext=tuple(CENTER),
               arrowprops=dict(arrowstyle="->", color=GOLD, lw=2))
    a.annotate("net pull of present letters\n= −(missing root of unity)", tr[-1],
               textcoords="offset points", xytext=(-4, -34), color=GOLD, fontsize=8.5)
    rec = CENTER + 0.28 * mdir
    a.annotate("", xy=rec, xytext=tuple(CENTER),
               arrowprops=dict(arrowstyle="->", color=NEURON, lw=2.2))
    a.annotate(f"negated → '{LETTERS[m]}' = number {count(W,b,mdir)+1}", rec,
               textcoords="offset points", xytext=(-6, -26), color=NEURON, fontsize=9)
    a.plot(*CENTER, "+", color="white", ms=10)
    a.set_xlim(0.12, 0.92); a.set_ylim(0.16, 0.86)
    a.set_aspect("equal")
    a.set_xlabel(r"$\mathrm{Re}\,\tau$"); a.set_ylabel(r"$\mathrm{Im}\,\tau$")
    a.set_title("skill A (find-missing): the bag sum of roots of unity,\ncarried by a confined driven particle",
                fontsize=12)

    b_ax = ax[1]
    truth = [r[0] + 1 for r in rows]
    pred = [r[4] + 1 for r in rows]
    okall = all(r[5] for r in rows)
    b_ax.plot(range(1, N + 1), truth, "o", color="#888", ms=16, mfc="none", mew=2,
              label="missing letter's true number")
    b_ax.plot(range(1, N + 1), pred, "x", color=GREEN, ms=12, mew=2.5,
              label="pipeline output (A→B)")
    for r in rows:
        b_ax.annotate(LETTERS[r[0]], (r[0] + 1, r[0] + 1), textcoords="offset points",
                      xytext=(8, -4), color=TEAL, fontsize=10)
    b_ax.plot([0.5, N + 0.5], [0.5, N + 0.5], color="#333", lw=0.8, zorder=0)
    b_ax.set_xticks(range(1, N + 1)); b_ax.set_yticks(range(1, N + 1))
    b_ax.set_xlabel("problem (which letter is missing)")
    b_ax.set_ylabel("answer: its number")
    b_ax.set_title(f"stacked: find-missing ∘ count  →  {'all correct' if okall else 'errors'}\n"
                   "(exact because Σ of all n-th roots of unity is 0)", fontsize=12)
    b_ax.legend(loc="upper left", fontsize=9, framealpha=0.3)

    fig.suptitle("Function composition (stacking): skill A is the identity Σ_present ω^p = −ω^missing, "
                 "skill B (trained apart) names the number;\nthe driven well transports the answer, "
                 "it does not compute it", fontsize=12.5, y=1.02)
    out = Path(__file__).parent.parent / "figures" / "29_skill_compose.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
