"""Train one single-zero model per class, separately; multiply the four forms; classify by
nearest zero. The zero union is exact; the classifier is nearest-centroid in q.

WHAT IT COMPUTES — Inputs x ∈ ℝ² are mapped by a fixed affine encoder to τ = 0.5 + 0.16x₁ +
    i(0.55 + 0.16x₂). For each class c the single-zero model F_c(q) = q − a_c is fitted
    alone on its own training points: the least-squares monic linear form has a_c = mean of
    the class's q-values, so its zero is the class centroid in the q-disk. The compound is
        F = F₀F₁F₂F₃,   coefficient vector = conv(a₀, a₁, a₂, a₃),
    and its zeros are recovered by polynomial root-finding and compared with the four parent
    zeros. Held-out inputs are classified by the nearest zero of the compound in τ.
WHAT THE FIGURE SHOWS — Left: decision regions of the composed model over the input plane
    with training (dots) and held-out (triangles) points. Right: the four parent zeros and
    the zeros of the product, which coincide.
WHAT IS PROVED / KNOWN — zeros(∏F_c) = ∪ zeros(F_c) (`mul_eq_zero`, ledger B1), with the
    coefficients given by the iterated Cauchy product; verified here to 1e−12 (the audit
    measured 1.76e−16, ledger B3). The nearest-zero rule on single-zero models is the
    nearest-centroid classifier in the q-plane, whose accuracy on Gaussian blobs is standard.
WHAT IS NOT CLAIMED — That composition contributed to the classification. The compound's
    zeros are exactly the parents' zeros, so "classify by nearest compound zero" is
    identical to "classify by nearest parent centroid"; the convolution is bookkeeping, and
    the 95% held-out accuracy is the centroid classifier's. No flow is run (nearest-zero is a
    detection readout, chosen precisely so that the basin cross-talk of figure 19 does not
    enter). This is the separate-training merge of system-spec §4.4(6) only in the weak sense
    of unioning detectors, not of composing behaviours (ledger F9).
HOW THIS WAS FOUND — We wanted to see the merge law do a job end to end: train pieces apart,
    multiply, use. The picture was four landmarks, one per class, placed by four people who
    never met, then put on one map by multiplying. The precise statement is the zero-union
    law; the honest reading of the result is that the map is a list of centroids.
"""
from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np

import modular
from cm_classifier import make_data

CNAME = ["AA", "AD", "DA", "DD"]
CCOL = ["#46f0d0", "#ffd166", "#c0a3ff", "#ff7b6b"]


def encode(X):
    """Fixed input->state map: input box -> a region of the strip."""
    X = np.atleast_2d(X)
    re = 0.5 + 0.16 * X[:, 0]
    im = 0.55 + 0.16 * X[:, 1]
    return re + 1j * im


def train_class(taus):
    """Single-zero model for one class, alone: the least-squares monic form q - a with |F|
    smallest on this class has a = mean(q_i). Returns (coefficient vector, zero as tau)."""
    q = np.exp(2j * np.pi * taus)
    a = q.mean()
    coeffs = np.array([-a, 1.0], dtype=complex)          # F(q) = q - a  (low->high)
    tau = (np.angle(a) % (2 * np.pi)) / (2 * np.pi) + 1j * (-np.log(abs(a)) / (2 * np.pi))
    return coeffs, tau


def run(make_fig=True):
    X, _B1, _B2, C = make_data()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    tr, te = idx[:160], idx[160:]
    Tau = encode(X)

    # --- fit each class model completely separately --- #
    vecs, zs = [], []
    for c in range(4):
        mask = tr[C[tr] == c]
        coeffs, tau = train_class(Tau[mask])
        vecs.append(coeffs); zs.append(tau)
        print(f"class {c} ({CNAME[c]}): zero tau={tau:.3f}  "
              f"j={complex(modular.jinv(np.array([tau]))[0]).real:8.1f}")
    zs = np.array(zs)

    # --- compose by multiplying the forms = convolving the coefficient vectors --- #
    compound = reduce(np.convolve, vecs)
    print(f"\ncompound coefficient vector of degree {len(compound)-1} (4 forms multiplied)")

    czeros = modular.qpoly_zeros(compound)
    czeros = czeros[(czeros.imag > 0.05) & (czeros.imag < 1.5)]
    err = max(min(abs(z - n) for z in czeros) for n in zs)
    print(f"compound zeros: {np.round(np.sort_complex(czeros),3)}")
    print(f"max |compound zero - parent zero| = {err:.2e}  (zero-set union, exact up to root-finding)")
    assert err < 1e-12 and len(czeros) == 4

    # --- classify held-out inputs by nearest zero = nearest centroid --- #
    def classify(taus):
        d = np.abs(taus[:, None] - zs[None, :])
        return d.argmin(axis=1)

    pred_tr = classify(Tau[tr]); pred_te = classify(Tau[te])
    tr_acc = (pred_tr == C[tr]).mean()
    te_acc = (pred_te == C[te]).mean()
    print(f"\ntrain acc {tr_acc:.3f}   held-out acc {te_acc:.3f}  (4-class, 80 points; "
          f"this is the nearest-centroid classifier in the q-plane)")
    assert te_acc >= 0.85

    if make_fig:
        _figure(X, C, tr, te, zs, czeros, te_acc)
    return te_acc


def _figure(X, C, tr, te, zs, czeros, te_acc):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import style  # noqa: F401

    fig, ax = plt.subplots(1, 2, figsize=(14.0, 6.0))

    a = ax[0]
    g = 60
    gx = np.linspace(-1.3, 1.3, g)
    grid = np.array([[xi, yi] for yi in gx for xi in gx])
    gt = encode(grid)
    gcls = np.abs(gt[:, None] - zs[None, :]).argmin(axis=1).reshape(g, g)
    a.imshow(gcls, origin="lower", extent=[-1.3, 1.3, -1.3, 1.3], alpha=0.30,
             cmap=ListedColormap(CCOL), vmin=0, vmax=3, aspect="auto")
    for cls in range(4):
        m = tr[C[tr] == cls]; mt = te[C[te] == cls]
        a.scatter(X[m, 0], X[m, 1], s=16, color=CCOL[cls], edgecolor="white", lw=0.3, zorder=3)
        a.scatter(X[mt, 0], X[mt, 1], s=42, marker="^", color=CCOL[cls],
                  edgecolor="black", lw=0.5, zorder=4)
    a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.3)
    a.set_xlabel("input $x_1$"); a.set_ylabel("input $x_2$")
    a.set_title(f"nearest zero of the composed model = nearest class centroid\n"
                f"(held-out acc {te_acc*100:.0f}%)   ● train   ▲ held-out", fontsize=12)

    b = ax[1]
    b.set_facecolor("#0b0b14")
    for c in range(4):
        b.plot(zs[c].real, zs[c].imag, "o", color=CCOL[c], ms=15, mfc="none",
               mew=2.5, zorder=4, label="parent zero" if c == 0 else None)
        b.annotate(CNAME[c], (zs[c].real, zs[c].imag),
                   textcoords="offset points", xytext=(10, 6), color=CCOL[c], fontsize=10)
    b.plot(czeros.real, czeros.imag, "x", color="white", ms=9, mew=2.0, zorder=5,
           label="zeros of the product")
    b.set_xlim(0.3, 0.7); b.set_ylim(0.38, 0.72)
    b.set_xlabel(r"$\mathrm{Re}\,\tau$"); b.set_ylabel(r"$\mathrm{Im}\,\tau$")
    b.set_title(r"$F=F_0 F_1 F_2 F_3$:  zeros = exact union" "\n"
                "(four coefficient vectors convolved into one)", fontsize=12)
    b.legend(loc="upper center", fontsize=8.5, framealpha=0.3)
    b.text(0.5, 0.04, "each parent fitted alone on its class;\ncompound assembled by one "
           "convolution", transform=b.transAxes, color="#aaa", fontsize=8.5, ha="center")

    fig.suptitle("Fit four single-zero models separately, multiply the forms, classify by nearest zero: "
                 "the zero union is exact,\nand the classifier it yields is nearest-centroid in the q-plane",
                 fontsize=12.5, y=1.03)
    out = Path(__file__).parent.parent / "figures" / "26_cm_tensor.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
