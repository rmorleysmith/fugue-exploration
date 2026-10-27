"""A fixed landscape with a trained linear encoder: reservoir computing with a CM-symbol code.

WHAT IT COMPUTES — The landscape is FIXED: F has zeros at the two CM representatives
    A = (0.5, 0.5) ~ i (j = 1728) and D = (0.5, 0.866) = ρ (j = 0), plus one lower zero as a
    basin floor. What is trained is a linear encoder (W ∈ ℝ^{2×2}, b ∈ ℝ², six parameters)
    from the 2-D input to the initial state, through the unit-speed descent on |F|² (150
    steps), with a cross-entropy on the distances of the endpoint to A and to D. Two such
    coordinates, each a binary classifier on one input axis, form a block-diagonal degree-2
    state; decoding each endpoint (reduce → j → nearest of {1728, 0}) gives a 2-bit codeword
    AA, AD, DA, DD, i.e. four classes. Data: four Gaussian blobs at the quadrant centres,
    240 points, 160 for training and 80 held out.
WHAT THE FIGURE SHOWS — Left: the decoded class over the input plane with the training
    points (dots) and the held-out points (triangles). Right: where the held-out inputs stop
    on each coordinate, beside the two CM representatives.
WHAT IS PROVED / KNOWN — The zeros are the attractors of the flow (`Fugue.isLocalMin_norm_of_eq_zero`,
    ledger A9); j(i) = 1728, j(ρ) = 0 (`Fugue.E₆_I`, `Fugue.E₄_rho`, ledger A3). Training only a
    readout or encoder around a fixed dynamical system is reservoir computing (Jaeger 2001).
    The accuracy reached is a measurement (ledger F8: 95% on 80 held-out points).
WHAT IS NOT CLAIMED — That the landscape learned anything: it is fixed by hand, and the June
    phrase "learned classifier" over-reads six encoder parameters (ledger F8). The held-out
    set shows only that six parameters do not memorise 160 points, which is no surprise.
    Four Gaussian blobs in the quadrants are separable by their signs; the task is easy by
    construction. Nothing is modular about F: it is a cubic in q with chosen roots.
HOW THIS WAS FOUND — We asked how to use the two reachable CM symbols (figure 24) for more
    than two classes, and the picture was a code: two coordinates, two bits. Keeping the
    landscape fixed and training the encoder was the pragmatic move that trained cleanly;
    the precise description of what that is — reservoir computing — is stated here rather
    than hidden.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, lax, vmap, value_and_grad

import modular

jax.config.update("jax_enable_x64", True)

A_REP = (0.5, 0.5)                      # j = 1728  (~ i)
D_REP = (0.5, float(np.sqrt(3) / 2))    # j = 0     (rho)
TARGET = {0: jnp.array(A_REP), 1: jnp.array(D_REP)}   # bit 0 -> A, bit 1 -> D
SYM = {0: "A", 1: "D"}


# --- energy and relaxation flow (degree 1) --------------------------------- #
def energy(coeffs, xy):
    q = jnp.exp(2j * jnp.pi * (xy[0] + 1j * xy[1]))
    f = jnp.sum(coeffs * q ** jnp.arange(coeffs.shape[0]))
    return jnp.real(f * jnp.conj(f))


_genergy = grad(energy, argnums=1)


def roll(coeffs, xy0, T=150, dt=0.02):
    def step(xy, _):
        g = _genergy(coeffs, xy)
        gn = g / (jnp.sqrt(g[0] ** 2 + g[1] ** 2) + 1e-4)
        nx = jnp.clip(xy[0] - dt * gn[0], 0.02, 0.98)
        ny = jnp.clip(xy[1] - dt * gn[1], 0.05, 0.95)
        n = jnp.array([nx, ny])
        return n, n
    _, tr = lax.scan(step, jnp.asarray(xy0), jnp.arange(T))
    return jnp.mean(tr[-25:], axis=0)


# FIXED landscape: zeros of F at the two CM representatives A=(0.5,0.5) and D=(0.5,0.866),
# plus a lower one for a stable basin floor. Only the input encoder is learned.
_Z = np.array([0.5 + 0.5j, 0.5 + 0.866j, 0.5 + 0.30j])
_C = np.poly(np.exp(2j * np.pi * _Z))[::-1]
_C = _C / np.max(np.abs(_C))
FIXED_COEFFS = jnp.asarray(_C.astype(complex))


# --- params = linear input encoder (W 2x2, b 2) ---------------------------- #
def _unpack(p):
    return p[:4].reshape(2, 2), p[4:6]


def _encode(W, b, x):
    # Im restricted to [0.45, 0.90], where j is monotone: ~1728 (A) near Im 0.5,
    # falling to 0 (D) by Im 0.85. (Below 0.45 j drops again, so we avoid it.)
    z = W @ x + b
    return jnp.array([0.5 + 0.3 * jnp.tanh(z[0]), 0.675 + 0.225 * jnp.tanh(z[1])])


def _endpoint(p, x):
    W, b = _unpack(p)
    return roll(FIXED_COEFFS, _encode(W, b, x))


def _loss(p, X, bits, beta=15.0):
    """Cross-entropy on (distance to A, distance to D): rewards landing nearer the correct
    representative than the wrong one, so it cannot minimise by collapsing to the midpoint."""
    ends = vmap(lambda x: _endpoint(p, x))(X)
    dA = jnp.sum((ends - TARGET[0]) ** 2, axis=1)
    dD = jnp.sum((ends - TARGET[1]) ** 2, axis=1)
    logits = -beta * jnp.stack([dA, dD], axis=1)
    logp = logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)
    ll = logp[jnp.arange(ends.shape[0]), bits]
    return -jnp.mean(ll) + 1e-4 * jnp.sum(p ** 2)


def _init(seed):
    rng = np.random.default_rng(seed)
    # warm start: first input axis -> Im (the axis separating A from D)
    enc = np.array([0.0, 0.0, 1.5, 0.0, 0.0, 0.0]) + 0.1 * rng.standard_normal(6)
    return jnp.array(enc)


# Per-coordinate input "view": each bit-classifier sees its discriminating axis first.
V1 = [1, 0]   # bit1 separates on x2 -> present (x2, x1)
V2 = [0, 1]   # bit2 separates on x1 -> present (x1, x2)


def train_bit(X, bits, seed=0, steps=800, lr=0.02):
    X = jnp.asarray(X)
    bits = jnp.asarray(bits, int)
    lg = jax.jit(value_and_grad(_loss))
    p = _init(seed)
    m = jnp.zeros_like(p); v = jnp.zeros_like(p)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        L, g = lg(p, X, bits)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        p = p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
    return p, float(L)


def decode_bit(p, x):
    """Run the flow, reduce tau*, read j, snap to A(1728)/D(0) -> bit."""
    end = np.array(_endpoint(p, jnp.asarray(x, float)))
    j = complex(modular.jinv(np.array([end[0] + 1j * end[1]]))[0]).real
    return 0 if abs(j - 1728) < abs(j - 0.0) else 1, end


def train_bit_best(X, bits, seeds=(1, 2, 3, 4, 5)):
    """Try several seeds (the CE landscape has a symmetric ln2 plateau); keep the best."""
    best_p, best_a = None, -1.0
    for s in seeds:
        p, _ = train_bit(X, bits, seed=s)
        a = sum(decode_bit(p, X[i])[0] == bits[i] for i in range(len(X))) / len(X)
        if a > best_a:
            best_p, best_a = p, a
        if a > 0.9:
            break
    return best_p, best_a


# --------------------------------------------------------------------------- #
def make_data(n=240, seed=3):
    """4 Gaussian blobs at the quadrant centres; class = 2*b1 + b2 where
    b1 = (x2>0), b2 = (x1>0). So class codeword (b1,b2) in {A,D}^2."""
    rng = np.random.default_rng(seed)
    centers = {(0, 0): (-0.55, -0.55), (0, 1): (0.55, -0.55),
               (1, 0): (-0.55, 0.55), (1, 1): (0.55, 0.55)}
    X, B1, B2, C = [], [], [], []
    per = n // 4
    for (b1, b2), ctr in centers.items():
        pts = rng.normal(ctr, 0.26, size=(per, 2))
        X.append(pts); B1 += [b1] * per; B2 += [b2] * per
        C += [2 * b1 + b2] * per
    X = np.concatenate(X)
    return X, np.array(B1), np.array(B2), np.array(C)


def run(make_fig=True):
    X, B1, B2, C = make_data()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    tr, te = idx[:160], idx[160:]

    p1, a1 = train_bit_best(X[tr][:, V1], B1[tr])
    p2, a2 = train_bit_best(X[tr][:, V2], B2[tr])
    print(f"fixed landscape, 6 encoder parameters per coordinate")
    print(f"bit1 train acc {a1:.3f}   bit2 train acc {a2:.3f}")

    def predict(x):
        b1, _ = decode_bit(p1, np.asarray(x)[V1])
        b2, _ = decode_bit(p2, np.asarray(x)[V2])
        return 2 * b1 + b2, (b1, b2)

    def acc(split):
        return sum(predict(X[i])[0] == C[i] for i in split) / len(split)

    def bit_acc(p, view, split, bits):
        return sum(decode_bit(p, X[i][view])[0] == bits[i] for i in split) / len(split)
    print(f"bit1 held-out acc {bit_acc(p1, V1, te, B1):.3f}   "
          f"bit2 held-out acc {bit_acc(p2, V2, te, B2):.3f}")

    tr_acc, te_acc = acc(tr), acc(te)
    print(f"train acc {tr_acc:.3f}   held-out acc {te_acc:.3f}  (4-class, 80 points)")
    assert te_acc >= 0.85, "held-out accuracy below 85%"
    for i in te[:5]:
        c, (b1, b2) = predict(X[i])
        print(f"  x={np.round(X[i],2)}  ->  {SYM[b1]}{SYM[b2]}  = class {c}  (true {C[i]})")

    if make_fig:
        _figure(p1, p2, X, C, tr, te, te_acc)
    return te_acc


def _figure(p1, p2, X, C, tr, te, te_acc):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import style  # noqa: F401

    CCOL = ["#46f0d0", "#ffd166", "#c0a3ff", "#ff7b6b"]
    CNAME = ["AA", "AD", "DA", "DD"]

    fig, ax = plt.subplots(1, 2, figsize=(14.0, 6.0), gridspec_kw={"width_ratios": [1, 1]})

    a = ax[0]
    g = 46
    gx = np.linspace(-1.3, 1.3, g)
    grid = np.array([[xi, yi] for yi in gx for xi in gx])
    e1 = np.array(vmap(lambda x: _endpoint(p1, x))(jnp.asarray(grid[:, V1])))
    e2 = np.array(vmap(lambda x: _endpoint(p2, x))(jnp.asarray(grid[:, V2])))
    j1 = modular.jinv(e1[:, 0] + 1j * e1[:, 1]).real
    j2 = modular.jinv(e2[:, 0] + 1j * e2[:, 1]).real
    gb1 = (np.abs(j1 - 1728) >= np.abs(j1 - 0.0)).astype(int)
    gb2 = (np.abs(j2 - 1728) >= np.abs(j2 - 0.0)).astype(int)
    gc = (2 * gb1 + gb2).reshape(g, g)
    a.imshow(gc, origin="lower", extent=[-1.3, 1.3, -1.3, 1.3], alpha=0.30,
             cmap=ListedColormap(CCOL), vmin=0, vmax=3, aspect="auto")
    for cls in range(4):
        m = C[tr] == cls
        a.scatter(X[tr][m, 0], X[tr][m, 1], s=18, color=CCOL[cls], edgecolor="white",
                  lw=0.3, zorder=3)
        mt = C[te] == cls
        a.scatter(X[te][mt, 0], X[te][mt, 1], s=42, marker="^", color=CCOL[cls],
                  edgecolor="black", lw=0.5, zorder=4)
    a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.3)
    a.set_xlabel("input $x_1$"); a.set_ylabel("input $x_2$")
    a.set_title(f"decision regions of the trained encoders  (held-out acc {te_acc*100:.0f}%)\n"
                "● train   ▲ held-out", fontsize=12)

    b = ax[1]
    b.set_facecolor("#0b0b14")
    for p_set, view, mk, lab in [(p1, V1, "o", "coordinate 1 (bit 1)"), (p2, V2, "s", "coordinate 2 (bit 2)")]:
        ends = np.array([np.array(_endpoint(p_set, jnp.asarray(X[i][view], float))) for i in te])
        jv = modular.jinv(ends[:, 0] + 1j * ends[:, 1]).real
        col = np.where(np.abs(jv - 1728) < np.abs(jv), "#46f0d0", "#ffd166")
        b.scatter(ends[:, 0], ends[:, 1], s=26, marker=mk, c=col, edgecolor="#222",
                  lw=0.3, zorder=3, label=lab)
    b.plot(*A_REP, "*", color="#46f0d0", ms=26, mec="white", mew=1.0, zorder=6)
    b.plot(*D_REP, "*", color="#ffd166", ms=26, mec="white", mew=1.0, zorder=6)
    b.annotate("A  (j=1728)", A_REP, textcoords="offset points", xytext=(12, -2),
               color="#46f0d0", fontsize=10)
    b.annotate("D  (j=0)", D_REP, textcoords="offset points", xytext=(12, -2),
               color="#ffd166", fontsize=10)
    txt = "code:  " + "   ".join(f"{n}={i}" for i, n in enumerate(CNAME))
    b.text(0.02, 0.06, txt, transform=b.transAxes, color="#bbb", fontsize=9)
    b.set_xlim(0.2, 0.95); b.set_ylim(0.35, 0.95)
    b.set_xlabel(r"$\mathrm{Re}\,\tau$"); b.set_ylabel(r"$\mathrm{Im}\,\tau$")
    b.set_title("each coordinate stops at a CM symbol of the FIXED landscape;\nthe pair is the class codeword",
                fontsize=12)
    b.legend(loc="upper right", fontsize=8.5, framealpha=0.3)

    fig.suptitle("Reservoir computing with a CM-symbol code: the landscape is fixed, a 6-parameter linear "
                 "encoder per coordinate is trained,\ntwo binary coordinates encode 4 classes (AA, AD, DA, DD)",
                 fontsize=12.5, y=1.03)
    out = Path(__file__).parent.parent / "figures" / "25_cm_classifier.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
