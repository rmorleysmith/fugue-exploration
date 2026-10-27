"""Foundations: E4 = Theta_E8, its energy landscape and flow, the zero-union law, degree 2 (figures 01-13).

WHAT IT COMPUTES — E4(tau) = Theta_E8(tau) = 1 + 240 sum sigma_3(n) q^n on the upper half-plane H
    and on the Poincare disk (`modular`, `geom`, `coloring`); the energy |F|^2 of E4 and of the
    truncated newform 11.4.a.a; the Euclidean gradient flow tau' = -alpha grad|F|^2 for designed
    q-polynomials F = prod (q - q_i) (`dna.dna_from_neurons`), both as unit-speed direction-field
    paths and as the actual ODE; sums and products of such polynomials and their zeros; the 240
    roots of E8 in the Coxeter plane and the coefficients 240 sigma_3(n); the degree-2 Siegel
    theta series Theta^(2)_E8(Z) = sum_T r2(T) exp(pi i tr(TZ)) (`siegel`) on complex-line slices
    of H_2, and its zero set on the slice family Z = [[tau, z], [z, tau + w]].
WHAT THE FIGURE SHOWS —
    01  phase portrait of E4 on H with the fundamental domain and the zeros at tau = rho (mod 1).
    02  the same on the Poincare disk; the real axis is the rim.
    03  log10 |F|^2 as a surface: E4 (one well per fundamental domain) and the truncated
        11.4.a.a (four wells).
    04  left, gradient-flow paths coloured by the zero they reach; right, the unforced ODE from
        three seeds, with |F(tau(t))| and the exact speed |tau'| = 2 alpha |F||F'|, which
        vanishes at the zero.
    05  F_A, F_B and F_A F_B: the product's zeros are the union of the factors' zeros.
    06  the 240 roots of E8 projected to the Coxeter plane (8 rings of 30), and the coefficients
        240 sigma_3(n).
    07  Theta^(2)_E8 on three slices of H_2: diag(tau, tau) (= E4(tau)^2), a coupled slice, and
        the off-diagonal plane.
    08  the flow paths alone, for 3 and for 10 prescribed zeros.
    09  the zero set of Theta^(2) on the slice family Z = [[tau, z], [z, tau + w]], decoupled
        (z = 0) and coupled (z = 0.35).
    10  the same for four couplings z: the swept curve bends and reconnects.
    11  F_A + F_B against F_A F_B: the sum recombines the zeros, the product unions them.
    12  Theta_{L+M} = Theta_L Theta_M, illustrated with two q-polynomials: a fixed factor times a
        second factor.
    13  E4 on the disk with no axes (a cover image).
WHAT IS PROVED / KNOWN — Theta_E8 = E4 (Serre, *A Course in Arithmetic*, ch. VII; Lean
    `Fugue.thetaE8_eq_E₄`, `Fugue.reprCount_e8`; audit ledger A2). E4 has one zero per fundamental
    domain, at rho = exp(2 pi i/3), simple (valence formula; Lean `Fugue.E₄_rho`; A3). For
    holomorphic non-constant F the local minima of |F| are exactly its zeros (minimum-modulus
    principle; Lean `Fugue.eq_zero_of_isLocalMin_norm`, `Fugue.isLocalMin_norm_of_eq_zero`; A9),
    and the Euclidean gradient of |F|^2 has modulus 2|F||F'|, so the flow freezes at zeros (A11).
    zeros(F G) = zeros(F) u zeros(G) in any integral domain (`mul_eq_zero`; B1), and multiplying
    q-series is the Cauchy product of coefficient vectors (B2). Theta_{L+M} = Theta_L Theta_M
    (Lean `Fugue.theta_directSum`, `Fugue.theta_e8_sum_e8`; C13) and Theta_{E8+E8} = E4^2 = E8
    (Lean `Fugue.thetaE8_sq_eq_E₈`). Theta^(2)_E8 is a Siegel modular form of weight 4 for Sp(4,Z)
    (Freitag, *Siegelsche Modulfunktionen*, 1983), equal to E4(tau1) E4(tau2) on the diagonal.
    The coupling matrix of the degree-2 flow is positive definite iff |kappa| < 1 (Lean
    `Fugue.couplingMatrix_posDef_iff`; E2).
WHAT IS NOT CLAIMED — the designed coefficient vectors of figures 04, 05, 08, 11, 12 are
    polynomials in q, not modular forms (no weight-8 level-1 form has four distinct zeros; ledger
    B5), so those figures illustrate the zero-union law for polynomials. Nothing here is trained,
    and no "spiking" or "liquid" behaviour is shown: the June version of figure 04 drove the
    state with a force 10^7 times the landscape gradient and thresholded 1/|F| at a percentile of
    its own trace (ledger A12); that panel is replaced by the unforced flow. The zero set of
    Theta^(2) is a divisor of complex dimension 2, not a curve (A13); figures 09-10 show its
    trace on a two-real-parameter slice family and keep only the roots with Im tau in a window
    where the norm <= 8 truncation is accurate. The number of roots of a truncated q-series is
    its degree, not a count of anything about the form (A7, A8).
HOW THIS WAS FOUND — we imagined a dynamical system whose state is a point of the upper
    half-plane and whose "energy" is a modular form, with the zeros of the form as the resting
    places; the precise statement is the minimum-modulus principle, which makes the zeros the
    only attractors of gradient descent on |F|^2. Counting E8 vectors shell by shell gave 240,
    2160, 6720 = 240 sigma_3(n), so the "DNA" of the E8 landscape was the Eisenstein series E4.
    We imagined merging two trained landscapes by multiplying them and saw every zero kept; the
    precise statement is that a product vanishes where a factor does. We imagined the degree-2
    Siegel space as two coupled copies with the off-diagonal entry as a "gravity" dial; the
    precise statement is that Z12 = 0 factorises the theta series as E4 (x) E4 and Z12 != 0 is
    the genuine period, with positivity |kappa| < 1. What did not survive was the bookkeeping that
    turned these facts into a reservoir with a controllable number of zeros.

Run from `scripts/`: `python render.py all`, or `python render.py <name> ...` with a name from
`FIGURES`. Writes `../figures/<NN>_<name>.png`, prints its checks (each one an assertion), and
exits 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import coloring
import dna
import e8
import geom
import modular
import siegel
import style

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
FIGDIR.mkdir(exist_ok=True)
RHO = np.exp(2j * np.pi / 3.0)  # the zero of E4: one per fundamental domain
CYAN = "#46f0d0"
DIM = "#9a98aa"

_REP: dict[int, dict] = {}  # representation numbers of E8 at degree 2, by norm cutoff


def _rep(maxn: int) -> dict:
    if maxn not in _REP:
        _REP[maxn] = siegel.representation_numbers(maxn=maxn)
    return _REP[maxn]


def _check(label: str, ok: bool, detail: str = "") -> None:
    """Print a check and assert it."""
    print(f"check  {label}: {'OK' if ok else 'FAILED'}  {detail}")
    assert ok, f"{label}: {detail}"


def _save(fig, name, **kw):
    path = FIGDIR / f"{name}.png"
    fig.savefig(path, **kw)
    plt.close(fig)
    print(f"wrote {path}")
    return path


# ============================================================ checks

def checks():
    """The classical facts the figures rest on, re-derived numerically."""
    v = abs(modular.e4(RHO))
    _check("E4(rho) = 0", v < 1e-9, f"|E4(rho)| = {v:.1e}")
    t = 0.3 + 0.7j
    res = abs(modular.e4(-1.0 / t) - t ** 4 * modular.e4(t))
    _check("E4(-1/tau) = tau^4 E4(tau)", res < 1e-9, f"residual {res:.1e}")
    sh = e8.shells(6.0)
    counts = {k: int(sh[k].shape[0]) for k in (2, 4, 6)}
    _check("E8 shell counts = 240 sigma3(n)", counts == {2: 240, 4: 2160, 6: 6720}, f"{counts}")
    a = e8.theta1_coeffs(3)
    _check("Theta_E8 coefficients 1, 240, 2160, 6720",
           np.array_equal(a, [1.0, 240.0, 2160.0, 6720.0]), f"{a.astype(int).tolist()}")


# ============================================================ degree 1: E4 = Theta_E8

def fig_e4_halfplane(res=1500):
    """01: phase portrait of E4 = Theta_E8 over the upper half-plane."""
    TAU, extent = geom.half_plane_grid((-2.0, 2.0), (0.045, 2.3), res)
    F = modular.e4(TAU)
    rgb = coloring.domain_color(F, cmap="twilight")

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.imshow(rgb, extent=extent, origin="upper", interpolation="none", aspect="auto")

    fre, fim = coloring.fundamental_domain_path()
    ax.plot(fre, fim, color=style.PAPER, lw=1.0, alpha=0.5)
    for shift in (-1, 0, 1):
        ax.plot(RHO.real + shift, RHO.imag, "o", color=style.NEURON, ms=7,
                mec="white", mew=0.8, zorder=5)
    ax.text(-1.95, 2.08, r"$\bullet$  zeros of $E_4$:  $\tau\equiv\rho=e^{2\pi i/3}$,"
            " one per fundamental domain", color=style.NEURON,
            ha="left", va="top", fontsize=10)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(r"Re $\tau$")
    ax.set_ylabel(r"Im $\tau$")
    ax.set_title("The theta series of E8 is the Eisenstein series of weight 4:   "
                 r"$\Theta_{E_8}(\tau) = E_4(\tau)$" "\n"
                 r"phase portrait on the upper half-plane  $\mathcal{H}$",
                 fontsize=14)
    fig.text(0.5, -0.02,
             "hue = arg F   ·   brightness bands = level curves of |F|   ·   "
             "fundamental domain of SL(2,Z) outlined",
             ha="center", fontsize=9, color=DIM)
    return _save(fig, "01_e4_halfplane")


def _e4_disk_rgba(res):
    W, mask, extent = geom.disk_grid(res)
    TAU = geom.inv_cayley(W)
    F = np.full(W.shape, np.nan, dtype=np.complex128)
    F[mask] = modular.e4(TAU[mask])
    rgb = coloring.domain_color(F, cmap="twilight")
    return np.dstack([rgb, mask.astype(float)]), mask, extent


def fig_e4_disk(res=1500):
    """02: E4 = Theta_E8 on the Poincare disk via the Cayley transform."""
    rgba, _, extent = _e4_disk_rgba(res)

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    ax.imshow(rgba, extent=extent, origin="upper", interpolation="none")
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=style.PAPER, lw=1.2, alpha=0.6))

    fre, fim = coloring.fundamental_domain_path()
    fd = geom.cayley(fre + 1j * fim)
    ax.plot(fd.real, fd.imag, color=style.PAPER, lw=1.0, alpha=0.55)
    for shift in (-1, 0, 1):
        wz = geom.cayley(RHO + shift)
        ax.plot(wz.real, wz.imag, "o", color=style.NEURON, ms=7, mec="white",
                mew=0.8, zorder=5)

    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"$\Theta_{E_8} = E_4$ on the Poincaré disk" "\n"
                 r"real axis of $\mathcal{H}\to$ the rim," "\n"
                 r"where the cusps $\mathbb{Q}\cup\{\infty\}$ lie",
                 fontsize=14, loc="left", x=0.02)

    wheel = fig.add_axes([0.80, 0.80, 0.16, 0.16], projection="polar")
    style.phase_wheel(wheel)
    return _save(fig, "02_e4_disk")


# ============================================================ energy landscape and zeros

def fig_energy_zeros(res=200):
    """03: the energy |F|^2 as a surface; its minima are exactly the zeros."""
    fig = plt.figure(figsize=(14, 6.3))

    # ---- left: E4, one well per fundamental domain, at rho ----
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    TAU, _ = geom.half_plane_grid((-1.0, 1.0), (0.45, 1.5), res)
    F = modular.e4(TAU)
    Z = np.log10(np.abs(F) ** 2 + 1e-3)
    rgba = np.dstack([coloring.domain_color(F, cmap="twilight"), np.ones(F.shape)])
    ax1.plot_surface(TAU.real, TAU.imag, Z, facecolors=rgba, rstride=2, cstride=2,
                     linewidth=0, antialiased=False, shade=False)
    floor = Z.min()
    for shift in (-0.5, 0.5):
        ax1.scatter([shift], [RHO.imag], [floor], color=style.NEURON, s=60,
                    edgecolor="white", depthshade=False, zorder=10)
    ax1.set_title(r"$E_4=\Theta_{E_8}$:  energy $\log_{10}|F|^2$" "\n"
                  r"one zero per fundamental domain, at $\rho$", fontsize=13)
    ax1.set_xlabel(r"Re $\tau$"); ax1.set_ylabel(r"Im $\tau$")
    ax1.set_zlabel(r"$\log_{10}|F|^2$")
    ax1.view_init(elev=52, azim=-58)

    # ---- right: the level-11 newform 11.4.a.a truncated to n <= 5 ----
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    TAU2, _ = geom.half_plane_grid((0.0, 1.0), (0.02, 0.62), res)
    F2 = modular.qpoly(dna.LEVEL11_4, TAU2)
    Z2 = np.log10(np.abs(F2) ** 2 + 1e-3)
    rgba2 = np.dstack([coloring.domain_color(F2, cmap="twilight"), np.ones(F2.shape)])
    ax2.plot_surface(TAU2.real, TAU2.imag, Z2, facecolors=rgba2, rstride=2, cstride=2,
                     linewidth=0, antialiased=False, shade=False)
    zeros = modular.qpoly_zeros(dna.LEVEL11_4)
    floor2 = Z2.min()
    ax2.scatter(zeros.real, zeros.imag, np.full(zeros.size, floor2),
                color=style.NEURON, s=55, edgecolor="white", depthshade=False, zorder=10)
    ax2.set_title(r"newform 11.4.a.a truncated to $n\leq5$" "\n"
                  rf"$F=\sum_{{n\leq5}}a_n q^n$:  {zeros.size} zeros of the $q$-polynomial",
                  fontsize=13)
    ax2.set_xlabel(r"Re $\tau$"); ax2.set_ylabel(r"Im $\tau$")
    ax2.set_zlabel(r"$\log_{10}|F|^2$")
    ax2.view_init(elev=52, azim=-58)
    _check("truncated 11.4.a.a has 4 roots in the q-disk", zeros.size == 4, f"{zeros.size}")

    fig.suptitle("The energy landscape |F|² of a holomorphic F: its minima are exactly the zeros "
                 "(minimum-modulus principle)", fontsize=14, y=1.06)
    return _save(fig, "03_energy_zeros")


# ============================================================ gradient flow

def _velocity(coeffs, tau, alpha=1.0):
    """Velocity of tau' = -alpha grad|F|^2 (Euclidean gradient in (Re tau, Im tau))."""
    F, Fp = modular.qpoly_grad(coeffs, tau)
    g = np.conj(F) * Fp
    return -alpha * 2.0 * np.real(g), alpha * 2.0 * np.imag(g), F


def integrate_paths(coeffs, seeds, zeros=None, n_steps=700, dt=0.012,
                    im_floor=0.03, stop_tol=0.02):
    """Vectorised unit-speed paths along the direction of -grad|F|^2, for all seeds at once.

    Following the direction at unit speed gives the same orbits (hence the same basins) as the
    ODE, is stable at any scale of the coefficient vector, and discards the ODE's own clock;
    the right panel of figure 04 shows that clock. Particles freeze once within `stop_tol` of a
    zero. Cost O(n_steps * P * len(coeffs)). Returns history (n_steps + 1, 2, P).
    """
    state = np.stack([seeds.real.astype(float), seeds.imag.astype(float)])
    hist = np.empty((n_steps + 1, 2, state.shape[1]))
    hist[0] = state
    active = np.ones(state.shape[1], dtype=bool)

    def direction(s):
        vx, vy, _ = _velocity(coeffs, s[0] + 1j * s[1])
        v = np.stack([vx, vy])
        return v / (np.hypot(vx, vy) + 1e-12)

    for k in range(n_steps):
        d = direction(state)
        d[:, ~active] = 0.0
        new = state + dt * d
        new[1] = np.clip(new[1], im_floor, im_floor + 3.0)
        bad = ~np.isfinite(new)
        new[bad] = state[bad]
        state = new
        if zeros is not None:
            ends = state[0] + 1j * state[1]
            mind = np.min(np.abs(ends[:, None] - zeros[None, :]), axis=1)
            active &= mind > stop_tol
        hist[k + 1] = state
    return hist


def basin_palette(n):
    return plt.colormaps["turbo"](np.linspace(0.06, 0.95, n))


def _basin_paths(ax, coeffs, zeros, re_lim, im_lim, nx=34, ny=22, tol=0.05):
    """Draw flow paths coloured by the zero they reach, and the zeros as stars."""
    xs = np.linspace(re_lim[0] + 0.01, re_lim[1] - 0.01, nx)
    ys = np.linspace(im_lim[0] + 0.01, im_lim[1] - 0.01, ny)
    SX, SY = np.meshgrid(xs, ys)
    seeds = (SX + 1j * SY).ravel()
    hist = integrate_paths(coeffs, seeds, zeros=zeros, im_floor=im_lim[0])
    ends = hist[-1, 0] + 1j * hist[-1, 1]
    dist = np.abs(ends[:, None] - zeros[None, :])
    nearest = np.argmin(dist, axis=1)
    captured = dist[np.arange(ends.size), nearest] < tol
    pal = basin_palette(len(zeros))
    for p in range(seeds.size):
        if captured[p]:
            ax.plot(hist[:, 0, p], hist[:, 1, p], color=pal[nearest[p]], lw=0.7, alpha=0.7)
        else:
            ax.plot(hist[:, 0, p], hist[:, 1, p], color=(0.5, 0.5, 0.55), lw=0.5, alpha=0.2)
    for i, z in enumerate(zeros):
        ax.plot(z.real, z.imag, "*", color=pal[i], ms=19, mec="white", mew=1.0, zorder=6)
    ax.set_xlim(*re_lim); ax.set_ylim(*im_lim)
    ax.set_facecolor("#07070d")
    n_cap = int(captured.sum())
    print(f"       basins: {n_cap} of {seeds.size} seeds reached a zero within {tol}")
    return n_cap


def fig_basins():
    """08: zeros and flow paths only, no field on a grid; 3 zeros and 10 zeros."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3))

    n3 = np.array([0.30 + 0.46j, 0.68 + 0.52j, 0.50 + 0.27j])
    c3 = dna.dna_from_neurons(n3)
    cap3 = _basin_paths(axes[0], c3, n3, (0.0, 1.0), (0.05, 0.92), nx=36, ny=24)
    axes[0].set_title("3 prescribed zeros — basins traced by flow paths", fontsize=13)
    axes[0].set_xlabel(r"Re $\tau$"); axes[0].set_ylabel(r"Im $\tau$")

    re = np.linspace(0.08, 0.92, 10)
    nrow = re + 1j * (0.30 + 0.05 * np.sin(2 * np.pi * re))
    crow = dna.dna_from_neurons(nrow)
    cap10 = _basin_paths(axes[1], crow, nrow, (0.0, 1.0), (0.08, 0.52), nx=64, ny=26)
    axes[1].set_title(f"{nrow.size} prescribed zeros — the same vectorised integration",
                      fontsize=13)
    axes[1].set_xlabel(r"Re $\tau$")
    _check("flow paths reach zeros (fig 08)", cap3 > 0 and cap10 > 0, f"{cap3}, {cap10}")

    fig.suptitle(r"Zeros and gradient-flow paths of $q$-polynomials $\prod(q-q_i)$ "
                 "(direction field at unit speed; no field evaluated on a grid)",
                 fontsize=14, y=1.0)
    return _save(fig, "08_basins_paths")


def _flow_ode(coeffs, seed, alpha, T=30.0, n=4000):
    """Integrate the ODE tau' = -alpha grad|F|^2 itself (not reparametrised) from one seed.

    Samples are geometric in time, because arrival times differ by three orders of magnitude
    between a seed near the real axis (|F| large, fast) and one high in the strip (F nearly
    constant, slow).
    """
    def rhs(t, s):
        vx, vy, _ = _velocity(coeffs, s[0] + 1j * s[1], alpha=alpha)
        return [vx, vy]
    ts = np.concatenate([[0.0], np.geomspace(1e-3, T, n)])
    sol = solve_ivp(rhs, [0.0, T], [seed.real, seed.imag], t_eval=ts,
                    max_step=0.01, rtol=1e-9, atol=1e-12)
    tau_t = sol.y[0] + 1j * sol.y[1]
    F, Fp = modular.qpoly_grad(coeffs, tau_t)
    predicted = 2.0 * alpha * np.abs(F) * np.abs(Fp)   # exact: |tau'| = 2 alpha |F| |F'|
    vx, vy, _ = _velocity(coeffs, tau_t, alpha=alpha)
    measured = np.hypot(vx, vy)
    return sol.t, tau_t, np.abs(F), predicted, measured


def fig_dynamics():
    """04: basins (left) and the ODE's own clock (right): the speed law 2 alpha |F||F'|."""
    zeros = np.array([0.30 + 0.46j, 0.68 + 0.52j, 0.50 + 0.27j])
    coeffs = dna.dna_from_neurons(zeros)

    fig = plt.figure(figsize=(14, 6.2))

    # ---- left: basin paths ----
    ax1 = fig.add_subplot(1, 2, 1)
    _basin_paths(ax1, coeffs, zeros, (0.0, 1.0), (0.05, 0.92), nx=36, ny=24)
    ax1.set_xlabel(r"Re $\tau$"); ax1.set_ylabel(r"Im $\tau$")
    ax1.set_title(r"Gradient flow  $\dot\tau=-\alpha\,\nabla|F|^2$  (Euclidean gradient)"
                  "\npaths coloured by the zero they reach · ★ = zeros", fontsize=13)

    # ---- right: the actual ODE from three seeds; one global clock ----
    # Rescaling the coefficient vector by c multiplies every rate by |c|^2, so alpha is one
    # clock for the whole landscape: chosen here so that the linearised relaxation time
    # 1/(2 alpha |F'(z_0)|^2) at the first zero is 0.5.
    _, Fp0 = modular.qpoly_grad(coeffs, zeros)
    alpha = 1.0 / (2.0 * 0.5 * np.abs(Fp0[0]) ** 2)
    ax2 = fig.add_subplot(1, 2, 2)
    seeds = [0.15 + 0.60j, 0.95 + 0.40j, 0.50 + 0.12j]
    pal = basin_palette(len(zeros))
    worst_ratio = 0.0
    for s in seeds:
        t, tau_t, absF, pred, meas = _flow_ode(coeffs, s, alpha)
        worst_ratio = max(worst_ratio, float(np.max(np.abs(meas - pred) / (pred.max() + 1e-300))))
        end = tau_t[-1]
        k = int(np.argmin(np.abs(zeros - end)))
        dist = float(abs(zeros[k] - end))
        _check(f"ODE from {s:.2f} reaches zero {k + 1}", dist < 1e-6, f"distance {dist:.1e}")
        ax2.loglog(t[1:], absF[1:], color=pal[k], lw=1.6,
                   label=rf"$|F(\tau(t))|$, seed {s:.2f} → zero {k + 1}")
        ax2.loglog(t[1:], pred[1:], color=pal[k], lw=1.0, ls="--", alpha=0.8)
    _check("speed law |tau'| = 2 alpha |F||F'| along the ODE", worst_ratio < 1e-9,
           f"max relative deviation {worst_ratio:.1e}")
    ax2.plot([], [], color=style.PAPER, lw=1.0, ls="--", label=r"speed $|\dot\tau|=2\alpha|F||F'|$")
    ax2.set_ylim(1e-12, 10.0)
    ax2.set_xlabel("time  (log scale; one clock for the whole landscape)")
    ax2.set_ylabel(r"$|F|$  and  $|\dot\tau|$")
    ax2.set_title("The unforced ODE from three seeds\n"
                  r"speed $=2\alpha|F||F'|$ exactly: fast where $|F|$ is large, frozen at the zero",
                  fontsize=13)
    ax2.legend(loc="lower left", fontsize=8, framealpha=0.25)
    fig.suptitle(r"Three prescribed zeros of a $q$-polynomial: the basins, and the flow's own clock",
                 fontsize=14, y=1.0)
    return _save(fig, "04_dynamics")


# ============================================================ products union zero sets

def _phase_panel(ax, coeffs, zeros, title, re_lim=(0.0, 1.0), im_lim=(0.05, 0.95), res=700):
    TAU, extent = geom.half_plane_grid(re_lim, im_lim, res)
    F = modular.qpoly(coeffs, TAU)
    rgb = coloring.domain_color(F, cmap="twilight")
    ax.imshow(rgb, extent=extent, origin="upper", interpolation="none", aspect="auto")
    ax.plot(zeros.real, zeros.imag, "*", color=CYAN, ms=15, mec="black", mew=0.7, zorder=6)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(r"Re $\tau$")
    ax.set_title(title, fontsize=12)


def _union_check(label, product_zeros, parents):
    d = np.min(np.abs(product_zeros[:, None] - parents[None, :]), axis=1)
    ok = product_zeros.size == parents.size and float(d.max()) < 1e-6
    _check(label, ok, f"{product_zeros.size} zeros, max distance to a parent zero {d.max():.1e}")


def fig_merge(res=700):
    """05: multiplying two q-polynomials unions their zero sets."""
    nA = np.array([0.28 + 0.40j, 0.70 + 0.58j])
    nB = np.array([0.50 + 0.30j, 0.15 + 0.62j])
    A = dna.dna_from_neurons(nA)
    B = dna.dna_from_neurons(nB)
    M = np.convolve(A, B)  # Cauchy product of the coefficient vectors = product of the forms
    nM = modular.qpoly_zeros(M)
    _union_check("zeros(F_A F_B) = zeros(F_A) u zeros(F_B) (fig 05)", nM, np.concatenate([nA, nB]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    _phase_panel(axes[0], A, nA, r"$F_A$:  a $q$-polynomial with 2 prescribed zeros", res=res)
    _phase_panel(axes[1], B, nB, r"$F_B$:  2 other prescribed zeros", res=res)
    _phase_panel(axes[2], M, nM,
                 r"product $F_A\cdot F_B$"
                 "\n"
                 r"zeros$(F_A F_B)=$ zeros$(F_A)\cup$ zeros$(F_B)$", res=res)
    axes[0].set_ylabel(r"Im $\tau$")
    fig.suptitle("Multiplying two forms unions their zero sets — the coefficient vectors "
                 "convolve (Cauchy product)", fontsize=14, y=1.02)
    fig.text(0.5, -0.02,
             f"checked: the product has {nM.size} zeros = A's {nA.size} + B's {nB.size}, "
             "each within 1e−6 of a parent zero  ·  cyan ★ = zeros",
             ha="center", fontsize=9, color=DIM)
    return _save(fig, "05_merge")


# ============================================================ E8 roots and theta coefficients

def fig_e8_roots():
    """06: the 240 roots of E8 in the Coxeter plane, and the coefficients 240 sigma_3(n)."""
    fig = plt.figure(figsize=(13.5, 6.6))

    ax1 = fig.add_subplot(1, 2, 1)
    roots = e8.shells(2.0)[2]
    P = e8.coxeter_plane(roots)
    r = np.hypot(P[:, 0], P[:, 1])
    rings = sorted(set(np.round(r, 4)))
    _check("240 roots fall on 8 rings in the Coxeter plane",
           roots.shape[0] == 240 and len(rings) == 8, f"{roots.shape[0]} roots, {len(rings)} rings")
    cmap = plt.colormaps["plasma"]
    for i, rad in enumerate(rings):
        sel = np.isclose(r, rad, atol=1e-3)
        pts = P[sel]
        ang = np.arctan2(pts[:, 1], pts[:, 0])
        order = np.argsort(ang)
        loop = np.vstack([pts[order], pts[order][:1]])
        col = cmap(i / (len(rings) - 1))
        ax1.plot(loop[:, 0], loop[:, 1], color=col, lw=0.8, alpha=0.7)
        ax1.scatter(pts[:, 0], pts[:, 1], color=col, s=22, edgecolor="white", lw=0.3, zorder=4)
    ax1.set_aspect("equal"); ax1.axis("off")
    ax1.set_title("The 240 roots of E8\nCoxeter-plane (Petrie) projection: "
                  "8 rings × 30 = 240", fontsize=13)

    ax2 = fig.add_subplot(1, 2, 2)
    a = e8.theta1_coeffs(8)
    n = np.arange(a.size)
    ax2.bar(n, a, color=style.ACCENT, edgecolor="white", lw=0.5)
    ax2.set_yscale("symlog")
    for ni, ai in zip(n, a):
        ax2.text(ni, ai * 1.25 + 1, f"{int(ai)}", ha="center", fontsize=8, color=style.PAPER)
    ax2.set_xlabel("n  (Fourier index)")
    ax2.set_ylabel(r"$a_n$  (symlog)")
    ax2.set_title(r"$\Theta_{E_8}=E_4=1+240\sum_{n\geq1}\sigma_3(n)\,q^n$"
                  "\nthe Fourier coefficients $a_n = 240\\,\\sigma_3(n)$ "
                  "(the June files' “DNA”)", fontsize=13)
    ax2.grid(True, alpha=0.15)
    return _save(fig, "06_e8_roots_coefficients")


# ============================================================ degree 2: Siegel slices

def fig_siegel(res=620):
    """07: the truncated Theta^(2)_E8 over complex-line slices of H_2."""
    dct = _rep(6)

    # the diagonal factorisation, checked at one point against truncated E4 (x) E4
    tau1, tau2 = 0.2 + 1.1j, -0.1 + 1.3j
    a_e4 = e8.theta1_coeffs(3)
    q1, q2 = np.exp(2j * np.pi * tau1), np.exp(2j * np.pi * tau2)
    rhs = sum(a_e4[k] * q1 ** k for k in range(4)) * sum(a_e4[k] * q2 ** k for k in range(4))
    diag_res = abs(siegel.theta2(dct, tau1, 0.0, tau2) - rhs)
    _check("Theta^(2) on the diagonal = E4(tau1) E4(tau2) (truncations, norm <= 6)",
           diag_res < 1e-7, f"residual {diag_res:.1e}")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))

    # slice 1: Z = diag(tau, tau) -> E4(tau)^2
    TAU, ext = geom.half_plane_grid((-0.6, 0.6), (0.45, 1.45), res)
    F1 = siegel.theta2(dct, TAU, np.zeros_like(TAU), TAU)
    axes[0].imshow(coloring.domain_color(F1, cmap="twilight"), extent=ext,
                   origin="upper", interpolation="none", aspect="auto")
    axes[0].set_title(r"$Z=\mathrm{diag}(\tau,\tau)$" "\n"
                      r"$\Theta^{(2)}=E_4(\tau)^2$  (decoupled)", fontsize=12)
    axes[0].set_ylabel(r"Im $\tau$"); axes[0].set_xlabel(r"Re $\tau$")

    # slice 2: Z = [[tau, z0],[z0, tau]] with a real off-diagonal coupling z0
    z0 = 0.35
    F2 = siegel.theta2(dct, TAU, np.full_like(TAU, z0), TAU)
    axes[1].imshow(coloring.domain_color(F2, cmap="twilight"), extent=ext,
                   origin="upper", interpolation="none", aspect="auto")
    axes[1].set_title(rf"$Z=[[\tau,{z0}],[{z0},\tau]]$" "\n"
                      r"the coupling $z$ deforms the picture", fontsize=12)
    axes[1].set_xlabel(r"Re $\tau$")

    # slice 3: the off-diagonal plane. Z = [[i s, z],[z, i s]], vary z
    s = 1.15
    Zx = np.linspace(-0.6, 0.6, res)
    Zy = np.linspace(0.9, -0.9, res)
    ZX, ZY = np.meshgrid(Zx, Zy)
    zz = ZX + 1j * ZY
    diag = 1j * s
    F3 = siegel.theta2(dct, np.full_like(zz, diag), zz, np.full_like(zz, diag))
    pd = (np.abs(zz.imag) < s)  # Im(Z) positive definite <=> |Im z| < s
    F3 = np.where(pd, F3, np.nan)
    axes[2].imshow(coloring.domain_color(F3, cmap="twilight"),
                   extent=(-0.6, 0.6, -0.9, 0.9), origin="upper", interpolation="none",
                   aspect="auto")
    axes[2].set_title(rf"$Z=[[{s}i,\,z],[z,\,{s}i]]$,  vary $z$" "\n"
                      r"the coupling plane  ($\mathrm{Im}\,Z\succ0$ iff $|\mathrm{Im}\,z|<"
                      rf"{s}$)", fontsize=12)
    axes[2].set_xlabel(r"Re $z$"); axes[2].set_ylabel(r"Im $z$")

    fig.suptitle(r"The degree-2 Siegel theta series of E8,  $\Theta^{(2)}_{E_8}(Z)$,  on "
                 r"slices of $\mathcal{H}_2$ (complex dimension 3; weight 4 for Sp(4,$\mathbb{Z}$))",
                 fontsize=14, y=1.02)
    fig.text(0.5, -0.02, "Fourier terms with E8 vector norms ≤ 6 (exact representation numbers)  ·  "
             f"diagonal slice checked against E₄⊗E₄: residual {diag_res:.0e}",
             ha="center", fontsize=9, color=DIM)
    return _save(fig, "07_siegel_degree2")


def degree2_divisor_points(dct, z0, ws, im_lo=0.6, im_hi=1.3):
    """Zeros of the degree-2 form on the slice Z = [[tau, z0],[z0, tau + w]] for each w.

    On this slice F is a polynomial in q = exp(2 pi i tau) of degree max((a + c)/2); its roots are
    the points where the zero divisor meets the slice. Returns (N, 3): (Re tau, Im tau, w), keeping
    only roots with im_lo < Im tau < im_hi, the window in which the norm cutoff is accurate.
    """
    a = np.array([k[0] for k in dct], float)
    b = np.array([k[1] for k in dct], float)
    c = np.array([k[2] for k in dct], float)
    r = np.array([dct[k] for k in dct], float)
    npow = ((a + c) / 2).astype(int)
    deg = int(npow.max())
    base = r * np.exp(2j * np.pi * b * z0)
    pts = []
    for w in ws:
        vals = base * np.exp(1j * np.pi * c * w)
        coeff = np.zeros(deg + 1, dtype=complex)
        np.add.at(coeff, npow, vals)
        q = np.roots(coeff[::-1])
        q = q[np.abs(q) > 1e-12]
        tau = np.log(q) / (2j * np.pi)
        t = tau[(tau.imag > im_lo) & (tau.imag < im_hi)]
        re = (t.real + 0.5) % 1.0 - 0.5
        pts.extend(zip(re, t.imag, np.full(t.size, w)))
    return np.array(pts) if pts else np.empty((0, 3))


def fig_divisor(nw=200):
    """09: the zero set of Theta^(2) on a slice family; decoupled and coupled."""
    dct = _rep(8)
    ws = np.linspace(-1.0, 1.0, nw)
    fig = plt.figure(figsize=(14.5, 6.6))
    for idx, (z0, lab) in enumerate([(0.0, r"decoupled  ($z = 0$):  $\Theta^{(2)}=E_4(\tau)\,E_4(\tau+w)$,"
                                           "\nso the zeros sit at Im τ = √3/2 for every w"),
                                     (0.35, r"coupled  ($z = 0.35$):  the branch bends and reconnects")]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        P = degree2_divisor_points(dct, z0, ws)
        if z0 == 0.0:
            dev = float(np.max(np.abs(P[:, 1] - np.sqrt(3) / 2)))
            _check("decoupled slice: zeros at Im tau = sqrt(3)/2 for every w", dev < 1e-6,
                   f"max deviation {dev:.1e} over {P.shape[0]} points")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=P[:, 2], cmap="viridis", s=7)
        w0 = ws[np.argmin(np.abs(ws))]
        m0 = np.isclose(P[:, 2], w0)
        ax.scatter(P[m0, 0], P[m0, 1], P[m0, 2], color=style.NEURON, s=55,
                   edgecolor="white", depthshade=False, zorder=8,
                   label="one complex slice → point(s)")
        ax.set_xlabel(r"Re $\tau$"); ax.set_ylabel(r"Im $\tau$"); ax.set_zlabel("w  (slice)")
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(0.6, 1.0); ax.set_zlim(-1, 1)
        ax.set_title(lab, fontsize=11)
        ax.view_init(elev=20, azim=-62)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(r"Degree 2: the zero set of $\Theta^{(2)}_{E_8}$ in $\mathcal{H}_2$ is a divisor "
                 r"(complex dimension 2)" "\n"
                 r"on the slice family $Z=[[\tau,z],[z,\tau+w]]$ it meets each complex slice in "
                 r"points (red) and, swept over $w$, traces a curve", fontsize=13, y=1.1)
    fig.text(0.5, -0.02, "Fourier terms with E8 vector norms ≤ 8  ·  roots kept with 0.6 < Im τ < 1.3 "
             "(2 of the 8 roots per slice; the rest lie where the truncation is not accurate)  ·  "
             "in the decoupled panel the lines are straight; any apparent bend is the 3-D projection",
             ha="center", fontsize=9, color=DIM)
    return _save(fig, "09_degree2_divisor")


def fig_divisor_sweep(nw=220):
    """10: the degree-2 zero set as the coupling z grows."""
    dct = _rep(8)
    ws = np.linspace(-1.0, 1.0, nw)
    z0s = [0.10, 0.25, 0.40, 0.55]
    fig = plt.figure(figsize=(13.5, 11.5))
    for i, z0 in enumerate(z0s):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        P = degree2_divisor_points(dct, z0, ws, im_lo=0.5, im_hi=1.45)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=P[:, 2], cmap="plasma", s=7)
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(0.5, 1.1); ax.set_zlim(-1, 1)
        ax.set_xlabel(r"Re $\tau$"); ax.set_ylabel(r"Im $\tau$"); ax.set_zlabel("w")
        ax.set_title(f"coupling  z = {z0:.2f}", fontsize=12)
        ax.view_init(elev=20, azim=-62)
    fig.suptitle("The degree-2 zero set on the slice family Z = [[τ, z], [z, τ+w]] as the coupling z "
                 "grows — the swept curve bends and reconnects",
                 fontsize=14, y=0.99)
    fig.text(0.5, 0.01, "Fourier terms with E8 vector norms ≤ 8  ·  roots kept with 0.5 < Im τ < 1.45",
             ha="center", fontsize=9, color=DIM)
    return _save(fig, "10_divisor_sweep")


# ============================================================ sum versus product

def _compose_panel(ax, coeffs, stars, hollow, title, res=720):
    re_lim, im_lim = (0.0, 1.0), (0.05, 0.95)
    TAU, ext = geom.half_plane_grid(re_lim, im_lim, res)
    F = modular.qpoly(coeffs, TAU)
    ax.imshow(coloring.domain_color(F), extent=ext, origin="upper",
              interpolation="none", aspect="auto")
    if hollow is not None and len(hollow):
        ax.plot(hollow.real, hollow.imag, "o", mfc="none", mec="#cfcfcf", mew=1.4,
                ms=14, zorder=5)
    if stars is not None and len(stars):
        ax.plot(stars.real, stars.imag, "*", color=CYAN, ms=16, mec="black",
                mew=0.7, zorder=6)
    ax.set_xlim(*re_lim); ax.set_ylim(*im_lim)
    ax.set_title(title, fontsize=12)


def fig_compose(res=720):
    """11: addition recombines the zeros; multiplication unions them."""
    nA = np.array([0.30 + 0.55j, 0.66 + 0.40j])
    nB = np.array([0.50 + 0.62j, 0.20 + 0.34j, 0.80 + 0.50j])
    A = dna.dna_from_neurons(nA)
    B = dna.dna_from_neurons(nB)
    L = max(A.size, B.size)
    Apad = np.zeros(L, complex); Apad[:A.size] = A
    Bpad = np.zeros(L, complex); Bpad[:B.size] = B
    S = Apad + Bpad
    P = np.convolve(A, B)
    nS = modular.qpoly_zeros(S)
    nP = modular.qpoly_zeros(P)
    union = np.concatenate([nA, nB])
    _union_check("zeros(F_A F_B) = zeros(F_A) u zeros(F_B) (fig 11)", nP, union)
    dS = np.min(np.abs(nS[:, None] - union[None, :]), axis=1)
    _check("zeros(F_A + F_B) are not the parents' zeros (fig 11)", float(dS.min()) > 1e-3,
           f"closest sum-zero to a parent zero: {dS.min():.3f}")

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 11))
    _compose_panel(ax[0, 0], A, nA, None, r"$F_A$  ($\bigstar$ = its zeros)", res)
    _compose_panel(ax[0, 1], B, nB, None, r"$F_B$", res)
    _compose_panel(ax[1, 0], S, nS, union,
                   r"$F_A + F_B$  (sum): the zeros recombine — new $\bigstar$, parents' ○ gone", res)
    _compose_panel(ax[1, 1], P, nP, None,
                   r"$F_A \cdot F_B$  (product): the zeros union — every parent zero kept", res)
    ax[0, 0].set_ylabel(r"Im $\tau$"); ax[1, 0].set_ylabel(r"Im $\tau$")
    ax[1, 0].set_xlabel(r"Re $\tau$"); ax[1, 1].set_xlabel(r"Re $\tau$")
    fig.suptitle("Two ways to combine coefficient vectors:  addition recombines the zeros · "
                 "multiplication unions them", fontsize=14, y=1.0)
    return _save(fig, "11_compose_add_vs_mul")


def fig_direct_sum(res=720):
    """12: Theta_{L+M} = Theta_L Theta_M, illustrated with two q-polynomials."""
    nS = np.array([0.25 + 0.52j, 0.52 + 0.66j, 0.76 + 0.46j, 0.40 + 0.30j])  # fixed factor
    nG = np.array([0.35 + 0.42j, 0.66 + 0.38j])                              # second factor
    FS = dna.dna_from_neurons(nS)
    FG = dna.dna_from_neurons(nG)
    FP = np.convolve(FS, FG)  # Cauchy product of the coefficient vectors; zeros union
    _union_check("zeros(F_S F_G) = zeros(F_S) u zeros(F_G) (fig 12)",
                 modular.qpoly_zeros(FP), np.concatenate([nS, nG]))

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.7))
    _compose_panel(ax[0], FS, nS, None, r"fixed factor $F_S$  (4 prescribed zeros)", res)
    _compose_panel(ax[1], FG, nG, None, r"second factor $F_G$  (2 prescribed zeros)", res)
    _compose_panel(ax[2], FP, None, None,
                   r"product $F_S\!\cdot\!F_G$  (coefficient vectors convolved)", res)
    ax[2].plot(nS.real, nS.imag, "*", color=CYAN, ms=15, mec="black", mew=0.7, zorder=6,
               label=r"zeros of $F_S$")
    ax[2].plot(nG.real, nG.imag, "*", color=style.ACCENT, ms=15, mec="black", mew=0.7,
               zorder=6, label=r"zeros of $F_G$")
    ax[2].legend(loc="upper right", fontsize=8, framealpha=0.3)
    ax[0].set_ylabel(r"Im $\tau$")
    for a in ax:
        a.set_xlabel(r"Re $\tau$")
    fig.suptitle(r"A direct sum of lattices multiplies theta series:  "
                 r"$\Theta_{L\oplus M}=\Theta_L\,\Theta_M$,  so  $\Theta_{E_8\oplus E_8}=E_4^2=E_8$" "\n"
                 r"illustrated with two $q$-polynomials: a second factor reshapes the field of the "
                 r"first and  zeros$(F_S F_G)$ = zeros$(F_S)\,\cup\,$zeros$(F_G)$",
                 fontsize=13, y=1.03)
    return _save(fig, "12_heterotic_e8xe8")


def fig_disk_clean(res=900):
    """13: E4 on the disk with no axes or title (a cover image)."""
    rgba, _, _ = _e4_disk_rgba(res)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(rgba, origin="upper", interpolation="none")
    return _save(fig, "13_e4_disk_clean", dpi=res / 6, transparent=True)


FIGURES = {
    "e4_half": fig_e4_halfplane,
    "e4_disk": fig_e4_disk,
    "energy": fig_energy_zeros,
    "dynamics": fig_dynamics,
    "merge": fig_merge,
    "roots": fig_e8_roots,
    "siegel": fig_siegel,
    "basins": fig_basins,
    "divisor": fig_divisor,
    "divisor_sweep": fig_divisor_sweep,
    "compose": fig_compose,
    "direct_sum": fig_direct_sum,
    "disk_clean": fig_disk_clean,
}


def main(argv):
    names = list(FIGURES) if (not argv or argv[0] == "all") else argv
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        sys.exit(f"unknown figure(s) {unknown}; have {list(FIGURES)}")
    checks()
    for name in names:
        FIGURES[name]()
    print("render OK")


if __name__ == "__main__":
    main(sys.argv[1:])
