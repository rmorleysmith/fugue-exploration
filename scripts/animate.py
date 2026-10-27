"""Animations of the landscape (needs ffmpeg; without it the script says so and exits 0).

WHAT IT COMPUTES — Run `python animate.py <name|all>`. Each animation writes an .mp4 to
    ../figures/ with matplotlib's FFMpegWriter:
      degree2         : the degree-2 theta series Θ^{(2)}_{E8} of siegel.py on the slice
                        Z = [[τ, z],[z, τ]], domain-coloured in τ while the off-diagonal z
                        runs round a loop of radius 0.45 in ℂ.
      tweak           : one coefficient of a degree-5 coefficient vector is scaled by 10^s,
                        s ∈ [−2.2, 2.2], and the roots move in the q-disk; the count inside
                        |q| < 1 changes as roots cross the circle.
      sector_dial     : a frozen factor F_state with four zeros times a second factor whose
                        two zeros orbit; the product's zeros are the union of both, so the
                        product field reshapes around the moving pair (the June file called
                        the second factor "gravity").
      flow, divisor_orbit, sector_divisor : need the renderer `render.py` of the Foundations
                        family (its path integrator and degree-2 divisor sampler); they
                        import it lazily and are skipped with a message if it is absent.
WHAT THE FIGURE SHOWS — No static figure; the June file had no figure number. Each frame is
    the corresponding still of figures 12 (degree-2 slice), the q-disk root plots, and 18.
WHAT IS PROVED / KNOWN — Θ_{L⊕M} = Θ_L·Θ_M (`Fugue.theta_directSum`, ledger C13) is what the
    `sector_dial` product illustrates; zeros(FG) = zeros(F) ∪ zeros(G) (`mul_eq_zero`,
    ledger B1); the roots of a polynomial depend continuously on its coefficients, and the
    number inside the unit disk changes only when a root crosses |q| = 1 (`tweak`).
WHAT IS NOT CLAIMED — That the number of roots inside the q-disk is "the number of neurons"
    of a modular form: it is the number of roots of a truncation (THE-STORY §3). No
    animation is checked numerically; this script is illustration only.
HOW THIS WAS FOUND — Still pictures of the degree-2 slice (figure 12) made us want to see
    the off-diagonal move; the picture was the coupling as a dial. The precise content is
    the product law for theta series of direct sums; the dial itself is a parameter sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

import coloring
import dna
import geom
import modular
import siegel
import style

FIGDIR = Path(__file__).parent.parent / "figures"


def _render():
    try:
        import render
        return render
    except ImportError:
        print("render.py (Foundations family) is not present; skipping this animation")
        return None


def degree2(frames=96, res=420, fps=20):
    dct = siegel.representation_numbers(maxn=6)
    TAU, ext = geom.half_plane_grid((-0.6, 0.6), (0.45, 1.45), res)
    tau_flat = TAU.ravel()

    # Theta^(2)(diag tau,tau; off-diag z) = sum_T r_T e^{i pi (a+c) tau} e^{i 2 pi b z}.
    terms = list(dct.items())
    P = np.empty((len(terms), tau_flat.size), dtype=np.complex128)
    bs = np.empty(len(terms))
    for i, ((a, b, c), r) in enumerate(terms):
        P[i] = r * np.exp(1j * np.pi * (a + c) * tau_flat)
        bs[i] = b

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    img = ax.imshow(np.zeros((res, res, 3)), extent=ext, origin="upper",
                    interpolation="none", aspect="auto")
    ax.set_xlabel(r"Re $\tau$"); ax.set_ylabel(r"Im $\tau$")
    ttl = ax.set_title("")

    def update(k):
        z = 0.45 * np.exp(2j * np.pi * k / frames)
        w = np.exp(1j * 2 * np.pi * bs * z)
        F = (w @ P).reshape(TAU.shape)
        img.set_data(coloring.domain_color(F, cmap="twilight"))
        ttl.set_text(r"$\Theta^{(2)}_{E8}$ on $Z=[[\tau,z],[z,\tau]]$,  "
                     f"$z={z.real:+.2f}{z.imag:+.2f}i$")
        return [img, ttl]

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = FIGDIR / "anim_genus2_coupling.mp4"
    anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=4500), dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def flow(fps=24, n_zeros=3):
    """Precompute paths once, animate dots flowing along them into the zeros."""
    render = _render()
    if render is None:
        return

    if n_zeros == 3:
        zs = np.array([0.30 + 0.46j, 0.68 + 0.52j, 0.50 + 0.27j])
        coeffs = dna.dna_from_neurons(zs, lead=40.0)
    else:
        gx, gy = np.meshgrid(np.linspace(0.16, 0.84, 4), np.linspace(0.30, 0.66, 3))
        zs = (gx + 1j * gy).ravel()
        coeffs = dna.dna_from_neurons(zs, lead=1.0)

    re_lim, im_lim = (0.0, 1.0), (0.05, 0.90)
    xs = np.linspace(0.02, 0.98, 46)
    ys = np.linspace(0.08, 0.88, 30)
    SX, SY = np.meshgrid(xs, ys)
    seeds = (SX + 1j * SY).ravel()

    n_steps = 700
    hist = render.integrate_paths(coeffs, seeds, neurons=zs, n_steps=n_steps, im_floor=im_lim[0])
    ends = hist[-1, 0] + 1j * hist[-1, 1]
    dist = np.abs(ends[:, None] - zs[None, :])
    nearest = np.argmin(dist, axis=1)
    captured = dist[np.arange(ends.size), nearest] < 0.05
    pal = render.basin_palette(len(zs))

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    ax.set_facecolor("#07070d")
    for p in range(seeds.size):
        col = pal[nearest[p]] if captured[p] else (0.5, 0.5, 0.55, 0.18)
        ax.plot(hist[:, 0, p], hist[:, 1, p], color=col, lw=0.5, alpha=0.45)
    ax.plot(zs.real, zs.imag, "*", color="white", ms=20, mec="black", mew=0.8, zorder=8)
    dot_colors = np.array([pal[nearest[p]] if captured[p] else (0.7, 0.7, 0.75, 1)
                           for p in range(seeds.size)])
    scat = ax.scatter(hist[0, 0], hist[0, 1], s=14, c=dot_colors, edgecolor="white",
                      linewidth=0.2, zorder=9)
    ax.set_xlim(*re_lim); ax.set_ylim(*im_lim)
    ax.set_xlabel(r"Re $\tau$"); ax.set_ylabel(r"Im $\tau$")
    ax.set_title(r"Relaxation along $\dot\tau=-\alpha\,\nabla|F|^2$"
                 "\nparticles flow down the paths into the zeros of $F$")

    step = 3

    def update(k):
        i = min(k * step, n_steps)
        scat.set_offsets(np.column_stack([hist[i, 0], hist[i, 1]]))
        return [scat]

    anim = FuncAnimation(fig, update, frames=n_steps // step + 1, blit=False)
    out = FIGDIR / "anim_reservoir_flow.mp4"
    anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=3500), dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def tweak(frames=120, fps=20):
    """Scale one coefficient and watch the roots move in the q-disk.

    The degree d caps the root count, and as the coefficient changes roots cross |q| = 1
    (enter or leave the disk) and collide: the count inside is not a free dial even though
    the coefficients are.
    """
    base = np.array([0.10 + 0.09j, 0.32 + 0.07j, 0.55 + 0.13j,
                     0.78 + 0.08j, 0.45 + 0.20j])  # near the |q|=1 edge
    c0 = dna.dna_from_neurons(base)
    d = c0.size - 1
    half = frames // 2
    ss = np.concatenate([np.linspace(-2.2, 2.2, half),
                         np.linspace(2.2, -2.2, frames - half)])
    all_roots = []
    for s in ss:
        c = c0.copy()
        c[0] = c0[0] * (10.0 ** s)
        all_roots.append(np.roots(c[::-1]))
    bg = np.concatenate(all_roots)

    fig, ax = plt.subplots(figsize=(6.6, 6.8))
    ax.set_facecolor("#07070d")
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="white", lw=1.3, alpha=0.75)
    ax.scatter(bg.real, bg.imag, s=3, color=(0.55, 0.55, 0.62), alpha=0.12, zorder=2)
    ax.plot(0, 0, "+", color="#888", ms=9)
    sc_in = ax.scatter([], [], s=90, color="#46f0d0", edgecolor="white", lw=0.6, zorder=5)
    sc_out = ax.scatter([], [], s=45, color="#777", edgecolor="none", zorder=4)
    txt = ax.text(-1.38, -1.34, "", color="white", fontsize=11)
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_aspect("equal")
    ax.set_xlabel("Re q"); ax.set_ylabel("Im q")
    ax.set_title(rf"degree-{d} coefficient vector · scaling one coefficient" "\n"
                 r"inside |q|<1 = a zero in $\mathbb{H}$ · outside = gone (Im $\tau\leq0$)", fontsize=12)

    def update(k):
        r = all_roots[k]
        inside = np.abs(r) < 1.0
        sc_in.set_offsets(np.column_stack([r[inside].real, r[inside].imag]))
        sc_out.set_offsets(np.column_stack([r[~inside].real, r[~inside].imag]))
        txt.set_text(f"roots inside |q|<1: {int(inside.sum())} / {d}")
        return [sc_in, sc_out, txt]

    anim = FuncAnimation(fig, update, frames=len(all_roots), blit=False)
    out = FIGDIR / "anim_tweak_zeros.mp4"
    anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=3000), dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def divisor_orbit(fps=24, frames=144):
    """Orbit the camera around the coupled degree-2 zero divisor curve."""
    render = _render()
    if render is None:
        return

    dct = siegel.representation_numbers(maxn=8)
    ws = np.linspace(-1.0, 1.0, 240)
    P = render.genus2_divisor_points(dct, 0.35, ws, im_lo=0.5, im_hi=1.45)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=P[:, 2], cmap="viridis", s=9)
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(0.5, 1.1); ax.set_zlim(-1, 1)
    ax.set_xlabel(r"Re $\tau$"); ax.set_ylabel(r"Im $\tau$"); ax.set_zlabel("w")
    ax.set_title("Degree-2 zero divisor (coupled, z = 0.35): the zero set is a curve")

    def update(k):
        ax.view_init(elev=18, azim=-62 + 360.0 * k / frames)
        return []

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = FIGDIR / "anim_divisor_orbit.mp4"
    anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=3500), dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def sector_dial(frames=120, fps=20, res=480):
    """Freeze one factor; dial the second factor's zeros; watch the product field respond.

    F = F_state * F_dial. The frozen zeros stay; the two dialled zeros orbit, and the product's
    energy landscape reshapes around them (zeros of the product = union).
    """
    nS = np.array([0.25 + 0.52j, 0.52 + 0.66j, 0.76 + 0.46j, 0.40 + 0.30j])
    FS = dna.dna_from_neurons(nS)
    TAU, ext = geom.half_plane_grid((0.0, 1.0), (0.05, 0.9), res)

    def dial_zeros(t):
        c = 0.5 + 0.46j
        return np.array([c + 0.20 * np.exp(2j * np.pi * t),
                         c + 0.20 * np.exp(2j * np.pi * (t + 0.5))])

    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    img = ax.imshow(np.zeros((res, res, 3)), extent=ext, origin="upper",
                    interpolation="none", aspect="auto")
    ax.plot(nS.real, nS.imag, "*", color="#46f0d0", ms=17, mec="black", mew=0.7,
            zorder=6, label="frozen factor's zeros")
    gv = ax.scatter([], [], marker="*", s=240, color=style.ACCENT, edgecolor="black",
                    linewidth=0.7, zorder=7, label="dialled factor's zeros")
    ax.set_xlim(0, 1); ax.set_ylim(0.05, 0.9)
    ax.set_xlabel(r"Re $\tau$"); ax.set_ylabel(r"Im $\tau$")
    ax.set_title(r"Product of two factors: dialling the second reshapes the frozen field"
                 "\n"
                 r"$F=F_{\rm state}\cdot F_{\rm dial}$,  zeros = union")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.3)

    def update(k):
        nG = dial_zeros(k / frames)
        FG = dna.dna_from_neurons(nG)
        F = modular.qpoly(np.convolve(FS, FG), TAU)
        img.set_data(coloring.domain_color(F, cmap="twilight"))
        gv.set_offsets(np.column_stack([nG.real, nG.imag]))
        return [img, gv]

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = FIGDIR / "anim_sector_dial.mp4"
    anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=4000), dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def sector_divisor(fps=20, frames=110):
    """Degree-2 version of the dial: one divisor curve frozen, the second factor's coupling
    dialled; the product's zero set is the union of the two divisors."""
    render = _render()
    if render is None:
        return

    dct = siegel.representation_numbers(maxn=8)
    ws = np.linspace(-1.0, 1.0, 200)
    state = render.genus2_divisor_points(dct, 0.35, ws, im_lo=0.5, im_hi=1.45)  # frozen
    zgs = 0.05 + 0.55 * (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frames) / frames))

    fig = plt.figure(figsize=(7.4, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(state[:, 0], state[:, 1], state[:, 2], color="#46f0d0", s=8,
               label="first factor  (coupling z = 0.35, frozen)")
    gsc = ax.scatter([], [], [], color=style.ACCENT, s=11, label="second factor  (coupling dialled)")
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(0.5, 1.1); ax.set_zlim(-1, 1)
    ax.set_xlabel(r"Re $\tau$"); ax.set_ylabel(r"Im $\tau$"); ax.set_zlabel("w")
    ax.legend(loc="upper left", fontsize=9)
    ttl = ax.set_title("")

    def update(k):
        zg = zgs[k]
        g = render.genus2_divisor_points(dct, zg, ws, im_lo=0.5, im_hi=1.45)
        gsc._offsets3d = (g[:, 0], g[:, 1], g[:, 2])
        ax.view_init(elev=18, azim=-62 + 24 * np.sin(2 * np.pi * k / frames))
        ttl.set_text(r"$E_8\oplus E_8$ zero divisor = union of the two factors' divisors"
                     f"   ·   second coupling z = {zg:.2f}")
        return [gsc]

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = FIGDIR / "anim_sector_divisor.mp4"
    anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=4000), dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


ANIMS = {"degree2": degree2, "flow": flow, "tweak": tweak, "divisor_orbit": divisor_orbit,
         "sector_dial": sector_dial, "sector_divisor": sector_divisor}


def main(argv):
    if not FFMpegWriter.isAvailable():
        print("ffmpeg is not available on this machine: no animation written (exit 0). "
              "Install ffmpeg and run `python animate.py all`.")
        return 0
    names = list(ANIMS) if (not argv or argv[0] == "all") else argv
    for n in names:
        ANIMS.get(n, lambda: print(f"unknown: {n}"))()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
