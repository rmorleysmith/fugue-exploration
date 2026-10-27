"""Domain colouring: phase as hue, |F| level bands as brightness (Wegert's phase portraits).

WHAT IT COMPUTES — for a complex array F, hue = arg F through a cyclic colormap (twilight by
    default, or pure HSV) and a sawtooth shade in log_base |F| that draws the level curves of |F|;
    and the boundary path of the fundamental domain D = {|tau| >= 1, |Re tau| <= 1/2}.
WHAT IT PROVIDES — `domain_color(F, cmap, darkest, lam, base, bands)` -> RGB array in [0, 1] of
    shape F.shape + (3,); `fundamental_domain_path(n)` -> (re, im).
WHAT IS PROVED / KNOWN — around a simple zero the argument winds once and the level bands
    accumulate; around a pole it winds the other way (the argument principle). E. Wegert,
    *Visual Complex Functions*, Birkhauser 2012.
WHAT IS NOT CLAIMED — the colouring is a visualisation device; the winding it shows is the local
    degree of F and says nothing about whether F is modular.
HOW THIS WAS FOUND — we needed the zeros to be visible at a glance in a field spanning many
    orders of magnitude; a phase portrait makes each zero a point where every hue meets, which
    is how the June figures read their "neurons".
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np


def domain_color(
    F,
    cmap: str = "twilight",
    darkest: float = 0.35,
    lam: float = 6.0,
    base: float = 1.6,
    bands: bool = True,
):
    """Map complex array F to an RGB image array in [0,1], shape F.shape + (3,)."""
    F = np.asarray(F, dtype=np.complex128)
    finite = np.isfinite(F)
    Fc = np.where(finite, F, 0.0)

    arg = np.angle(Fc)
    hue = (arg + np.pi) / (2.0 * np.pi)

    if cmap == "hsv":
        h = hue
        s = np.ones_like(h)
        v = np.ones_like(h)
        rgb = mcolors.hsv_to_rgb(np.stack([h, s, v], axis=-1))
    else:
        rgb = mpl.colormaps[cmap](hue)[..., :3]

    if bands:
        mag = np.abs(Fc)
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log(np.maximum(mag, 1e-300)) / np.log(base)
        t = lg - np.floor(lg)  # sawtooth in [0,1)
        shade = darkest + (1.0 - darkest) * np.power(t, 1.0 / lam)
        rgb = rgb * shade[..., None]

    rgb = np.where(finite[..., None], rgb, 0.0)
    return np.clip(np.nan_to_num(rgb), 0.0, 1.0)


def fundamental_domain_path(n: int = 400):
    """Boundary of the SL(2,Z) fundamental domain D = {|tau|>=1, |Re|<=1/2}.

    Returns (re, im) arrays tracing the two vertical walls and the unit-circle arc.
    """
    arc_t = np.linspace(np.pi / 3.0, 2.0 * np.pi / 3.0, n)
    arc = np.cos(arc_t) + 1j * np.sin(arc_t)
    top = 3.5
    left = np.array([-0.5 + 1j * top, -0.5 + 1j * np.sin(2 * np.pi / 3)])
    right = np.array([0.5 + 1j * np.sin(np.pi / 3), 0.5 + 1j * top])
    path = np.concatenate([left, arc[::-1], right])
    return path.real, path.imag
