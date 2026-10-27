"""Geometry helpers: the upper half-plane, the Poincare disk, and sampling grids.

WHAT IT COMPUTES — the Cayley transform w = (tau - i)/(tau + i) from H onto the unit disk and its
    inverse tau = i (1 + w)/(1 - w); rectangular grids over H (row 0 at the top, so `imshow` with
    origin='upper' reads the way H is drawn) and over the disk, with a mask.
WHAT IT PROVIDES — `cayley`, `inv_cayley`, `half_plane_grid`, `disk_grid`.
WHAT IS PROVED / KNOWN — the Cayley transform is a biholomorphism H -> D sending the real axis to
    the boundary circle and i to 0 (Ahlfors, *Complex Analysis*, ch. 3).
WHAT IS NOT CLAIMED — the disk picture is a change of coordinates only; nothing in the exploration
    depends on it.
HOW THIS WAS FOUND — we wanted the whole of H, cusps included, in one bounded picture; the disk
    model does that, and it is where figures 02 and 13 live.
"""
from __future__ import annotations

import numpy as np


def cayley(tau):
    """Cayley transform H -> unit disk: tau |-> (tau - i)/(tau + i)."""
    tau = np.asarray(tau, dtype=np.complex128)
    return (tau - 1j) / (tau + 1j)


def inv_cayley(w):
    """Inverse Cayley, unit disk -> H: w |-> i (1 + w)/(1 - w)."""
    w = np.asarray(w, dtype=np.complex128)
    return 1j * (1.0 + w) / (1.0 - w)


def half_plane_grid(re_lim, im_lim, res):
    """Grid over the upper half-plane. Returns (TAU, extent) for imshow.

    re_lim = (xmin, xmax), im_lim = (ymin, ymax). Row 0 of TAU is the TOP
    (largest Im) so imshow with origin='upper' renders the way we read H.
    """
    xs = np.linspace(re_lim[0], re_lim[1], res)
    ys = np.linspace(im_lim[1], im_lim[0], res)  # top row = high Im
    X, Y = np.meshgrid(xs, ys)
    extent = (re_lim[0], re_lim[1], im_lim[0], im_lim[1])
    return X + 1j * Y, extent


def disk_grid(res):
    """Grid over the unit disk. Returns (W, mask, extent)."""
    xs = np.linspace(-1, 1, res)
    ys = np.linspace(1, -1, res)
    X, Y = np.meshgrid(xs, ys)
    W = X + 1j * Y
    mask = np.abs(W) <= 1.0 - 1e-9
    return W, mask, (-1, 1, -1, 1)
