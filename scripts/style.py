"""Shared matplotlib styling for the figure set.

WHAT IT COMPUTES — nothing. It sets the rcParams (dark ink background, pale text, DejaVu Sans,
    200 dpi on save) and the palette: INK, PAPER, ACCENT, and NEURON, the marker colour for
    zeros (the June name, kept because every script imports it).
WHAT IT PROVIDES — the constants above and `phase_wheel(ax, cmap)`, a small polar legend for the
    hue = arg F convention of `coloring.py`.
WHAT IS PROVED / KNOWN — nothing.
WHAT IS NOT CLAIMED — nothing.
HOW THIS WAS FOUND — one style file, so that the sixty-odd June figures read as one set.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

INK = "#0a0a12"
PAPER = "#e8e6f0"
ACCENT = "#ffd166"
NEURON = "#ff3b6b"

mpl.rcParams.update({
    "figure.facecolor": INK,
    "savefig.facecolor": INK,
    "axes.facecolor": INK,
    "text.color": PAPER,
    "axes.edgecolor": "#444455",
    "axes.labelcolor": PAPER,
    "xtick.color": PAPER,
    "ytick.color": PAPER,
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def phase_wheel(ax, cmap="twilight"):
    """Draw a small phase-legend wheel (arg color key) on a polar inset axis."""
    import numpy as np

    n = 256
    theta = np.linspace(0, 2 * np.pi, n)
    r = np.linspace(0.6, 1.0, 16)
    T, R = np.meshgrid(theta, r)
    C = (T) / (2 * np.pi)
    ax.pcolormesh(T, R, C, cmap=cmap, shading="auto")
    ax.set_yticks([])
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(["0", "π/2", "±π", "−π/2"], fontsize=8)
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor(INK)
    ax.set_title("arg F", fontsize=9, pad=2)
