"""The E8 lattice: vectors by norm, the 240 roots, and the coefficients of its theta series.

WHAT IT COMPUTES — E8 = { x in Z^8 u (Z+1/2)^8 : sum(x_i) in 2Z } (the even coordinate system);
    all vectors with ||v||^2 <= M by coordinate-wise enumeration with norm pruning; the shells by
    squared norm; sigma_3(n) and the coefficients a_n = 240 sigma_3(n) of Theta_E8; the Bourbaki
    simple roots, the Coxeter element c = s_1 ... s_8, and the projection onto its Coxeter plane
    (the eigenplane of exp(2 pi i/30)). All vectors are float64 arrays of shape (N, 8).
WHAT IT PROVIDES — `e8_vectors`, `shells`, `sigma`, `theta1_coeffs`, `simple_roots`,
    `coxeter_element`, `coxeter_plane`. The `__main__` block asserts the shell counts 240, 2160,
    6720, 17520 and reports the 8 radii of the Petrie projection.
WHAT IS PROVED / KNOWN — E8 is the even unimodular lattice of rank 8 (Conway and Sloane, *Sphere
    Packings, Lattices and Groups*, ch. 4; Lean `Fugue.e8Form_even`, `Fugue.e8Form_pos`, Mathlib
    `CartanMatrix.E₈_det`). #{v : ||v||^2 = 2n} = 240 sigma_3(n), i.e. Theta_E8 = E4 (Lean
    `Fugue.reprCount_e8`, `Fugue.thetaE8_eq_E₄`). The Coxeter number of E8 is 30, and the 240
    roots project to 8 concentric 30-gons (Bourbaki, *Lie Groups and Lie Algebras*, ch. VI).
    Audit ledger rows A1, A2.
WHAT IS NOT CLAIMED — nothing here is specific to a network. The lattice enters the exploration
    only through its theta series, and the Petrie projection is a picture of the root system, not
    of any state space.
HOW THIS WAS FOUND — we counted lattice vectors shell by shell, saw 240, 2160, 6720, and
    recognised 240 sigma_3(n): the theta series of E8 is the Eisenstein series E4, so the
    "DNA" of the E8 landscape was a classical divisor sum. We imagined the lattice itself as the
    reservoir; the precise statement is that the state space is the upper half-plane and E8
    contributes the function on it.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _coset_vectors(vals: np.ndarray, max_norm2: float) -> np.ndarray:
    """All 8-tuples drawn from `vals` with squared norm <= max_norm2.

    Built coordinate-by-coordinate with norm pruning so we never materialise
    the full |vals|**8 product.
    """
    vals = np.asarray(vals, dtype=np.float64)
    sq = vals ** 2
    cur = vals.reshape(-1, 1)
    nrm = sq.copy()
    keep = nrm <= max_norm2 + _EPS
    cur, nrm = cur[keep], nrm[keep]
    for _ in range(7):
        k = vals.shape[0]
        n = cur.shape[0]
        cur = np.concatenate(
            [np.repeat(cur, k, axis=0), np.tile(vals.reshape(-1, 1), (n, 1))],
            axis=1,
        )
        nrm = np.repeat(nrm, k) + np.tile(sq, n)
        keep = nrm <= max_norm2 + _EPS
        cur, nrm = cur[keep], nrm[keep]
    return cur


def e8_vectors(max_norm2: float) -> np.ndarray:
    """All E8 vectors v with ||v||^2 <= max_norm2 (includes the origin)."""
    m = int(np.floor(np.sqrt(max_norm2) + _EPS))
    int_vals = np.arange(-m, m + 1, dtype=np.float64)
    half_vals = np.arange(-(m + 1), m + 1, dtype=np.float64) + 0.5
    half_vals = half_vals[half_vals ** 2 <= max_norm2 + _EPS]

    out = []
    for vals in (int_vals, half_vals):
        v = _coset_vectors(vals, max_norm2)
        if v.shape[0] == 0:
            continue
        s = np.rint(v.sum(axis=1)).astype(np.int64)
        v = v[s % 2 == 0]  # even coordinate sum
        out.append(v)
    return np.concatenate(out, axis=0)


def shells(max_norm2: float) -> dict[int, np.ndarray]:
    """E8 vectors grouped by squared norm (excluding the origin)."""
    v = e8_vectors(max_norm2)
    n2 = np.rint((v ** 2).sum(axis=1)).astype(int)
    return {int(k): v[n2 == k] for k in sorted(set(n2.tolist())) if k > 0}


# --- theta series coefficients (degree 1) --------------------------------------

def sigma(k: int, n: int) -> int:
    """Divisor power sum sigma_k(n) = sum_{d | n} d^k."""
    return int(sum(d ** k for d in range(1, n + 1) if n % d == 0))


def theta1_coeffs(num: int) -> np.ndarray:
    """Coefficients of Theta_E8(tau) = E4(tau) = 1 + 240 sum_{n>=1} sigma3(n) q^n.

    Returns array a[0..num] with a[0]=1, a[n]=240*sigma3(n).
    """
    a = np.zeros(num + 1, dtype=np.float64)
    a[0] = 1.0
    for n in range(1, num + 1):
        a[n] = 240.0 * sigma(3, n)
    return a


# --- root system + Coxeter (Petrie) projection ---------------------------------

def simple_roots() -> np.ndarray:
    """The 8 Bourbaki simple roots of E8 (each of squared norm 2), shape (8, 8)."""
    e = np.eye(8)
    a = np.zeros((8, 8))
    a[0] = 0.5 * np.array([1, -1, -1, -1, -1, -1, -1, 1], dtype=float)
    a[1] = e[0] + e[1]
    a[2] = e[1] - e[0]
    a[3] = e[2] - e[1]
    a[4] = e[3] - e[2]
    a[5] = e[4] - e[3]
    a[6] = e[5] - e[4]
    a[7] = e[6] - e[5]
    return a


def coxeter_element() -> np.ndarray:
    """Coxeter element c = s_1 s_2 ... s_8 as an 8x8 orthogonal matrix.

    For a root alpha with ||alpha||^2 = 2 the reflection is s(v) = v - (v.alpha) alpha,
    i.e. the matrix I - alpha alpha^T.
    """
    roots = simple_roots()
    c = np.eye(8)
    for alpha in roots:
        refl = np.eye(8) - np.outer(alpha, alpha)
        c = c @ refl
    return c


def coxeter_plane(points: np.ndarray) -> np.ndarray:
    """Project `points` (N,8) onto the Coxeter plane of E8.

    The Coxeter number is h = 30; the Coxeter element has a complex eigenvalue
    exp(2 pi i / 30) whose eigenvector spans the invariant 2-plane in which the
    240 roots fall into 8 concentric 30-gons (the famous Petrie projection).
    Returns an (N, 2) array.
    """
    c = coxeter_element()
    w, V = np.linalg.eig(c)
    target = np.exp(2j * np.pi / 30.0)
    idx = int(np.argmin(np.abs(w - target)))
    u = V[:, idx]
    # Re(u), Im(u) are orthogonal and equal length; rescale to orthonormal axes.
    ax = np.real(u)
    ay = np.imag(u)
    ax = ax / np.linalg.norm(ax)
    ay = ay / np.linalg.norm(ay)
    # Remove any residual non-orthogonality (Gram-Schmidt).
    ay = ay - (ay @ ax) * ax
    ay = ay / np.linalg.norm(ay)
    return np.stack([points @ ax, points @ ay], axis=1)


if __name__ == "__main__":
    # Self-tests: shell counts must be 240 * sigma3(n).
    sh = shells(8.0)
    expected = {2: 240, 4: 2160, 6: 6720, 8: 17520}
    for k, cnt in expected.items():
        got = sh[k].shape[0]
        assert got == cnt, f"shell {k}: got {got}, expected {cnt}"
        print(f"shell norm^2={k}: {got} vectors  (expected {cnt})  OK")

    # Coxeter projection: 240 roots should fall into 8 radius clusters.
    roots = sh[2]
    proj = coxeter_plane(roots)
    radii = np.sort(np.unique(np.round(np.linalg.norm(proj, axis=1), 4)))
    print(f"distinct projected radii: {len(radii)} -> {radii}")
    print("E8 module OK")
