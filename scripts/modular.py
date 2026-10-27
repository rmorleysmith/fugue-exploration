"""Level-one modular forms E4, E6, Delta, j by q-series with fundamental-domain reduction.

WHAT IT COMPUTES — E4(tau) = 1 + 240 sum sigma3(n) q^n, E6(tau) = 1 - 504 sum sigma5(n) q^n,
    Delta = (E4^3 - E6^2)/1728 and j = E4^3/Delta, q = exp(2 pi i tau), for arrays of tau in the
    upper half-plane H. Each tau is first reduced into the fundamental domain
    D = {|tau| >= 1, |Re tau| <= 1/2} by some g in SL(2,Z), the series (24 terms) is summed there,
    where it converges fast, and the automorphy factor (c tau + d)^k is undone. Also truncated
    q-polynomials F(tau) = sum_{n<=N} a_n q^n (`qpoly`), their tau-derivative, and their roots in
    the q-disk (`qpoly_zeros`, via `np.roots`).
WHAT IT PROVIDES — `e4`, `e4_grad`, `e6`, `delta`, `jinv`, `to_fundamental`, `qpoly`,
    `qpoly_grad`, `qpoly_zeros`. The `__main__` block checks E4(5i) ~ 1, E4(rho) ~ 0, the
    modularity E4(-1/tau) = tau^4 E4(tau), and the gradient against a finite difference.
WHAT IS PROVED / KNOWN — Theta_E8 = E4 (Serre, *A Course in Arithmetic*, ch. VII; Lean
    `Fugue.thetaE8_eq_E₄`). E4 vanishes at rho = exp(2 pi i/3), its only zero per fundamental
    domain, simple, by the valence formula (Serre VII.3; Lean `Fugue.E₄_rho`,
    `Fugue.eq_zero_at_rho`). Delta = (E4^3 - E6^2)/1728 (Mathlib
    `ModularForm.discriminant_eq_E₄_cube_sub_E₆_sq`). Audit ledger rows A3, A4, A6.
WHAT IS NOT CLAIMED — the roots of a truncated q-polynomial are not the zeros of a modular form:
    a degree-N truncation has N roots (fundamental theorem of algebra) whatever the form, and only
    their limit points are zeros of the form (Hurwitz's theorem; ledger A7, A8). `qpoly_zeros`
    returns the roots of a polynomial, nothing more.
HOW THIS WAS FOUND — we wanted to draw E4 over the whole half-plane and the plain q-series blew
    up near the real axis; reducing to the fundamental domain first is the standard cure, and it
    made the pictures show the modular tiling for free. We imagined the zeros of a form as the
    "neurons" of a network and its Fourier coefficients as its "DNA"; the precise statement is
    that E4 has one zero per fundamental domain, whose SL(2,Z)-orbit maps to countably many
    points of the q-disk, at |q| = exp(-pi sqrt(3)/(c^2 - cd + d^2)).
"""
from __future__ import annotations

import numpy as np

_NTERMS = 24
_n = np.arange(1, _NTERMS + 1)
_SIG3 = np.array([sum(d ** 3 for d in range(1, k + 1) if k % d == 0) for k in _n], dtype=float)
_SIG5 = np.array([sum(d ** 5 for d in range(1, k + 1) if k % d == 0) for k in _n], dtype=float)


def to_fundamental(tau, iters: int = 80):
    """Reduce tau into the fundamental domain.

    Returns (z, (a, b, c, d)) with z = (a tau + b)/(c tau + d) in D and
    [[a,b],[c,d]] in SL(2,Z). Arrays are broadcast over the input shape.
    """
    tau = np.asarray(tau, dtype=np.complex128)
    shape = tau.shape
    z = tau.ravel().copy()
    a = np.ones(z.size)
    b = np.zeros(z.size)
    c = np.zeros(z.size)
    d = np.ones(z.size)
    for _ in range(iters):
        k = np.round(z.real)
        z = z - k
        a = a - k * c
        b = b - k * d
        m = np.abs(z) < 1.0 - 1e-12
        if m.any():
            zi = z[m]
            z[m] = -1.0 / zi
            am, bm, cm, dm = a[m].copy(), b[m].copy(), c[m].copy(), d[m].copy()
            a[m] = -cm
            b[m] = -dm
            c[m] = am
            d[m] = bm
    return (
        z.reshape(shape),
        (a.reshape(shape), b.reshape(shape), c.reshape(shape), d.reshape(shape)),
    )


def _e4_series(z):
    q = np.exp(2j * np.pi * z)
    qn = q[..., None] ** _n
    return 1.0 + 240.0 * (qn * _SIG3).sum(axis=-1)


def _e4_series_deriv(z):
    """d/dz of the E4 series."""
    q = np.exp(2j * np.pi * z)
    qn = q[..., None] ** _n
    return 240.0 * (qn * (_SIG3 * 2j * np.pi * _n)).sum(axis=-1)


def _e6_series(z):
    q = np.exp(2j * np.pi * z)
    qn = q[..., None] ** _n
    return 1.0 - 504.0 * (qn * _SIG5).sum(axis=-1)


def e4(tau):
    """Weight-4 Eisenstein series = theta function of E8."""
    z, (a, b, c, d) = to_fundamental(tau)
    return (c * tau + d) ** (-4) * _e4_series(z)


def e4_grad(tau):
    """Return (E4(tau), dE4/dtau), the latter via the chain rule through reduction."""
    z, (a, b, c, d) = to_fundamental(tau)
    jac = c * tau + d
    e4z = _e4_series(z)
    e4pz = _e4_series_deriv(z)
    val = jac ** (-4) * e4z
    grad = -4.0 * c * jac ** (-5) * e4z + jac ** (-6) * e4pz
    return val, grad


def e6(tau):
    z, (a, b, c, d) = to_fundamental(tau)
    return (c * tau + d) ** (-6) * _e6_series(z)


def delta(tau):
    """Modular discriminant Delta = (E4^3 - E6^2)/1728 (weight 12, nonvanishing on H)."""
    z, (a, b, c, d) = to_fundamental(tau)
    e4z = _e4_series(z)
    e6z = _e6_series(z)
    return (c * tau + d) ** (-12) * (e4z ** 3 - e6z ** 2) / 1728.0


def jinv(tau):
    """Klein j-invariant E4^3/Delta (weight 0, SL(2,Z)-invariant)."""
    z, _ = to_fundamental(tau)
    e4z = _e4_series(z)
    e6z = _e6_series(z)
    dz = (e4z ** 3 - e6z ** 2) / 1728.0
    return e4z ** 3 / dz


# --- truncated q-polynomials (the finite landscape the figures draw; the June files called
# --- the coefficient vector the "DNA") ----------------------------------------------

def qpoly(coeffs, tau):
    """Evaluate a truncated form F(tau) = sum_n coeffs[n] q^n, q = exp(2 pi i tau)."""
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    q = np.exp(2j * np.pi * np.asarray(tau, dtype=np.complex128))
    powers = q[..., None] ** np.arange(coeffs.size)
    return (powers * coeffs).sum(axis=-1)


def qpoly_grad(coeffs, tau):
    """Return (F, dF/dtau) for the truncated form."""
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    idx = np.arange(coeffs.size)
    q = np.exp(2j * np.pi * np.asarray(tau, dtype=np.complex128))
    powers = q[..., None] ** idx
    val = (powers * coeffs).sum(axis=-1)
    grad = (powers * (coeffs * 2j * np.pi * idx)).sum(axis=-1)
    return val, grad


def qpoly_zeros(coeffs):
    """Zeros of the truncated form F in tau-space: roots of the q-polynomial with |q|<1.

    Returns tau values with Im(tau) > 0 and Re(tau) in [0,1).
    """
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    # np.roots wants the highest power first; roots at q = 0 (a leading-zero block) are dropped
    qroots = np.roots(coeffs[::-1])
    qroots = qroots[np.abs(qroots) > 1e-12]
    qroots = qroots[np.abs(qroots) < 1.0 - 1e-9]
    tau = np.log(qroots) / (2j * np.pi)
    tau = tau.real % 1.0 + 1j * tau.imag
    return tau[tau.imag > 1e-6]


if __name__ == "__main__":
    # E4 q-expansion check.
    tau = 1j * 5.0  # large imaginary -> q ~ 0, E4 ~ 1
    print("E4(5i) ~ 1:", e4(tau))
    # Zero of E4 at rho = exp(2 pi i / 3).
    rho = np.exp(2j * np.pi / 3.0)
    print("E4(rho) ~ 0:", abs(e4(rho + 1e-9j)))
    # Modularity: E4(-1/tau) = tau^4 E4(tau).
    t = 0.3 + 0.7j
    lhs = e4(-1.0 / t)
    rhs = t ** 4 * e4(t)
    print("modularity residual:", abs(lhs - rhs))
    # Gradient finite-difference check.
    h = 1e-6
    _, g = e4_grad(t)
    fd = (e4(t + h) - e4(t - h)) / (2 * h)
    print("grad residual:", abs(g - fd))
    print("modular module OK")
