"""Truncated power-law-mixture forms of Sec. 4.3 (equations "index mixture", "index moment expansion" and
"index cumulant expansion" of the paper), fitted through the energy-cell basis.

With t = ln(gamma/GAMMA0) and GAMMA0 = 1e4 (ensembles.GAMMA0), the forms are

  mom<N>: dN/dgamma = (1/GAMMA0) (gamma/GAMMA0)^-P0 sum_{n=0}^{N} c_n (-t)^n / n!,   c_n = A m_n,
          the moment expansion about the fixed index P0 = 2.6, linear in its N + 1 coefficients c_n;
  cum<N>: dN/dgamma = (A/GAMMA0) exp[sum_{n=1}^{N} k_n (-t)^n / n!],
          the cumulant expansion, nonlinear in k_1..k_N, with the amplitude A solved linearly.

cum2 is the eLP family of baseline_models.py with k_1 = p and k_2 = -2 kappa, and its bounds
(k_1 in [0.5, 6], k_2 in [-1, 1], those of the paper's Appendix B) and starting points are those of eLP,
so cum2 fits reproduce eLP fits. cum3 adds k_3 in [-1, 1] and starts from the cum2 solution (moved 1e-6
inside the bounds) as well as from a fixed grid; cum4 adds k_4 in [-1, 1] and starts in the same way
from the cum3 solution, so the cumulant fits are nested and the chi2 does not rise with the order. The
moment form has no bounds and no positivity constraint. A mixture with a non-negative index density has ln N convex in ln gamma (k_2 = var p >= 0), so
neither form describes a steepening or a cut-off as a mixture; fitted to such spectra (E2 above its break,
E4-JP, E6) they act as curvature parametrisations, with k_2 < 0 or m_2 < 0, and the cumulant fits can end
on the bounds (fit_form reports this).
All forms use the energy cells of config.PAPER; the columns passed in must share them, and have N + 1
columns per cell for the basis order N passed to fit_form (default 2).

Every form gives energy-cell moments (n_bins, N + 1) through ensembles._smooth_family_moments (32-node
GL per cell), and its channel spectrum through the full separable columns of the same order (all cells,
the paper's field statistics, azimuth factor and screen), as the electron families of
fits_extrap.py do. fit_form fits I, Q and U of one realisation and evaluates the fitted moments on the
prediction columns; it is a module-level function so that spawned worker processes can import it.
"""
from __future__ import annotations

from functools import lru_cache
from math import factorial

import numpy as np
from scipy.optimize import least_squares

import fits as F  # first: puts lib/ on the path
import ensembles as E
from config import PAPER

GAMMA0 = E.GAMMA0
P0 = 2.6  # index of the moment expansion: the index of the reference weights (Eq. (B.2)) and of E1, E3, E4
FORMS = ("mom2", "cum2", "mom3", "cum3", "mom4", "cum4")
K_BOUNDS = ((0.5, 6.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))  # k_1, k_2, k_3, k_4
CUM2_STARTS = tuple((2.6, -2.0 * kap) for kap in (-0.05, 0.0, 0.05))  # eLP starts (p0 = 2.6, kappa), as k
K3_STARTS = (-0.02, 0.0, 0.02)
K4_STARTS = (-0.02, 0.0, 0.02)


def kind_order(form: str) -> tuple[str, int]:
    kind, order = form[:3], int(form[3:])
    if kind not in ("mom", "cum") or order not in (2, 3, 4):
        raise ValueError(form)
    return kind, order


def n_params(form: str) -> int:
    """Free parameters: N + 1 coefficients (mom) or A and k_1..k_N (cum)."""
    return kind_order(form)[1] + 1


@lru_cache(maxsize=None)
def moment_basis(order: int, N: int = 2, cfg=PAPER) -> np.ndarray:
    """(order + 1, n_bins, N + 1): energy-cell moments (basis order N) of (1/GAMMA0)(gamma/GAMMA0)^-P0 (-t)^n/n!,
    n = 0..order.

    Cached per process; callers must not modify the returned array.
    """
    out = []
    for n in range(order + 1):
        def f(g, th, n=n):
            t = np.log(g / GAMMA0)
            return (g / GAMMA0) ** (-P0) * (-t) ** n / factorial(n) + 0.0 * th[:, 0]
        out.append(E._smooth_family_moments(cfg, f, np.array([1.0, 0.0]), N)[0])
    M = np.stack(out)
    M.flags.writeable = False
    return M


def cumulant_moments(k, N: int = 2, cfg=PAPER) -> np.ndarray:
    """(n_bins, N + 1): unit-amplitude energy-cell moments (basis order N) of (1/GAMMA0) exp[sum_n k_n (-t)^n/n!]."""
    k = np.asarray(k, float)

    def f(g, th):
        t = np.log(g / GAMMA0)
        expo = sum(th[:, n] * (-t) ** (n + 1) / factorial(n + 1) for n in range(th.shape[1]))
        return np.exp(expo)

    return E._smooth_family_moments(cfg, f, np.r_[1.0, k], N)[0]


def _fit_moment(order, C_rows, d, sig, N=2):
    """Weighted linear least squares for the coefficients c_n; returns (moments, chi2, params)."""
    M = moment_basis(order, N)
    G = np.stack([C_rows @ M[n].ravel() for n in range(order + 1)], axis=1) / sig[:, None]
    c, *_ = np.linalg.lstsq(G, d / sig, rcond=None)
    chi2 = float(np.sum((G @ c - d / sig) ** 2))
    return np.tensordot(c, M, axes=1), chi2, c, False


def _fit_cumulant(order, C_rows, d, sig, N=2):
    """Variable projection over k_1..k_N with A solved linearly (as baseline_models.fit_family)."""
    dw = d / sig

    def bounds(n):
        return np.array([b[0] for b in K_BOUNDS[:n]]), np.array([b[1] for b in K_BOUNDS[:n]])

    def gvec(k):
        return (C_rows @ cumulant_moments(k, N).ravel()) / sig

    def resid(k):
        g = gvec(k)
        return (g @ dw) / (g @ g) * g - dw

    def solve(x0):
        lo, hi = bounds(len(x0))
        x0 = np.clip(np.asarray(x0, float), lo + 1e-6, hi - 1e-6)
        return least_squares(resid, x0, bounds=(lo, hi), method="trf", xtol=1e-12, ftol=1e-12,
                             gtol=1e-12, max_nfev=3000)

    def best_of(starts):
        best = None
        for x0 in starts:
            r = solve(x0)
            if best is None or r.cost < best.cost:
                best = r
        return best

    best = best_of(CUM2_STARTS)
    if order >= 3:
        best = best_of([np.r_[best.x, 0.0]] + [np.r_[s, k3] for s in CUM2_STARTS for k3 in K3_STARTS])
    if order >= 4:
        best = best_of([np.r_[best.x, 0.0]] + [np.r_[s, 0.0, k4] for s in CUM2_STARTS for k4 in K4_STARTS])
    g = gvec(best.x)
    amp = (g @ dw) / (g @ g)
    lo, hi = bounds(order)
    at_bound = bool(np.any(np.isclose(best.x, lo, atol=1e-5) | np.isclose(best.x, hi, atol=1e-5)))
    return amp * cumulant_moments(best.x, N), float(2 * best.cost), np.r_[amp, best.x], at_bound


def fit_form(args):
    """One realisation, one form: fit I, Q, U rows; return (prediction (J, 3), chi2, params, at_bound).

    args = (form, C_rows, d, sig, C_pred[, N]): N is the Taylor order of the energy-cell basis of the columns
    (default 2); the columns must have N + 1 per cell.
    """
    form, C_rows, d, sig, C_pred, *rest = args
    N = rest[0] if rest else 2
    kind, order = kind_order(form)
    try:
        a, chi2, params, at_bound = (_fit_moment if kind == "mom" else _fit_cumulant)(order, C_rows, d, sig, N)
    except Exception:  # pragma: no cover - a failed nonlinear fit is recorded, not hidden
        return None
    return F.B.predict(C_pred, a.ravel()), chi2, params, at_bound
