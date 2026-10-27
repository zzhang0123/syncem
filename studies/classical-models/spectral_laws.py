"""Generic spectral laws used as baselines, fitted by weighted least squares in I.

Laws (L = ln(nu/nu0), nu0 = 1 GHz unless stated; every law is evaluated at each channel
quadrature frequency and integrated with the same unit-area channel weights as the moment
basis, or monochromatically with a single unit weight):

  s1        PySM s1 power law:            I = A exp(beta L)                          k = 2
  s3        PySM s3 curved power law:     I = A exp(beta L + C L^2)                  k = 3
            (PySM writes A (nu/nu0)^(beta + C ln(nu/nu_c)); the family is the same for
             any curvature pivot nu_c; C is pivot-invariant, beta shifts by -C ln(nu_c/nu0))
  logcubic  ln I = c0 + c1 L + c2 L^2 + c3 L^3                                       k = 4
  bmom3     second-order beta-moment SED (Chluba, Hill & Abitbol 2017) with the pivot
            beta_bar chosen so that the first moment vanishes (omega_1 = 0):
            I = A exp(beta L) [1 + omega_2 L^2 / 2]                                   k = 3
  bmom4     the same with omega_1 free:
            I = A exp(beta L) [1 + omega_1 L + omega_2 L^2 / 2]                       k = 4

PySM s1/s3 are defined in brightness temperature; in specific intensity beta_I = beta_T + 2,
so the families are identical.  Fits use variable projection: the linear coefficients
(A, A omega_1, A omega_2) are solved exactly for given nonlinear parameters; one nonlinear
parameter is searched on a grid and refined by bounded Brent; two or three are refined by
scipy.optimize.least_squares from a weighted log-space polynomial start.

The laws carry no polarisation.  ``polarisation_with_assumption`` adds the explicit extra
assumption Pi(nu) = (p_loc + 1)/(p_loc + 7/3) with p_loc = 1 + 2 alpha_loc(nu), alpha_loc =
-d ln I/d ln nu of the fitted law (the unbounded-power-law fraction applied locally),
times the supplied azimuth and screen factors inside the channel integral.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

BETA_GRID = np.arange(-4.0, 2.0 + 1e-9, 0.05)


@dataclass(frozen=True)
class Grid:
    """Evaluation rows: L (n_rows, T) = ln(nu/nu0) at quadrature nodes, w (T,) unit-area weights."""

    L: np.ndarray
    w: np.ndarray

    def rows(self, idx) -> "Grid":
        return Grid(self.L[idx], self.w)

    @property
    def centre_L(self) -> np.ndarray:
        """ln(nu/nu0) of the weighted mean frequency (used only for start values)."""
        return np.log(np.exp(self.L) @ self.w / self.w.sum())


def channel_grid(chan_nu, chan_w, nu0=1e9) -> Grid:
    return Grid(np.log(np.asarray(chan_nu) / nu0), np.asarray(chan_w, float))


def mono_grid(nu, nu0=1e9) -> Grid:
    return Grid(np.log(np.asarray(nu, float) / nu0)[:, None], np.array([1.0]))


# ----------------------------------------------------------------------------- law definitions
def _basis(name, phi, L):
    """Basis functions g_i(L; phi), shape (n_lin, ...L.shape)."""
    if name == "s1":
        return np.exp(phi[0] * L)[None]
    if name == "s3":
        return np.exp(phi[0] * L + phi[1] * L**2)[None]
    if name == "logcubic":
        return np.exp(phi[0] * L + phi[1] * L**2 + phi[2] * L**3)[None]
    e = np.exp(phi[0] * L)
    if name == "bmom3":
        return np.stack([e, 0.5 * e * L**2])
    if name == "bmom4":
        return np.stack([e, e * L, 0.5 * e * L**2])
    raise ValueError(name)


N_NONLIN = {"s1": 1, "s3": 2, "logcubic": 3, "bmom3": 1, "bmom4": 1}
N_LIN = {"s1": 1, "s3": 1, "logcubic": 1, "bmom3": 2, "bmom4": 3}
LAWS = ("s1", "s3", "bmom3", "logcubic", "bmom4")


def n_params(name) -> int:
    return N_NONLIN[name] + N_LIN[name]


def design(name, phi, grid: Grid) -> np.ndarray:
    """(n_rows, n_lin) channel-integrated basis functions."""
    g = _basis(name, phi, grid.L)
    return np.einsum("irt,t->ri", g, grid.w)


def predict(name, params, grid: Grid) -> np.ndarray:
    """I at every row of grid for params = (phi..., c...)."""
    k = N_NONLIN[name]
    return design(name, params[:k], grid) @ params[k:]


def dlnI_dL(name, params, L):
    """d ln I/d ln nu of the law at L."""
    k = N_NONLIN[name]
    phi, c = params[:k], params[k:]
    if name == "s1":
        return np.full_like(L, phi[0])
    if name == "s3":
        return phi[0] + 2 * phi[1] * L
    if name == "logcubic":
        return phi[0] + 2 * phi[1] * L + 3 * phi[2] * L**2
    w1 = c[1] / c[0] if name == "bmom4" else 0.0
    w2 = c[-1] / c[0]
    return phi[0] + (w1 + w2 * L) / (1 + w1 * L + 0.5 * w2 * L**2)


def describe(name, params) -> dict:
    """Named parameters (physical convention of each law)."""
    k = N_NONLIN[name]
    phi, c = params[:k], params[k:]
    if name == "s1":
        return {"A": c[0], "beta": phi[0]}
    if name == "s3":
        return {"A": c[0], "beta": phi[0], "C": phi[1]}
    if name == "logcubic":
        return {"c0": float(np.log(c[0])) if c[0] > 0 else float("nan"), "c1": phi[0], "c2": phi[1], "c3": phi[2]}
    out = {"A": c[0], "beta_bar": phi[0], "omega_2": c[-1] / c[0]}
    if name == "bmom4":
        out["omega_1"] = c[1] / c[0]
    return out


# ----------------------------------------------------------------------------- fitting
def _solve_linear(G, d, sig):
    Gw, dw = G / sig[:, None], d / sig
    c, *_ = np.linalg.lstsq(Gw, dw, rcond=None)
    r = Gw @ c - dw
    return c, r


def _chi2_phi(name, phi, grid, d, sig):
    _, r = _solve_linear(design(name, np.atleast_1d(phi), grid), d, sig)
    return float(r @ r)


def _start(name, grid, d, sig):
    """Weighted log-space polynomial fit at the channel mean frequencies (start values only)."""
    L = grid.centre_L
    deg = {"s3": 2, "logcubic": 3}[name]
    wts = np.where(d > 0, np.abs(d) / sig, 0.0)
    coef = np.polynomial.polynomial.polyfit(L, np.log(np.clip(d, 1e-300, None)), deg, w=wts)
    return coef[1:]


def fit(name, grid: Grid, d, sig, start=None) -> dict:
    """Weighted least squares of law `name` to data d (sd sig) on grid rows; returns params, chi2."""
    d, sig = np.asarray(d, float), np.asarray(sig, float)
    if N_NONLIN[name] == 1:
        vals = np.array([_chi2_phi(name, b, grid, d, sig) for b in BETA_GRID])
        i = int(np.argmin(vals))
        lo, hi = BETA_GRID[max(i - 1, 0)], BETA_GRID[min(i + 1, len(BETA_GRID) - 1)]
        res = minimize_scalar(lambda b: _chi2_phi(name, b, grid, d, sig), bounds=(lo, hi),
                              method="bounded", options={"xatol": 1e-10, "maxiter": 500})
        phi = np.array([res.x])
        edge = bool(i in (0, len(BETA_GRID) - 1))
    else:
        x0 = _start(name, grid, d, sig) if start is None else np.asarray(start, float)

        def resid(phi):
            return _solve_linear(design(name, phi, grid), d, sig)[1]

        res = least_squares(resid, x0, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=4000)
        phi = res.x
        edge = not res.success
    c, r = _solve_linear(design(name, phi, grid), d, sig)
    return {"params": np.concatenate([phi, c]), "chi2": float(r @ r), "edge_or_fail": edge}


def multistart_check(name, grid: Grid, d, sig, best_chi2, n_starts=12, seed=0) -> dict:
    """Full-parameter nonlinear least squares from random starts; best chi2 relative to VarPro."""
    rng = np.random.default_rng(seed)
    k = N_NONLIN[name]
    base = fit(name, grid, d, sig)["params"]  # call with d, sig scaled to O(1) amplitudes
    best = np.inf
    for _ in range(n_starts):
        p0 = base.copy()
        p0[:k] += rng.normal(scale=0.3, size=k) * np.maximum(np.abs(base[:k]), 0.1)
        p0[k:] *= np.exp(rng.normal(scale=0.3, size=len(base) - k))

        def resid(p):
            return (predict(name, p, grid) - d) / sig

        try:
            r = least_squares(resid, p0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
            best = min(best, float(r.fun @ r.fun))
        except (ValueError, FloatingPointError):
            continue
    return {"best_multistart_chi2": best, "varpro_chi2": best_chi2,
            "varpro_not_worse": bool(best_chi2 <= best * (1 + 1e-7) + 1e-12)}


# ----------------------------------------------------------------------------- polarisation
def pol_fraction_local(alpha):
    """(p+1)/(p+7/3) with p = 1 + 2 alpha (unbounded power law, applied locally)."""
    return (2 * alpha + 2) / (2 * alpha + 10.0 / 3.0)


def polarisation_with_assumption(name, params, grid: Grid, factor, screen):
    """P_j = -factor sum_t w_t screen_jt Pi(alpha_loc) I_law  (EXTRA ASSUMPTION; see module doc).

    The minus sign is the continuum Q-kernel sign (K_Q = -A G) of the moment basis.
    """
    k = N_NONLIN[name]
    f = np.einsum("irt,i->rt", _basis(name, params[:k], grid.L), params[k:])
    alpha = -dlnI_dL(name, params, grid.L)
    return -factor * np.einsum("rt,rt,t->r", screen, pol_fraction_local(alpha) * f, grid.w)
