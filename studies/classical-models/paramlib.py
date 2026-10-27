"""Fitting machinery for the fits and physical parameters, continuum kernels only.

* ``Whitened``: noise-whitened fitted rows W = Sigma^{-1/2} C_fit, SVD W = U D V^T,
  UPRE (Mallows C_p) rank R* = argmin_R [chi2_fit(R) + 2R], beta_hat = D_R^{-1} U_R^T d_w.
* ``Family``: a parametric electron family in fit coordinates phi (ln A, ...), mapped to
  the library families of ensembles.py; moments a(phi) and their phi-Jacobian (central
  differences, vectorised).
* ``fit``: compressed chi2_beta(phi) = |D_R (beta_hat - V_R^T a(phi))|^2 (Eq. (4.24)
  of the paper) or full |d_w - W a(phi)|^2, Levenberg-Marquardt, with the
  Gauss-Newton covariance (J^T J)^{-1} at the solution.
* ``bias_response``: linear map from a whitened data perturbation to the phi shift of
  either estimator (used for the representation bias and its envelope |G| E).
* Invariants (functions of phi and of the assumed field statistics): amplitude
  C <B_perp^{(p+1)/2}> and nu_b = a1 <B_perp> gamma_b^2 (B in muG).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

import ensembles as E
import kernels as K

LN_PIVOT = np.log(E.GAMMA0)
NU_STAR = 1e9  # pivot frequency of the E3 local invariants [Hz]


# ----------------------------------------------------------------------------- linear problem
class Whitened:
    """Fitted rows of d = C a + eps + n, whitened by the known noise sd (0.01 I_j^direct)."""

    def __init__(self, cfg, C, S_direct):
        self.cfg = cfg
        self.C = C
        self.n = C.shape[-1]
        self.sig = cfg.noise_sigma(S_direct)  # (J, 3)
        self.fit, self.held = cfg.fit_idx, cfg.held_idx
        self.sf = self.sig[self.fit].ravel()
        self.sh = self.sig[self.held].ravel()
        self.W = C[self.fit].reshape(-1, self.n) / self.sf[:, None]
        self.Wh = C[self.held].reshape(-1, self.n) / self.sh[:, None]
        self.U, self.s, self.Vt = np.linalg.svd(self.W, full_matrices=False)

    def whiten(self, d_fit):
        return np.asarray(d_fit).ravel() / self.sf

    def whiten_held(self, d_held):
        return np.asarray(d_held).ravel() / self.sh

    @property
    def numerical_rank(self):
        """Singular values above 1e-13 s_1 (the scan range of UPRE, as in fits.py)."""
        return int(np.sum(self.s > 1e-13 * self.s[0]))

    def upre(self, dw):
        """(R*, chi2_fit(R), criterion chi2 + 2R) for R = 1..numerical rank."""
        proj = self.U.T @ dw
        chi2 = (dw @ dw - np.cumsum(proj**2))[:self.numerical_rank]
        R = np.arange(1, chi2.size + 1)
        crit = chi2 + 2.0 * R
        return int(R[np.argmin(crit)]), chi2, crit

    def beta_hat(self, dw, R):
        return (self.U[:, :R].T @ dw) / self.s[:R]

    def tsvd_moments(self, dw, R):
        return self.Vt[:R].T @ self.beta_hat(dw, R)


# ----------------------------------------------------------------------------- families
def _bperp_moment(cfg, q):
    """<B_perp^q> over cfg's field-strength and eta quadrature (B in muG)."""
    b, wb = cfg.b_nodes_weights
    _, we = cfg.eta_nodes_weights
    q = np.asarray(q, float)
    return np.sum(wb * b ** q[..., None], axis=-1) * np.sum(we * cfg.eta_s ** q[..., None], axis=-1)


def _mixture_kappa(cfg, theta, N):
    """theta = (A, pbar, kappa): N = (A/GAMMA0) exp(-pbar y + kappa y^2/2), kappa = sigma_p^2."""
    f = lambda g, th: np.exp(-th[:, 0] * np.log(g / E.GAMMA0) + 0.5 * th[:, 1] * np.log(g / E.GAMMA0) ** 2)  # noqa: E731
    return E._smooth_family_moments(cfg, f, theta, N)


@dataclass(frozen=True)
class Family:
    """Parametric family in fit coordinates; ``key`` selects the mapping below."""

    key: str
    nu_conv: float = 0.0  # ln(a1 <B_perp>) used for nu_b <-> gamma_c (JP) [ln Hz]

    @property
    def names(self):
        return {"PL": ("lnA", "p"), "SBPL_s2": ("lnA", "p1", "p2", "ln_gamma_b"),
                "SBPL_sfree": ("lnA", "p1", "p2", "ln_gamma_b", "ln_s"),
                "MIX": ("lnA", "pbar", "kappa=sigma_p^2"), "JP": ("lnA", "p", "ln_nu_b")}[self.key]

    @property
    def k(self):
        return len(self.names)

    def to_lib(self, phi):
        phi = np.atleast_2d(np.asarray(phi, float))
        A = np.exp(phi[:, :1])
        if self.key == "PL":
            return np.hstack([A, phi[:, 1:2]])
        if self.key == "SBPL_s2":
            return np.hstack([A, phi[:, 1:4], np.full((len(phi), 1), 2.0)])
        if self.key == "SBPL_sfree":
            return np.hstack([A, phi[:, 1:4], np.exp(phi[:, 4:5])])
        if self.key == "MIX":
            return np.hstack([A, phi[:, 1:3]])
        if self.key == "JP":
            return np.hstack([A, phi[:, 1:2], 0.5 * (phi[:, 2:3] - self.nu_conv)])
        raise ValueError(self.key)

    def from_lib(self, theta):
        th = np.asarray(theta, float)
        lnA = np.log(th[0])
        if self.key == "PL":
            return np.array([lnA, th[1]])
        if self.key == "SBPL_s2":
            return np.array([lnA, th[1], th[2], th[3]])
        if self.key == "SBPL_sfree":
            return np.array([lnA, th[1], th[2], th[3], np.log(th[4])])
        if self.key == "MIX":
            return np.array([lnA, th[1], th[2] ** 2])
        if self.key == "JP":
            return np.array([lnA, th[1], self.nu_conv + 2.0 * th[2]])
        raise ValueError(self.key)

    def moments(self, cfg, phi, N=2):
        """(m, n_bins*(N+1)) moments a(phi) in cfg's bins and metric."""
        th = self.to_lib(phi)
        if self.key == "PL":
            a = E.powerlaw_moments(cfg, th, N)
        elif self.key.startswith("SBPL"):
            a = E.smooth_broken_moments(cfg, th, N)
        elif self.key == "MIX":
            a = _mixture_kappa(cfg, th, N)
        else:
            a = E.jp_moments(cfg, th, N)
        return a.reshape(len(th), -1)

    def steps(self, phi):
        return 1e-5 * np.maximum(1.0, np.abs(phi))

    def moments_and_jacobian(self, cfg, phi, N=2):
        """a(phi) (n,) and da/dphi (n, k) by central differences (one vectorised call)."""
        phi = np.asarray(phi, float)
        h = self.steps(phi)
        pts = np.vstack([phi[None], phi[None] + np.diag(h), phi[None] - np.diag(h)])
        a = self.moments(cfg, pts, N)
        jac = (a[1:self.k + 1] - a[self.k + 1:]).T / (2.0 * h[None, :])
        return a[0], jac

    # ------------------------------------------------------------ invariants
    def invariant_names(self):
        return {"PL": ("ln[C <B_perp^((p+1)/2)>]",),
                "SBPL_s2": ("ln nu_b [Hz]", "ln[C1 <B_perp^((p1+1)/2)>]", "ln[C2 <B_perp^((p2+1)/2)>]"),
                "SBPL_sfree": ("ln nu_b [Hz]", "ln[C1 <B_perp^((p1+1)/2)>]", "ln[C2 <B_perp^((p2+1)/2)>]"),
                "MIX": ("p_local(nu*=1GHz)", "ln[C* <B_perp^((p*+1)/2)>] at nu*=1GHz", "kappa"),
                "JP": ("ln nu_b [Hz]", "ln[C <B_perp^((p+1)/2)>]")}[self.key]

    def invariants(self, cfg_field, phi):
        """Invariants under B -> lam B, N(gamma) -> lam^-1/2 N(gamma sqrt(lam)); C in N = C gamma^-p."""
        phi = np.asarray(phi, float)
        lnA = phi[0]
        lnnu = np.log(K.a1 * cfg_field.mean_bperp)
        if self.key == "PL":
            p = phi[1]
            return np.array([lnA + (p - 1.0) * LN_PIVOT + np.log(_bperp_moment(cfg_field, 0.5 * (p + 1)))])
        if self.key.startswith("SBPL"):
            p1, p2, lgb = phi[1], phi[2], phi[3]
            c1 = lnA - LN_PIVOT + p1 * lgb + np.log(_bperp_moment(cfg_field, 0.5 * (p1 + 1)))
            c2 = lnA - LN_PIVOT + p2 * lgb + np.log(_bperp_moment(cfg_field, 0.5 * (p2 + 1)))
            return np.array([lnnu + 2.0 * lgb, c1, c2])
        if self.key == "MIX":
            pbar, kap = phi[1], phi[2]
            lng = 0.5 * (np.log(NU_STAR) - lnnu)  # gamma_* with a1 <B_perp> gamma_*^2 = nu_*
            y = lng - LN_PIVOT
            pst = pbar - kap * y
            lnN = lnA - LN_PIVOT - pbar * y + 0.5 * kap * y**2
            return np.array([pst, lnN + pst * lng + np.log(_bperp_moment(cfg_field, 0.5 * (pst + 1))), kap])
        if self.key == "JP":
            p = phi[1]
            return np.array([phi[2], lnA + (p - 1.0) * LN_PIVOT + np.log(_bperp_moment(cfg_field, 0.5 * (p + 1)))])
        raise ValueError(self.key)

    def invariants_with_errors(self, cfg_field, phi, cov):
        phi = np.asarray(phi, float)
        v = self.invariants(cfg_field, phi)
        h = self.steps(phi)
        G = np.stack([(self.invariants(cfg_field, phi + h[i] * np.eye(self.k)[i])
                       - self.invariants(cfg_field, phi - h[i] * np.eye(self.k)[i])) / (2 * h[i])
                      for i in range(self.k)], axis=1)
        return v, np.sqrt(np.diag(G @ cov @ G.T))


# ----------------------------------------------------------------------------- fitting
class _Model:
    """Caches a(phi), da/dphi between residual and Jacobian calls of least_squares."""

    def __init__(self, fam, cfg, N):
        self.fam, self.cfg, self.N = fam, cfg, N
        self._key, self._val = None, None

    def __call__(self, phi):
        key = tuple(np.asarray(phi, float))
        if key != self._key:
            self._key, self._val = key, self.fam.moments_and_jacobian(self.cfg, phi, self.N)
        return self._val


def residual_maps(prob: Whitened, mode: str, dw, R=None):
    """(residual(a), jac(Ja)) linear maps for 'comp' (rank R) or 'full'."""
    if mode == "comp":
        bh = prob.beta_hat(dw, R)
        sR, VR = prob.s[:R], prob.Vt[:R]
        return (lambda a: sR * (bh - VR @ a)), (lambda Ja: -(sR[:, None] * (VR @ Ja)))
    if mode == "full":
        return (lambda a: dw - prob.W @ a), (lambda Ja: -(prob.W @ Ja))
    raise ValueError(mode)


def fit(prob: Whitened, fam: Family, cfg_basis, dw, mode="comp", R=None, phi0=None, N=2):
    """Levenberg-Marquardt fit; returns dict(phi, cov, sd, chi2, nfev, status)."""
    model = _Model(fam, cfg_basis, N)
    res_a, jac_a = residual_maps(prob, mode, dw, R)
    fun = lambda p: res_a(model(p)[0])  # noqa: E731
    jac = lambda p: jac_a(model(p)[1])  # noqa: E731
    sol = least_squares(fun, np.asarray(phi0, float), jac=jac, method="lm", xtol=1e-13, ftol=1e-13,
                        gtol=1e-13, max_nfev=2000)
    J = jac(sol.x)
    cov = np.linalg.inv(J.T @ J)
    return {"phi": sol.x, "cov": cov, "sd": np.sqrt(np.diag(cov)), "chi2": float(2.0 * sol.cost),
            "nfev": int(sol.nfev), "status": int(sol.status), "success": bool(sol.success)}


def fisher_sd(prob: Whitened, fam: Family, cfg_basis, phi, mode="comp", R=None, N=2):
    """Gauss-Newton sd at phi (no data needed)."""
    _, Ja = fam.moments_and_jacobian(cfg_basis, phi, N)
    J = (prob.s[:R, None] * (prob.Vt[:R] @ Ja)) if mode == "comp" else prob.W @ Ja
    cov = np.linalg.inv(J.T @ J)
    return np.sqrt(np.diag(cov)), cov


def bias_response(prob: Whitened, fam: Family, cfg_basis, phi, mode="comp", R=None, N=2):
    """G (k, 63) with delta phi = G delta d_w to first order (Gauss-Newton)."""
    _, Ja = fam.moments_and_jacobian(cfg_basis, phi, N)
    if mode == "comp":
        J = prob.s[:R, None] * (prob.Vt[:R] @ Ja)
        return np.linalg.solve(J.T @ J, J.T @ prob.U[:, :R].T)
    J = prob.W @ Ja
    return np.linalg.solve(J.T @ J, J.T)


def profile_amplitude(prob: Whitened, fam: Family, cfg_basis, dw, phi, N=2):
    """Replace phi[0] = ln A by its full-likelihood optimum given the other coordinates."""
    phi = np.asarray(phi, float).copy()
    phi[0] = 0.0
    a1 = fam.moments(cfg_basis, phi, N)[0]
    m = prob.W @ a1
    A = float(m @ dw / (m @ m))
    phi[0] = np.log(A) if A > 0 else np.log(1e-300)
    return phi


def multistart(prob, fam, cfg_basis, dw, mode, R, starts, N=2):
    """Fits from several starts (ln A profiled first); returns all results, best first."""
    out = []
    for s0 in starts:
        x0 = profile_amplitude(prob, fam, cfg_basis, dw, s0, N)
        try:
            r = fit(prob, fam, cfg_basis, dw, mode, R, x0, N)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        out.append(r)
    return sorted(out, key=lambda r: r["chi2"])


@dataclass(frozen=True)
class Fixed:
    """A Family with coordinate ``idx`` held at ``value`` (for profile grids)."""

    fam: Family
    idx: int
    value: float

    @property
    def k(self):
        return self.fam.k - 1

    def _full(self, phi):
        phi = np.atleast_2d(np.asarray(phi, float))
        return np.insert(phi, self.idx, self.value, axis=1)

    def moments(self, cfg, phi, N=2):
        return self.fam.moments(cfg, self._full(phi), N)

    def steps(self, phi):
        return 1e-5 * np.maximum(1.0, np.abs(phi))

    def moments_and_jacobian(self, cfg, phi, N=2):
        phi = np.asarray(phi, float)
        h = self.steps(phi)
        pts = np.vstack([phi[None], phi[None] + np.diag(h), phi[None] - np.diag(h)])
        a = self.moments(cfg, pts, N)
        return a[0], (a[1:self.k + 1] - a[self.k + 1:]).T / (2.0 * h[None, :])


def summarise(values, ref, sd):
    """Pull statistics of (values - ref)/sd over realisations (rows; NaN rows ignored)."""
    pulls = (values - ref[None, :]) / sd
    ok = np.all(np.isfinite(pulls), axis=1)
    p = pulls[ok]
    n = int(ok.sum())
    cov1 = np.mean(np.abs(p) < 1.0, axis=0)
    return {"n": n, "pull_mean": p.mean(axis=0), "pull_sd": p.std(axis=0, ddof=1),
            "pull_mean_stderr": p.std(axis=0, ddof=1) / np.sqrt(n),
            "coverage_|pull|<1": cov1, "coverage_binomial_sd": np.sqrt(0.6827 * 0.3173 / n),
            "estimate_mean": values[ok].mean(axis=0), "estimate_sd": values[ok].std(axis=0, ddof=1),
            "sd_hat_median": np.median(sd[ok], axis=0)}


# ----------------------------------------------------------------------------- realisations
def noise_draw(seed, k, cfg):
    """Standard normals (J, 3) for realisation k: default_rng(SeedSequence([seed, 1, k]))."""
    return np.random.default_rng([seed, 1, k]).normal(size=(cfg.n_channels, 3))


def _fit_record(prob, fam, cfg_basis, dw, dhw, mode, R, phi0, phi_retry):
    """One fit; invariants use the assumed field statistics of cfg_basis."""
    for start in (phi0, phi_retry):
        try:
            r = fit(prob, fam, cfg_basis, dw, mode, R, start)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        if r["success"] and np.all(np.isfinite(r["sd"])):
            a = fam.moments(cfg_basis, r["phi"])[0]
            inv, inv_sd = fam.invariants_with_errors(cfg_basis, r["phi"], r["cov"])
            held = float(np.sum((dhw - prob.Wh @ a) ** 2))
            return np.concatenate([r["phi"], r["sd"], [r["chi2"], held], inv, inv_sd, [1.0]])
    m = len(fam.invariant_names())
    return np.full(2 * fam.k + 2 + 2 * m + 1, np.nan)


def realisation_chunk(payload, ks):
    """Fits for realisations ks: UPRE R*, comp at R*, comp at fixed R_seed, full.

    Returns (len(ks), 1 + 3 * width) array: [R*, rec(comp_Rstar), rec(comp_Rseed), rec(full)],
    rec = [phi (k), sd (k), chi2, chi2_held, invariants (m), invariant sd (m), ok].
    """
    cfg, fam = payload["cfg_basis"], payload["family"]
    prob = Whitened(cfg, payload["C"], payload["S"])
    S = payload["S"]
    rows = []
    for k in ks:
        z = noise_draw(payload["seed"], k, cfg)
        dw = prob.whiten(S[cfg.fit_idx] + z[cfg.fit_idx] * prob.sig[cfg.fit_idx])
        dhw = prob.whiten_held(S[cfg.held_idx] + z[cfg.held_idx] * prob.sig[cfg.held_idx])
        R = prob.upre(dw)[0]
        row = [float(R)]
        for mode, RR in (("comp", max(R, fam.k)), ("comp", payload["R_seed"]), ("full", None)):
            row.extend(_fit_record(prob, fam, cfg, dw, dhw, mode, RR,
                                   payload["phi_start"], payload["phi_retry"]))
        rows.append(row)
    return np.array(rows)
