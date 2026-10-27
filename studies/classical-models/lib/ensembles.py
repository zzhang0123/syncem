"""Reference ensembles: densities, direct spectra, true moments, families.

Conventions
  * Unit electron column on the stated support; densities N(gamma) per unit gamma.
  * Moments in the paper metric: a_br = (1/w_b) int_bin x_b^r dN, w_b = reference bin
    masses of p0 ~ gamma^-2.6 (cfg.w_b).  Arrays (n_bins, N+1).
  * Direct spectra (J, 3) = [I, Q, U] per unit column, erg s^-1 Hz^-1 sr^-1, computed by
    quadrature over gamma (panels split at every bin edge, kink and support edge;
    Gauss-Jacobi on panels ending at a (1 - gamma/gamma_c)^alpha cut-off) with the
    eta-resolved continuum channel kernels of basis.py.  ``direct_check`` doubles the
    energy (and, for KP, angular) nodes and reports max |change| / I_j.
  * Parametric families: ``FAMILIES[name](cfg, theta, N)`` returns moments of
    N(gamma) = (A / GAMMA0) f(gamma; theta) for theta of shape (..., k), vectorised.

Ensembles: E1 power law, E2 smoothly broken power law, E3 index mixture, E4-JP and
E4-KP ageing, E5 two-zone line of sight, E1M (E1 with a Maxwellian field; direct spectrum
needs the Maxwellian Config).  E2a (Orlando 2018 LIS) is not implemented here; params.py has an
exploratory E2a branch that needs an input file not distributed with the study.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import integrate
from scipy.special import eval_legendre, roots_jacobi

import basis as B
import kernels as K
from config import PAPER, Config

GAMMA0 = 1e4  # amplitude pivot of the parametric families


# ----------------------------------------------------------------------------- quadrature
def _panels(cfg: Config, lo: float, hi: float, kinks=()):
    """ln-gamma panels covering [lo, hi], split at bin edges and kinks."""
    pts = [np.log(lo), np.log(hi)]
    pts += [e for e in cfg.log_edges if np.log(lo) < e < np.log(hi)]
    pts += [np.log(k) for k in kinks if lo < k < hi]
    pts = np.unique(np.array(pts))
    return list(zip(pts[:-1], pts[1:]))


def _bin_of(cfg: Config, u):
    b = np.searchsorted(cfg.log_edges, u, side="right") - 1
    return np.clip(b, 0, cfg.n_bins - 1)


@dataclass(frozen=True)
class Measure:
    """Quadrature of a measure dN: int f dN ~ sum_i w_i f(gamma_i); nodes carry bin and x_b."""

    gamma: np.ndarray
    w: np.ndarray
    b: np.ndarray
    x: np.ndarray

    def moments(self, cfg: Config, N: int) -> np.ndarray:
        a = np.zeros((cfg.n_bins, N + 1))
        for r in range(N + 1):
            np.add.at(a[:, r], self.b, self.w * self.x**r)
        return a / cfg.w_b[:, None]

    def abs_moments(self, cfg: Config, m: int) -> np.ndarray:
        """int_bin |x_b|^m dN (not scaled by w_b)."""
        out = np.zeros(cfg.n_bins)
        np.add.at(out, self.b, np.abs(self.w) * np.abs(self.x) ** m)
        return out

    def scaled(self, c: float) -> "Measure":
        return Measure(self.gamma, c * self.w, self.b, self.x)


def measure_1d(cfg: Config, density, lo, hi, n, kinks=(), cut=None, alpha=None, regular=None):
    """Measure of N(gamma) dgamma on [lo, hi].

    If cut is given (= hi), the last panel uses Gauss-Jacobi with weight (u_c - u)^alpha,
    u = ln gamma, and ``regular(gamma)`` must return N(gamma) gamma / (u_c - u)^alpha.
    """
    t, wt = leggauss(n)
    gs, ws = [], []
    for ua, ub in _panels(cfg, lo, hi, kinks):
        if cut is not None and np.isclose(ub, np.log(cut), rtol=0, atol=1e-13):
            tj, wj = roots_jacobi(n, alpha, 0.0)
            u = ua + 0.5 * (ub - ua) * (tj + 1.0)
            g = np.exp(u)
            gs.append(g)
            ws.append((0.5 * (ub - ua)) ** (alpha + 1.0) * wj * regular(g))
        else:
            u = 0.5 * (ub - ua) * t + 0.5 * (ub + ua)
            g = np.exp(u)
            gs.append(g)
            ws.append(0.5 * (ub - ua) * wt * g * density(g))
    g, w = np.concatenate(gs), np.concatenate(ws)
    u = np.log(g)
    b = _bin_of(cfg, u)
    x = (u - np.log(cfg.gamma_b[b])) / cfg.h
    return Measure(g, w, b, x)


def direct_from_measure(cfg: Config, m: Measure, factor=None, table=None):
    """(J, 3) direct spectrum of a separable measure (isotropic pitch, eta ~ q)."""
    factor = cfg.azimuth_factor if factor is None else factor
    kI, kQ = B.kbar(cfg, m.gamma, order=0, table=table)
    return B.to_stokes(m.w @ kI[0], m.w @ kQ[0], factor)


def _max_change_over_I(s1, s2):
    return [float(v) for v in np.max(np.abs(s1 - s2) / s2[:, 0:1], axis=0)]


# ----------------------------------------------------------------------------- 1-D ensembles
@dataclass(frozen=True)
class Ensemble1D:
    """Separable ensemble: N(gamma) on [lo, hi], isotropic pitch, paper angles/screen."""

    name: str = ""
    lo: float = 300.0
    hi: float = 1e5
    kinks: tuple = ()

    # subclasses define shape(g) (unnormalised); optional cut-off exponent
    def shape(self, g):  # pragma: no cover
        raise NotImplementedError

    cut: float | None = None
    alpha: float | None = None

    def shape_regular(self, g):  # N gamma / (u_c - u)^alpha (unnormalised), cut-off panel
        raise NotImplementedError

    @cached_property
    def norm(self) -> float:
        """1 / int shape dgamma (adaptive quad on ln gamma panels; QAWS on a cut-off panel)."""
        tot = 0.0
        for ua, ub in _panels(PAPER, self.lo, self.hi, self.kinks):
            if self.cut is not None and np.isclose(ub, np.log(self.cut), rtol=0, atol=1e-13):
                v, _ = integrate.quad(lambda u: self.shape_regular(np.exp(u)), ua, ub, weight="alg",
                                      wvar=(0.0, self.alpha), epsabs=0, epsrel=1e-13, limit=400)
            else:
                v, _ = integrate.quad(lambda u: np.exp(u) * self.shape(np.exp(u)), ua, ub,
                                      epsabs=0, epsrel=1e-13, limit=400)
            tot += v
        return 1.0 / tot

    def density(self, g):
        g = np.asarray(g, float)
        out = np.where((g >= self.lo) & (g <= self.hi), self.norm * self.shape(g), 0.0)
        return out

    def measure(self, cfg: Config = PAPER, n=24) -> Measure:
        reg = (lambda g: self.norm * self.shape_regular(g)) if self.cut is not None else None
        return measure_1d(cfg, lambda g: self.norm * self.shape(g), self.lo, self.hi, n,
                          kinks=self.kinks, cut=self.cut, alpha=self.alpha, regular=reg)

    def moments(self, cfg: Config = PAPER, N=2, n=24):
        return self.measure(cfg, n).moments(cfg, N)

    def abs_moments(self, cfg: Config = PAPER, m=3, n=24):
        return self.measure(cfg, n).abs_moments(cfg, m)

    def direct(self, cfg: Config = PAPER, n=24, table=None):
        return direct_from_measure(cfg, self.measure(cfg, n), table=table)

    def direct_check(self, cfg: Config = PAPER, n=24, table=None):
        s1, s2 = self.direct(cfg, n, table), self.direct(cfg, 2 * n, table)
        a1, a2 = self.moments(cfg, 4, n), self.moments(cfg, 4, 2 * n)
        return {"nodes_per_panel": [n, 2 * n], "max_change_over_I_IQU": _max_change_over_I(s1, s2),
                "moments_max_abs_change": float(np.max(np.abs(a1 - a2)))}

    def describe(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["norm"] = self.norm
        return d


@dataclass(frozen=True)
class PowerLaw(Ensemble1D):
    name: str = "E1"
    p: float = 2.6

    def shape(self, g):
        return (np.asarray(g) / GAMMA0) ** (-self.p) / GAMMA0

    @property
    def theta(self):
        return np.array([self.norm, self.p])


@dataclass(frozen=True)
class SmoothBroken(Ensemble1D):
    name: str = "E2"
    p1: float = 2.0
    p2: float = 3.2
    s: float = 2.0
    gamma_br: float = 7800.0

    def shape(self, g):
        y = np.asarray(g) / self.gamma_br
        return y ** (-self.p1) * (1.0 + y**self.s) ** (-(self.p2 - self.p1) / self.s) / GAMMA0

    @property
    def theta(self):
        return np.array([self.norm, self.p1, self.p2, np.log(self.gamma_br), self.s])


@dataclass(frozen=True)
class IndexMixture(Ensemble1D):
    name: str = "E3"
    pbar: float = 2.6
    sigma_p: float = 0.2
    gamma_0: float = GAMMA0

    def shape(self, g):
        y = np.log(np.asarray(g) / self.gamma_0)
        return np.exp(-self.pbar * y + 0.5 * self.sigma_p**2 * y**2) / GAMMA0

    @property
    def theta(self):
        return np.array([self.norm, self.pbar, self.sigma_p])


@dataclass(frozen=True)
class JaffePerola(Ensemble1D):
    """N = K gamma^-p (1 - gamma/gamma_c)^(p-2) on [300, gamma_c] (isotropised pitch)."""

    name: str = "E4-JP"
    p: float = 2.6
    gamma_c: float = 2e4

    def __post_init__(self):
        object.__setattr__(self, "hi", self.gamma_c)
        object.__setattr__(self, "cut", self.gamma_c)
        object.__setattr__(self, "alpha", self.p - 2.0)

    def shape(self, g):
        g = np.asarray(g, float)
        return (g / GAMMA0) ** (-self.p) * np.clip(1.0 - g / self.gamma_c, 0.0, None) ** (self.p - 2.0) / GAMMA0

    def shape_regular(self, g):
        g = np.asarray(g, float)
        d = np.log(self.gamma_c) - np.log(g)
        ratio = np.where(d > 0, -np.expm1(-d) / np.where(d > 0, d, 1.0), 1.0)
        return g * (g / GAMMA0) ** (-self.p) * ratio ** (self.p - 2.0) / GAMMA0

    @property
    def theta(self):
        return np.array([self.norm, self.p, np.log(self.gamma_c)])


# ----------------------------------------------------------------------------- ageing setup
@dataclass(frozen=True)
class Ageing:
    """Loss coefficients for E4: synchrotron at <B^2> of cfg's B, Thomson IC on U_rad."""

    u_rad_ev: float = 1.0
    t_myr: float = 19.0
    cfg: Config = PAPER

    @property
    def b_s(self):  # isotropic-pitch synchrotron coefficient [s^-1]
        return K.LOSS_COEFF * self.cfg.mean_b2 * K.MUG**2 / (8.0 * np.pi)

    @property
    def b_c(self):
        return K.LOSS_COEFF * self.u_rad_ev * K.EV_ERG

    @property
    def t(self):
        return self.t_myr * K.SEC_PER_MYR

    @property
    def gamma_c_jp(self):
        return 1.0 / ((self.b_s + self.b_c) * self.t)

    def b_kp(self, mu):
        return 1.5 * self.b_s * (1.0 - np.asarray(mu) ** 2) + self.b_c

    def gamma_c_kp(self, mu):
        return 1.0 / (self.b_kp(mu) * self.t)

    def mu_at_cut(self, gamma):
        """|mu| with gamma_c(mu) = gamma (nan if outside [gamma_c(0), gamma_c(1)])."""
        v = 1.0 - (1.0 / (np.asarray(gamma) * self.t) - self.b_c) / (1.5 * self.b_s)
        return np.where((v >= 0) & (v <= 1), np.sqrt(np.clip(v, 0, 1)), np.nan)

    def describe(self):
        return {"u_rad_eV_cm3": self.u_rad_ev, "t_Myr": self.t_myr, "b_s_s^-1": self.b_s,
                "b_c_s^-1": self.b_c, "mean_B2_muG2": self.cfg.mean_b2,
                "gamma_c_JP": self.gamma_c_jp, "gamma_c_KP(mu=0)": float(self.gamma_c_kp(0.0)),
                "gamma_c_KP(mu=1/2)": float(self.gamma_c_kp(0.5)),
                "gamma_c_KP(mu=1)": float(self.gamma_c_kp(1.0)),
                "losses": "dgamma/dt = -b gamma^2; b_s = (4 sigma_T/3 m_e c) <B^2>/(8 pi), "
                          "b_c = (4 sigma_T/3 m_e c) U_rad (Thomson); KP b(mu) = 1.5 b_s (1-mu^2) + b_c"}


@dataclass(frozen=True)
class KardashevPacholczyk:
    """N(gamma, mu) = (K/2) gamma^-p [1 - b(mu) gamma t]^(p-2) on 300 <= gamma < gamma_c(mu).

    Pitch fixed per electron; normalised so int dmu int dgamma N = 1.  Beaming limit:
    the observed emission at eta uses 2 N(gamma, mu = eta).
    """

    name: str = "E4-KP"
    p: float = 2.6
    ageing: Ageing = field(default_factory=Ageing)
    lo: float = 300.0

    @property
    def alpha(self):
        return self.p - 2.0

    def _shape_regular(self, g, gc):
        d = np.log(gc) - np.log(g)
        ratio = np.where(d > 0, -np.expm1(-d) / np.where(d > 0, d, 1.0), 1.0)
        return 0.5 * g * (g / GAMMA0) ** (-self.p) * ratio**self.alpha / GAMMA0

    def _shape(self, g, gc):
        return 0.5 * (g / GAMMA0) ** (-self.p) * np.clip(1 - g / gc, 0, None) ** self.alpha / GAMMA0

    def gamma_measure(self, cfg: Config, mu: float, n: int, scale: float = 1.0) -> Measure:
        """Measure of N(gamma, mu) dgamma at fixed mu (times scale)."""
        gc = float(self.ageing.gamma_c_kp(mu))
        c = scale * self.norm_unscaled_inv
        return measure_1d(cfg, lambda g: c * self._shape(g, gc), self.lo, gc, n, cut=gc,
                          alpha=self.alpha, regular=lambda g: c * self._shape_regular(g, gc))

    def _mu_panels(self, cfg: Config, mu_max: float):
        """[0, mu_max] split where gamma_c(mu) crosses a bin edge (moments are kinked there)."""
        cuts = [float(self.ageing.mu_at_cut(e)) for e in cfg.edges]
        pts = sorted({0.0, mu_max, *[c for c in cuts if np.isfinite(c) and 0 < c < mu_max]})
        return list(zip(pts[:-1], pts[1:]))

    @cached_property
    def norm_unscaled_inv(self) -> float:
        """K such that the density integrates to one (nested QAWS + GL in mu)."""
        tot = 0.0
        for ma, mb in self._mu_panels(PAPER, 1.0):
            t, w = leggauss(32)
            for mu, wm in zip(0.5 * (mb - ma) * t + 0.5 * (ma + mb), 0.5 * (mb - ma) * w):
                gc = float(self.ageing.gamma_c_kp(mu))
                inner = 0.0
                for ua, ub in _panels(PAPER, self.lo, gc):
                    if np.isclose(ub, np.log(gc), rtol=0, atol=1e-13):
                        v, _ = integrate.quad(lambda u: self._shape_regular(np.exp(u), gc), ua, ub,
                                              weight="alg", wvar=(0.0, self.alpha), epsabs=0,
                                              epsrel=1e-12, limit=200)
                    else:
                        v, _ = integrate.quad(lambda u: np.exp(u) * self._shape(np.exp(u), gc), ua, ub,
                                              epsabs=0, epsrel=1e-12, limit=200)
                    inner += v
                tot += 2.0 * wm * inner  # mu in [-1, 1], even density
        return 1.0 / tot

    # ---------------------------------------------------------------- truth quantities
    def _outer(self, cfg: Config, mu_max: float, n_mu: int):
        t, w = leggauss(n_mu)
        nodes, weights = [], []
        for ma, mb in self._mu_panels(cfg, mu_max):
            nodes.append(0.5 * (mb - ma) * t + 0.5 * (ma + mb))
            weights.append(0.5 * (mb - ma) * w)
        return np.concatenate(nodes), np.concatenate(weights)

    def direct(self, cfg: Config = PAPER, n=24, n_eta=16, table=None):
        """int deta q(eta) int dgamma 2 N(gamma, eta) K^iso(gamma, B, eta), separable azimuth."""
        eta, we = self._outer(cfg, cfg.eta_max, n_eta)
        we = we / cfg.eta_max  # q(eta) = 1/(2 eta_max) on [-eta_max, eta_max], even integrand
        sI = np.zeros(cfg.n_channels)
        sQ = np.zeros(cfg.n_channels, complex)
        for e, w in zip(eta, we):
            m = self.gamma_measure(cfg, e, n, scale=2.0)
            KI, KQ = B.channel_kernels(cfg, m.gamma, eta_s=[np.sqrt(1 - e**2)], order=0, table=table)
            sI += w * (m.w @ KI[0, :, 0])
            sQ += w * (m.w @ KQ[0, :, 0])
        return B.to_stokes(sI, sQ, cfg.azimuth_factor)

    def effective_moments(self, cfg: Config = PAPER, N=2, n=24, n_eta=16, absolute_m=None):
        """Effective diagonal moments of N_eff(gamma) = int deta q(eta) 2 N(gamma, eta).

        With absolute_m set, returns int_bin |x_b|^m N_eff dgamma instead (unscaled).
        """
        eta, we = self._outer(cfg, cfg.eta_max, n_eta)
        we = we / cfg.eta_max
        out = np.zeros((cfg.n_bins, N + 1)) if absolute_m is None else np.zeros(cfg.n_bins)
        for e, w in zip(eta, we):
            m = self.gamma_measure(cfg, e, n, scale=2.0)
            out += w * (m.moments(cfg, N) if absolute_m is None else m.abs_moments(cfg, absolute_m))
        return out

    def legendre_moments(self, cfg: Config = PAPER, ells=(0, 2, 4, 8, 16), N=2, n=24, n_mu=24):
        """a_{l,br} = (1/w_b) int_bin x_b^r n_l(gamma) dgamma, n_l = int dmu N P_l(mu); dict l -> array."""
        mu, wm = self._outer(cfg, 1.0, n_mu)
        out = {l: np.zeros((cfg.n_bins, N + 1)) for l in ells}
        for m_, w in zip(mu, wm):
            m = self.gamma_measure(cfg, m_, n, scale=2.0 * w)  # factor 2: mu in [-1,1], even l
            a = m.moments(cfg, N)
            for l in ells:
                out[l] += eval_legendre(l, m_) * a
        return out

    def p2_conditional(self, gammas, n_mu=400):
        """<P_2(mu) | gamma> on a gamma grid (diagnostic; GL on |mu| > mu_c)."""
        res = []
        for g in np.atleast_1d(gammas):
            mc = self.ageing.mu_at_cut(g)
            mc = 0.0 if not np.isfinite(mc) else float(mc)
            if g >= self.ageing.gamma_c_kp(1.0):
                res.append(np.nan)
                continue
            if mc > 0:
                tj, wj = roots_jacobi(n_mu, 0.0, self.alpha)  # weight (1+t)^alpha at mu = mu_c
                mu = mc + 0.5 * (1 - mc) * (tj + 1)
                # 1 - b(mu) g t = 1.5 b_s g t (mu - mc)(mu + mc) for mu > mc
                wts = wj * (1.5 * self.ageing.b_s * g * self.ageing.t * (mu + mc)) ** self.alpha
            else:
                tj, wj = leggauss(n_mu)
                mu = 0.5 * (tj + 1)
                wts = wj * np.clip(1 - self.ageing.b_kp(mu) * g * self.ageing.t, 0, None) ** self.alpha
            res.append(float(np.sum(wts * eval_legendre(2, mu)) / np.sum(wts)))
        return np.array(res)

    def direct_check(self, cfg: Config = PAPER, n=24, n_eta=16, table=None):
        s1 = self.direct(cfg, n, n_eta, table)
        s2 = self.direct(cfg, 2 * n, 2 * n_eta, table)
        a1, a2 = self.effective_moments(cfg, 4, n, n_eta), self.effective_moments(cfg, 4, 2 * n, 2 * n_eta)
        return {"nodes": {"gamma_per_panel": [n, 2 * n], "eta_per_panel": [n_eta, 2 * n_eta]},
                "max_change_over_I_IQU": _max_change_over_I(s1, s2),
                "effective_moments_max_abs_change": float(np.max(np.abs(a1 - a2)))}

    def describe(self):
        return {"name": self.name, "p": self.p, "lo": self.lo, "K": self.norm_unscaled_inv,
                **self.ageing.describe()}


# ----------------------------------------------------------------------------- two-zone E5
@dataclass(frozen=True)
class TwoZone:
    """E5: zones A (steep) and B (flat), power laws on [300, 1e5], zone azimuth densities
    [1 + 0.8 cos 2(phi - phi_z)]/(2 pi); fractions set so that I_A = I_B at the crossing
    channel (1 GHz).  Common known screen."""

    name: str = "E5"
    p_a: float = 3.1
    p_b: float = 2.1
    phi_a: float = 0.25
    dphi: float = np.pi / 4
    amp: float = 0.8
    nu_cross: float = 1e9

    @property
    def zones(self):
        return PowerLaw(name="E5-A", p=self.p_a), PowerLaw(name="E5-B", p=self.p_b)

    def factors(self):
        return (0.5 * self.amp * np.exp(2j * self.phi_a),
                0.5 * self.amp * np.exp(2j * (self.phi_a + self.dphi)))

    def zone_spectra(self, cfg: Config = PAPER, n=24, table=None):
        """Per unit column: list of (I (J,), Qc (J,) complex, no azimuth factor)."""
        out = []
        for z in self.zones:
            m = z.measure(cfg, n)
            kI, kQ = B.kbar(cfg, m.gamma, order=0, table=table)
            out.append((m.w @ kI[0], m.w @ kQ[0]))
        return out

    def fractions(self, cfg: Config = PAPER, n=24, table=None):
        (ia, _), (ib, _) = self.zone_spectra(cfg, n, table)
        j = int(np.argmin(np.abs(cfg.centres - self.nu_cross)))
        fa = ib[j] / (ia[j] + ib[j])
        return float(fa), float(1 - fa), j

    def direct(self, cfg: Config = PAPER, n=24, table=None, fractions=None):
        zs = self.zone_spectra(cfg, n, table)
        fa, fb, _ = self.fractions(cfg, 48, table) if fractions is None else fractions
        ca, cb = self.factors()
        sI = fa * zs[0][0] + fb * zs[1][0]
        sP = fa * ca * zs[0][1] + fb * cb * zs[1][1]
        return np.stack([sI, sP.real, sP.imag], axis=-1)

    def moments(self, cfg: Config = PAPER, N=2, n=24, fractions=None):
        """(a (nb, N+1) real, M2 (nb, N+1) complex) with M2 = <e^{2i phi} x^r>_bin / w_b."""
        fa, fb, _ = self.fractions(cfg, 48) if fractions is None else fractions
        ca, cb = self.factors()
        A, Bz = (z.moments(cfg, N, n) for z in self.zones)
        return fa * A + fb * Bz, fa * ca * A + fb * cb * Bz

    def zone_abs_moments(self, cfg: Config = PAPER, m=3, n=24, fractions=None):
        fa, fb, _ = self.fractions(cfg, 48) if fractions is None else fractions
        A, Bz = (z.abs_moments(cfg, m, n) for z in self.zones)
        return [(fa, A), (fb, Bz)]

    def direct_check(self, cfg: Config = PAPER, n=24, table=None):
        fr = self.fractions(cfg, 48, table)
        s1, s2 = self.direct(cfg, n, table, fr), self.direct(cfg, 2 * n, table, fr)
        return {"nodes_per_panel": [n, 2 * n], "max_change_over_I_IQU": _max_change_over_I(s1, s2)}

    def describe(self):
        ca, cb = self.factors()
        return {"name": self.name, "p_a": self.p_a, "p_b": self.p_b, "phi_a": self.phi_a,
                "phi_b": self.phi_a + self.dphi, "azimuth_factor_a": [ca.real, ca.imag],
                "azimuth_factor_b": [cb.real, cb.imag], "nu_cross_Hz": self.nu_cross}


def derotated_angle(cfg: Config, S):
    """chi_j = (1/2) arg(P_j exp(-2 i lambda_j^2 Psi_bar)) at the channel centre [rad]."""
    lam2 = (K.C_SI / cfg.centres) ** 2
    P = (S[:, 1] + 1j * S[:, 2]) * np.exp(-2j * lam2 * cfg.screen_mean)
    return 0.5 * np.angle(P)


# ----------------------------------------------------------------------------- families
_XG, _WG = leggauss(32)


def _smooth_family_moments(cfg: Config, f, theta, N):
    """Moments of (A/GAMMA0) f(gamma, theta[...,1:]) by 32-node GL per bin, vectorised."""
    theta = np.atleast_2d(np.asarray(theta, float))
    g = cfg.gamma_b[:, None] * np.exp(cfg.h * _XG[None, :])  # (nb, 32)
    vals = f(g[None], theta[:, 1:, None, None]) * theta[:, 0, None, None] / GAMMA0  # (nt, nb, 32)
    wx = cfg.h * _WG[None, :] * g  # dgamma = gamma h dx
    pw = np.stack([_XG**r for r in range(N + 1)])  # (N+1, 32)
    return np.einsum("tbx,bx,rx->tbr", vals, wx, pw) / cfg.w_b[None, :, None]


def powerlaw_moments(cfg: Config, theta, N=2):
    """theta = (A, p): N = (A/GAMMA0)(gamma/GAMMA0)^-p on [300, 1e5]."""
    return _smooth_family_moments(cfg, lambda g, th: (g / GAMMA0) ** (-th[:, 0]), theta, N)


def smooth_broken_moments(cfg: Config, theta, N=2):
    """theta = (A, p1, p2, ln gamma_br, s): N = (A/GAMMA0) y^-p1 (1 + y^s)^(-(p2-p1)/s), y = gamma/gamma_br."""

    def f(g, th):
        p1, p2, lgb, s = th[:, 0], th[:, 1], th[:, 2], th[:, 3]
        y = g / np.exp(lgb)
        return y ** (-p1) * (1.0 + y**s) ** (-(p2 - p1) / s)

    return _smooth_family_moments(cfg, f, theta, N)


def index_mixture_moments(cfg: Config, theta, N=2):
    """theta = (A, pbar, sigma_p): N = (A/GAMMA0) exp(-pbar y + sigma_p^2 y^2/2), y = ln(gamma/GAMMA0)."""

    def f(g, th):
        y = np.log(g / GAMMA0)
        return np.exp(-th[:, 0] * y + 0.5 * th[:, 1] ** 2 * y**2)

    return _smooth_family_moments(cfg, f, theta, N)


_XV, _WV = leggauss(64)


def jp_moments(cfg: Config, theta, N=2, n_v=None):
    """theta = (A, p, ln gamma_c): N = (A/GAMMA0)(gamma/GAMMA0)^-p (1-gamma/gamma_c)^(p-2), 300 <= gamma < gamma_c.

    Per bin, u = ln gamma in [u_lo, min(u_hi, u_c)] is mapped to v = ln(u_c - u); the
    integrand e^v (1 - e^{-e^v})^alpha ~ e^{(1+alpha) v} is smooth in v for any distance of
    the cut-off from the bin, so one 64-node GL rule per bin is uniformly accurate; the
    lower v limit is truncated where e^{-(1+alpha) V} < e^{-40}.
    """
    xv, wv = (_XV, _WV) if n_v is None else leggauss(n_v)
    theta = np.atleast_2d(np.asarray(theta, float))
    A, p, uc = theta[:, 0, None], theta[:, 1, None], theta[:, 2, None]
    alpha = p - 2.0
    ulo, uhi = cfg.log_edges[None, :-1], cfg.log_edges[None, 1:]
    uup = np.minimum(uhi, uc)
    valid = uup > ulo
    vhi = np.log(np.where(valid, uc - ulo, 1.0))
    with np.errstate(divide="ignore"):
        vlo = np.log(np.where(valid, uc - uup, 1.0))
    vlo = np.maximum(vlo, vhi - 40.0 / (1.0 + alpha))
    v = 0.5 * (vhi - vlo)[..., None] * xv + 0.5 * (vhi + vlo)[..., None]  # (nt, nb, nv)
    wvv = 0.5 * (vhi - vlo)[..., None] * wv
    ev = np.exp(v)
    u = uc[..., None] - ev
    g = np.exp(u)
    integrand = (A[..., None] / GAMMA0) * (g / GAMMA0) ** (-p[..., None]) * g \
        * (-np.expm1(-ev)) ** alpha[..., None] * ev
    x = (u - np.log(cfg.gamma_b)[None, :, None]) / cfg.h
    integrand = np.where(valid[..., None], integrand, 0.0)
    out = np.stack([np.sum(wvv * integrand * x**r, axis=-1) for r in range(N + 1)], axis=-1)
    return out / cfg.w_b[None, :, None]


def two_zone_moments(cfg: Config, theta, factors, N=2):
    """theta = (A_a, p_a, A_b, p_b); factors = (c_a, c_b) complex azimuth factors.

    Returns (a (nt, nb, N+1), M2 (nt, nb, N+1) complex)."""
    theta = np.atleast_2d(np.asarray(theta, float))
    ma = powerlaw_moments(cfg, theta[:, 0:2], N)
    mb = powerlaw_moments(cfg, theta[:, 2:4], N)
    return ma + mb, factors[0] * ma + factors[1] * mb


FAMILIES = {"E1": powerlaw_moments, "E2": smooth_broken_moments, "E3": index_mixture_moments,
            "E4-JP": jp_moments, "E1M": powerlaw_moments}


# ----------------------------------------------------------------------------- registry
def age_for_break(cfg: Config = PAPER, nu_c=8e9, u_rad_ev=1.0, round_myr=0.5):
    """Age (rounded) with a1 <B_perp> gamma_c,JP^2 ~ nu_c."""
    ag = Ageing(u_rad_ev=u_rad_ev, t_myr=1.0, cfg=cfg)
    gc = np.sqrt(nu_c / (K.a1 * cfg.mean_bperp))
    t = 1.0 / ((ag.b_s + ag.b_c) * gc) / K.SEC_PER_MYR
    return round(t / round_myr) * round_myr


AGEING = Ageing(t_myr=19.0)


def reference_ensembles(cfg: Config = PAPER):
    """Dict name -> ensemble object (E1M uses E1's density; its direct spectrum needs PAPER_MAXWELL)."""
    return {
        "E1": PowerLaw(),
        "E2": SmoothBroken(),
        "E3": IndexMixture(),
        "E4-JP": JaffePerola(p=2.6, gamma_c=AGEING.gamma_c_jp),
        "E4-KP": KardashevPacholczyk(p=2.6, ageing=AGEING),
        "E5": TwoZone(),
        "E1M": PowerLaw(name="E1M"),
    }
