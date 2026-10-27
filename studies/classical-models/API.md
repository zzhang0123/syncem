# Core library API (`studies/classical-models/lib`)

Continuum (beaming-limit) radiation only: K_I = A_θ F(x), K_Q = −A_θ G(x), no V. Never call results from this library harmonic.

## Import

```python
import sys
sys.path.insert(0, "lib")  # from studies/classical-models
import kernels as K, basis as B, ensembles as E
from config import PAPER, PAPER_MAXWELL, Config
```

The modules import each other by bare name (`kernels`, `config`, `basis`, `ensembles`, `provenance`), so `lib/` must be on `sys.path`. The scripts run from `studies/classical-models` and put `lib/` on `sys.path` themselves. Set `OPENBLAS_NUM_THREADS` before importing numpy when you run many processes. All functions are single-process. Every call below is cheap enough to run in each worker of a process pool. Kernel tables are cached on disk in `cache/` (atomic write), keyed by the B distribution and by the source hashes of `kernels.py` and `basis.py`.

## Conventions

- Stokes layout: arrays `(..., J, 3)` with components `[I, Q, U]`, where P = Q + iU and J = 41 channels.
- Units: erg s⁻¹ Hz⁻¹ sr⁻¹ per unit electron column. Channel responses have unit area in Hz.
- Bins: 12 equal bins in ln γ on [300, 1e5]. Bin b has centre `cfg.gamma_b[b]`, half-width `cfg.h` (0.242048) and coordinate x_b = ln(γ/γ_b)/h.
- Moments: a_br = (1/w_b) ∫_bin x_b^r dN, with `cfg.w_b` the reference bin masses of p0 ∝ γ^−2.6 (sum 1). Moment arrays have shape `(12, N+1)`.
- Column index: `b*(N+1) + r`. For N = 2 this is the harmonic pipeline's 3b + r.
- Derivatives: `kbar(..., order=n)[r]` is d^r K̄/d(ln γ)^r. The columns use D_{j;br} = (h^r/r!) d^r K̄/d(ln γ)^r = (1/r!) ∂^r_{x_b} K̄ at x_b = 0.
- Equation labels: some docstrings cite LaTeX labels of the paper. They correspond to these equations:

| Label | Equation |
|---|---|
| `eq: harmonic frequencies` | (2.1) |
| `eq: harmonic amplitudes` | (2.2) |
| `eq: harmonic prefactor` | (2.3) |
| `eq: In` | (2.4a) |
| `eq: Vn` | (2.4b) |
| `eq: beaming limit` | (2.7) |
| `eq: directional continuum` | (2.8) |

## `config.py`

`Config` is a frozen dataclass. Its fields and defaults (= paper geometry) are:

| Field | Default | Meaning |
|---|---|---|
| `nu_lo`, `nu_hi`, `n_channels` | 1e8, 1e10, 41 | log-spaced channel centres |
| `chan_half` | 0.2 | h_j = 0.2 ν_j (support half-width of ρ₁) |
| `n_chan_nodes` | 400 | GL nodes per channel |
| `b_kind` | "uniform" | "uniform" on [`b_lo`, `b_hi`] = [4, 6] µG, or "maxwell" with `b_rms` = 5 |
| `n_b` | 16 | B nodes |
| `maxwell_cut` | 8.0 | Maxwellian truncated at 8 B_rms |
| `eta_max`, `n_eta` | 0.5, 12 | η ~ U[−½, ½]; GL on [0, ½] by symmetry |
| `azimuth_amp`, `azimuth_phase` | 0.8, 0.25 | azimuth factor 0.4 e^{0.5i} |
| `screen_mean`, `screen_sd` | 8, 0.5 | Gaussian screen [rad m⁻²] |
| `noise_frac`, `seed` | 0.01, 20260928 | noise model |
| `gamma_lo`, `gamma_hi`, `n_bins`, `ref_slope` | 300, 1e5, 12, 2.6 | energy bins and w_b metric |

Derived quantities, all cached: `centres` (41), `fit_idx` (0, 2, …, 40), `held_idx` (1, 3, …, 39), `chan_nu` (J, T), `chan_w` (T,) (unit-area weights, Σ_t w_t f(ν_t) = ∫R_j f dν), `screen` (J, T) complex, `azimuth_factor` (complex), `b_nodes_weights`, `eta_nodes_weights`, `eta_s` = √(1−η²), `mean_b2` (25.333 µG²), `mean_bperp`, `log_edges`, `edges`, `h`, `gamma_b`, `w_b`.

Methods:
- `noise_sigma(direct (J,3)) -> (J,3)` returns 0.01 I_j for I, Q and U.
- `draw_noise(n_rows, seed=None) -> (n_rows, 3)` returns standard normals from `default_rng(seed)`. `draw_noise(21)` with the paper seed is the harmonic pipeline's draw: data = S[fit_idx] + draw × σ[fit_idx].
- `describe() -> dict`.

Module objects: `PAPER = Config()`, `PAPER_MAXWELL = replace(PAPER, b_kind="maxwell", n_b=96)` (E1M), `Z1` (= ∫ρ₁ = 1.2069003), `rho1(t)`, `gl_interval(n, lo, hi)`.

Variants use `dataclasses.replace`:
- intrinsic polarisation: `replace(PAPER, screen_mean=0, screen_sd=0, azimuth_amp=2.0, azimuth_phase=0.0)`, which gives azimuth factor 1;
- refinement checks: `replace(PAPER, n_chan_nodes=800, n_b=32, n_eta=24)`.

## `kernels.py`

- Constants: `E_ESU`, `M_E`, `C_CGS`, `C_SI` (identical to `syncem.constants`, unchanged since SyncMoments d51f9be), `SIGMA_T`, `EV_ERG`, `MUG`, `SEC_PER_MYR`.
  - `A1` = A_θ per µG of B⊥ (1.86558e-29).
  - `a1` = a_B per µG (4.198873 Hz).
  - `LOSS_COEFF` = 4σ_T/(3 m_e c).
  - `F_SMALL_X_COEFF` = 2.149528 (F ≈ 2.1495 x^{1/3}).
- `log_derivatives(x, order=0, which="F"|"G", nodes=200, cutoff=70.) -> x.shape+(order+1,)` returns (−x d/dx)^r W(x) for r = 0..order, order ≤ 8.
- `scaled_log_derivatives(x, order, which) -> (ln E0, f)` is the log-safe form W^{(r)} = x e^{−x} E0 f_r.
- `F(x)`, `G(x)`, `G_bessel(x)` (G via scipy kve).
- `random_field_kernel(x)` is R(x) of Crusius & Schlickeiser / Ghisellini et al.
- `mellin_F(m)`, `mellin_G(m)`, `powerlaw_pol_fraction(p)` = (p+1)/(p+7/3).

## `basis.py`

- `get_table(cfg, max_order=6) -> KernelTable` returns the B-averaged spline tables of ln T_W and of the signed ratios τ_r (no clamped logs).
  - `KernelTable.direct(y, which)` gives the same quantities by quadrature, without the spline.
  - `KernelTable.spline_check(which)` reports the spline error.
- `channel_kernels(cfg, gammas, eta_s=None, order=0, table=None) -> (KI, KQ)` gives η-resolved channel kernels.
  - Shapes: `(order+1, n_g, n_e, J)`. KQ is complex, includes the screen and excludes the azimuth factor.
  - `eta_s` are values of √(1−η²); the default is `cfg.eta_s`.
- `eta_weights(cfg, pitch_l=None)` returns the η weights: q(η) for the isotropic average, or q(η)(2l+1)P_l(η) for pitch column l.
- `kbar(cfg, gammas, order=0, weights=None, pitch_l=None, table=None) -> (KIbar, KQbar)` with shapes `(order+1, n_g, J)`.
  - These are the η-averaged kernels. Separable P = `cfg.azimuth_factor * KQbar`.
  - `weights` may be `(n_e,)` or `(n_g, n_e)`.
- `to_stokes(kI, kQ, factor) -> (..., 3)` returns `[I, Re(factor·kQ), Im(factor·kQ)]`.
- `derivative_spectra(cfg, N, pitch_l=None) -> (DI (12, N+1, J), DQ (12, N+1, J) complex)` uses analytic derivatives.
- `stencil_derivative_spectra(cfg, step=0.04, pitch_l=None) -> (DI, DQ)` for r = 0, 1, 2. This is the paper's 5-point stencil, provided for cross-checks.
- Column builders:
  - `separable_columns(cfg, DI, DQ, factor=None) -> (J, 3, 12(N+1))`: P columns = factor × D^Q, sharing a_br with I.
  - `joint_columns(cfg, DI, DQ) -> (J, 3, 3·12(N+1))`: unknowns `[a_br, Re M2_br, Im M2_br]`, with I rows from a and P rows from M2 (E5).
  - `q_columns(cfg, DQ) -> (J, 12(N+1)) complex`: w_b D^Q. Use these for the free-complex-factor comparator, P = c · q_columns @ a.
  - `build_basis(cfg, N=2, variant="separable"|"joint"|"q", pitch_l=None)` wraps the three builders.
- `predict(C, a) -> (J, 3)`.
- `bin_derivative_grid(cfg, order, n_grid=65, pitch_l=None) -> (x, dI (12, n_grid, J), dQ (complex))` gives ∂^order_{x_b} K̄ on a grid, for remainder envelopes.

## `ensembles.py`

`reference_ensembles(cfg=PAPER) -> dict` contains the following entries:

| Key | Object | Parameters |
|---|---|---|
| `E1` | `PowerLaw()` | p = 2.6 on [300, 1e5] |
| `E2` | `SmoothBroken()` | p1 = 2.0, p2 = 3.2, s = 2, γ_br = 7800 |
| `E3` | `IndexMixture()` | p̄ = 2.6, σ_p = 0.2, γ₀ = 1e4 |
| `E4-JP` | `JaffePerola(p=2.6, gamma_c=AGEING.gamma_c_jp)` | γ_c = 19672.7 |
| `E4-KP` | `KardashevPacholczyk(p=2.6, ageing=AGEING)` | joint (γ, μ) density |
| `E5` | `TwoZone()` | p_A = 3.1, p_B = 2.1, φ_A = 0.25, Δφ = π/4; I_A = I_B at 1 GHz |
| `E1M` | `PowerLaw(name="E1M")` | direct spectrum must use `PAPER_MAXWELL` |

`AGEING = Ageing(u_rad_ev=1.0, t_myr=19.0, cfg=PAPER)` provides `b_s`, `b_c`, `t`, `gamma_c_jp`, `b_kp(mu)`, `gamma_c_kp(mu)`, `mu_at_cut(gamma)` and `describe()`.

Methods of the 1-D ensembles (E1, E2, E3, E4-JP, E1M):
- `density(g)` is the unit-column N(γ). `norm` is the normalisation and `theta` the true family parameters.
- `measure(cfg, n=24) -> Measure(gamma, w, b, x)` is a quadrature of dN, split at bin edges, kinks and the cut-off. The cut-off panel uses Gauss–Jacobi.
- `moments(cfg, N=2, n=24) -> (12, N+1)`.
- `abs_moments(cfg, m, n=24) -> (12,)` returns ∫_bin |x_b|^m dN, unscaled.
- `direct(cfg, n=24) -> (J, 3)` and `direct_check(cfg, n=24)` (node doubling).

`KardashevPacholczyk` (E4-KP) methods:
- `direct(cfg, n=24, n_eta=16)` computes ∫dη q ∫dγ 2N(γ, η) K^iso(γ, B, η) with the separable azimuth.
- `effective_moments(cfg, N, n=24, n_eta=16, absolute_m=None)` returns the effective diagonal moments of N_eff(γ) = ∫dη q(η) 2N(γ, η). They are the reference truth of E4-KP for the isotropic basis.
- `legendre_moments(cfg, ells, N, n=24, n_mu=24) -> {l: (12, N+1)}` returns moments of n_l(γ) = ∫dμ N P_l(μ), for use with `build_basis(cfg, N, pitch_l=l)`. l = 0 gives the marginal moments.
- `gamma_measure(cfg, mu, n, scale)`, `p2_conditional(gammas)`, `direct_check(cfg)`, `describe()`.

`TwoZone` (E5) methods:
- `fractions(cfg) -> (f_A, f_B, j_cross)`.
- `factors() -> (c_A, c_B)`.
- `direct(cfg, n=24, fractions=None)`.
- `moments(cfg, N, n=24, fractions=None) -> (a, M2 complex)`.
- `zone_abs_moments(cfg, m)`, `zone_spectra(cfg)`.

`derotated_angle(cfg, S) -> chi_j` [rad] de-rotates by the mean depth at the channel centre.

Parametric families (vectorised; `theta` has shape `(n_theta, k)` or `(k,)`; they return `(n_theta, 12, N+1)`):
- `powerlaw_moments(cfg, theta, N)`, θ = (A, p): N = (A/1e4)(γ/1e4)^−p.
- `smooth_broken_moments(cfg, theta, N)`, θ = (A, p1, p2, ln γ_br, s).
- `index_mixture_moments(cfg, theta, N)`, θ = (A, p̄, σ_p).
- `jp_moments(cfg, theta, N)`, θ = (A, p, ln γ_c). It uses a log-distance map that is uniformly accurate for any cut-off position.
- `two_zone_moments(cfg, theta, factors, N) -> (a, M2)`, θ = (A_a, p_a, A_b, p_b).
- `FAMILIES = {"E1", "E2", "E3", "E4-JP", "E1M"}` maps names to these functions.

Each 1-D ensemble's `theta` reproduces its quadrature moments to ≤ 1.3e-14 (`forward_results.json` → `family_checks`). One call with 1000 θ at N = 2 took 3.8–25.2 ms in the distributed run (`forward_results.json` → `family_checks.<name>.seconds_per_1000_theta_N2`) and 4.2–28.0 ms in the original run of the study, on the machine named in `README.md`; Sec. 5.4 of the paper quotes 4–28 ms.

## `provenance.py`

- `provenance(script, t_start, inputs=()) -> dict` returns the `provenance` block of a results JSON: the script and input paths relative to `studies/classical-models`, their SHA-256, the SHA-256 of `kernels.py`, `config.py`, `basis.py`, `ensembles.py` and `provenance.py`, the versions of Python, numpy and scipy (and of matplotlib, jax and syncem when the script has imported them), the wall-clock time and the finishing time.
- `relative(path) -> str` returns a path inside `studies/classical-models` relative to it. It raises `ValueError` for a path outside it.
- `sha256(path)` and `write_json(path, payload)` (numpy arrays and scalars are converted).

## `figure_style.py`

A byte copy of the style module used for the paper's figures: `configure()` sets TeX fonts (newtx) and 8–9 pt text, and `WIDTH = 6.1` in is the final text width. The figure scripts use `configure` and `WIDTH` and save with their own `save` functions. `configure()` needs a LaTeX installation with the newtx fonts.

## `syncem_compat.py`

`use_syncem()` imports `syncem` from the package root of this repository and registers it under its earlier name, so that the harmonic-pipeline modules in `inputs/harmonic-radio-fit` (`radio_numerics.py`, `stable_lines.py`) run without code changes; `inputs/README.md` records the one path edit in `radio_numerics.py`. It first checks that `syncem.constants` and `syncem.model.high_order_harmonic` equal their SyncMoments d51f9be sources up to the rename. `bridge_harmonic_probe.py`, `verify_bridge.py` and part G of `kp_bridge.py` use it. `check_record()` returns what it checked, for a results JSON.

## Examples

```python
import numpy as np
from dataclasses import replace

cfg = PAPER
C = B.build_basis(cfg, N=2)                       # (41, 3, 36) separable
ens = E.reference_ensembles(cfg)
S = ens["E2"].direct(cfg)                         # (41, 3) direct spectrum
a = ens["E2"].moments(cfg, N=2)                   # (12, 3) true moments
sig = cfg.noise_sigma(S)                          # (41, 3)
data = S[cfg.fit_idx] + cfg.draw_noise(len(cfg.fit_idx)) * sig[cfg.fit_idx]
design = C[cfg.fit_idx].reshape(-1, 36) / sig[cfg.fit_idx].reshape(-1, 1)
resid = np.max(np.abs(S - B.predict(C, a)) / S[:, :1])   # representation error

a_theta = E.FAMILIES["E2"](cfg, ens["E2"].theta, N=2)[0]  # equals a

# E5 joint-azimuth model
CJ = B.build_basis(cfg, 2, "joint")               # (41, 3, 108)
a5, M2 = ens["E5"].moments(cfg, 2)
pred = B.predict(CJ, np.concatenate([a5.ravel(), M2.real.ravel(), M2.imag.ravel()]))

# KP with pitch-Legendre columns (use n_eta = 32 for l up to 16)
cfgp = replace(cfg, n_eta=32)
leg = ens["E4-KP"].legendre_moments(cfg, ells=range(0, 17, 2), N=2)
pred_kp = sum(B.predict(B.build_basis(cfgp, 2, pitch_l=l), leg[l]) for l in range(0, 9, 2))  # L = 8

# E1M: Maxwellian-field data, fitted with the U[4,6] basis
S_m = ens["E1M"].direct(PAPER_MAXWELL)
```

Precomputed direct spectra: `forward_spectra.npz`, with keys E1, E2, E3, E4_JP, E4_KP, E5, E1M and centres. `python forward.py` writes it; like the other npz products it is not in git and is attached to the GitHub Release. The same spectra are in `forward_results.json` → `direct_spectra_IQU_per_unit_column`.
