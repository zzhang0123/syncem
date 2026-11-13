# Classical-model study (`studies/classical-models`)

This directory holds the scripts, settings, inputs, results and checksums behind Fig. 3 of Sec. 3.4, Sec. 5 (examples and physical interpretation) and Appendix B (test ensembles and comparisons) of the paper "A statistical framework for the modelling of synchrotron emission". From this directory, `python run_all.py` runs the producing scripts in dependency order, then the verify scripts, then the figure scripts (`make_figures.py`, `render_classical_limit.py`, `fig_cmb_extrapolation.py`, `emission_windows.py` and `derivative_spectra_figure.py`). It stops at the first script that fails, and after a complete run it rewrites `SHA256SUMS`. The run recorded here took about 83 minutes on the machine named below, the sum of the per-script wall times that `run_all.py` prints, with the kernel tables already in `cache/`. `cost_reference_check.py` takes 30 of these minutes. Peak memory was not recorded for this run.

`SHA256SUMS` lists the SHA-256 checksums of the inputs and of every results JSON, table, npz product and figure, with paths relative to this directory. The results JSONs and `tables/` are in git. The npz products and `figures/` are not. They are attached to the GitHub Release of SyncEM v0.7.5 as the archive `classical-models-outputs-v0.7.5.tar.gz`, with paths relative to this directory. After extracting the archive here, `sha256sum -c SHA256SUMS` (or `shasum -a 256 -c SHA256SUMS`) checks every file. Without it, `sha256sum -c --ignore-missing SHA256SUMS` checks the files in git. For the outputs of v0.6.0, a rerun on the machine and versions named below left the checksums of the inputs, the npz products, the figures and the tables other than `tables/cost.tex` unchanged. This check has not been repeated for any output of 0.7.1. On another platform, BLAS library or TeX installation the npz products and figures may differ in the last bits. The results JSONs and `tables/cost.tex` change on every run, because they record timings, finishing times and the hashes of other results. Their other numbers reproduce, so compare them by value or with `git diff`.

The results were produced on an Apple M3 Ultra with Python 3.12.9, NumPy 2.3.5, SciPy 1.16.3, Matplotlib 3.10.8, mpmath 1.3.0, JAX 0.10.0 and Equinox 0.13.7, with `syncem` 0.7.1.dev0 imported from the root of this repository. `verify_bridge_results.json` records `syncem` 0.7.1, because `verify_bridge.py` was rerun after the version was set. The package at 0.7.1 differs from 0.7.1.dev0 only in the version string. For 0.7.2, `verify_extrap.py`, `fig_cmb_extrapolation.py` and `emission_windows.py` were rerun with `syncem` 0.7.1, whose package code equals that of 0.7.2. For 0.7.3, `make_figures.py` was rerun to draw the true spectrum of Fig. 5 as a wider band. For 0.7.4, `fig_cmb_extrapolation.py` and `emission_windows.py` were rerun to add legends to Figs. 6 and 7. Their numbers are unchanged. For 0.7.5, `fits_cmb.py` was added for Fig. 6(c) and `fig_cmb_extrapolation.py` was rerun, with the error panels of Fig. 6(a, b) removed and the 16–84% prediction ranges of the drawn families shaded in their place. `bridge.py` and the figure scripts need LaTeX with the newtx packages on the TeX search path.

`inputs/` holds every external file that a script reads or hashes, in subfolders named by origin, and `inputs/README.md` gives the origin and SHA-256 of each. `API.md` documents the library in `lib/` and gives the equation numbers of the paper for the LaTeX labels that docstrings cite. Each producing script writes its numbers to a results JSON, and most JSONs record the script, input and library hashes and the package versions under `provenance`. `bridge_mpmath_check.json`, `bridge_harmonic_probe.json`, `verify_baselines_results.json` and `verify_bridge_results.json` have no provenance block. The `verify_*.py` scripts check the producing scripts and write `verify_*_results.json`. `make_figures.py` compares the plotted values with the results JSONs and records these checks in `make_figures_results.json`, and `fig_cmb_extrapolation.py`, `emission_windows.py` and `derivative_spectra_figure.py` record their checks in their own results JSONs. Kernel tables are cached in `cache/`, which the scripts rebuild when it is empty. `params.py` contains an exploratory branch, E2a, that uses an unverified transcription of a published local electron spectrum (Table 5 of Orlando E., 2018, MNRAS 475, 2724, doi:10.1093/mnras/stx3280). The transcription is not distributed, so `params.py` skips E2a, prints a line saying so and records the skip under `E2a_input` in `params_results.json`. No number in the paper or in this README comes from E2a.

The energy range is 300 ≤ γ ≤ 2.96e6 in 19 cells (`lib/config.py`). `regression.py` keeps the 12 cells on [300, 1e5] of the frozen harmonic fit, and `bridge.py`, `verify_bridge.py` and `bridge_harmonic_probe.py` compare with that fit. These scripts, `core_checks.py`, `ridge_sed.py`, `kp_bridge.py`, `kp_bridge_tables.py`, `verify_kpbridge.py` support values of this README that the paper does not state (groups 3, 5 and 15), and `kp_bridge_results.json` is an input of `harmonic_correction.py`.

## Numerical settings and checks

This section records the numerical settings and checks of the study by topic, including values that the paper does not state. Groups 1, 2 and 11 record settings and checks of the harmonic and Faraday-screen computations of Secs. 2–4, which no script in this directory produces.

Values marked **checked** were compared with the results JSON named with them, and all agree to the stated rounding. Values quoted with a JSON key were read from that key. The other values were not re-derived from the results JSONs here.

**Error categories.** Each check carries one of four labels:
- *strict bound*: holds given the stated inputs.
- *numerical check*: holds for the finite case stated.
- *extrapolated estimate*: rests on a measured scaling outside the computed cases.
- *not bounded*: no bound was derived.

Fit statistics (coverage, withheld χ², percentiles over noise realisations) are labelled numerical checks for the realisations stated. Timings are measurements on the stated machine, not error checks. [inferred] marks an interpretation.

**Paper.** Each group ends with a line on where its values appear in the paper, by section, equation, table or figure number. These numbers follow the version of the manuscript that contains the extrapolation test of Sec. 5.3. "README only" lists values that the paper does not state. The section "Paper items and scripts" at the end maps each figure, table and section of the paper to its scripts and outputs.

**Producing scripts.** Each group names the script and results file in this directory that produce its numbers. Where no script here produces them, the group says so.

Notation: I_j is the channel intensity, ν_j the channel centre, η the cosine of the angle between the field and the line of sight, φ the projected-field azimuth and Ψ the Faraday depth. x = ν/(a_B γ²) is the continuum variable, except in the Airy formulae of group 1 and in group 11, which define x locally. x_b = ln(γ/γ_b)/h_b is the cell coordinate, u_b in the paper. Section, equation, table and figure numbers refer to the paper.

### 1. Harmonic forward spectra

**Produced by.** The spectra and checks of this group come from the harmonic pipeline, which is not in this directory. No script in this directory produces these numbers. Fig. 3 is drawn by `render_classical_limit.py` (→ `figures/classical_limit.{pdf,png}`, `render_classical_limit_results.json`) from `inputs/validation/harmonic_continuum_results.json` and from `recover_classical_limit_a_results.json`, which `recover_classical_limit_a.py` recovers from the vector paths of the earlier figure (`inputs/manuscript-fig3/`). No physics is recomputed for Fig. 3.

**Ensembles.**
- Low-energy spectra of Fig. 3(a): uniform energy densities on γ ∈ [1.1, 1.2] and [2, 4]. Fields are uniform on [2, 8] µG (broad) or [4.5, 5.5] µG (narrow, this panel only). The rounded response (k = 1) has f = 0.08.
- Channel-averaging example of Fig. 2(c): N(γ) ∝ γ^−2.6 on [300, 1e5], fields uniform on [2, 8] µG.
- Both: isotropic pitch angles, η uniform on [−1/2, 1/2], projected-field azimuth density [1 + 0.8 cos(2φ)]/(2π), so that ⟨e^{2iφ}⟩_F = 0.4, and an independent Gaussian Faraday screen with mean 8 and standard deviation 0.5 rad m⁻². All these variables are independent.

**Channel responses** (Eq. (3.9) is the case k = 1):

```text
ρ_k(t)       = exp[−t^(2k) / (1 − t²)]   for |t| < 1,      ρ_k(t) = 0   for |t| ≥ 1
R_j^(k)(ν)   = ρ_k[(ν − ν_j)/h_j] / (h_j Z_k)
Z_k          = ∫_{−1}^{1} ρ_k(t) dt
s_k²         = (1/Z_k) ∫_{−1}^{1} t² ρ_k(t) dt
h_j          = f ν_j / s_k
```

k = 1 gives the rounded and k = 2 the flat-centred response. Both are C^∞. f is the fractional rms width.

**Screen factor** of the Gaussian screen at each harmonic frequency:

```text
⟨exp(2i λ_m² Ψ)⟩_scr = exp(2i λ_m² Ψ̄ − 2 λ_m⁴ σ_Ψ²)
```

**Line amplitudes at high order.** At high harmonic order, J′_m formed from differences of neighbouring Bessel values loses accuracy. Line amplitudes near their turning point therefore use

```text
J_m(m z)  ≃  k Ai(x)          J′_m(m z)  ≃  −k² Ai′(x)
k = (2/(m z))^(1/3)           x = m (1 − z) k           z = β_⊥ √(1 − η²) / D
```

([DLMF 10.9.2](https://dlmf.nist.gov/10.9.E2) and [DLMF 9.5.1](https://dlmf.nist.gov/9.5.E1)). This branch is used for m ≥ 1e6, z ≥ 0.9 and x ≤ 10. Lower orders use Bessel evaluations. Lines outside the retained region are bounded by the envelope of [DLMF 10.14.5](https://dlmf.nist.gov/10.14.E5). The dense channel sum is evaluated as an integral over harmonic order plus half-endpoint corrections ([DLMF 2.10.1](https://dlmf.nist.gov/2.10.E1)).

**Checks.**

| Check | Result | Category |
|---|---|---|
| Liénard power: low-energy harmonics summed and integrated over photon directions and isotropic pitch angles, against 2e²γ⁴ω_B²β_⊥²/(3c) averaged over isotropic pitch angles | 6e-15 relative at γ = 1.5 and 2 | numerical check |
| Quadrature refinement, γ^−2.6 spectra of Fig. 2(c) | I_j changes by at most 5.5e-5 relative, and complex P_j by at most 2.1e-5 I_j | numerical check |
| Omitted-line envelope, same spectra | below 2.6e-7 I_j | numerical check |
| Remainders of the retained Airy approximation and of the accelerated integer sum over the whole ensemble | no bound | not bounded |

A related check in this directory, part G of `kp_bridge.py` (`kp_bridge_results.json → G_cubic_airy`), measures the per-line error of the pipeline's cubic-Airy branch. It does not bound the remainder over an ensemble.

**Paper.** Eq. (3.9) in Sec. 3.3, the response with k = 1. Sec. 3.3 gives the ensemble of Fig. 2(c): γ^−2.6 on [300, 1e5], [2, 8] µG, isotropic pitch angles, η on [−1/2, 1/2], ⟨e^{2iφ}⟩_F = 0.4 and the Gaussian screen (mean 8, standard deviation 0.5 rad m⁻²). The caption of Fig. 3 gives B_0 = 5 µG. Sec. 5.4 states that the cone-width estimate of group 4 does not cover the discreteness of the harmonic lines. README only: the low-energy ensemble of Fig. 3(a), the screen-factor formula, the Airy formulae, the branch limits, the DLMF references, the Liénard-power check (6e-15), the refinement values 5.5e-5 and 2.1e-5 and the omitted-line envelope 2.6e-7.

### 2. Low-order harmonic expansion check

**Produced by.** No script in this directory.

This check tests angular and Taylor truncation separately for a correlated ensemble at low harmonic order.

**Ensemble.** γ_0 = 20, ν_* = eB_0/(2πγ_0 m_e c) and ζ = 2(c/ν_*)²Ψ, with wavelength in metres. For independent uniform u, v ∈ [−1, 1]:

```text
γ = 20 (1 + 0.2 u)              B/B_0 = 1 + 0.2 (0.6 u + 0.4 v)
ζ = 4 + 2u + v                  φ = 0.2 + 0.4 μ + 0.2 v   (mod 2π)
ρ(μ, η | u, v) ∝ exp[(2 + u) μ + (1 + 0.5 v) η + 0.75 μ η]   on [−1, 1]²
```

**Channels.** Three channels with unit-peak responses exp[1 − 1/(1 − y_j²)] for |y_j| < 1, where y_j = (ν − ν_j)/(0.65 ν_j) and ν_j/ν_* = 2, 4, 8. All contributing harmonics have m ≤ 40.

**Reference and approximation.** The direct reference averages the line powers over the ensemble, weighting each line by its channel response and, for Q + iU, by its own Faraday phase. The approximation uses L_μ = L_η = L and total Taylor degree N. The reported quantity is

```text
E_{L,N} = 100 max_j max_{X ∈ {I,Q,U,V}} |S^(L,N)_{X,j} − S^harm_{X,j}| / I^harm_j   [%]
```

**Results**: maximum sampled channel discrepancy E_{L,N} in per cent, over I, Q, U, V and the three channels.

| Angular order | N = 0 | N = 1 | N = 2 |
|---|---|---|---|
| L = 2 | 6.506 | 6.748 | 6.698 |
| L = 4 | 1.920 | 1.559 | 0.274 |
| L = 8 | 1.856 | 1.450 | 0.175 |

- Raising N alone does not improve the L = 2 result, which the angular representation limits to about 6.7%.
- At L = 8 the maximum decreases from 1.856% at N = 0 to 0.175% at N = 2.
- Refining quadrature nodes and finite-difference steps changes the N = 2, L = 8 response by less than 7e-7 I_j.
- Category: numerical check. These are sampled discrepancies for this ensemble, not uniform bounds.

**Paper.** Not in the paper. README only: every value of this group.

### 3. Observing configuration and moment bases

**Produced by.**
- Configuration: `lib/config.py` (the defaults of `Config`, module object `PAPER`). The results of `core_checks.py`, `forward.py`, `fits.py`, `params.py`, `baselines.py`, `sensitivity.py` and `cost.py` store the configuration under `config`.
- Continuum columns: `lib/basis.py` and `lib/kernels.py`, checked by `core_checks.py` → `core_checks_results.json` (`kernels_vs_mpmath`, `kbar_quadrature`).
- Sampled Taylor envelope: `forward.py` → `forward_results.json` (`envelope_definition`, `ensembles.<name>.forward.N2`).
- Ranks: `fits.py` → `fits_results.json` (`ensembles.<name>.seeded.rank` and `.n_columns`).
- Continuum–harmonic comparison on the 12 cells of the frozen harmonic fit: `regression.py` → `regression_results.json` (continuum library against the frozen harmonic caches, with `replace(PAPER, gamma_hi=1e5, n_bins=12)`). `bridge.py` → `bridge_results.json` repeats the comparison at the harmonic quadrature nodes (continuum kernel in `bridge_continuum.py`, checked against mpmath by `bridge_mpmath_check.py`, and harmonic kernel on a γ grid from `bridge_harmonic_probe.py`), and `verify_bridge.py` → `verify_bridge_results.json` checks it.
- Harmonic columns and their refinement numbers: harmonic pipeline outside this directory. No script here produces them. `bridge.py` reads the frozen harmonic matrix (`inputs/harmonic-radio-fit/`, listed in `inputs/README.md`) but does not produce these refinement numbers.

**Observing configuration** (basis of Sec. 5).
- 41 channel centres logarithmically spaced over 0.1–10 GHz, each with response ρ_1[(ν − ν_j)/(0.2 ν_j)]/(0.2 ν_j Z_1) (group 1).
- Alternate centres give the 21 fitted channels. The other 20 are withheld.
- Fields uniform on [4, 6] µG, and η uniform on [−1/2, 1/2].
- Projected-azimuth density [1 + 0.8 cos 2(φ − 0.25)]/(2π), so that ⟨e^{2iφ}⟩_F = 0.4 e^{0.5i}.
- Independent Gaussian screen with mean 8 and standard deviation 0.5 rad m⁻², entering at each frequency through the screen factor of group 1.
- Pitch angles isotropic unless an ensemble states otherwise.
- Simultaneous reversal of the two direction cosines cancels V.

**Energy cells and coefficient scales** (Eq. (B.2), checked). Nineteen equal intervals in ln γ on [300, γ_max] with γ_max = 300 (1e5/300)^(19/12) = 2962642.6 (`fits_results.json → config.gamma_hi`). The first 12 cells are the cells of [300, 1e5] of the frozen harmonic fit. The half-width is h_b = ln(γ_max/300)/38 = 0.2420476 (`config.h_b`), unchanged from the 12-cell grid. The fixed scaling

```text
w_b = ∫_{Γ_b} p_0(γ) dγ,        p_0(γ) = 1.6 γ^−2.6 / [300^−1.6 − γ_max^−1.6]
```

is chosen before the singular-value decomposition and defines the Euclidean metric on the 57 coefficients a_br. It does not constrain the fitted moments to follow a power law. Direct spectra are evaluated at energy quadrature nodes within each cell, independently of the derivative columns. The noise-weighted design has 57 columns and numerical rank 26 for the energy-only basis, and 171 columns and numerical rank 61 for the E5 joint basis (`fits_results.json → ensembles.<name>.seeded.n_columns` and `.rank`).

**Binwise Taylor remainder** (not in the paper). For a supplied envelope M^X_{j,b} ≥ sup_{|x_b| ≤ 1} |∂³_{x_b} K̄_{X,j}(γ_b e^{h_b x_b})|,

```text
|ε^bin_{X,j}| ≤ (1/6) Σ_b M^X_{j,b} ∫_{Γ_b} |x_b|³ dN(γ)
```

which propagates through |K|. With M supplied, the category is strict bound. `forward.py` samples the suprema on 65 points per cell, which gives an estimate rather than a bound (checked). At N = 2 no channel error of E1–E4-JP exceeds the sampled envelope (`forward_results.json → ensembles.<name>.forward.N2.cells_resid_exceeds_envelope` = 0). For E1 the largest envelope in I is 25 times the largest error (`max_envelope_over_I_IQU[0]` = 0.0213 against `max_resid_over_I_IQU[0]` = 8.58e-4). As evaluated, the category is numerical check, not a bound. The actual N = 2 errors are in the N = 2 column of Table 2, and the N = 0, 2, 4 table is in group 6.

**Harmonic columns** of the frozen harmonic fit (12 cells on [300, 1e5]).
- (8, 7, 18, 64) nodes in field strength, field-direction cosine, each mapped pitch panel and channel order.
- Five-point differences with step 0.04 in x_b.

| Check | Result | Category |
|---|---|---|
| Quadrature refinement, spectra at the true moments of the γ^−2.6 ensemble | at most 7.8e-8 I_j | numerical check |
| Step halved | prediction changes by at most 2.4e-9 I_j | numerical check |
| Integrated omitted-line envelope | below 1.2e-8 I_j | numerical check |
| Remainders of the Airy-line approximation and the accelerated sum | no bound (group 1) | not bounded |

**Continuum columns** (Sec. 5, checked). Kernels of Eq. (3.11) with analytic logarithmic derivatives of F and G.

| Check | Result | Category |
|---|---|---|
| Analytic log-derivatives of F and G against arbitrary-precision evaluations | agree to 7.5e-14 (`core_checks_results.json → kernels_vs_mpmath`, largest `max_err_over_W0(1+x)^r_by_order` 7.48e-14) | numerical check |
| Averages over 16 field, 12 η and 400 channel nodes, all doubled | averaged kernel and its first two derivatives change by at most 1.0e-10 (`kbar_quadrature → adopted(400,16,12)_vs_(800,32,24)`, largest 1.02e-10) | numerical check |

**Continuum–harmonic comparison** (12 cells, checked against `regression_results.json`).

| Check | Result | Category |
|---|---|---|
| Continuum columns against harmonic columns | differ by at most 1.8e-7 I_j (`bridge_continuum_minus_harmonic.columns_max_abs_diff_over_I` = 1.79e-7) | numerical check |
| Direct spectra, γ^−2.6 ensemble | differ by 4.06e-7 I_j in I and 1.5e-8 I_j in Q and U (`truth_max_abs_diff_over_I_IQU`) | numerical check |
| Fit to the same noise realisation with a fixed threshold D_ii ≥ 10 on the whitened singular values in place of the UPRE rule | seven combinations retained with either basis (`match.retained_7`) | numerical check |
| Adjacent singular values | 10.2128794 and 5.4532910 (continuum, `continuum.s7_s8`), differing from the harmonic values by 2.8e-6 and 2.6e-6 (`s7_s8_diff`) | numerical check |
| Largest representation bias, 0.263 standard deviations (`continuum.max_bias_sigma`) | changes by 5e-8 (`max_bias_sigma_diff`) | numerical check |

These differences combine the γ^−2 correction of the continuum kernel with the error of the Airy-line branch of group 1, which also scales as γ^−2. Group 4 separates them.

**Paper.** Sec. 5.1 gives the 41 channels over 0.1–10 GHz, the nineteen cells and N = 2. App. B gives h_j = 0.2ν_j (f = 8.0%), 300 ≤ γ ≤ γ_max with γ_max ≃ 2.96e6, h_b = ln(γ_max/300)/38 ≃ 0.242, the 57 energy moments, Eq. (B.2), "The singular-value decomposition depends on these weights, which do not force the fitted moments to follow a power law", the 171 real columns of E5, the six monomials of E6 and the numerical ranks 26 and 61. Sec. 5.2 gives the 21 fitted and 20 withheld channels. Table 5 gives [4, 6] µG, η on [−1/2, 1/2], isotropic pitch angles, 0.4e^{0.5i} and the screen. README only: the binwise remainder bound with the 65-point sampling and the factor 25, the harmonic- and continuum-column node counts and refinement numbers (7.8e-8, 2.4e-9, 1.2e-8, 7.5e-14, 1.0e-10) and the 12-cell continuum–harmonic comparison (1.8e-7, 4.06e-7, 1.5e-8, the singular values, 0.263 and 5e-8).

### 4. Harmonic correction to the continuum kernel (Sec. 5.4)

**Produced by.** `harmonic_correction.py`, with `harmonic_cells.py` and `kp_harmonic.py`, → `harmonic_correction_results.json` (`tables_c_I`, `tables_c_Q`, `gamma_scaling`, `checks`, `propagation`) and `harmonic_correction_tables.npz`. It reads the 19-cell direct spectra of `forward_spectra.npz`.

**Method.**
- The frequency-averaged harmonic emissivity is evaluated with Bessel functions of real order: SciPy up to order 1e8 and Olver's uniform expansion beyond, without the Airy branch.
- Isotropic pitch angles, γ = 100 and 300, 1e-3 ≤ x ≤ 30 and 0 ≤ η ≤ 1/2.
- The relative correction to the continuum kernel is written as c_S(x, η)/γ².

**Results** (checked).
- At γ = 300: −63 ≤ c_I ≤ −0.77 and −64 ≤ c_Q ≤ −0.95 (`gamma_scaling.c_I.range_gamma300`, `c_Q.range_gamma300`).
- c_S changes by at most 0.26% between the two energies (`max_rel_change`, 0.00264 and 0.00263).
- Extrapolating c_S/γ² to the energies of the reference ensembles gives corrections of at most 3.9e-7, 3.0e-7 and 4.2e-7 of I_j in I for E1, E2 and E4-JP, all at 0.1 GHz, and at most 1.9e-8 I_j in P (`propagation.<name>.gamma300_xm23.max_abs_dI_over_I` and `.max_abs_dP_over_I`).
- For E1 the mean of γ^−2 weighted by the emission in each channel is 2.3e-7, 2.3e-8 and 2.3e-9 at 0.1, 1 and 10 GHz (`propagation.E1.mean_inv_g2_at_0.1_1_10_GHz`).
- Category: extrapolated estimate. The values rest on the measured γ^−2 scaling and are estimates, not bounds.

**Paper.** Sec. 5.4 gives the correction for the 1/γ width of the emission cone for E1, E2 and E4-JP at about 4.2e-7 I_j in I, Q and U from a two-point γ^−2 extrapolation, and states that it does not cover the discreteness of the harmonic lines or the correction near the pitch-angle cut-offs of E4-KP. App. B gives the real-order Bessel evaluations at γ = 100 and 300. README only: 3.9e-7 and 3.0e-7 per ensemble, 1.9e-8 in P, the E1 values of the mean γ^−2, the method details (SciPy to order 1e8, Olver's expansion beyond, no Airy approximation), the x and η ranges, the c_I and c_Q ranges at γ = 300 and the 0.26% change between γ = 100 and 300.

### 5. KP pitch-angle edge

**Produced by.** `kp_bridge.py`, with `kp_harmonic.py`, → `kp_bridge_results.json`, which holds the full results. `kp_bridge_tables.py` prints tables from that JSON, and `verify_kpbridge.py` → `verify_kpbridge_results.json` checks it. The cut-offs are in `forward_results.json → ensembles.E4-KP.params` (`forward.py`).

**Settings and results.**
- The E4-KP pitch density at fixed γ is ∝ (μ² − μ_c²)₊^{p−2}. The cut-off γ_c(μ) is 16489 at μ = 0, 18767 at μ = 1/2 and 32049 at μ = 1 (checked, `gamma_c_KP(mu=0)`, `gamma_c_KP(mu=1/2)`, `gamma_c_KP(mu=1)`), so μ_c(γ) ≤ 1/2 for 16489 ≤ γ ≤ 18767. The density is therefore not smooth on scales of 1/γ near μ_c.
- Harmonic sums at γ = 20–1000 with the edge held at fixed μ_c show that the factorisation ∫ P_μ(μ) K_X^nat dμ ≃ 2 P_μ(η) K_X^iso, the anisotropic form of Eq. (3.10), fails within a few beam widths of the edge.
- One beam width inside the edge, the relative error of I is −6.9%, −4.9% and −1.6% at x = 0.1, 1 and 3, independent of γ.
- Integrated over η at fixed γ, or over γ in an ordered field, the correction returns to the form c_I/γ² in tests at γ = 20–80, with −3.32 ≤ c_I ≤ 1.80.
- This suggests a correction for E4-KP of the same order as for E4-JP. [inferred]: the correction was not computed for the E4-KP spectra.
- Category: numerical check. No bound was derived. An η distribution whose edge coincides with μ_c was not tested.

**Paper.** Table 5 gives γ_c(μ) = 16489–32049. Sec. 5.4 states that the cone-width estimate does not cover the correction near the pitch-angle cut-offs of E4-KP. README only: the density form (μ² − μ_c²)₊^{p−2}, the failure of the factorisation near the edge, the range γ = 20–1000, −6.9%, −4.9% and −1.6% at x = 0.1, 1 and 3, the return to c_I/γ² at γ = 20–80 and −3.32 ≤ c_I ≤ 1.80.

### 6. Reference ensembles and forward predictions (Sec. 5.1, App. B)

**Produced by.** `lib/ensembles.py` defines the densities. `forward.py` → `forward_results.json` (`ensembles.<name>.params`: loss rates, age, cut-offs, zone parameters, node-doubling checks of every direct spectrum, `e1_analytic`: the E1 analytic and Mellin checks, `summary_max_resid_over_I_IQU`: the forward-prediction table, `ensembles.E4-KP.forward_pitch_legendre_vs_L`: the KP pitch-Legendre columns) and `forward_spectra.npz`. `verify_core.py` → `verify_core_results.json → values` repeats the E1 polarisation fraction and the forward errors. The JP(t_eff) row of the forward table comes from `kp_pseudotruth.py` → `kp_pseudotruth_results.json → forward`. E1M uses `PAPER_MAXWELL` in `lib/config.py`. The joint ensemble is built in `cost.py` → `cost_results.json → joint_gamma_B`. `make_figures.py` with `figure_tables.py` writes `tables/ensembles.tex`. The parameter values are in Table 5 of the paper.

**Densities** (Eq. (B.1)), up to normalisation to unit column, on [300, 2.96e6]:

```text
E1:  N ∝ γ^−2.6
E2:  N ∝ (γ/γ_br)^−p_1 [1 + (γ/γ_br)^s]^(−(p_2 − p_1)/s)
E3:  N ∝ (γ/γ_0)^−p̄ exp[(1/2) σ_p² ln²(γ/γ_0)]
E4:  N ∝ γ^−p (1 − b γ t)^(p−2),    γ < γ_c = 1/(b t)
```

**E4 loss rates and support** (checked against `forward_results.json → ensembles.E4-KP.params`).
- b = b_s + b_c for JP and b(μ) = (3/2) b_s (1 − μ²) + b_c for KP.
- b_s = 3.274e-20 s⁻¹ from synchrotron losses at ⟨B²⟩_col = 25.33 µG² (`b_s_s^-1`, `mean_B2_muG2`).
- b_c = 5.204e-20 s⁻¹ from Thomson inverse-Compton losses on 1 eV cm⁻³ (`b_c_s^-1`).
- t = 19 Myr and γ_c = 19672.7 for JP (`t_Myr`, `gamma_c_JP`).
- The aged form holds for an impulsive power-law injection that extends to arbitrarily high energy. An injection cut at γ_{0,max} ages to a cut at γ_{0,max}/(1 + γ_{0,max} b t) < γ_c.
- The aged form is kept on [300, γ_c(μ)), and each pitch angle is normalised to unit column.

**E5.** Two power laws on [300, 2.96e6] with p_A = 3.1 and p_B = 2.1, and azimuth densities [1 + 0.8 cos 2(φ − φ_z)]/(2π) with φ_A = 0.25 and φ_B = φ_A + π/4. The zone-A fraction f_A = 0.938 equalises the zone intensities in the 1 GHz channel (checked, `ensembles.E5.fractions_A_B_crossing_channel[0]` = 0.93788).

**E1M.** B is drawn from √(2/π) B² σ_B^−3 exp[−B²/(2σ_B²)] with σ_B = B_rms/√3.

**Joint ensemble** (E6 of the paper, group 10). It ages the same injection with pitch-angle isotropisation (JP), synchrotron losses in its local field with B uniform on [4, 6] µG, and the E4 inverse-Compton losses. At the outermost field quadrature nodes γ_c is 16949.5 and 22901.4 (checked, `cost_results.json → joint_gamma_B.gamma_c_range`).

**Direct-spectrum quadrature** (checked). Gauss–Legendre panels are split at every break, cut-off and support edge.

| Check | Result | Category |
|---|---|---|
| Energy nodes doubled | direct spectra change by at most 3.3e-15 I_j (`ensembles.<name>.direct_check_energy_nodes.max_change_over_I_IQU`, largest for E1M) | numerical check |
| Channel, field and η nodes doubled | at most 2.9e-14 I_j (`direct_check_config_doubling_...`, largest for E2) | numerical check |

**E1 against the analytic power law** (checked). Source: `forward_results.json → e1_analytic`.

| Check | Result | Category |
|---|---|---|
| Linear-polarisation fraction of the direct E1 spectrum at 0.1 GHz, ordered field without azimuth or screen factors, against (p + 1)/(p + 7/3) for p = 2.6 | 0.7297298 against 0.7297297, difference 2.1e-8 (`Pi_direct_no_screen_at_0.1_0.316_1_3.16_10GHz[0]`, `Pi_unbounded_(p+1)/(p+7/3)`, `abs(Pi_direct(0.1GHz) - (p+1)/(p+7/3))`) | numerical check |
| Finite support evaluated through incomplete Mellin integrals of F, against the direct quadrature | agree to 3.4e-12 in I (`max_abs(I_direct/I_exact_incomplete_mellin - 1)` = 3.39e-12) and to 9.9e-13 I_j in P (`max_abs(P_direct - P_exact)/I`) | numerical check |
| Upper support edge γ = 2.96e6, over the 41 channels | intensity at most 1.7e-5 below the unbounded power law (`max_abs(I_direct/I_unbounded - 1)` = 1.68e-5) | numerical check |

**Forward predictions from moments** (checked). The entry is 100 max_{j,X} |S^direct_{X,j} − (C a)_{X,j}|/I_j over the 41 channels and X ∈ {I, Q, U}, in per cent, for Taylor order N. The basis has isotropic pitch angles and one azimuth factor unless stated. For E4-KP, "η-averaged" uses the moments of the effective density ∫dη P_η(η) 2P_μ(μ = η | γ) N(γ), with P_η uniform on [−1/2, 1/2] and N the electron density. "JP(t_eff)" uses the moments of the JP spectrum with t_eff = 1.1475 t. "marginal" uses the energy moments without pitch dependence, and "pitch-Legendre" adds pitch-angle Legendre columns along μ = η up to ℓ = 8.

| Ensemble | Moments and basis | N = 0 | N = 2 | N = 4 |
|---|---|---|---|---|
| E1 | energy | 2.6 | 0.086 | 0.0088 |
| E2 | energy | 3.8 | 0.091 | 0.012 |
| E3 | energy | 2.7 | 0.10 | 0.0096 |
| E4-JP | energy | 18 | 0.36 | 0.045 |
| E4-KP | η-averaged | 23 | 2.6 | 1.7 |
| | JP(t_eff) | – | 0.89 | 0.064 |
| | pitch-Legendre, ℓ ≤ 8 | – | 0.88 | 0.066 |
| | marginal | 103 | 78 | 78 |
| E5 | joint energy–azimuth | 3.5 | 0.18 | 0.014 |
| | energy, one azimuth factor | 29 | 29 | 29 |
| E1M | energy, Maxwellian-field basis | 2.5 | 0.062 | 0.0018 |
| | energy, uniform-field basis | 6.4 | 3.8 | 3.8 |

- First order alone: adding the N = 1 terms changes the error by 10% of its N = 0 value for E4-JP (18.5% to 16.6% of I_j) and by less than 3% for E1–E3 (`summary_max_resid_over_I_IQU`, entries N0 and N1).
- From N = 0 to N = 2 the error of E1–E4-JP falls by factors of 27–51, and from N = 2 to N = 4 by factors of 8–11 (ratios of the entries above at full precision).
- Source: `forward_results.json → summary_max_resid_over_I_IQU` (maximum over I, Q and U, keys `E1` … `E1M`, `E4-KP_marginal(L=0)`, `E5_separable_paper_factor`, `E1M_U46_basis`), `forward_results.json → ensembles.E4-KP.forward_pitch_legendre_vs_L.<N>.L8` for the pitch-Legendre row and `kp_pseudotruth_results.json → forward.N2` and `forward.N4` (`max_abs_S_KP_minus_C_a_over_I_IQU`, 8.87e-3 and 6.41e-4) for the JP(t_eff) row. The JP spectrum with t_eff reproduces the E4-KP spectrum to 7.7e-5 I_j (`forward.max_abs_S_JP_minus_S_KP_over_I_IQU[0]`). Every entry agrees to its rounding.
- Category: numerical check. These are measured differences for the stated ensembles and parameters, not bounds.

**Figures not in the paper.** `figures/reference_ensembles_compact.pdf` shows the spectra and polarisation fractions of the reference ensembles. `figures/reference_ensembles.pdf` has two further panels: (c), the ordered-field polarisation fraction of all seven ensembles against (p + 1)/(p + 7/3) for p = 2.6, and (d), the E5 polarisation angle without a screen and after removal of the mean screen rotation where |P_j|/I_j > 3%, with each zone alone. `make_figures.py` with `figure_plots.py` draws both files.

**Paper.** Table 5 gives the parameters, the cut-off range γ_c(μ) = 16489–32049 of E4-KP, the E6 cut-off range 16950–22900 at the outermost field quadrature nodes and f_A = 0.938. App. B gives Eq. (B.1), the JP and KP forms of b with b_s and b_c named but not evaluated and ⟨B²⟩_col = 25.3 µG², the impulsive injection without an upper cut-off, the E5 and E1M densities, the statement that E1 realises the power law of Eq. (3.13) over a finite energy range, the E4-KP pitch-Legendre values 0.88% and 0.066%, the JP(t_eff) match 7.7e-5 I_j and "Doubling the quadrature nodes of the true spectrum changes it by at most 3e-14 I_j". Table 2 gives the N = 0, 2 and 4 errors of E1–E4-JP, E4-KP (JP(t_eff)), E5 (both bases) and E1M (uniform-field basis). Sec. 5.1 gives the factors 27–51 and 8–11 and the E5 values 29% and 0.18%. README only: ⟨B²⟩_col = 25.33 µG², the statement on an injection cut at γ_{0,max}, the values of b_s and b_c, 3.3e-15 and the panel splitting, the E1 checks (0.7297298 against 0.7297297, 3.4e-12, 9.9e-13, 1.7e-5), the N = 1 result, the η-averaged and marginal E4-KP rows, the E1M row with the Maxwellian-field basis and the panels of `figures/reference_ensembles.pdf`.

### 7. Fit settings and fit statistics (Secs. 5.2 and 5.4)

**Produced by.**
- Seeds: the seeded realisation of `fits.py`, `params.py` and `baselines.py` (`lib/config.py`, field `seed`, and `fits_results.json → settings`). The seeds of the repeated noise realisations are set in these scripts and recorded in their results JSONs.
- Coverage realisations: `fits.py` → `fits_results.json`, with the rank tail in `fits_tails.py` → `fits_tails_results.json` and the E4-KP reference in `kp_pseudotruth.py` → `kp_pseudotruth_results.json`. `verify_fits.py` → `verify_fits_results.json → checks.joint_coverage_vs_representation_bias` compares the measured coverage with the coverage predicted from the representation shifts.
- Low-noise fits of Fig. 5: `fits_lownoise.py` → `fits_lownoise_results.json` and `fits_lownoise_products.npz`, checked by `verify_fits.py` (`checks.low_noise`).
- Parameter realisations and starting points: `params.py` with `paramlib.py` → `params_results.json` and `params_realisations.npz`, and `verify_params.py` → `verify_params_results.json` checks them.
- Baseline realisations: `baselines.py` → `baselines_results.json`, and `baselines_noise_sweep.py` → `baselines_noise_sweep_results.json`.
- Rank search, E5 pairs and nonnegative witness: `fits.py`. `params.py` uses the same search.
- Fit statistics not in the paper: `fits_results.json → ensembles.<name>.seeded.chi2_held_predictive`, `.realisations.coverage_fixed_R.joint_ellipsoid_analytic` and `.joint_ellipsoid_empirical`, and `.realisations.coverage_post_selection.joint_ellipsoid_empirical`, the expectations in `fits_tails_results.json → ensembles.<name>.expectation_predictive_fixed_R_seeded`, the E4-KP row in `kp_pseudotruth_results.json → fit_summary.pseudo_truth_JP_t_eff` and, against the η-averaged density, `fit_summary.spec_truth_effective_diagonal`, and the E5 one-factor row in `fits_results.json → E5.separable_free_factor.seeded`. `make_figures.py` with `figure_tables.py` writes `tables/fit_summary.tex` (the table with these statistics) and `tables/fit_summary_compact.tex` (the fit columns of Table 2 and a Bias column).
- E1M and E4-KP blocks of the parameter table: `params.py` → `params_results.json → tasks.E1M_U46basis` and `tasks.E4-KP_JPfamily`. `tables/parameter_recovery.tex` holds the full table, with these blocks.

**Settings.**
- The fitted rows use the noise realisation of seed 20260928 and the withheld rows that of seed 20260929. All fit rows of Table 2 use this one realisation and the same kernel columns.
- Coverage uses 4000 realisations, the parameters 1000 and the intensity baselines 500.
- The rank search runs up to the numerical rank of each design: R = 1, …, 26 for the 57 energy columns and R = 1, …, 61 for the 171 columns of E5 (`fits_results.json → ensembles.<name>.seeded.rank`, group 3). For E5, exactly degenerate pairs of singular values are retained or discarded together.
- In the Bias column of `tables/fit_summary_compact.tex`, each exactly degenerate E5 pair of singular values enters through its two-dimensional norm.
- The intensity-only baselines limit the search to R ≤ 14 (`baseline_models.R_CAP`).
- The nonnegative witness uses 17 Chebyshev–Lobatto points per cell and `scipy.optimize.nnls`.
- Parametric fits minimise Eq. (4.18) or the full linear likelihood from 12 starting points.

**Fit statistics not in Table 2** (checked). One noise realisation with 63 fitted and 60 withheld data, R_⋆ chosen by UPRE.
- χ²_held = rᵀ(1 + H_R H_Rᵀ)⁻¹ r is the withheld residual weighted by its predictive covariance, where H_R maps fitted-channel noise to the withheld predictions. "Exp." is its expectation at fixed R_⋆, 60 + μᵀ(1 + H_R H_Rᵀ)⁻¹μ, which includes the mean withheld residual μ left by truncation and representation error.
- Coverage is that of the joint 68.3% region of the retained combinations at fixed R_⋆, predicted from the representation shifts ("pred.") and measured over 4000 realisations ("meas."), and measured with R_⋆ reselected in each realisation.

| Ensemble | χ²_held (exp.) | Cov. fixed, pred. | Cov. fixed, meas. | Cov. reselected |
|---|---|---|---|---|
| E1 | 51.9 (62.4) | 0.676 | 0.666 | 0.557 |
| E2 | 48.1 (60.7) | 0.681 | 0.674 | 0.566 |
| E3 | 52.5 (62.8) | 0.674 | 0.664 | 0.560 |
| E4-JP | 46.8 (60.9) | 0.648 | 0.640 | 0.559 |
| E4-KP† | 47.9 (61.4) | 0.574 | 0.568 | 0.504 |
| E5, joint moments | 52.9 (63.0) | 0.670 | 0.658 | 0.580 |
| E1M | 51.9 (62.4) | 0.000 | 0.000 | 0.000 |
| E5, one azimuth factor | 261.7 | – | – | – |

† Against the moments of the JP spectrum with t_eff/t = 1.1475. Against the moments of the η-averaged KP density the bias is 3.8 (`fit_summary.spec_truth_effective_diagonal.max_abs_rep_bias_over_sigma` = 3.77) and the coverage at fixed R_⋆ is 0.042 predicted and 0.041 measured, and 0.037 reselected (`fits_results.json → ensembles.E4-KP`).

- At fixed R_⋆ the joint region covers the reference moments in 0.640–0.674 of the realisations for E1–E4-JP and E5. The coverage predicted from the representation shifts is 0.648–0.681, and the measured fractions lie within 1.6 binomial standard deviations of it (`verify_fits_results.json → checks.joint_coverage_vs_representation_bias.max_abs_empirical_minus_predicted_over_sd` = 1.63, for E5). Reselecting R_⋆ lowers the coverage to 0.557–0.580. E4-KP gives 0.568 and 0.504, and E1M zero.
- χ²_held lies 10–14 below its expectation in every row, about one standard deviation of χ²₆₀ (√120 ≈ 11). Over the 4000 realisations the median predictive χ²_held is 61.5–63.1 (`fits_results.json → summary.<name>.chi2_held_predictive_p50_realisations`), close to the expectations. [inferred] The common offset belongs to the one noise realisation that all rows share.
- The E5 one-factor value is `E5.separable_free_factor.seeded.chi2_held_predictive_fixed_c`.
- Category: numerical check, for this realisation and these 4000 realisations. All entries agree with the JSON keys above to the rounding.

**Low-noise fits** (checked). The noise realisation of Table 2 scaled to 0.01% of I_j, fitted with the same basis and rank rule (`fits_lownoise_results.json → summary.0.0001`). The 1% fits reproduce the predictions and ranks of `fits_products.npz` (`checks`, all differences 0).

| Ensemble | R_⋆ at 1% | R_⋆ at 0.01% | χ²_fit/dof at 0.01% | Max. error at 0.01% [%] |
|---|---|---|---|---|
| E2 | 7 | 16 | 40.7/47 | 0.026 |
| E4-KP | 9 | 16 | 40.8/47 | 0.027 |
| E5 | 17 | 34 | 29.5/29 | 0.039 |

Category: numerical check, for this realisation.

**UPRE rank tail** (checked). For E1–E4 and E1M, UPRE selects R_⋆ ≥ 15 in 3–5% of the 4000 realisations, and the 99th percentile of the largest error over all channels reaches 5–24% of I_j, against 2.9–3.2% if the search is limited to R ≤ 14. The predictive χ²_held does not reveal these cases, because it divides out the prediction variance. Source: `fits_tails_results.json → ensembles.<name>.R_star_frac_ge_15` (0.030–0.054), `.max_err_over_I_at_R_star.p99` (0.052–0.244) and `.sensitivity_capped.max_err_over_I_at_R_star_p99` (0.0289–0.0318). Category: numerical check.

**Parameter recovery, E1M and E4-KP blocks** (checked, not in Table 7). Columns as in `tables/parameter_recovery.tex`: truth or reference, compressed and full estimates ± 1σ for the noise realisation of Table 2, σ_c/σ_f, and the bias of a noise-free compressed fit in units of the Fisher σ at the reference.

| Ensemble (R_⋆) | Parameter | Truth or ref. | Compressed | Full | σ_c/σ_f | Bias [σ] |
|---|---|---|---|---|---|---|
| E1M† (7) | ln A | −5.1405 | −5.1794 ± 0.0022 | −5.1794 ± 0.0022 | 1.0000 | −17.90 |
| | p | 2.6000 | 2.5977 ± 0.0031 | 2.5976 ± 0.0031 | 1.0002 | −0.01 |
| E4-KP‡ (9) | ln A | −5.1157 | −5.1175 ± 0.0050 | −5.1175 ± 0.0050 | 1.0000 | +0.17 |
| | p | 2.5998 | 2.5974 ± 0.0061 | 2.5973 ± 0.0061 | 1.0001 | −0.22 |
| | ln(ν_b/Hz) | 22.499 | 22.495 ± 0.011 | 22.495 ± 0.011 | 1.0001 | −0.48 |

† Maxwellian-field data fitted with the uniform-field basis. The reference is the E1 truth. ‡ KP data fitted with the JP family. The reference is the exact forward JP fit to the noise-free KP spectrum.

- Sec. 5.4 of the paper states the noise-free E1M amplitude 3.7% low, 17.2 standard deviations for the assumed noise (`tasks.E1M_U46basis.noisefree.comp.phi_minus_truth[0]` = −0.0374 and `pull_vs_truth[0]` = −17.24). The Bias column divides the same shift by the Fisher σ at the reference (`shift_over_fisher_sd[0]` = −17.90). The 17.6σ in the Bias column of `tables/fit_summary_compact.tex` refers to the best-measured combination (`fits_results.json → summary.E1M.max_abs_rep_bias_sigma` = 17.57).
- Sec. 5.4 states the E4-KP effective age (1.150 ± 0.006) t over 1000 realisations (`tasks.E4-KP_JPfamily.realisations.t_eff_over_t_comp_Rstar_mean_sd`) and 1.1475 t without noise (`kp_pseudotruth_results.json → pseudo_truth.t_eff_over_t`).
- Source: `params_results.json → tasks.E1M_U46basis` and `tasks.E4-KP_JPfamily` (`reference_point`, `seeded_comp`, `seeded_full`, `fisher_sd_comp_over_full_at_R_seed`, `noisefree.comp.shift_over_fisher_sd`). All entries agree. Category: numerical check.

**Figures not in the paper.** `figures/moment_fit_gallery.pdf` has the layout of Fig. 5 with an extra E1 row. `figures/moments_and_parameters.pdf` has four panels: (a) the E2 moments, (b) the shift of every retained combination with the representation-bias bars, (c) the 68% and 95% regions of (p_2, ln γ_br) for E2 from the compressed likelihood at R_⋆ = 7 and from the full linear likelihood, and (d) the ridge contours with the Δχ² along the ridge below. `figures/moments_and_parameters_compact.pdf` has panels (a) and (d), and `figures/field_energy_ridge.pdf` (group 15) is panel (d) without its letter. The largest bias per ensemble is in the Bias column of `tables/fit_summary_compact.tex`, and σ_c/σ_f ≤ 1.0030 in `tables/parameter_recovery.tex`.

**Paper.** Table 2 gives R_⋆, χ²_fit/dof and Max. error. Sec. 5.2 gives the largest errors 0.86–1.1% of the fits other than the E5 one-factor fit, the one-factor χ² of 271 for 54 degrees of freedom, and the 0.01% fits with R_⋆ = 16, 16 and 34 and errors of at most 0.039%. Its lower panels in Fig. 5 show the 1% and 0.01% fits. Sec. 5.2 also gives the rank tail (3–5% of 4000 realisations, 5–24% of I_j). Sec. 5.4 gives the coverage at fixed rank (64–67%, predicted 64.8–68.1%, within 1.6 binomial standard deviations), the reselected coverage 56–58%, the E1M amplitude shift (3.7%, 17.2 standard deviations), the E4-KP age and the compressed against full agreement within 0.3%. Table 7 gives the compressed and full estimates for E1, E2, E3 and E4-JP. App. B gives the rank search up to the numerical rank, 26 and 61. README only: the seeds, the 12 starting points, the Chebyshev–Lobatto placement and `nnls`, the E5 pair rules, the 17 witness points per cell, the Bias column, χ²_held with its definition and expectation, the per-ensemble coverage values, the E4-KP coverage, the E5 one-factor χ²_held, the χ²_fit/dof of the 0.01% fits, 2.9–3.2% for the capped search, the σ_c/σ_f and Bias columns of the parameter table and its E1M and E4-KP blocks, and the figures listed above.

### 8. Intensity-only baselines (Sec. 5.2, Table 6)

**Produced by.** `spectral_laws.py` (laws), `baseline_models.py` (moment TSVD and electron families through the basis), `baselines.py` → `baselines_results.json` and `baselines_plotdata.npz` (1% noise and the noise-free fits), and `baselines_noise_sweep.py` → `baselines_noise_sweep_results.json` (the same fits at 1%, 0.1% and 0.01% noise). The 1% run of the sweep reproduces the medians of `baselines_results.json` (`archive_1pct_max_abs_diff_rms_held` = 0). `make_figures.py` with `figure_tables.py` writes `tables/baselines.tex` (the 1% table with the 95th percentiles and the uncapped-tail row) and `tables/baselines_compact.tex`. `verify_baselines.py` → `verify_baselines_results.json` checks the baselines, the noise sweep and two values of `cost.py`.

**Laws.** With L = ln(ν/1 GHz) (y in the paper):
- s1: A e^{βL}.
- s3: A e^{βL + CL²}.
- log-cubic: exp(c_0 + c_1 L + c_2 L² + c_3 L³).
- β-moment SED: A e^{β̄L}(1 + ω_1 L + ω_2 L²/2), with ω_1 = 0 or free.

All laws are integrated through the channel responses.

**Electron families** fitted through the 57 basis columns:
- power law (A, p).
- log-parabola N ∝ (γ/1e4)^−p exp[−κ ln²(γ/1e4)] (A, p, κ).
- the E2 form with s = 2 (A, p_1, p_2, γ_br).

**Settings.** Rms of (fit − direct)/I_j over the withheld channels in per cent (20 for E1–E5, 12 native nodes for GALPROP), over 500 realisations at each noise level, with R_⋆ reselected in each realisation and R ≤ 14. Table 6 gives the medians at 1%, 0.1% and 0.01% noise and the noise-free R = 4 row (`baselines_results.json → results.<name>.noise_free.moment_R4`, `baselines_noise_sweep_results.json → results.<noise>.compact_median_rms_held.<name>.<model>`).

**95th percentiles at 1% noise** (checked, not in Table 6). Source: `baselines_results.json → results.<name>.monte_carlo.summary.<model>.rms_res_held.p95`.

| Model | E1 | E2 | E3 | E4-JP | E4-KP | E5 | GALPROP |
|---|---|---|---|---|---|---|---|
| moments, UPRE | 0.97 | 0.96 | 0.98 | 1.0 | 1.1 | 0.98 | 1.3 |
| moments, R = 4 | 4.2 | 2.5 | 4.5 | 5.2 | 6.4 | 5.6 | 3.0 |
| PySM s1 | 0.52 | 8.0 | 0.96 | 21 | 26 | 4.7 | 8.1 |
| PySM s3 | 0.58 | 0.63 | 0.58 | 5.8 | 7.4 | 0.58 | 1.5 |
| β-mom., ω_1 = 0 | 0.58 | 0.76 | 0.58 | 5.6 | 7.1 | 0.59 | 1.6 |
| log-cubic | 0.63 | 0.66 | 0.63 | 1.6 | 2.0 | 0.64 | 0.88 |
| β-mom., ω_1 free | 0.65 | 0.78 | 0.64 | 0.82 | 0.78 | 0.63 | 1.0 |
| electron PL | 0.52 | 8.0 | 0.96 | 21 | 26 | 4.7 | 8.1 |
| electron log-parabola | 0.58 | 0.62 | 0.58 | 6.3 | 10.5 | 0.58 | 1.5 |
| electron broken PL | 0.60 | 0.63 | 0.61 | 3.2 | 4.8 | 0.64 | 0.84 |

**Rank and noise** (checked).
- The 1%-noise medians of the R = 4 row (4.1, 2.4, 4.4, 5.1, 6.3, 5.5 and 2.8) differ from its noise-free values by 0.008–0.055 percentage points (largest for GALPROP: 2.830% against 2.776%), and its 95th percentiles exceed them by 0.09–0.26 percentage points (largest for GALPROP: 3.031%). Table 6 has the noise-free R = 4 row only.
- The median UPRE rank over the 500 realisations at 1% noise is 7, 7, 8, 9, 9, 9 and 7 for E1, E2, E3, E4-JP, E4-KP, E5 and GALPROP, at 0.1% noise 12, 11, 12, 11, 11, 12 and 11, and at 0.01% noise 14 for E1–E5 and 12 for GALPROP (`compact_median_rms_held.<name>.R_star_median`).
- Without the limit R ≤ 14, R_⋆ ≥ 15 occurs in a fraction 0.028, 0.018, 0.028, 0.052, 0.056, 0.034 and 0 of the realisations at 1% noise (`frac_cap_binding`), 0.10–0.15 for E1–E5 at 0.1% noise and 0.27–0.91 at 0.01% noise. At 1% noise the 99th percentile of the moment-basis rms on E1–E5 is then 1.8–16% instead of 1.1–1.2% (`baselines_results.json → results.<name>.monte_carlo.summary.moment_UPRE_uncapped.rms_res_held.p99` and `moment_UPRE.rms_res_held.p99`).
- Removing the limit changes the median of the moment basis by at most 1.5% of its value at 1% noise and 3.1% at 0.1% noise. At 0.01% noise it changes the medians of E1–E5 by 3.6–11% of their values, upwards for E1–E3 and downwards for E4-JP, E4-KP and E5 (`moment_UPRE_uncapped` against `moment_UPRE` in `compact_median_rms_held`).
- Category: numerical check (Monte Carlo over 500 realisations).

**Paper.** Sec. 5.2 gives the comparison with the laws at 1% noise (0.32–0.39% for the best three- or four-parameter law on E1–E3 and E5 against 0.62–0.68% for the basis, 0.26% for the power-law form on E1, 5.7–26% for the PySM forms on the aged spectra), the effect of the rank limit (2–6% of the realisations at 1% noise, at most 1.5% and 3.1% of the medians at 1% and 0.1% noise, 27–91% and 4–11% at 0.01% noise) and the statement that every E1–E5 fit selects R_⋆ = 14 at 0.01% noise. Table 6 gives the medians at three noise levels, the noise-free R = 4 row and the ranges of the median UPRE rank (12–14, 11–12 and 7–9), and its caption states that the GALPROP fits select 12 or 13 at 0.01% noise. App. B gives the definitions of the laws and the electron families, with the parameter bounds. Sec. 6 gives the ratios 1.6–2.1, two to three and a half and 3–44 between laws and basis. README only: the 95th percentiles, the 1%-noise R = 4 row, the per-ensemble median ranks, the per-ensemble uncapped fractions and 1.8–16% against 1.1–1.2%.

### 9. GALPROP spectrum (Sec. 5.2)

**Produced by.** `baselines.py`, part B, → `baselines_results.json`, and `baselines_noise_sweep.py`, with input `inputs/galprop-benchmark/regional_spectra.csv` (column `high_latitude`). `make_figures.py` draws `figures/galprop_fit.pdf`, which the paper does not include. `galprop_absorption.py`, run by `baselines.py`, gives the absorption estimate (`baselines_results.json → absorption_caveat`).

**Source.** Orlando E., Strong A. W., 2013, MNRAS 436, 2127, doi:10.1093/mnras/stt1718. `inputs/README.md` gives the published file, its SHA-256, the regional averaging and the caveats of its metadata. Only the regional means are redistributed here.

**Settings.**
- The spectrum is evaluated at its native frequencies 1e-3 × 1.2^(26+k) GHz, k = 0, …, 24 (25 nodes from 0.114 to 9.10 GHz).
- Even k are fitted (13 nodes) and odd k withheld (12 nodes).
- Weights are σ = f I with f = 1%, 0.1% or 0.01%.

**Absorption estimate** (checked). The published spectrum turns over near 7.4 MHz. A mixed emitting and absorbing slab with τ ∝ ν^−2.1, fitted together with a polynomial for the unabsorbed spectrum to the nodes up to 0.022, 0.046 or 0.095 GHz, implies 0.055–0.073% attenuation at 0.1145 GHz over the nine slab fits (polynomial degrees 2–4, three ranges). Dividing the central slab estimate out changes the noise-free withheld rms by at most 0.003 percentage points. Source: `baselines_results.json → absorption_caveat` (`slab_attenuation_at_lowest_node_min_max` = 5.47e-4 and 7.26e-4, `turnover.peak_node_GHz` = 0.00743, `refit_withheld`). Category: extrapolated estimate. The script labels it a model-dependent estimate, not a bound. It depends on the polynomial degree, the fitted range and the geometry.

**Noise-free fits** (checked, not in the paper). The withheld rms of the noise-free moment fit is 2.8% at R = 4, 0.28% at R = 7, 0.099% at R = 9 and 0.031% at R = 12 (`results.GALPROP.noise_free_tsvd_curve`), and 0.33% and 0.27% for the log-cubic law and the broken power-law family (`results.GALPROP.noise_free.<model>.rms_res_held`). `figures/galprop_fit.pdf` shows noise-free fits at the 25 native nodes with residuals.

**Paper.** Sec. 5.2 gives the GALPROP map, the fit to every other output frequency, the median withheld rms 0.78% at 1% noise against 0.54–0.57% for the log-cubic law and the broken power-law family, 0.033% at 0.01% noise against 0.27–0.33%, and the median R_⋆ of 7 and 12. Table 6 gives the GALPROP column. App. B gives the model z10LMPD_SUNfE averaged over |b| ≥ 60°, the native frequencies 10^−3 × 1.2^(26+k) GHz (0.114–9.10 GHz) with k = 0, …, 24 and the even and odd split. README only: the absorption estimate (7.4 MHz, the slab model τ ∝ ν^−2.1, 0.055–0.073%, 0.003 percentage points), the noise-free fits and `figures/galprop_fit.pdf`.

### 10. Timings and cost (Sec. 5.2, Table 3)

**Produced by.** `cost.py` → `cost_results.json` (`machine`, `machine_start`, `machine_end`, `one_d`, `build`, `joint_gamma_B`), and `make_figures.py` writes `tables/cost.tex`. `cost_reference_check.py` → `cost_reference_check_results.json` gives the largest-error column of Table 3. `forward.py` → `forward_results.json` times the parametric families. The harmonic per-call times quoted below come from a timing memo of the harmonic pipeline, which is not distributed. `cost.py` copies them into `cost_results.json` with the label "read". It does not run the harmonic kernel.

**Settings.** Timings use one thread of an Apple M3 Ultra with Python 3.12.9, NumPy 2.3.5 and SciPy 1.16.3, and report medians after warm-up. They come from one run, and `cost_results.json` records the load of the machine during it (`machine_start`, `machine_end` and `machine_after` of each timing). Timings change between runs with that load. The timings printed in Table 3 and the ratios of Sec. 5.2 come from an earlier run of `cost.py` with the 19-cell configuration, whose results file is not distributed. The timings of this run differ from them by up to 20% (E6 joint-basis evaluation, 717 µs against 598 µs).

**Cost table** (checked). One parametric evaluation of the 123 data rows, all routes with the same continuum kernel: (i) recomputing the energy, field-direction, channel and screen averages from a cached table of F and G averaged over field strength, (ii) a cached table of the averaged kernel at fixed energy nodes, (iii) the moment basis. "vs (i)" is the largest difference from route (i) relative to I_j, as recorded by `cost.py`. Route (ii) for E2 reuses the energy nodes of route (i), so that comparison is circular. "vs ref." is the largest difference from the independent reference of `cost_reference_check.py`: four times as many energy (and field) nodes on panels offset by half a cell from the table grid.

| Ensemble | Route | Columns | Build | Per evaluation | vs (i) | vs ref. |
|---|---|---|---|---|---|---|
| E2 | (i) recompute | 456 nodes | – | 1.81 s | – | – |
| | (ii) γ table | 456 | 1.81 s | 23.7 µs | 3.3e-15 | 5.7e-15 |
| | (iii) moment basis | 57 | 0.150 s | 27.4 µs | 1.2e-3 | 1.2e-3 |
| E4-JP | (i) recompute | 216 nodes | – | 0.856 s | – | – |
| | (ii) γ table | 456 | 1.80 s | 22.9 µs | 3.7e-3 | 3.7e-3 |
| | (iii) moment basis | 57 | 0.150 s | 58.8 µs | 1.1e-2 | 1.1e-2 |
| joint (γ, B) | (i) recompute | 3456 nodes | – | 13.6 s | – | – |
| | (ii) γ × B table | 7296 | 28.7 s | 218 µs | 8.9e-4 | 8.9e-4 |
| | (iii) joint moments | 114 | 0.152 s | 717 µs | 8.7e-3 | 8.7e-3 |
| | field-averaged basis, marginal moments* | 57 | – | – | 8.4e-2 | – |
| | energy columns at B_0, rescaled energies* | 57 | – | – | 3.7e-3 | – |

\* Error at the true parameters only.

- Sources: `cost_results.json → one_d.<name>.route_i_recompute`, `.route_ii_table_456`, `.route_iii_moment_basis_36` (57 columns despite the key name), `build.continuum_analytic_basis`, `joint_gamma_B.route_*`, and `cost_reference_check_results.json → <name>.max_rel_err.<route>.vs_new_ref`.
- The N = 2 continuum basis needs 19 averaged-kernel calls with analytic derivatives (0.150 s). With five-point differences it needs 95 (0.382 s), as does the harmonic basis, whose calls take 1.3–2.9 s each (read from that memo, `build.harmonic_stencil_basis_read_from_survey`).
- The 114 columns of the joint ensemble need 19 kernel calls because Eq. (4.16) supplies the field derivatives. A kernel without that identity needs 475 (`joint_gamma_B.counts`).
- The matrix–vector product of route (iii) takes 0.96 µs (`one_d.<name>.route_iii_parts.matvec_123x36_only`), and computing a(θ) by quadrature takes the rest, 93% of the evaluation time for E2 and 96% for E4-JP (`family_moments_only`).
- At the true parameters the joint basis reaches 7.8e-3 (`joint_gamma_B.moment_route_72_max_rel_err`), and mapping each field strength onto the fixed-field energy columns through Eq. (4.16) reaches 3.7e-3 without any field expansion (`pushforward_36_fixed_B0_max_rel_err`). The field-averaged basis with marginal energy moments misses by 8.4e-2 (`preaveraged_36_with_marginal_moments_max_rel_err`), because it ignores the energy–field correlation.
- For E6 the table takes 189 times longer to build than the joint basis (`table_build_s` over `joint_basis_build_s`). It evaluates a spectrum 3.3 times faster and is 9.7 times more accurate against the reference. For E2 and E4-JP the 456-energy tables evaluate 1.2–2.6 times faster than the basis.
- A 152-energy table gives 8.4e-14 for E2 and 1.9e-2 for E4-JP against the reference (`cost_reference_check_results.json → <name>.max_rel_err.table_152.vs_new_ref`).
- Parameter vectors: for E2 and E4-JP the errors are maxima over 60 vectors, the reference and 59 draws around it. For the joint ensemble the errors are maxima over the true parameters and 4 draws (`cost_reference_check_results.json → <name>.n_theta`).
- The reference has converged: an eight-fold reference changes it by at most 4.1e-15 (`new_ref_8x_vs_4x_max_rel`, `new_ref_8x_vs_4x_max_rel_theta0`).
- Parametric families: the moments of 1000 parameter vectors at N = 2 take 5.8, 7.6, 16.1 and 39.1 ms for E3, E1, E2 and E4-JP (`forward_results.json → family_checks.<name>.seconds_per_1000_theta_N2`).
- Category: timings are medians on the stated machine. The error columns are numerical checks at the stated parameter vectors.

**Paper.** Table 3 gives the kernel evaluations (456, 19, 7296), build and evaluation times of an earlier run and the largest error against the independent reference. Its caption states that 93–97% of the basis evaluation time is the computation of the moments, the 60 and 5 parameter vectors and the offsets of the draws. Sec. 5.2 gives the ratios "about 190", 2.7 and ten for E6 and 1.3–2.6 for E2 and E4-JP from that earlier run. Table 2 gives the E6 errors 8.4% and 0.78% at the true parameters. Sec. 6 compares build cost and evaluation time in words. README only: the hardware and versions, the timings of this run, the errors against route (i), the 152-energy table, the 475 calls without Eq. (4.16), the harmonic per-call time 1.3–2.9 s, the 0.96 µs product, the convergence of the reference, the family timings and the value 3.7e-3 of the energy columns at B_0.

### 11. Illustrated Faraday screens (Sec. 4.3)

**Produced by.** No script in this directory (Fig. 4 of the paper).

**Screens.** With x = (Ψ − Ψ̄)/σ_Ψ and t = 2λ²σ_Ψ, the rotation-phase averages are characteristic functions C(t) = ⟨e^{itx}⟩_scr:

| Screen | Density | C(t) | Third and fourth cumulants |
|---|---|---|---|
| Gaussian | standard normal | e^{−t²/2} | (0, 0) |
| Laplace | p(x) = e^{−√2 \|x\|}/√2 | (1 + t²/2)^−1 | (0, 3) |
| Gamma | x = Y/2 − 2, Y ~ Gamma(4, 1) (unit scale) | e^{−2it}(1 − it/2)^−4 | (1, 3/2) |

- All three have zero mean and unit variance.
- The two- and four-cumulant approximations follow from Eq. (4.13) with these cumulants.
- The Gamma residual position angle is −t + 2 arctan(t/2), continued from zero after subtracting the mean-depth rotation.
- The Laplace and Gamma logarithmic series have radii √2 and 2. Fourth order need not meet a tolerance inside these radii, and adding cumulants outside them can increase the error.

**Exponent remainder** (not in the paper). On a zero-free interval, write the logarithm of the screen factor as g = g_4 + R_5. Taylor's theorem gives |R_5(τ)| ≤ |τ|⁵ sup_{u ∈ [0, τ]} |g^(5)(u)|/5!, where the derivative envelope is additional information. With P̂_ν = P_ν^em e^{g_4}, a supplied |R_5| ≤ ε_5 gives

```text
|P_ν^obs − P̂_ν| ≤ |P̂_ν| (e^{ε_5} − 1)
```

The channel error follows by integrating this envelope against |R_j|. For harmonic spectra, the absolute line amplitudes are summed before ensemble averaging. Category: strict bound given ε_5.

| Screen | sup over real t of \|g^(5)(t)\| | ε_5 | Category |
|---|---|---|---|
| Laplace (standardised) | ≤ 6√2 | \|t\|⁵/(10√2) | strict bound |
| Gamma (standardised) | ≤ 3 | \|t\|⁵/40 | strict bound |
| Gaussian | remainder vanishes | 0 | strict bound (R_5 = 0) |

These bounds become uninformative at large phase spread and control only screen truncation at fixed intrinsic emission. At characteristic-function zeros, the direct phase average or the joint representation is used instead.

**Paper.** The caption of Fig. 4 names the Laplace and skewed Gamma depth distributions with the same mean and variance. Sec. 4.3 gives the cumulant truncation of Eq. (4.13), the vanishing of all cumulants above second order for the Gaussian screen, the comparison of four and two cumulants for the Laplace (crossing near τσ_Ψ ≃ 1.5) and Gamma (up to τσ_Ψ = 2) screens, and |Ĉ_4| reaching 1 at τσ_Ψ = 2 for the Laplace screen. README only: the densities and characteristic functions, the cumulant values, the Gamma position angle, the radii √2 and 2, the exponent remainder with the propagation rule for a supplied ε_5, the ε_5 values and the derivative suprema.

### 12. Secondary checks of Secs. 5.2 and 5.4

**Produced by.** `fits.py` → `fits_results.json`, `verify_fits.py` → `verify_fits_results.json`, and `params.py` with `paramlib.py` and `params_degeneracy.py` → `params_results.json`.

**Checks** (checked, not in the paper). Values refer to the seeded noise realisation unless a number of realisations is given.

| Statement | Value | Source | Category |
|---|---|---|---|
| E1M: the part of the representation error outside the retained directions adds to χ² | 0.020 | `verify_fits_results.json → ensembles.E1M.eps_w_outside_retained_R_star` (0.0202) | numerical check |
| E5: the joint fit stays within this many standard deviations of the true polarisation angle in every channel with \|P_j\|/I_j > 3% (29 channels) | 1.6 | `fits_results.json → E5.chi_nu.max_abs_joint_minus_true_over_sigma_selected` (1.64) | numerical check |
| E5: UPRE prefers the joint model to one free azimuth factor | in 400 of 400 realisations | `fits_results.json → E5.joint_vs_separable_realisations.fraction_upre_prefers_joint` (1.0, n = 400) | numerical check |
| E1–E4 and E1M: the minimum-norm moments violate the moment conditions of a nonnegative distribution in at least one cell | in every one of 4000 realisations | `fits_results.json → ensembles.<name>.realisations.hausdorff_tsvd_fraction_with_violation` (1.0) | numerical check |
| Standard deviations of the parameter pulls (compressed fit at each realisation's R_⋆, E1, E2 with s = 2, E3, E4-JP) | 0.97–1.01 over 1000 realisations | `params_results.json → tasks.<task>.realisations.comp_Rstar.params_vs_reference.pull_sd` (0.970–1.010) | numerical check |
| E2 with s free: fits that fail to converge | 13–15% of 1000 (132 compressed at R_⋆, 146 at fixed R, 146 full) | `params_results.json → tasks.E2_sfree.realisations.{comp_Rstar,comp_Rseed,full}.n_failed` | numerical check |
| E2 with s free: growth of σ(p_1) at the truth | fourfold (0.063 to 0.268) | `params_results.json → tasks.E2_s2.fisher_sd_full_at_ref[1]` and `tasks.E2_sfree.fisher_sd_full_at_ref[1]` | numerical check |
| First relation of Eq. (4.16) for the numerical averaged kernel, λ = 0.5, 0.8, 1.25 and 2 | to 9e-12 of the largest channel value | `params_results.json → degeneracy.kernel_scaling_identity[*].kbar_I_max_abs_diff_over_max_j` (largest 9.3e-12) | numerical check |
| E2 ridge with the energy support rescaled with the field: change of χ² | at most 4e-13 | `params_results.json → degeneracy.E2_ridge.summary["transformed/seeded/full/delta_chi2_vs_lnlam"]` (largest 4.3e-13, noise-free 5.2e-15) | numerical check |
| Same ridge: d ln γ_br/d ln B | −0.5000000 | `degeneracy.E2_ridge.summary["transformed/noisefree/full/slope_dln_gamma_b_dln_B"]` (−0.50000000008) | numerical check |

- [inferred] The uncertainty of the E1M invariant C⟨B_⊥^{(p+1)/2}⟩ is dominated by that of p through C = A (1e4)^p. The amplitude invariants have errors 11–14 times σ(ln A) for E1 and E4-JP (`params_results.json → tasks.<task>.seeded_comp.invariants_sd` against `sd`).
- Values that the paper states are not repeated in the table: the E1 index 2.5976 ± 0.0031 (Table 7), the joint-fit rotation 24.4° against 25.2° and the one-factor rotation 8.5° (Sec. 5.2, `E5.chi_nu.swing_deg_selected`), the per-parameter coverage 65.7–69.1% with binomial standard deviation 1.5% (Sec. 5.4, `realisations.comp_Rstar.params_vs_reference.coverage_|pull|<1` and `coverage_binomial_sd`), the largest representation shift 0.26 standard deviations (Sec. 5.4, `noisefree.comp.shift_over_fisher_sd`), the E1M field-moment ratio 0.964 (Sec. 5.4, `tasks.E1M_U46basis.field_moment_ratio_<B_perp^1.8>_Maxwell_over_U46`) and the E4-JP break-frequency uncertainty 1.3% (Sec. 5.4, `tasks.E4-JP.seeded_comp.sd[2]` = 0.0135 in ln ν_b).

**Paper.** Sec. 5.2 states the E5 rotations. Sec. 5.4 states the per-parameter coverage, the largest representation shift, the compressed against full agreement within 0.3%, the field-scaling degeneracy, the E1M field-moment ratio and the invariant recovered to 0.08%. README only: every value in the table above.

### 13. Expansion order and reference points (Secs. 5.1 and 5.2)

**Produced by.** `sensitivity.py` → `sensitivity_results.json → sensitivity` (one record per ensemble, order N and reference offset q) and `checks`.

**Settings.**
- E1, E2 and E4-JP with the parameters of group 6, intensity only, in the 41 paper channels with 48 quadrature nodes per channel (`sensitivity_results.json → config`) and the 19 cells.
- Taylor orders N = 0, …, 4. The reference point of cell b is γ_b e^{q h}, with γ_b the cell centre, h the half-width in ln γ and q ∈ {−1/2, 0, +1/2}, so q = ±1/2 moves it by a quarter of the cell width. The true moments about the moved points are integrated from the density and checked against the binomial shift of the centred moments.
- The forward error is max_j |(C a)_j − I_j|/I_j over the 41 channels, with the true moments.
- Fits use the 21 fitted intensities with noise sd 0.01 I_j and noise-whitened TSVD at the UPRE rank, searched over R ≤ 14 (`baseline_models.R_CAP`). There is one noise-free fit and there are 200 noise realisations of seed 20261102. The withheld error is the rms of (fit − direct)/I_j over the 20 withheld channels. Moving the reference points changes the columns and with them the TSVD metric, so these fits compare bases as well as orders.
- The direct spectra change by at most 3.3e-10 I_j when the quadrature is refined to 96 channel nodes, 24 η nodes, 32 field-strength nodes and 48 energy nodes per panel (checked, `checks.<name>_refinement_max_fraction`, largest 3.28e-10). The intensity columns agree with `basis.kbar` to 3.8e-16 of their largest value (`checks.intensity_vs_existing_kbar_scaled`).

**Forward error and fitted withheld error** (checked). Forward errors in per cent of I_j for N = 0 to 4, and the range over N = 0 to 4 of the median withheld rms over the 200 realisations, in per cent.

| Ensemble | q | N = 0 | N = 1 | N = 2 | N = 3 | N = 4 | Median withheld rms |
|---|---|---|---|---|---|---|---|
| E1 | −1/2 | 15.6 | 3.41 | 0.496 | 0.119 | 0.0608 | 0.59–0.68 |
| | 0 | 2.56 | 2.60 | 0.0858 | 0.0474 | 0.00876 | 0.59–0.66 |
| | +1/2 | 24.5 | 5.94 | 1.01 | 0.262 | 0.142 | 0.58–0.66 |
| E2 | −1/2 | 18.9 | 5.05 | 0.687 | 0.159 | 0.0810 | 0.57–0.62 |
| | 0 | 3.83 | 3.94 | 0.0907 | 0.0567 | 0.0116 | 0.55–0.61 |
| | +1/2 | 32.5 | 8.97 | 1.24 | 0.310 | 0.197 | 0.57–0.61 |
| E4-JP | −1/2 | 30.9 | 18.5 | 4.29 | 0.410 | 0.235 | 0.60–0.70 |
| | 0 | 18.5 | 16.6 | 0.364 | 0.194 | 0.0447 | 0.61–0.69 |
| | +1/2 | 92.7 | 40.5 | 6.39 | 1.34 | 0.922 | 0.59–0.68 |

- With q = 0 the forward errors agree with the forward table of group 6 to its rounding. That table takes the maximum over I, Q and U and uses the default quadrature.
- Moving the reference points raises the error of E1, E2 and E4-JP by factors of 6–18 at N = 2 and 5–21 at N = 4, to at most 6.4% and 0.92% (ratios of the entries above at full precision).
- At q = 0 the median withheld rms rises from N = 0 to N = 4 by 12% of its value for E1 (0.593% to 0.664%), 13% for E2 (0.546% to 0.615%) and 13% for E4-JP (0.611% to 0.691%).
- Over all orders and reference points the median withheld rms spans 0.576–0.679% for E1, 0.546–0.622% for E2 and 0.593–0.698% for E4-JP. The noise-free fits have withheld rms 0.03–0.35% (`noiseless_held_rms`), and the median UPRE rank is 5–9 (`median_rank`).
- Source: `sensitivity_results.json → sensitivity[*].forward_max` and `.noisy_held_rms_percentiles[1]`. Every entry agrees to the digits shown. Category: numerical check, for these ensembles, reference points and realisations.

**Paper.** Table 2 gives the shifted columns at N = 2 and N = 4 for E1, E2 and E4-JP. Sec. 5.1 gives the factors 6–18 and 5–21 and the largest values 6.4% and 0.92%. Sec. 5.2 gives the rise of the median withheld error by 12–13% from N = 0 to N = 4 over 200 realisations. README only: the entries at N = 0, 1 and 3, the medians at each reference point, the noise-free fits, the ranks and the quadrature checks.

### 14. Prescribed scalar absorption

**Produced by.** `sensitivity.py` → `sensitivity_results.json → absorption` and `checks.slab_tau<τ>_refinement`.

**Settings.**
- E1 seen through a passive homogeneous slab, I^slab = I^thin (1 − e^−τ_ν)/τ_ν with τ_ν = τ_0.1 (ν/0.1 GHz)^−2.1 and the factor set to 1 at τ_ν = 0, for τ_0.1 ∈ {0, 0.1, 1, 10}. The factor is applied at each channel quadrature node, before channel integration. No noise is added.
- Free-coefficient fit: the N = 2 intensity basis at the centred reference points, fitted to the 21 fitted intensities by TSVD at the UPRE rank, with weights from a noise sd of 0.01 I_j.
- Nonnegative fit: `scipy.optimize.nnls` on direct channel kernels at 192 or 384 log-spaced energies on [300, 2.96e6], with the same weights and a free normalisation for each energy column. This tests a finite nonnegative grid, not all nonnegative electron distributions.
- At τ_0.1 = 0 the slab spectrum equals the E1 spectrum of group 13 (`checks.transparent_limit` = 0). The refined quadrature of group 13 changes every slab spectrum by at most 3.0e-10 I_j (checked, `checks.slab_tau<τ>_refinement`).

**Emission-only fits to the attenuated spectrum** (checked). rms of (fit − direct)/I_j in per cent, over the 20 withheld channels unless the column says fitted.

| τ_0.1 | UPRE rank | Free basis, fitted | Free basis, withheld | NNLS 192, withheld | NNLS 384, withheld | NNLS 384, fitted |
|---|---|---|---|---|---|---|
| 0 | 7 | 0.311 | 0.280 | 7.4e-5 | 7.4e-5 | 1.2e-14 |
| 0.1 | 7 | 0.271 | 0.241 | 1.9e-4 | 2.0e-4 | 1.6e-5 |
| 1 | 8 | 0.168 | 0.164 | 0.597 | 0.596 | 0.778 |
| 10 | 8 | 0.319 | 0.330 | 16.900 | 16.867 | 18.9 |

- At τ_0.1 = 10, 192 energies in place of 384 change the withheld rms by 0.033 percentage points.
- Source: `sensitivity_results.json → absorption[*]` (`rank`, `free_train_rms`, `free_held_rms`, `nnls192_held_rms`, `nnls384_held_rms`, `nnls384_train_rms`). Every entry agrees to the digits shown. Category: numerical check, for this slab, this ensemble and these grids.

**Paper.** Not in the paper. README only: every value of this group.

### 15. Spectrum error along the E2 field–energy ridge

**Produced by.** `ridge_sed.py` with `paramlib.py` and `params_degeneracy.py` → `ridge_sed_results.json` (`rows`, `summary`, `checks`). The script reads `params_results.json` and stops without writing if its N = 2 fits differ from `params_results.json → degeneracy.E2_ridge` by more than 1e-8 in relative χ² or in φ. In the run recorded here they agree exactly on all 11 points (`checks`).

**Settings.**
- The E2 data and the SBPL_s2 family (ln A, p_1, p_2, ln γ_br, with s = 2) of the E2 ridge in `params.py` (group 12): full linear likelihood, noise sd 0.01 I_j, fits started at the transformed truth, for the noise-free data and for the seeded realisation of `params.py`.
- Fields uniform on [4λ, 6λ] µG at 17 points ln λ = Δln B = −0.40, −0.35, …, +0.40. The energy support is either scaled as λ^−1/2 ("scaled", `transformed` in the JSON) or fixed at [300, 2.96e6]. Orders N = 2, 3 and 4.
- The spectrum error is max over the 41 channels and I, Q, U of |C a(φ̂) − S|/I_j at the fitted parameters φ̂. The 1σ band is the largest sqrt(diag(J cov Jᵀ))/I_j of the fit.
- The χ² minimum along the ridge is the vertex of the parabola through the smallest noise-free grid value and its two neighbours. The curvature c at the true field is the second difference of the noise-free χ² divided by 2h², with h = 0.05, so that χ² ≈ χ²(0) + c (Δln B)² near the true field and σ(ln B) = c^−1/2 where c > 0.

**Spectrum error along the ridge** (checked). Ranges over the 17 points, in per cent of I_j. With the support scaled every point gives the same value to the digits shown, and the noise-free χ² is flat, so it has no minimum.

| Support | N | Noise-free | Seeded | 1σ band, seeded | χ² minimum at Δln B | Curvature c |
|---|---|---|---|---|---|---|
| scaled | 2 | 0.0407 | 0.931 | 0.712 | – | 6e-13 |
| | 3 | 0.0391 | 0.932 | 0.712 | – | 1e-12 |
| | 4 | 0.0110 | 0.943 | 0.711 | – | −1e-13 |
| fixed | 2 | 0.0407–0.0489 | 0.931–0.948 | 0.711–0.712 | −0.370 | 0.0086 |
| | 3 | 0.0346–0.0422 | 0.931–0.950 | 0.711–0.712 | +0.207 | −0.038 |
| | 4 | 0.0109–0.0133 | 0.938–0.943 | 0.711–0.712 | −0.360 | −8.3e-5 |

- At the true field the noise-free error is 0.0407% at N = 2, 0.0391% at N = 3 and 0.0110% at N = 4, for both supports.
- With the support fixed, the N = 2 curvature at the true field is c = 0.0086, so σ(ln B) = 10.8 at 1% noise. σ(ln B) is proportional to the noise sd. At N = 3 and 4 the curvature at the true field is negative. With the support scaled the curvature is below 1e-12 in magnitude, since the scaling symmetry holds for every λ.
- All 204 fits converged (17 points, 2 supports, 3 orders, 2 data sets).
- Source: `ridge_sed_results.json → summary` (`<support>/N<N>/<data>/sed_max_range`, `.../sed_max`, `.../sed_band_1sigma_max_range`, `.../noisefree/chi2_min_dlnB_parabola`, `.../noisefree/chi2_curvature_per_lnB2_at_truth`, `.../all_fits_converged`). Every entry agrees to the digits shown. Category: numerical check, for this grid, these data and this seed. The χ² minimum is interpolated between grid points 0.05 apart in ln B.

**Paper.** Not in the paper. README only: every value of this group. `make_figures.py` draws `figures/field_energy_ridge.pdf`, with the contours from `params_results.json` and the errors from `ridge_sed_results.json`.

### 16. Extrapolation from 1–40 GHz to 40–300 GHz and free-moment fit over 10–300 GHz (Sec. 5.3)

**Produced by.** `fits_extrap.py` → `fits_extrap_results.json` and `fits_extrap_products.npz`, `fits_extrap_noisefree.py` → `fits_extrap_noisefree_results.json`, `extrap_curves.py` → `extrap_curves.json` and `extrap_curves.npz` (the curves of one realisation in Fig. 6(a, b)), `fits_cmb.py` → `fits_cmb_results.json` and `fits_cmb_products.npz` (Fig. 6(c)), and `fig_cmb_extrapolation.py` → `figures/cmb_extrapolation.{pdf,png}` and `fig_cmb_extrapolation_results.json`. `verify_extrap.py` → `verify_extrap_results.json` recomputes the values of Table 4 from the npz products and checks the summaries, the noise-free fractions and the curves (`failed` is empty).

**Settings** (checked against `fits_extrap_results.json`).
- Fit channels: 21 logarithmically spaced centres over 1–40 GHz (`fit_config`). Prediction channels: 25 over 40–300 GHz (`pred_config`). Both use the response of group 1 with h_j = 0.2 ν_j.
- The 19 cells of group 3 and N = 2. The free-moment basis keeps cells 4–19 (γ ≥ 1281.9, 48 energy columns, `retained_cells`). The omitted electrons contribute at most 5.5e-14 of I_j (`omitted_cells_max_over_I`).
- Gaussian noise with standard deviation f I_j in each of I, Q and U, f = 1% or 0.01%, 500 realisations of seed 20261112 (`noise_fracs`, `n_real`, `seed`).
- The electron families of group 8 (ePL, eLP, eSBPL) are evaluated through all 19 cells with the azimuth factor and screen of group 3 and fitted to I, Q and U of E2 and E3. E5, whose azimuth depends on energy, has no family fit.
- A fit is rejected at p < 0.001 of its in-band χ². The selection threshold 6.635 is the p = 0.01 point of χ² for one degree of freedom (`verify_extrap_results.json → selection.threshold`).

**Free-moment fits** (checked). At 1% noise the median R_⋆ is 7 for E2 and E3 and 19 for E5 (`summary.0.01.<name>.basis.R_star_median`). The norm fraction of the 300 GHz intensity row outside the retained directions is 0.881, 0.898 and 0.908 for E2, E3 and E5 (`fits_extrap_noisefree_results.json → results.0.01.<name>.row300_norm_fraction_outside_VR`). At 0.01% noise the median R_⋆ is 14, 13 and 37, and the fractions are 0.399, 0.436 and 0.781. The noise-free minimum-norm prediction misses the intensity over 40–300 GHz by up to 42%, 66% and 75% at the 1% ranks and 16%, 15% and 57% at the 0.01% ranks (`noisefree_max_dI_over_I`).

**Electron families at 1% noise** (checked). Median in-band χ²/dof, percentage of fits rejected, and the intensity error |ΔI|/I_j at 300 GHz in per cent (median, with the 16–84% range). Source: `verify_extrap_results.json → table_rows.<name>_<family>`. For every row the median error is largest at 300 GHz (`largest_median_at_300GHz`).

| Ensemble | Family | χ²/dof | Rej. [%] | Error at 300 GHz [%] |
|---|---|---|---|---|
| E2 | eSBPL (right) | 0.997 | 0 | 2.6 (0.95–5.2) |
| | eLP (wrong) | 1.089 | 0.2 | 20 (17–22) |
| | ePL | 7.81 | 100 | 35 (34–36) |
| E3 | eLP (right) | 0.987 | 0 | 1.9 (0.55–3.8) |
| | ePL (wrong) | 1.098 | 0.2 | 6.7 (5.9–7.5) |
| | eSBPL | 0.997 | 0 | 3.8 (1.5–12) |

- The in-band χ² of the right family is lower than that of the wrong family by more than 6.635 in 47.8% (E2) and 55.0% (E3) of the realisations. The median difference is 6.4 and 7.1 (`selection.<name>`).
- Without noise the right family misses the intensity over 40–300 GHz by at most 0.23% (E2) and 0.031% (E3), and the wrong family by 20% and 6.7% (`noisefree_family_fits.<name>_<family>.max_dI_over_I_40_300GHz`).
- Without a fit, the separable columns at the generating moments miss the intensity over 40–300 GHz by at most 0.25% (E2) and 0.077% (E3) at N = 2 and by 0.020% and 0.0081% at N = 4 (`verify_extrap_results.json → representation_at_generating_moments.<name>_N<order>.max_dI_over_I_40_300GHz`).
- Fig. 6(a, b) shades the 16–84% range of the predicted intensity of each drawn family over the 500 realisations above 40 GHz. `fig_cmb_extrapolation.py` repeats the 500 family fits of `fits_extrap.py` for these four families to get signed predictions, and stops unless they reproduce the saved |ΔI|/I arrays, the dense curves of `extrap_curves.npz` and each other at 300 GHz (`checks.<name>_<family>_refit_*`, all differences 0). At 300 GHz the signed 16–84% ranges of I_pred/I − 1 are −4.3% to 2.9% (E2 eSBPL), −22% to −17% (E2 eLP), −2.7% to 2.8% (E3 eLP) and −7.5% to −5.9% (E3 ePL) (`values.<name>_<family>.signed_rel_intensity_offset_300GHz_median_p16_p84`). The median and 84th-percentile curves recorded by `fig_cmb_extrapolation.py` equal the npz arrays (`fig_cmb_extrapolation_results.json → checks`, all differences 0), and the medians at 40 GHz are 0.46% and 1.1% for E2 and 0.42% and 1.1% for E3, right and wrong family (`values.<name>_<family>.dI_over_I_40GHz_median`).
- Category: numerical check, over 500 realisations at the stated noise.

**Free-moment fit over 10–300 GHz** (Fig. 6(c), checked against `fits_cmb_results.json`). E2 is fitted with free moments and no electron family in the 41 channels of 10–300 GHz (`config`), 21 fitted and 20 withheld, with the N = 2 separable basis restricted to cells 6–19 (γ ≥ 3375.4, 42 energy columns, `retained_cells`). The omitted electrons contribute at most 3.3e-20 of I_j (`omitted_cells_max_over_I`). One seeded realisation (paper seed) is scaled to 1% and 0.01% noise.

| Noise | R_⋆ | χ²_fit/dof | Max. \|ΔI\|/I_j over 41 channels [%] |
|---|---|---|---|
| 1% | 7 | 51.3/56 | 1.3 |
| 0.01% | 14 | 42.6/49 | 0.025 |

Source: `summary.<noise>.R_star`, `chi2_fit`, `dof_fit`, `max_dI_over_I`. The largest errors drawn in Fig. 6(c) equal these values (`fig_cmb_extrapolation_results.json → checks.c_<noise>_max_dI_over_I_rel_diff`, 0). Category: numerical check, one realisation.

**Paper.** Sec. 5.3 gives the setup, the median R_⋆ = 7, the fractions 88–90%, the median errors 2.6%, 1.9%, 20% and 6.7% at 300 GHz, the 499 of 500 accepted fits of each wrong family and the conclusion. App. B gives the channels, the noise, the evaluation of the families through all nineteen cells, the representation errors at the generating moments, 0.25% at N = 2 and 0.020% at N = 4, and the noise-free bias 0.23% of eSBPL for E2 at 300 GHz. The caption of Fig. 6 gives the largest errors 1.3% and 0.025% of the free-moment fit over 10–300 GHz. README only: the table values other than these, the 6.6 threshold with 48% and 55%, the E5 rows, the 0.01% fits, the noise-free minimum-norm errors, the noise-free errors of the wrong families and of eLP for E3, the representation errors of E3, the ePL row of E2 and the eSBPL row of E3, the retained cells and the 40 GHz medians, and the ranks, χ² values and retained cells of the free-moment fit over 10–300 GHz.

### 17. Emission windows and energy basis spectra (App. B, Figs. 7 and 8)

**Produced by.** `emission_windows.py` → `emission_windows_results.json`, `emission_windows_curves.npz` and `figures/emission_windows.{pdf,png}`. `derivative_spectra_figure.py` → `derivative_spectra_figure_results.json`, `derivative_spectra_figure.npz` and `figures/harmonic_derivative_spectra.{pdf,png}`. `run_all.py` runs the full computation of `emission_windows.py`, which also has a `--replot` option that redraws the figure from the npz.

**Emission windows** (checked against `emission_windows_results.json`). Monochromatic continuum emission with the default geometry of group 3.
- The intensity of a γ^−2.6 spectrum on [300, 2.96e6] divided by the untruncated power law is 0.99921 at 300 GHz (`I_over_PL_at_GHz.300.new_range`). On [300, 1e5] it is 0.96605 at 10 GHz (`10.old_range`).
- The E4-JP intensity divided by the power law is 0.00768 at 30 GHz (`30.aged_E4JP`).
- The mean screen rotates the polarisation angle by 41.2° at 1 GHz (`screen.rotation_deg_at_GHz.1`). At 0.2 GHz the Gaussian depolarisation factor is 0.080 and |P|/I is 0.024 (`screen.depolarisation_factor_exp(-2 lambda^4 sd^2)_at_GHz.0.2`, `screen.abs_P_over_I_at_GHz.0.2`). The frequency 0.2 GHz of the annotation in panel (b) is placed by hand (`depolarisation_annotation_note`).
- |V|/I for B = 5 µG and a field at 75° to the line of sight, from the fitting formula of Pandya et al. (2016), is 1.9e-4 at 0.1 GHz and 3.4e-6 at 300 GHz (`V_over_I_theta75_at_GHz`). Panel (c) draws it over 10 < ν/ν_c < 3e10, the range on which the formula was tested (`V_over_I_tested_nu_over_nu_c`), which ends at 420 GHz (`V_over_I_tested_upper_GHz`).
- The intensity ratios in the npz equal the computed values (`checks.I_over_PL_from_npz_minus_computed_max_abs` = 0). Against the study kernel with narrow channels the intensity shape agrees to 2.0e-8 (`checks.monochromatic_vs_study_kernel_E1_narrow_channels.I_shape_max_rel`).
- Category: numerical check.

**Energy basis spectra** (checked against `derivative_spectra_figure_results.json`). The columns D^X_{j;br}, r = 0, 1, 2, of cells 4, 7 and 9 (zero-based `bins` 3, 6, 8), with γ_b = 1632.9, 6977.2 and 18372.2 (`gamma_b`), in the frame of the mean projected field. In panel (i) the γ_b = 18372 curve stays within 1.4e-4 of zero in units of I_max, below the 2e-4 stated in the caption (`unlabelled_curves.panel_i_cell_2.max_abs_over_A`). Two labels are placed by hand (`hand_placed_labels`). Category: numerical check.

**Paper.** App. B gives 0.99921 as "within 10^−3 of the untruncated power law up to 300 GHz", 3.4% below the power law at 10 GHz for an upper limit of 1e5, the depolarisation below about 0.2 GHz and the end of the tested |V|/I range near 420 GHz. The caption of Fig. 8 gives γ_b = 18372 and 2e-4. README only: 0.8% for E4-JP at 30 GHz, the 41° rotation at 1 GHz, |V|/I ≈ 1.9e-4 and 3.4e-6, the depolarisation factor and |P|/I at 0.2 GHz, the kernel check, the cell indices and the other γ_b values.

## Paper items and scripts

Each figure, table and section of the paper with numbers from this study, the scripts that produce them and the files that hold them. Groups give the JSON keys. Scripts in `lib/` and the helper modules (`figure_data.py`, `figure_plots.py`, `figure_tables.py`, `baseline_models.py`, `spectral_laws.py`, `paramlib.py`) are not listed.

| Paper item | Scripts | Outputs | Group |
|---|---|---|---|
| Figs. 1, 2 and 4 | not in this directory | – | 1, 11 |
| Fig. 3 | `render_classical_limit.py`, with `recover_classical_limit_a.py` | `figures/classical_limit.{pdf,png}`, `render_classical_limit_results.json`, `recover_classical_limit_a_results.json` | 1 |
| Fig. 5 | `fits.py`, `fits_lownoise.py`, `make_figures.py` | `figures/moment_fit_gallery_compact.{pdf,png}`, `fits_results.json`, `fits_products.npz`, `fits_lownoise_results.json`, `fits_lownoise_products.npz`, `make_figures_results.json` | 7 |
| Fig. 6 | `fits_extrap.py`, `extrap_curves.py`, `fits_cmb.py`, `fig_cmb_extrapolation.py`, checked by `verify_extrap.py` | `figures/cmb_extrapolation.{pdf,png}`, `fits_extrap_products.npz`, `extrap_curves.npz`, `fits_cmb_results.json`, `fits_cmb_products.npz`, `fig_cmb_extrapolation_results.json` | 16 |
| Fig. 7 | `emission_windows.py` | `figures/emission_windows.{pdf,png}`, `emission_windows_results.json`, `emission_windows_curves.npz` | 17 |
| Fig. 8 | `derivative_spectra_figure.py` | `figures/harmonic_derivative_spectra.{pdf,png}`, `derivative_spectra_figure_results.json`, `derivative_spectra_figure.npz` | 17 |
| Table 1 | notation only | – | – |
| Table 2, prediction columns | `forward.py`, `sensitivity.py` (shifted), `kp_pseudotruth.py` (E4-KP), `cost.py` (E6) | `forward_results.json`, `sensitivity_results.json`, `kp_pseudotruth_results.json`, `cost_results.json` | 6, 10, 13 |
| Table 2, fit columns | `fits.py`, `make_figures.py` | `fits_results.json`, `tables/fit_summary_compact.tex` | 7 |
| Table 3 | `cost.py`, `cost_reference_check.py` | `cost_results.json` (timings of this run, which differ from the printed ones, group 10), `cost_reference_check_results.json` (largest-error column). `tables/cost.tex`, written by `make_figures.py`, has a different layout, with the errors against route (i). | 10 |
| Table 4 | `fits_extrap.py`, `verify_extrap.py` | `fits_extrap_results.json`, `fits_extrap_products.npz`, `verify_extrap_results.json` | 16 |
| Table 5 | `lib/ensembles.py`, `lib/config.py`, recorded by `forward.py` and `cost.py`, `make_figures.py` | `forward_results.json`, `cost_results.json`, `tables/ensembles.tex` | 6 |
| Table 6 | `baselines.py` (noise-free R = 4 row), `baselines_noise_sweep.py` | `baselines_noise_sweep_results.json → results.<noise>.compact_median_rms_held`, `baselines_results.json → results.<name>.noise_free.moment_R4`. No table in `tables/` has the layout of Table 6. | 8, 9 |
| Table 7 | `params.py`, `make_figures.py` | `params_results.json`, `tables/parameter_recovery.tex` | 7, 12 |
| Sec. 5.1 | `forward.py`, `sensitivity.py`, `cost.py` | `forward_results.json`, `sensitivity_results.json`, `cost_results.json` | 6, 10, 13 |
| Sec. 5.2 | `fits.py`, `fits_lownoise.py`, `fits_tails.py`, `sensitivity.py`, `baselines.py`, `baselines_noise_sweep.py`, `cost.py` | the JSONs of these scripts | 7–10, 13 |
| Sec. 5.3 | `fits_extrap.py`, `fits_extrap_noisefree.py`, `verify_extrap.py` | `fits_extrap_results.json`, `fits_extrap_noisefree_results.json`, `verify_extrap_results.json` | 16 |
| Sec. 5.4 | `params.py`, `fits.py`, `verify_fits.py`, `kp_pseudotruth.py`, `harmonic_correction.py` | the JSONs of these scripts | 4, 7, 12 |
| Sec. 6 | `baselines_noise_sweep.py`, `fits_extrap.py`, `verify_extrap.py` | `baselines_noise_sweep_results.json`, `verify_extrap_results.json` | 8, 16 |
| App. B | `forward.py`, `fits.py`, `kp_pseudotruth.py`, `harmonic_correction.py`, `fits_extrap.py`, `emission_windows.py`, `derivative_spectra_figure.py`, `baselines.py` | the JSONs of these scripts | 3, 4, 6, 8, 9, 16, 17 |

The parameter bounds of the families and laws in App. B are constants in `baseline_models.py` (`FAMILY_BOUNDS`).
