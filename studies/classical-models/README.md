# Classical-model study (`studies/classical-models`)

This directory holds the scripts, settings, inputs, results and checksums behind Sec. 5 (classical synchrotron models in the moment basis) of the paper "A statistical framework for the modelling of synchrotron emission", and behind the estimates in Sec. 6 that use the same ensembles. From this directory, `python run_all.py` runs the producing scripts in dependency order, then the verify scripts, then `make_figures.py`; it stops at the first script that fails, and after a complete run it rewrites `SHA256SUMS`. A complete run takes about 40 minutes on the machine named below and peaks at about 2.4 GB of resident memory. `SHA256SUMS` lists the SHA-256 checksums of the inputs and of every results JSON, table, npz product and figure, with paths relative to this directory. The results JSONs and `tables/` are in git. The npz products and `figures/` are not; they are attached to the GitHub Release of SyncEM v0.6.0 as the archive `classical-models-outputs-v0.6.0.tar.gz`, with paths relative to this directory. After extracting the archive here, `sha256sum -c SHA256SUMS` (or `shasum -a 256 -c SHA256SUMS`) checks every file; without it, `sha256sum -c --ignore-missing SHA256SUMS` checks the files in git. On the machine and versions named below, a rerun leaves the checksums of the inputs, the npz products, the figures and the tables other than `tables/cost.tex` unchanged; on another platform, BLAS library or TeX installation the npz products and figures may differ in the last bits. The results JSONs and `tables/cost.tex` change, because they record timings, finishing times and the hashes of other results; their other numbers reproduce, so compare them by value or with `git diff`. The results were produced on an Apple M3 Ultra with Python 3.12.9, NumPy 2.3.5, SciPy 1.16.3, Matplotlib 3.10.8, mpmath 1.3.0, JAX 0.10.0 and Equinox 0.13.7, with `syncem` imported from the root of this repository. `bridge.py` and `make_figures.py` need LaTeX with the newtx packages on the TeX search path.

`inputs/` holds every external file that a script reads or hashes, in subfolders named by origin, and `inputs/README.md` gives the origin and SHA-256 of each. `API.md` documents the library in `lib/` and gives the equation numbers of the paper for the LaTeX labels that docstrings cite. Each producing script writes its numbers to a results JSON; most JSONs record the script, input and library hashes and the package versions under `provenance`. The `verify_*.py` scripts check the producing scripts and write `verify_*_results.json`. `make_figures.py` compares the plotted values with the results JSONs and records these checks in `make_figures_results.json`. Kernel tables are cached in `cache/`, which the scripts rebuild when it is empty. `params.py` contains an exploratory branch, E2a, that uses an unverified transcription of a published local electron spectrum (Table 5 of Orlando E., 2018, MNRAS 475, 2724, doi:10.1093/mnras/stx3280). The transcription is not distributed, so `params.py` skips E2a, prints a line saying so and records the skip under `E2a_input` in `params_results.json`. No number in the paper or in this README comes from E2a.

## Numerical settings and checks

This section records the numerical settings and checks of the study by topic, including values that the paper does not state. Groups 1, 2 and 11 record settings and checks of the harmonic and Faraday-screen computations of Secs. 2–4, which no script in this directory produces.

Values marked **checked** were compared with the results JSON named with them; all agree to the stated rounding. Values quoted with a JSON key were read from that key. The other values were not re-derived from the results JSONs here.

**Error categories.** Each check carries one of four labels:
- *strict bound*: holds given the stated inputs;
- *numerical check*: holds for the finite case stated;
- *extrapolated estimate*: rests on a measured scaling outside the computed cases;
- *not bounded*: no bound was derived.

Fit statistics (coverage, withheld χ², percentiles over noise realisations) are labelled numerical checks for the realisations stated. Timings are measurements on the stated machine, not error checks. [inferred] marks an interpretation.

**Paper.** Each group ends with a line on where its values appear in the paper, by section, equation, table or figure number. Sec. 6 has no numbered subsections, so its paragraphs are named by their run-in headings. "README only" lists values that the paper does not state.

**Producing scripts.** Each group names the script and results file in this directory that produce its numbers. Where no script here produces them, the group says so.

Notation: I_j is the channel intensity, ν_j the channel centre, η the cosine of the angle between the field and the line of sight, φ the projected-field azimuth and Ψ the Faraday depth. x = ν/(a_B γ²) is the continuum variable, except in the Airy formulae of group 1 and in group 11, which define x locally. Section, equation, table and figure numbers refer to the paper.

### 1. Harmonic forward spectra

**Produced by.** The spectra and checks of this group come from the harmonic pipeline, which is not in this directory. No script in this directory produces these numbers.

**Ensembles.**
- Low-energy spectra of Fig. 3(a): uniform energy densities on γ ∈ [1.1, 1.2] and [2, 4]. Fields are uniform on [2, 8] µG (broad) or [4.5, 5.5] µG (narrow, this panel only). The rounded response (k = 1) has f = 0.08.
- Channel-averaging example of Fig. 5: N(γ) ∝ γ^−2.6 on [300, 1e5], fields uniform on [2, 8] µG.
- Both: isotropic pitch angles, η uniform on [−1/2, 1/2], projected-field azimuth density [1 + 0.8 cos(2φ)]/(2π), so that ⟨e^{2iφ}⟩_F = 0.4, and an independent Gaussian Faraday screen with mean 8 and standard deviation 0.5 rad m⁻². All these variables are independent.

**Channel responses** (Eq. (3.14)):

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
| Quadrature refinement, γ^−2.6 spectra of Fig. 5 | I_j changes by at most 5.5e-5 relative; complex P_j by at most 2.1e-5 I_j | numerical check |
| Omitted-line envelope, same spectra | below 2.6e-7 I_j | numerical check |
| Remainders of the retained Airy approximation and of the accelerated integer sum over the whole ensemble | no bound | not bounded |

A related check in this directory, part G of `kp_bridge.py` (`kp_bridge_results.json → G_cubic_airy`), measures the per-line error of the pipeline's cubic-Airy branch. It does not bound the remainder over an ensemble.

**Paper.** Eq. (3.14) in Sec. 3.3; in the caption of Fig. 3, γ ∈ [1.1, 1.2], the two field ranges, isotropic pitch angles, η on [−1/2, 1/2] and k = 1 channels of 8% rms width; in the caption of Fig. 5, γ^−2.6 on [300, 1e5], [2, 8] µG, isotropic pitch angles, η on [−1/2, 1/2], ⟨e^{2iφ}⟩_F = 0.4 and the Gaussian screen (mean 8, standard deviation 0.5 rad m⁻²); 6e-15 at γ = 1.5 and 2 in Sec. 2.1; in Sec. 6, paragraph "Harmonic numerics and the continuum limit", the statement that the remainders of the Airy approximation (m ≥ 1e6) and of the accelerated sum are not bounded over a whole ensemble, and "Refining the quadrature changes the spectra of Fig. 5 by less than 1e-4 of I_j", which covers 5.5e-5 and 2.1e-5. README only: the screen-factor formula, the Airy formulae, the branch limits, the DLMF references, the values 5.5e-5 and 2.1e-5 and the omitted-line envelope 2.6e-7.

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

**Paper.** Sec. 4.2: the ensemble described in words (correlated energy, field strength, depth, azimuth and angles; γ within 20% of 20, m ≤ 40, three channels); 6.506%–6.748% at L = 2 for N = 0–2; 1.856% (N = 0) to 0.175% (N = 2) at L = 8; labelled sampled discrepancies, not bounds. Sec. 4.2 also cites this check for the angular and Taylor truncation of Eq. (4.14). README only: the ensemble definition, the other table entries and the 7e-7 refinement.

### 3. Observing configuration and moment bases

**Produced by.**
- Configuration: `lib/config.py` (the defaults of `Config`, module object `PAPER`). The results of `core_checks.py`, `regression.py`, `forward.py`, `fits.py`, `params.py`, `baselines.py` and `cost.py` store the configuration under `config`.
- Continuum columns: `lib/basis.py` and `lib/kernels.py`, checked by `core_checks.py` → `core_checks_results.json` (`kernels_vs_mpmath`, `kbar_quadrature`).
- Sampled Taylor envelope: `forward.py` → `forward_results.json` (`envelope_definition`).
- Continuum–harmonic comparison: `regression.py` → `regression_results.json` (continuum library against the frozen harmonic caches). `bridge.py` → `bridge_results.json` repeats the comparison at the harmonic quadrature nodes (continuum kernel in `bridge_continuum.py`, checked against mpmath by `bridge_mpmath_check.py`; harmonic kernel on a γ grid from `bridge_harmonic_probe.py`), and `verify_bridge.py` → `verify_bridge_results.json` checks it. The UPRE rank for E1 comes from `fits.py` → `fits_results.json`.
- Harmonic columns and their refinement numbers: harmonic pipeline outside this directory; no script here produces them. `bridge.py` reads the frozen harmonic matrix (`inputs/harmonic-radio-fit/`, listed in `inputs/README.md`) but does not produce these refinement numbers.

**Observing configuration** (basis of Sec. 5 and Fig. 6).
- 41 channel centres logarithmically spaced over 0.1–10 GHz, each with response ρ_1[(ν − ν_j)/(0.2 ν_j)]/(0.2 ν_j Z_1) (group 1).
- Alternate centres give the 21 fitted channels; the other 20 are withheld.
- Fields uniform on [4, 6] µG; η uniform on [−1/2, 1/2].
- Projected-azimuth density [1 + 0.8 cos 2(φ − 0.25)]/(2π), so that ⟨e^{2iφ}⟩_F = 0.4 e^{0.5i}.
- Independent Gaussian screen with mean 8 and standard deviation 0.5 rad m⁻², entering at each frequency through the screen factor of group 1.
- Pitch angles isotropic unless an ensemble states otherwise.
- Simultaneous reversal of the two direction cosines cancels V.

**Energy bins and coefficient scales** (Eq. (5.1)). Twelve equal intervals in ln γ on [300, 1e5] give h_b = ln(1e5/300)/24 ≃ 0.24205. The fixed scaling

```text
w_b = ∫_{Γ_b} p_0(γ) dγ,        p_0(γ) = 1.6 γ^−2.6 / [300^−1.6 − (1e5)^−1.6]
```

is chosen before the singular-value decomposition and defines the Euclidean metric on the 36 coefficients a_br. It does not constrain the fitted moments to follow a power law. Direct spectra are evaluated at energy quadrature nodes within each bin, independently of the derivative columns.

**Binwise Taylor remainder** (not in the paper). For a supplied envelope M^X_{j,b} ≥ sup_{|x_b| ≤ 1} |∂³_{x_b} K̄_{X,j}(γ_b e^{h_b x_b})|,

```text
|ε^bin_{X,j}| ≤ (1/6) Σ_b M^X_{j,b} ∫_{Γ_b} |x_b|³ dN(γ)
```

which propagates through |K|. With M supplied, the category is strict bound. `forward.py` samples the suprema on 65 points per bin, which gives an estimate rather than a bound. With these sampled suprema the envelope exceeds every channel error of E1–E4-JP, and of E5 and E1M with their correct bases; at N = 2 it is 25 times the actual error for E1. As evaluated, the category is numerical check, not a bound. The actual N = 2 errors are in the Fwd. column of Table 4, and the N = 0, 2, 4 table is in group 6.

**Harmonic columns** (Fig. 6).
- (8, 7, 18, 64) nodes in field strength, field-direction cosine, each mapped pitch panel and channel order.
- Five-point differences with step 0.04 in x_b.

| Check | Result | Category |
|---|---|---|
| Quadrature refinement, spectra at the true moments of the γ^−2.6 ensemble | at most 7.8e-8 I_j | numerical check |
| Step halved | prediction changes by at most 2.4e-9 I_j | numerical check |
| Integrated omitted-line envelope | below 1.2e-8 I_j | numerical check |
| Remainders of the Airy-line approximation and the accelerated sum | no bound (group 1) | not bounded |

**Continuum columns** (Sec. 5). Kernels of Eq. (2.8) with analytic logarithmic derivatives of F and G.

| Check | Result | Category |
|---|---|---|
| Analytic log-derivatives of F and G against arbitrary-precision evaluations | agree to 7.5e-14 | numerical check |
| Averages over 16 field, 12 η and 400 channel nodes, all doubled | averaged kernel and its first two derivatives change by at most 1.2e-10 | numerical check |

**Continuum–harmonic comparison.**

| Check | Result | Category |
|---|---|---|
| Continuum columns against harmonic columns | differ by at most 1.8e-7 I_j | numerical check |
| Direct spectra, γ^−2.6 ensemble | differ by 4.06e-7 I_j in I and 1.5e-8 I_j in Q and U | numerical check |
| Fit to the same noise realisation with a fixed threshold D_ii ≥ 10 on the whitened singular values in place of the UPRE rule | seven combinations retained with either basis; UPRE also gives R_⋆ = 7 for E1 | numerical check |
| Adjacent singular values | 10.2128766 and 5.4532885 (harmonic), 10.2128794 and 5.4532910 (continuum); agree to 3e-6 | numerical check |
| Largest representation bias, 0.263 standard deviations | changes by 5e-8 | numerical check |

These differences combine the γ^−2 correction of Sec. 2.3 with the error of the Airy-line branch of group 1, which also scales as γ^−2. Group 4 separates them.

**Paper.** The twelve equal intervals in ln γ on 300 ≤ γ ≤ 1e5 in Sec. 4.3; in the opening of Sec. 5, 41 channels over 0.1–10 GHz with h_j = 0.2ν_j (k = 1), h_b = ln(1e5/300)/24 without its decimal value, N = 2 with 36 moments, Eq. (5.1), "These fixed scales set the metric of the singular-value decomposition; they do not force the fitted moments to follow a power law", and a pointer to Sec. 6 for the harmonic correction; [4, 6] µG, η on [−1/2, 1/2], isotropic pitch angles, 0.4e^{0.5i} and the screen in the caption of Table 3; 21 fitted and 20 withheld channels in Sec. 5.3; R_⋆ = 7 for E1 in Table 4. README only: the binwise remainder bound with the 65-point sampling and the factor 25; the harmonic- and continuum-column node counts and refinement numbers (7.8e-8, 2.4e-9, 1.2e-8, 7.5e-14, 1.2e-10), 4.06e-7, 1.5e-8, the singular values and 3e-6; h_b ≃ 0.24205, the continuum–harmonic column difference 1.8e-7 and the fixed-threshold comparison (seven combinations with either basis; 0.263 standard deviations, changed by 5e-8).

### 4. Harmonic correction to the continuum kernel

**Produced by.** `harmonic_correction.py`, with `harmonic_cells.py` and `kp_harmonic.py`, → `harmonic_correction_results.json` (`tables_c_I`, `tables_c_Q`, `gamma_scaling`, `checks`, `propagation`) and `harmonic_correction_tables.npz`.

**Method.**
- The frequency-averaged harmonic emissivity is evaluated with Bessel functions of real order: SciPy up to order 1e8 and Olver's uniform expansion beyond, without the Airy branch.
- Isotropic pitch angles, γ = 100 and 300, 1e-3 ≤ x ≤ 30 and 0 ≤ η ≤ 1/2.
- The relative correction to the continuum kernel is written as c_S(x, η)/γ².

**Results.**
- At γ = 300: −63 ≤ c_I ≤ −0.8 and −64 ≤ c_Q ≤ −0.9.
- c_S changes by at most 0.26% between the two energies.
- Extrapolating c_S/γ² to the energies of the reference ensembles gives corrections of at most 3.9e-7, 3.0e-7 and 4.2e-7 of I_j in I for E1, E2 and E4-JP, all at 0.1 GHz, and at most 1.9e-8 I_j in P.
- For E1 the mean of γ^−2 weighted by the emission in each channel is 2.3e-7, 2.3e-8 and 2.4e-9 at 0.1, 1 and 10 GHz.
- Category: extrapolated estimate. The values rest on the measured γ^−2 scaling and are estimates, not bounds.

**Paper.** In Sec. 6, paragraph "Harmonic numerics and the continuum limit", "Real-order Bessel evaluations at γ = 100 and 300, extrapolated as γ^−2, give a correction of at most 4.2e-7 I_j in I and 1.9e-8 I_j in P for E1, E2 and E4-JP; these values are estimates, not bounds"; the opening of Sec. 5 says that Sec. 6 estimates this correction for E1, E2 and E4-JP and gives no number. README only: 3.9e-7 and 3.0e-7 per ensemble, the E1 values of the mean γ^−2, the method details (SciPy to order 1e8, Olver's expansion beyond, no Airy approximation), the x and η ranges, the c_I and c_Q ranges at γ = 300 and the 0.26% change between γ = 100 and 300.

### 5. KP pitch-angle edge

**Produced by.** `kp_bridge.py`, with `kp_harmonic.py`, → `kp_bridge_results.json`, which holds the full results. `kp_bridge_tables.py` prints tables from that JSON, and `verify_kpbridge.py` → `verify_kpbridge_results.json` checks it. The cut-off range 16489–18767 is in `forward_results.json → ensembles.E4-KP.params` (`forward.py`).

**Settings and results.**
- The E4-KP pitch density at fixed γ is ∝ (μ² − μ_c²)₊^{p−2}, with μ_c(γ) ≤ 1/2 for 16489 ≤ γ ≤ 18767. It is therefore not smooth on scales of 1/γ near μ_c.
- Harmonic sums at γ = 20–1000 with the edge held at fixed μ_c show that Eq. (2.7) fails within a few beam widths of the edge.
- One beam width inside the edge, the relative error of I is −6.9%, −4.9% and −1.6% at x = 0.1, 1 and 3, independent of γ.
- Integrated over η at fixed γ, or over γ in an ordered field, the correction returns to the form c_I/γ² in tests at γ = 20–80, with −3.32 ≤ c_I ≤ 1.80.
- This suggests a correction for E4-KP of the same order as for E4-JP. [inferred]: the correction was not computed for the E4-KP spectra.
- Category: numerical check. No bound was derived. An η distribution whose edge coincides with μ_c was not tested.

**Paper.** 16489 and 18767 in Sec. 5.1; in Sec. 6, paragraph "Harmonic numerics and the continuum limit", with a pointer from Sec. 2.3, "The E4-KP pitch density has an edge at μ_c", the failure of Eq. (2.7) within a few beam widths of it at γ = 20–1000, the return "to order γ^−2 at γ = 20–80", "No bound was derived" and the untested edge case. README only: the density form (μ² − μ_c²)₊^{p−2}, −6.9%, −4.9% and −1.6% at x = 0.1, 1 and 3, and −3.32 ≤ c_I ≤ 1.80.

### 6. Reference ensembles and forward predictions (Sec. 5.1–5.2)

**Produced by.** `lib/ensembles.py` defines the densities. `forward.py` → `forward_results.json` (`ensembles.<name>.params`: loss rates, age, cut-offs, zone parameters; node-doubling checks of every direct spectrum; `e1_analytic`: the E1 analytic and Mellin checks; `summary_max_resid_over_I_IQU`: the forward-prediction table) and `forward_spectra.npz`. `verify_core.py` → `verify_core_results.json → values.E1_Pi_0.1GHz` repeats the E1 polarisation fraction. The JP(t_eff) row of the forward table comes from `kp_pseudotruth.py` → `kp_pseudotruth_results.json → forward`. E1M uses `PAPER_MAXWELL` in `lib/config.py`. The joint ensemble is built in `cost.py` → `cost_results.json → joint_gamma_B`. The parameter values are in Table 3 of the paper.

**Densities** (Eq. (5.2)), up to normalisation to unit column, on [300, 1e5]:

```text
E1:  N ∝ γ^−2.6
E2:  N ∝ (γ/γ_br)^−p_1 [1 + (γ/γ_br)^s]^(−(p_2 − p_1)/s)
E3:  N ∝ (γ/γ_0)^−p̄ exp[(1/2) σ_p² ln²(γ/γ_0)]
E4:  N ∝ γ^−p (1 − b γ t)^(p−2),    γ < γ_c = 1/(b t)
```

**E4 loss rates and support.**
- b = b_s + b_c for JP and b(μ) = (3/2) b_s (1 − μ²) + b_c for KP.
- b_s = 3.274e-20 s⁻¹ from synchrotron losses at ⟨B²⟩_col = 25.33 µG².
- b_c = 5.204e-20 s⁻¹ from Thomson inverse-Compton losses on 1 eV cm⁻³.
- The aged form holds for an impulsive power-law injection that extends to arbitrarily high energy. An injection cut at γ_{0,max} ages to a cut at γ_{0,max}/(1 + γ_{0,max} b t) < γ_c.
- The aged form is kept on [300, γ_c(μ)), which corresponds to an injection lower bound near 305, and each pitch angle is normalised to unit column. For JP the bound is 300/(1 − 300/γ_c) = 304.65, with γ_c = 19672.7 (`forward_results.json → ensembles.E4-JP.params.gamma_c`).

**E5.** Two power laws on [300, 1e5] with p_A = 3.1 and p_B = 2.1, and azimuth densities [1 + 0.8 cos 2(φ − φ_z)]/(2π) with φ_A = 0.25 and φ_B = φ_A + π/4. The zone-A fraction f_A = 0.937 equalises the zone intensities in the 1 GHz channel.

**E1M.** B is drawn from √(2/π) B² σ_B^−3 exp[−B²/(2σ_B²)] with σ_B = B_rms/√3.

**Joint ensemble** (group 10). It ages the same injection with pitch-angle isotropisation (JP), synchrotron losses in its local field with B uniform on [4, 6] µG, and the E4 inverse-Compton losses. γ_c runs from 16920 at 6 µG to 22940 at 4 µG. The values in `cost_results.json → joint_gamma_B.gamma_c_range` (16949.5, 22901.4) are γ_c at the outermost field quadrature nodes, B ≈ 5.989 and 4.010 µG; with b(B) = b_s B²/25.33 µG² + b_c and one age t ≈ 6.00e14 s, the endpoint values at 6 and 4 µG are 16920 and 22935, consistent with the range above to its rounding.

**Direct-spectrum quadrature.** Gauss–Legendre panels are split at every break, cut-off and support edge.

| Check | Result | Category |
|---|---|---|
| Energy nodes doubled | direct spectra change by at most 2e-15 I_j | numerical check |
| Channel, field and η nodes doubled | at most 3e-14 I_j | numerical check |

**E1 against the analytic power law** (checked). Source: `forward_results.json → e1_analytic`.

| Check | Result | Category |
|---|---|---|
| Linear-polarisation fraction of the direct E1 spectrum at 0.1 GHz, ordered field without azimuth or screen factors, against (p + 1)/(p + 7/3) for p = 2.6 | 0.729775 against 0.729730 (`Pi_direct_no_screen_at_0.1_0.316_1_3.16_10GHz[0]` = 0.7297746, `Pi_unbounded_(p+1)/(p+7/3)` = 0.7297297) | numerical check |
| Finite support evaluated through incomplete Mellin integrals of F, against the direct quadrature | agree to 1.4e-12 in I (`max_abs(I_direct/I_exact_incomplete_mellin - 1)` = 1.41e-12; P to 4.2e-13 I_j) | numerical check |
| Upper support edge γ = 1e5, 10 GHz channel | intensity 3.38% below the unbounded power law (`I_direct_over_I_unbounded_RL[40]` = 0.96616) | numerical check |

**Forward predictions from moments** (checked). The entry is 100 max_{j,X} |S^direct_{X,j} − (C a)_{X,j}|/I_j over the 41 channels and X ∈ {I, Q, U}, in per cent, for Taylor order N. The basis has isotropic pitch angles and one azimuth factor unless stated. For E4-KP, "η-averaged" uses the moments of the effective density ∫dη P_η(η) 2P_μ(μ = η | γ) N(γ), with P_η uniform on [−1/2, 1/2] and N the electron density; "JP(t_eff)" uses the moments of the JP spectrum with t_eff = 1.1475 t; "marginal" uses the energy moments without pitch dependence.

| Ensemble | Moments and basis | N = 0 | N = 2 | N = 4 |
|---|---|---|---|---|
| E1 | energy | 2.7 | 0.086 | 0.0088 |
| E2 | energy | 3.9 | 0.092 | 0.012 |
| E3 | energy | 2.7 | 0.10 | 0.0096 |
| E4-JP | energy | 18 | 0.36 | 0.045 |
| E4-KP | η-averaged | 23 | 2.6 | 1.7 |
| | JP(t_eff) | – | 0.89 | 0.064 |
| | marginal | 100 | 78 | 78 |
| E5 | joint energy–azimuth | 3.5 | 0.18 | 0.014 |
| | energy, one azimuth factor | 30 | 29 | 29 |
| E1M | energy, Maxwellian-field basis | 2.7 | 0.063 | 0.0018 |
| | energy, uniform-field basis | 6.4 | 3.8 | 3.8 |

- First order alone: adding the N = 1 terms changes the error by 10% of its N = 0 value for E4-JP (18.5% to 16.6% of I_j) and by less than 3% for E1–E3 (`summary_max_resid_over_I_IQU`, entries N0 and N1).
- Source: `forward_results.json → summary_max_resid_over_I_IQU` (maximum over I, Q and U; keys `E1` … `E1M`, `E4-KP_marginal(L=0)`, `E5_separable_paper_factor`, `E1M_U46_basis`) and `kp_pseudotruth_results.json → forward.N2` and `forward.N4` (`max_abs_S_KP_minus_C_a_over_I_IQU`, 8.87e-3 and 6.41e-4) for the JP(t_eff) row. Every entry agrees to its rounding.
- Category: numerical check. These are measured differences for the stated ensembles and parameters, not bounds.

**Figure panels not in the paper.** Fig. 8 shows panels (a) and (b) (`figures/reference_ensembles_compact.pdf`). `figures/reference_ensembles.pdf` has two further panels: (c), the ordered-field polarisation fraction of all seven ensembles against (p + 1)/(p + 7/3) for p = 2.6, and (d), the E5 polarisation angle without a screen and after removal of the mean screen rotation where |P_j|/I_j > 3%, with each zone alone. `make_figures.py` with `figure_plots.py` draws both files.

**Paper.** The parameters of Table 3 and, in its caption, ⟨B²⟩_col = 25.3 µG², 1 eV cm⁻³, the equal zone intensities at 1 GHz and the definition of f_A; in Sec. 5.1, Eq. (5.2), the JP and KP forms of b with b_s and b_c named but not evaluated, the aged-form statement with the injection lower bound near 305, the E5 and E1M densities (azimuth densities with φ_A and φ_B; the Maxwellian density with σ_B), "Without support edges it has the intensity spectrum of Sec. 2.3 and, in an ordered field without a Faraday screen, its polarisation fraction (p+1)/(p+7/3)", and the 3.38% support-edge effect at 10 GHz; in Sec. 5.2, "Doubling its quadrature nodes changes it by at most 3e-14 I_j", the E1–E4-JP ranges 2.7–18% (N = 0), 0.09–0.36% (N = 2) and below 0.05% (N = 4), E5 with one azimuth factor at 30% (N = 0) and 29% (N = 2 and 4) and with joint moments at 0.18%, E4-KP at 78% (marginal), 2.6% and 1.7% (η-averaged) and 0.89% and 0.064% (JP(t_eff)), and E1M at 0.063% and 3.8%; the N = 2 column of the forward table as the Fwd. column of Table 4; in the paragraph "Computational cost" of Sec. 5.4, the joint ensemble in one clause ("aged as E4-JP but with synchrotron losses in a local field uniform on [4, 6] µG"). README only: ⟨B²⟩_col = 25.33 µG², the statement on an injection cut at γ_{0,max}, the values of b_s and b_c, 2e-15 and the panel splitting, the E1 checks (0.729775 against 0.729730; 1.4e-12), the N = 1 result, the per-ensemble N = 0 and N = 4 entries of the forward table that Sec. 5.2 does not quote, panels (c) and (d) of `figures/reference_ensembles.pdf`, and the cut-off range 16920–22940 of the joint ensemble.

### 7. Fit settings and fit statistics (Sec. 5.3–5.4)

**Produced by.**
- Seeds: the seeded realisation of `fits.py`, `params.py` and `baselines.py` (`lib/config.py`, field `seed`; `fits_results.json → settings`). The seeds of the repeated noise realisations are set in these scripts and recorded in their results JSONs.
- Coverage realisations: `fits.py` → `fits_results.json`, with the rank tail in `fits_tails.py` → `fits_tails_results.json` and the E4-KP reference of Sec. 5.3 in `kp_pseudotruth.py` → `kp_pseudotruth_results.json`.
- Parameter realisations and starting points: `params.py` with `paramlib.py` → `params_results.json` and `params_realisations.npz`; `verify_params.py` → `verify_params_results.json` checks them.
- Baseline realisations: `baselines.py` → `baselines_results.json`.
- Rank search, E5 pairs and nonnegative witness: `fits.py`; `params.py` uses the same search.
- Fit statistics not in the paper: `fits_results.json → ensembles.<name>.seeded.chi2_held_predictive`, `.realisations.coverage_fixed_R.joint_ellipsoid_analytic` and `.realisations.coverage_post_selection.joint_ellipsoid_empirical`; the expectations in `fits_tails_results.json → ensembles.<name>.expectation_predictive_fixed_R_seeded`; the E4-KP row in `kp_pseudotruth_results.json → fit_summary.pseudo_truth_JP_t_eff` and, against the η-averaged density, `fit_summary.spec_truth_effective_diagonal`; the E5 one-factor row in `fits_results.json → E5.separable_free_factor.seeded`. `make_figures.py` with `figure_tables.py` writes `tables/fit_summary.tex` (the table with these statistics) and `tables/fit_summary_compact.tex` (the columns of Table 4).
- E1M and E4-KP blocks of the parameter table: `params.py` → `params_results.json → tasks.E1M_U46basis` and `tasks.E4-KP_JPfamily`; `tables/parameter_recovery.tex` holds the full table, with these blocks.

**Settings.**
- The fitted rows use the noise realisation of seed 20260928 and the withheld rows that of seed 20260929. All rows of Table 4 use this one realisation and the same kernel columns.
- Coverage uses 4000 realisations, the parameters 1000 and the intensity baselines 500.
- The rank search runs over R = 1, …, 26, the numerical rank of the separable design. For E5, exactly degenerate pairs of singular values are retained or discarded together.
- In the Bias column of Table 4, each exactly degenerate E5 pair of singular values enters through its two-dimensional norm.
- For the E5 joint design (108 columns) the search runs over R = 1..61. `fits.py` scans up to the numerical rank of each design (`LinearFit.scan`), and `fits_results.json → ensembles.E5.seeded.rank` gives 61. The paper states no range ("The rank search extends to the numerical rank of each design").
- The intensity-only baselines limit the search to R ≤ 14 (`baseline_models.R_CAP`).
- The nonnegative witness uses 17 Chebyshev–Lobatto points per bin and `scipy.optimize.nnls`.
- Parametric fits minimise Eq. (4.24) or the full linear likelihood from 12 starting points.

**Fit statistics not in Table 4** (checked). One noise realisation with 63 fitted and 60 withheld data, R_⋆ chosen by UPRE.
- χ²_held = rᵀ(1 + H_R H_Rᵀ)⁻¹ r is the withheld residual weighted by its predictive covariance, where H_R maps fitted-channel noise to the withheld predictions. "Exp." is its expectation at fixed R_⋆, 60 + μᵀ(1 + H_R H_Rᵀ)⁻¹μ, which includes the mean withheld residual μ left by truncation and representation error.
- Coverage is that of the joint 68.3% region of the retained combinations: analytic at fixed R_⋆, and Monte Carlo with R_⋆ reselected in each of 4000 realisations.

| Ensemble | χ²_held (exp.) | Cov. fixed | Cov. reselected |
|---|---|---|---|
| E1 | 50.9 (62.0) | 0.675 | 0.554 |
| E2 | 47.9 (60.6) | 0.681 | 0.567 |
| E3 | 51.3 (62.2) | 0.674 | 0.558 |
| E4-JP | 46.8 (60.9) | 0.648 | 0.560 |
| E4-KP† | 47.9 (61.4) | 0.574 | 0.504 |
| E5, joint moments | 51.7 (62.6) | 0.670 | 0.579 |
| E1M | 50.9 (62.0) | 0.000 | 0.000 |
| E5, one azimuth factor | 252.8 | – | – |

† Against the moments of the JP spectrum with t_eff/t = 1.1475. Against the moments of the η-averaged KP density the bias is 3.8 (Table 4 and Sec. 5.3) and the coverage 0.042 at fixed R_⋆ and 0.037 reselected.

- At fixed R_⋆ the joint region covers the reference moments with probability 0.65–0.68 for E1–E4-JP and E5, and reselecting R_⋆ lowers this to 0.55–0.58 (Sec. 5.3); E4-KP gives 0.57 and 0.50, and E1M zero.
- χ²_held lies 11–14 below its expectation in every row, about one standard deviation of χ²₆₀ (√120 ≈ 11). Over the 4000 realisations the median predictive χ²_held is 61.5–63.1 (`fits_results.json → summary.<name>.chi2_held_predictive_p50_realisations`), close to the expectations. [inferred] The common offset belongs to the one noise realisation that all rows share.
- Category: numerical check, for this realisation and these 4000 realisations. All entries agree with the JSON keys above to the rounding.

**UPRE rank tail** (checked). For E1–E4 and E1M, UPRE selects R_⋆ ≥ 15 in 3–5% of the 4000 realisations, and the 99th percentile of the largest error over all channels reaches 5–24% of I_j (both in Sec. 5.3), against 2.9–3.2% if the search is limited to R ≤ 14. The predictive χ²_held does not reveal these cases, because it divides out the prediction variance. Source: `fits_tails_results.json → ensembles.<name>.R_star_frac_ge_15` (0.029–0.054), `.max_err_over_I_at_R_star.p99` (0.051–0.243) and `.sensitivity_capped.max_err_over_I_at_R_star_p99` (0.0288–0.0318). Category: numerical check.

**Parameter recovery, E1M and E4-KP blocks** (checked; not in Table 6). Columns as in Table 6: truth or reference; compressed and full estimates ± 1σ for the noise realisation of Fig. 9; σ_c/σ_f; bias of a noise-free compressed fit in units of the Fisher σ at the reference.

| Ensemble (R_⋆) | Parameter | Truth or ref. | Compressed | Full | σ_c/σ_f | Bias [σ] |
|---|---|---|---|---|---|---|
| E1M† (7) | ln A | −5.1404 | −5.1789 ± 0.0022 | −5.1789 ± 0.0022 | 1.0000 | −17.62 |
| | p | 2.6000 | 2.5972 ± 0.0032 | 2.5971 ± 0.0032 | 1.0002 | −0.16 |
| E4-KP‡ (9) | ln A | −5.1157 | −5.1175 ± 0.0050 | −5.1175 ± 0.0050 | 1.0000 | +0.17 |
| | p | 2.5998 | 2.5974 ± 0.0061 | 2.5973 ± 0.0061 | 1.0001 | −0.22 |
| | ln(ν_b/Hz) | 22.499 | 22.495 ± 0.011 | 22.495 ± 0.011 | 1.0001 | −0.48 |

† Maxwellian-field data fitted with the uniform-field basis; reference is the E1 truth. ‡ KP data fitted with the JP family; reference is the exact forward JP fit to the noise-free KP spectrum.

- Sec. 5.4 of the paper states the E1M amplitude 3.6% low (−17.6 standard deviations), the E1M index at −0.16 standard deviations and the KP bias of 0.48 relative to the exact JP fit. The −17.6σ refers to the amplitude; the 17.5σ of Table 4 refers to the best-measured combination.
- Source: `params_results.json → tasks.E1M_U46basis` and `tasks.E4-KP_JPfamily` (`reference_point`, `seeded_comp`, `seeded_full`, `fisher_sd_comp_over_full_at_R_seed`, `noisefree.comp.shift_over_fisher_sd`). All entries agree. Category: numerical check.

**Figure panels not in the paper.** Fig. 9 (`figures/moment_fit_gallery_compact.pdf`) shows E2, E4-KP and E5; `figures/moment_fit_gallery.pdf` also has an E1 row. Fig. 10 (`figures/moments_and_parameters_compact.pdf`) shows panels (a) and (d) of `figures/moments_and_parameters.pdf`, relabelled (a) and (b). Panel (b) of that file, the shift of every retained combination with the representation-bias bars, and panel (c), the 68% and 95% regions of (p_2, ln γ_br) for E2 from the compressed likelihood at R_⋆ = 7 and from the full linear likelihood, are not in the paper. The largest bias per ensemble is in the Bias column of Table 4, and σ_c/σ_f ≤ 1.0026 in Table 6.

**Paper.** In Sec. 5.3, "The rank search extends to the numerical rank of each design", the reselected coverage ("with R_⋆ reselected in each of 4000 noise realisations, the joint 68.3% region of the retained combinations covers the reference moments with probability 0.55–0.58"), the rank tail ("UPRE selects R_⋆ ≥ 15 in 3–5% of the 4000 realisations. Those fits follow the noise of the fitted channels; over all realisations, the 99th percentile of the maximum error of Table 4 reaches 5–24% of I_j"), and, for the baselines, "we limit the rank search to R ≤ 14, which excludes the R_⋆ ≥ 15 tail"; Table 4 with R_⋆, χ²_fit/dof, Max. error and Bias (and the Fwd. column, group 6); 1000 parameter vectors and 1000 realisations in Sec. 5.4; Table 6 for E1, E2, E3 and E4-JP; 500 realisations in Sec. 5.3 and in the caption of Table 5, which also states R ≤ 14. README only: the seeds, the 12 starting points, the Chebyshev–Lobatto placement and `nnls`, the range R = 1, …, 26 and the E5 range R = 1..61, the E5 pair rules, the 17 witness points per bin, χ²_held with its definition and expectation, the coverage at fixed R_⋆ and the per-ensemble coverage values, the E4-KP coverage against the η-averaged density, 2.9–3.2%, the E5 one-factor χ²_held, the E1M and E4-KP blocks of the parameter table, and the figure panels listed above.

### 8. Baselines (Sec. 5.3)

**Produced by.** `spectral_laws.py` (laws), `baseline_models.py` (moment TSVD and electron families through the basis), `baselines.py` → `baselines_results.json` and `baselines_plotdata.npz`. `make_figures.py` with `figure_tables.py` writes `tables/baselines.tex` (the table with the 95th percentiles and the uncapped-tail row) and `tables/baselines_compact.tex` (a cross-check of Table 5). `verify_baselines.py` → `verify_baselines_results.json` checks the baselines and two values of `cost.py`.

**Laws.** With L = ln(ν/1 GHz):
- s1: A e^{βL};
- s3: A e^{βL + CL²};
- log-cubic: exp(c_0 + c_1 L + c_2 L² + c_3 L³);
- β-moment SED: A e^{β̄L}(1 + ω_1 L + ω_2 L²/2), with ω_1 = 0 or free.

All laws are integrated through the channel responses.

**Electron families** fitted through the basis columns:
- power law (A, p);
- log-parabola N ∝ (γ/1e4)^−p exp[−κ ln²(γ/1e4)] (A, p, κ);
- the E2 form with s = 2 (A, p_1, p_2, γ_br).

**Entries not in Table 5** (checked). Rms of (fit − direct)/I_j over the withheld channels in per cent (20 for E1–E5; 12 native nodes for GALPROP): 95th percentile over 500 realisations with 1% noise. The noise-free values and the 1%-noise medians are in Table 5.

| Model | E1 | E2 | E3 | E4-JP | E4-KP | E5 | GALPROP |
|---|---|---|---|---|---|---|---|
| moments, UPRE | 0.97 | 0.96 | 0.98 | 1.0 | 1.1 | 0.98 | 1.3 |
| moments, R = 4 | 4.1 | 2.5 | 4.3 | 5.2 | 6.3 | 5.3 | 3.0 |
| PySM s1 | 0.68 | 8.2 | 0.65 | 21 | 26 | 4.0 | 8.1 |
| PySM s3 | 0.61 | 0.64 | 0.61 | 5.8 | 7.4 | 0.67 | 1.5 |
| β-mom., ω_1 = 0 | 0.61 | 0.76 | 0.61 | 5.6 | 7.1 | 0.65 | 1.6 |
| log-cubic | 0.64 | 0.65 | 0.64 | 1.6 | 2.0 | 0.66 | 0.88 |
| β-mom., ω_1 free | 0.65 | 0.79 | 0.65 | 0.82 | 0.78 | 0.65 | 1.0 |
| electron PL | 0.52 | 7.8 | 0.91 | 21 | 26 | 4.3 | 7.9 |
| electron log-parabola | 0.58 | 0.62 | 0.58 | 6.3 | 10 | 0.59 | 1.5 |
| electron broken PL | 0.60 | 0.64 | 0.60 | 3.2 | 4.8 | 0.64 | 0.86 |

- The 1%-noise medians of the R = 4 row (4.0, 2.3, 4.2, 5.1, 6.3, 5.2 and 2.8) differ from its noise-free values by 0.008–0.055 percentage points (largest for GALPROP: 2.832% against 2.777%), and its 95th percentiles exceed them by 0.09–0.26 percentage points (largest for GALPROP: 3.033%); Table 5 has one R = 4 row, and its caption states the bound 0.06 on the medians.
- The median UPRE rank over the 500 realisations is 7, 7, 7, 9, 9, 9 and 7 for E1, E2, E3, E4-JP, E4-KP, E5 and GALPROP; the caption of Table 5 states these ranks.
- Without the limit R ≤ 14, R_⋆ ≥ 15 occurs in a fraction 0.028, 0.018, 0.026, 0.052, 0.056, 0.036 and 0 of the realisations (up to 6%), and the 99th percentile of the moment-basis rms on E1–E5 is 1.8–16% instead of 1.1–1.2%.
- Source: `baselines_results.json → results.<name>.monte_carlo.summary.<model>.rms_res_held` (`median`, `p95`, `p99`), `summary.moment_UPRE.R_star.median`, `summary.moment_UPRE.R_star_uncapped_frac_ge_15` and `summary.moment_UPRE_uncapped.rms_res_held.p99`. All entries agree to the rounding. Category: numerical check (Monte Carlo over 500 realisations).

**Paper.** The definitions of the four laws, the integration through the channel responses and the three electron families in the baselines paragraph of Sec. 5.3, with y = ln(ν/1 GHz) in place of L; in the same paragraph, the comparison in words ("laws with three or four parameters predict the withheld channels better than the basis at its UPRE rank, whose 7–9 retained combinations carry more noise", the aged-spectrum comparison, and "Electron families fitted through the basis columns are the most accurate without noise (0.02% rms) for the matching ensembles"); Table 5 with one "noise-free/1%-noise median" cell per model and spectrum and one R = 4 row, whose caption states the median UPRE ranks (7 for E1–E3 and GALPROP, 9 otherwise), the reselection of R_⋆ in each realisation, R ≤ 14, and that the 1%-noise medians of the R = 4 row differ from its noise-free values by at most 0.06 percentage points; the polarisation predictions of Sec. 5.3, including the minimum-norm moments at 0.53–0.84% for E1–E4-JP and 1.1% for E4-KP. The Conclusions (Sec. 7) quote table values (0.37–0.58% against 0.62–0.68%; 1.4–26% against 0.2%; 0.5%; 0.58–0.62% against 0.69–0.70%; 0.02%). README only: the 95th percentiles, the 1%-noise R = 4 row, the median-rank and uncapped-tail rows, and 1.8–16% against 1.1–1.2%.

### 9. GALPROP spectrum (Sec. 5.3)

**Produced by.** `baselines.py`, part B, → `baselines_results.json`, with input `inputs/galprop-benchmark/regional_spectra.csv` (column `high_latitude`). `make_figures.py` draws `figures/galprop_fit.pdf`, which the paper does not include. `galprop_absorption.py`, run by `baselines.py`, gives the absorption estimate (`baselines_results.json → absorption_caveat`).

**Source.** Orlando E., Strong A. W., 2013, MNRAS 436, 2127, doi:10.1093/mnras/stt1718. `inputs/README.md` gives the published file, its SHA-256, the regional averaging and the caveats of its metadata; only the regional means are redistributed here.

**Settings.**
- The spectrum is evaluated at its native frequencies 0.1145 × 1.2^k GHz, k = 0, …, 24 (25 nodes up to 9.10 GHz).
- Even k are fitted (13 nodes) and odd k withheld (12 nodes).
- Weights are σ = 0.01 I.

**Absorption estimate** (checked). The published spectrum turns over near 7.4 MHz. A mixed emitting and absorbing slab with τ ∝ ν^−2.1, fitted together with a polynomial for the unabsorbed spectrum to the nodes up to 0.022, 0.046 or 0.095 GHz, implies 0.055–0.073% attenuation at 0.1145 GHz over the nine slab fits (polynomial degrees 2–4, three ranges). Dividing the central slab estimate out changes the withheld rms by at most 0.003 percentage points. Source: `baselines_results.json → absorption_caveat` (`slab_attenuation_at_lowest_node_min_max` = 5.47e-4 and 7.26e-4; `turnover.peak_node_GHz` = 0.00743; `refit_withheld`). Category: extrapolated estimate. The script labels it a model-dependent estimate, not a bound; it depends on the polynomial degree, the fitted range and the geometry.

**Figure not in the paper.** `figures/galprop_fit.pdf` shows the noise-free fits at the 25 native nodes with the moment basis at R = 7 (the median UPRE rank at 1% noise) and R = 12, the log-cubic law and the broken power-law electron family, with residuals at the nodes. Its legend values are in the GALPROP column of Table 5 (0.28, 2.8, 0.33, 0.30) or in Sec. 5.3 (R = 12: 0.031%).

**Paper.** In the GALPROP paragraph of Sec. 5.3, the model z10LMPD_SUNfE averaged over |b| ≥ 60°, "native frequencies 0.1145×1.2^k GHz with k = 0, …, 24, fitting even k and withholding odd k", the withheld rms 0.28%, 0.099% and 0.031% at R = 7, 9 and 12 and 2.8% at R = 4 without noise, 0.78% with R_⋆ = 7 in the median at 1% noise, "the four-parameter log-cubic law and broken power-law family are close to the basis at R = 7 without noise and better at 1% noise (Table 5)", the statement that the fitted moments are not GALPROP's electron spectrum, and "The published spectrum includes absorption, which turns it over near 7.4 MHz; dividing out a slab estimate of the attenuation changes the withheld rms by at most 0.003 percentage points"; in the caption of Table 5, 12 native nodes and σ = 1% of the published value. README only: `figures/galprop_fit.pdf` with its values (25 nodes up to 9.10 GHz, 13 fitted and 12 withheld), the slab model τ ∝ ν^−2.1 and the attenuation 0.055–0.073%.

### 10. Timings and cost (Sec. 5.4)

**Produced by.** `cost.py` → `cost_results.json` (`machine`, `machine_start`, `machine_end`, `one_d`, `build`, `joint_gamma_B`); `make_figures.py` writes `tables/cost.tex`. `forward.py` → `forward_results.json` times the parametric families. The harmonic per-call times quoted below come from a timing memo of the harmonic pipeline, which is not distributed. `cost.py` copies them into `cost_results.json` with the label "read"; it does not run the harmonic kernel.

**Settings.** Timings use one thread of an Apple M3 Ultra with Python 3.12.9, NumPy 2.3.5 and SciPy 1.16.3, and report medians after warm-up. They come from one run; `cost_results.json` records the load of the machine during it (`machine_start`, `machine_end` and `machine_after` of each timing). Timings change between runs with that load.

**Cost table** (checked; not in the paper). One parametric evaluation of the 123 data rows, all routes with the same continuum kernel: (i) recomputing the energy, field-direction, channel and screen averages from a cached table of F and G averaged over field strength; (ii) a cached table of the averaged kernel at fixed energy nodes; (iii) the moment basis. Errors are the largest difference from route (i) relative to I_j. Route (ii) for E2 reuses the energy nodes of route (i), so its error is zero by construction.

| Ensemble | Route | Columns | Build | Per evaluation | Max. error |
|---|---|---|---|---|---|
| E2 | (i) recompute | 288 nodes | – | 1.33 s | – |
| | (ii) γ table | 288 | 1.34 s | 16.9 µs | 2.4e-15 |
| | (iii) moment basis | 36 | 93.0 ms | 22.5 µs | 1.2e-3 |
| E4-JP | (i) recompute | 216 nodes | – | 1.00 s | – |
| | (ii) γ table | 288 | 1.35 s | 17.3 µs | 3.7e-3 |
| | (iii) moment basis | 36 | 93.0 ms | 48.6 µs | 1.1e-2 |
| joint (γ, B) | (i) recompute | 3456 nodes | – | 16.6 s | – |
| | (ii) γ × B table | 4608 | 21.6 s | 148 µs | 8.9e-4 |
| | (iii) joint moments | 72 | 95.3 ms | 409 µs | 8.7e-3 |
| | field-averaged basis, marginal moments* | 36 | – | – | 8.4e-2 |
| | energy columns at B_0, rescaled energies* | 36 | – | – | 3.7e-3 |

\* Error at the true parameters only.

- The N = 2 continuum basis needs 12 averaged-kernel calls with analytic derivatives; with five-point differences it needs 60, as does the harmonic basis, whose calls take 1.3–2.9 s each (read from that memo; `build.harmonic_stencil_basis_read_from_survey`).
- The 72 columns of the joint ensemble need 12 kernel calls because Eq. (5.3) supplies the field derivatives; a kernel without that identity needs 300 (`joint_gamma_B.counts`).
- The matrix–vector product of route (iii) takes 0.8 µs (`one_d.<name>.route_iii_parts.matvec_123x36_only`, 0.79 and 0.79 µs); computing a(θ) by quadrature takes the rest.
- At the true parameters the joint basis reaches 7.8e-3 (`joint_gamma_B.moment_route_72_max_rel_err`), and mapping each field strength onto the fixed-field energy columns through Eq. (5.3) reaches 3.7e-3 without any field expansion (`pushforward_36_fixed_B0_max_rel_err`). The field-averaged basis with marginal energy moments misses by 8.4e-2 (`preaveraged_36_with_marginal_moments_max_rel_err`), because it ignores the energy–field correlation; Sec. 5.4 gives 8%.
- Parameter vectors: for E2 and E4-JP the errors of routes (ii) and (iii) are maxima over 60 vectors, the reference and 59 draws around it (`cost.py`, `theta_list`, seeds 7 and 8). For the joint ensemble route (i) is evaluated for the first 5 of its 60 vectors (`n_direct = 5`), so the joint errors 8.9e-4 and 8.7e-3 are maxima over the true parameters and 4 draws.
- Parametric families: the moments of 1000 parameter vectors at N = 2 take 3.8, 5.2, 10.5 and 25.2 ms for E3, E1, E2 and E4-JP (`forward_results.json → family_checks.<name>.seconds_per_1000_theta_N2`). Sec. 5.4 quotes 4–28 ms from the original run of this study, which gave 4.2, 4.9, 11.6 and 28.0 ms.
- Source: `cost_results.json` (`one_d`, `build`, `joint_gamma_B`); all entries agree to the rounding. Category: timings are medians on the stated machine; the error column is a numerical check at the stated parameter vectors.

**Paper.** In the paragraph "Computational cost" at the end of Sec. 5.4, "Building the N = 2 continuum basis takes 12 averaged-kernel calls with analytic derivatives, or 60 with five-point differences as for the harmonic kernel, against 96–288 for a cached table of the averaged kernel at fixed energy nodes"; for E2 or E4-JP, 22–49 µs per evaluation of the 123 data rows with the basis, against 17 µs with the 288-node table and about 1 s for recomputing the averages; "For the E4-JP parameters and 59 draws around them, with standard deviations of 5% in A and 0.05 in p and ln γ_c", up to 1.1e-2 of I_j for the basis and 3.7e-3 for the 288-node table; for the joint ensemble, 4608 kernel calls for a table against 12 for its 72 joint columns, 300 for a kernel without Eq. (5.3), up to 8.7e-3 of I_j for the joint basis over the true parameters and four draws (standard deviations of 5% in A and in the age and 0.05 in p; `cost.py` compares only these five vectors with the direct spectrum), and 8% of I_j at the true parameters for a field-averaged basis evaluated at the marginal energy moments; "The timings, hardware and further routes are recorded with the scripts (see Code availability)." The Conclusions (Sec. 7) quote 17 against 22–49 µs and refer to Sec. 5.4 for the kernel calls. README only: the hardware and versions, the cost table with its build and per-route times, the node counts 288, 216 and 3456, the errors 2.4e-15, 1.2e-3, 8.9e-4 and 8.4e-2, the harmonic per-call time 1.3–2.9 s, the 0.8 µs product, and the true-parameter values 7.8e-3 and 3.7e-3.

### 11. Illustrated Faraday screens (Sec. 4.4)

**Produced by.** No script in this directory (Fig. 7 of the paper).

**Screens.** With x = (Ψ − Ψ̄)/σ_Ψ and t = 2λ²σ_Ψ, the rotation-phase averages are characteristic functions C(t) = ⟨e^{itx}⟩_scr:

| Screen | Density | C(t) | Third and fourth cumulants |
|---|---|---|---|
| Gaussian | standard normal | e^{−t²/2} | (0, 0) |
| Laplace | p(x) = e^{−√2 \|x\|}/√2 | (1 + t²/2)^−1 | (0, 3) |
| Gamma | x = Y/2 − 2, Y ~ Gamma(4, 1) (unit scale) | e^{−2it}(1 − it/2)^−4 | (1, 3/2) |

- All three have zero mean and unit variance.
- The two- and four-cumulant approximations follow from Eq. (4.21) with these cumulants.
- The Gamma residual position angle is −t + 2 arctan(t/2), continued from zero after subtracting the mean-depth rotation.
- The Laplace and Gamma logarithmic series have radii √2 and 2. Fourth order need not meet a tolerance inside these radii, and adding cumulants outside them can increase the error.

**Exponent remainder** (Sec. 4.4). On a zero-free interval, write the logarithm of the screen factor as g = g_4 + R_5. Taylor's theorem gives |R_5(τ)| ≤ |τ|⁵ sup_{u ∈ [0, τ]} |g^(5)(u)|/5!, where the derivative envelope is additional information. With P̂_ν = P_ν^em e^{g_4}, a supplied |R_5| ≤ ε_5 gives

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

**Paper.** The Laplace density, the shifted-Gamma definition and the three characteristic functions in the caption of Fig. 7; in Sec. 4.4, the Laplace and Gamma screens with equal mean and variance, x and t, the propagation rule for a supplied ε_5, the ε_5 values, the vanishing Gaussian R_5 and the radii √2 and 2. README only: the cumulant values, the Gamma position angle and the derivative suprema.

### 12. Secondary checks of Sec. 5.3–5.4

**Produced by.** `fits.py` → `fits_results.json`; `verify_fits.py` → `verify_fits_results.json`; `params.py` with `paramlib.py` and `params_degeneracy.py` → `params_results.json`.

**Checks** (checked; not in the paper). Values refer to the seeded noise realisation unless a number of realisations is given.

| Statement | Value | Source | Category |
|---|---|---|---|
| E1M: the part of the representation error outside the retained directions adds to χ² | 0.018 | `verify_fits_results.json → ensembles.E1M.eps_w_outside_retained_R_star` (0.0180) | numerical check |
| E5: the joint fit stays within this many standard deviations of the true polarisation angle in every channel with \|P_j\|/I_j > 3% (29 channels) | 1.6 | `fits_results.json → E5.chi_nu.max_abs_joint_minus_true_over_sigma_selected` (1.61) | numerical check |
| E5: UPRE prefers the joint model to one free azimuth factor | in 400 of 400 realisations | `fits_results.json → E5.joint_vs_separable_realisations.fraction_upre_prefers_joint` (1.0, n = 400) | numerical check |
| E1–E4 and E1M: the minimum-norm moments violate the moment conditions of a nonnegative distribution in at least one bin | in every one of 4000 realisations | `fits_results.json → ensembles.<name>.realisations.hausdorff_tsvd_fraction_with_violation` (1.0) | numerical check |
| Standard deviations of the parameter pulls (compressed fit at each realisation's R_⋆; E1, E2 with s = 2, E3, E4-JP) | 0.97–1.01 over 1000 realisations | `params_results.json → tasks.<task>.realisations.comp_Rstar.params_vs_reference.pull_sd` (0.970–1.010) | numerical check |
| E2 with s free: fits that fail to converge | 16–17% of 1000 (156 compressed at R_⋆, 169 at fixed R, 164 full) | `params_results.json → tasks.E2_sfree.realisations.{comp_Rstar,comp_Rseed,full}.n_failed` | numerical check |
| E2 with s free: growth of σ(p_1) at the truth | fourfold (0.064 to 0.276) | `params_results.json → tasks.E2_s2.fisher_sd_full_at_ref[1]` and `tasks.E2_sfree.fisher_sd_full_at_ref[1]` | numerical check |
| First relation of Eq. (5.3) for the numerical averaged kernel, λ = 0.5, 0.8, 1.25 and 2 | to 7e-12 of the largest channel value | `params_results.json → degeneracy.kernel_scaling_identity[*].kbar_I_max_abs_diff_over_max_j` (largest 6.6e-12) | numerical check |
| E2 ridge with the energy support rescaled with the field: change of χ² | at most 4e-13 | `params_results.json → degeneracy.E2_ridge.summary["transformed/seeded/full/delta_chi2_vs_lnlam"]` (largest 3.8e-13; noise-free 4.4e-15) | numerical check |
| Same ridge: d ln γ_br/d ln B | −0.5000000 | `degeneracy.E2_ridge.summary["transformed/noisefree/full/slope_dln_gamma_b_dln_B"]` (−0.49999999978) | numerical check |

- [inferred] The uncertainty of the E1M invariant C⟨B_⊥^{(p+1)/2}⟩ is dominated by that of p through C = A (1e4)^p. The amplitude invariants have errors 11–14 times σ(ln A) for E1 and E4-JP (`params_results.json → tasks.<task>.seeded_comp.invariants_sd` against `sd`).
- Values that the paper states are not repeated in the table: the E1 index 2.5976 ± 0.0032 (Table 6), the JP–KP match 7.7e-5 I_j (Sec. 5.2) and the E1M χ² of 51.2 for 56 degrees of freedom (Table 4).

**Paper.** Sec. 5.3 states "The joint fit recovers 24.1°" and the 10–11 violated bins of the seeded realisation; Sec. 5.4 states "with s free, some fits fail to converge and the uncertainty of p_1 grows", the compressed-against-full agreement within 0.3% for R ≥ 7, σ(ln B) ≈ 2e5, the fixed-support ridge (0.012 over a factor 2.2 in B; −0.11 to +0.25 with noise; up to 0.42 standard deviations) and the E1M invariant at 0.5% (0.17 standard deviations). README only: every value in the table above.
