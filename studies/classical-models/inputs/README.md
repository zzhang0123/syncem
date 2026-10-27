# Inputs

External files that the scripts of `studies/classical-models` read or hash. Each subfolder is named by the origin of its files. Except for the two files listed under "Edited copies", every file is a byte copy of the file the study used, and its SHA-256 equals the value that the study's results recorded for it.

## `harmonic-radio-fit/`

Frozen outputs and code of the harmonic pipeline: the linear radio fit of the paper with finite-γ harmonic line powers (41 channels over 0.1–10 GHz, N = 2, 36 moments). The pipeline pinned SyncMoments commit d51f9bef4fa744a536a60c3da821b1033818a232.

| File | SHA-256 | Read or hashed by |
|---|---|---|
| `results_final.json` | `aa3d576c5e72a71d9078c0e9f84f4b22a73fd9bb82b6c5194aafaa25cd5cbbd0` | `regression.py`, `bridge.py`, `verify_bridge.py` |
| `trusted_basis.npz` | `94617e24610f75e7b5855e75490ad0f44a51bb883f5b5defbf6e688c5bee7329` | `bridge.py`, `verify_bridge.py` |
| `trusted_basis_halfstep.npz` | `a24c8cef839f9620f12935f89232e6ca4cccc56836573fb194f3e12b568a3478` | `bridge.py`, `verify_bridge.py` |
| `trusted_basis_refined.npz` | `a1d08f02c17dd00ea94782aff57e6e0813d79d20509dcb47d60ebf6d16f38c53` | `regression.py`, `bridge_harmonic_probe.py`, `bridge.py`, `verify_bridge.py` |
| `trusted_truth.npz` | `e9a5b95341412d4659abcee465fc0c51a9ad34577aae3a7437b72542ca0fb3e2` | `bridge.py`, `verify_bridge.py` |
| `trusted_truth_refined.npz` | `c4048e3d501259f960b2ed386ffb5a9301cd0104cf6915c3890806a42cecc939` | `regression.py`, `bridge.py`, `verify_bridge.py`, `verify_core.py` |
| `stable_lines.py` | `c661ba3235bae5c5343651b2fdc995d9fb7228dfc329619255456adc51cc6768` | imported by `radio_numerics.py` and by part G of `kp_bridge.py`; hashed by `bridge.py`, `verify_bridge.py`, `kp_bridge.py` |
| `radio_numerics.py` | `a70738408d68928c87c8557ffa11643d51a4c13380026b36bd2d02f273fce4b0` (edited copy) | imported by `bridge_harmonic_probe.py` and `verify_bridge.py`; hashed by `bridge.py`, `verify_bridge.py` |
| `finalize.py` | `d2604b42535a4d9c822b5273962b8c86a586caa866c85ad043eeba93b3419c3f` (edited copy) | hashed by `bridge.py` (the fit driver that wrote `results_final.json`; not run here) |

`radio_numerics.py` and `stable_lines.py` import the package by its earlier name, `syncmoments`. The scripts that import them call `lib/syncem_compat.py` first. It imports `syncem` from the package root of this repository, checks that `syncem.constants` and `syncem.model.high_order_harmonic` equal their d51f9be sources up to the rename, and registers `syncem` under the earlier name.

The five npz files were written by the pipeline's `generate_cache.py` through `cache_io.py`. Each records, under `sources` in its `metadata_json`, the SHA-256 of the pipeline's `generate_cache.py`, `cache_io.py`, `radio_numerics.py` and `stable_lines.py`. `generate_cache.py` and `cache_io.py` are not distributed. `finalize.py` wrote `results_final.json` and is included for its SHA-256 only. It imports `cache_io`, which is not distributed, and running it would rewrite `results_final.json`; do not run it here.

### Edited copies

Each of these two files contained one absolute path to a temporary directory of the machine that ran the pipeline. That line was replaced; the rest of each file is unchanged.

| File | SHA-256 of the pipeline's file | Edited line |
|---|---|---|
| `radio_numerics.py` | `bd477ecadf9b015fe7a7fc37c59c3cf9847a797cd93a3d135c90b67eecdd6320` | `PACKAGE`, the directory of the pinned package snapshot, is now `Path(__file__).resolve().parents[4]`, the package root of this repository. The module puts it on `sys.path`; the package is found through `lib/syncem_compat.py`. |
| `finalize.py` | `67c4db4b22bd8c0e9afcc6b5543853365e4b2771f0b8180def391214952b64d9` | The `MPLCONFIGDIR` default is now `.mplcache` next to the file. |

The study's earlier results record the pipeline's SHA-256 for these two files. Results written with these copies record the SHA-256 of the copies.

## `validation/`

Output of the validation script `harmonic_continuum_check.py` (harmonic against continuum kernels at θ = π/3). The script is not distributed.

| File | SHA-256 | Read by |
|---|---|---|
| `harmonic_continuum_results.json` | `aad5d7140bfa7e66818a0659c3982b6496fb18e87e439e7375e2feceba55cd12` | `harmonic_correction.py` (harmonic-versus-continuum kernels at θ = π/3) |

`lib/figure_style.py` is a byte copy of the style module used for the paper's figures (SHA-256 `e2f37e065b7e0b4f927a16819faba9f406e2733eb61977af70529e22d20cd003`), used by `bridge.py` and `make_figures.py`.

## `galprop-benchmark/`

Regional spectra of the published GALPROP synchrotron intensity cube of model z10LMPD_SUNfE, and two earlier fits to them.

**Source.** Orlando E., Strong A. W., 2013, MNRAS 436, 2127, doi:10.1093/mnras/stt1718, arXiv:1309.2947. The cube is the file https://www.mpe.mpg.de/~aws/test_healpix_standard_synchrotron_healpix_55_z10LMPD_SUNfE.gz (52,205,497 bytes, SHA-256 `29334988dfb630aa2b9adc95d4a2fc9979619b9fc2eacddd4327c52b1980f596`), linked from https://www.mpe.mpg.de/~aws/propagate.html. It holds a HEALPix map (NSIDE 128, RING order, 196608 pixels) at each of 76 frequencies 1e6 × 1.2^k Hz, k = 0, …, 75. Of the cube, only the regional means below are redistributed here.

**Reduction.** `regional_spectra.csv` gives, at each of the 76 frequencies (column `frequency_ghz`, in GHz), the intensity in erg cm⁻² s⁻¹ sr⁻¹ Hz⁻¹ averaged over equal-area pixels (the mean of the intensity, not of its logarithm) in three regions fixed before any fit: the whole sky (`all_sky`), 20° ≤ |b| < 60° (`mid_latitude`) and |b| ≥ 60° (`high_latitude`). The reduction script `curvature_pilot.py` wrote it together with `curvature_results.json` (polynomial fits of degree 1–3 to ln I against ln ν), and `moment_fit.py` wrote `moment_fit_results.json`. Neither script is distributed.

**Caveats of the published file.** The FITS frequency column is labelled MeV, but its values are frequencies in Hz, and the intensity unit reads `unknown`; the units above follow the GALPROP manual. The header has no COORDSYS keyword, and Galactic coordinates are assumed as stated in the publication. No matching galdef file, code revision or convergence record was obtained, so the absorption settings of the run cannot be audited from the file. Orlando & Strong (2013) include free-free absorption, so the spectrum is not treated as absorption-free. `galprop_absorption.py` estimates the absorption from the low-frequency turnover of the spectrum (`baselines_results.json → absorption_caveat`). `curvature_results.json` records the source URL, the cube SHA-256 and these caveats (`source_url`, `input_sha256`, `metadata_caveats`).

| File | SHA-256 | Read by |
|---|---|---|
| `regional_spectra.csv` | `e4a1bab569b8ae1b6b3ec954f5ab22ac9cee29b74e3e560184a7b9c2542516ea` | `baselines.py`, `verify_baselines.py` (column `high_latitude`, \|b\| ≥ 60°) |
| `curvature_results.json` | `9f67da7e32d9c9f4a659472a0893a1d21d2ab98730c7913cd966e1d5b4a0429a` | `baselines.py` (reproduces the log-space fits recorded there) |
| `moment_fit_results.json` | `37870ab52f6a3ad98601529c4f8f4a64ce4c4d3c9271fec415df6eaa7304e2c9` | `baselines.py` (log-cubic withheld error recorded there) |

## `beaming-survey/`

Output of `check_beaming.py` (not distributed), an earlier check of the beaming limit: exact harmonic sums against 2p(η)K^iso for four pitch densities at γ = 10–80 and θ = π/3.

| File | SHA-256 | Read by |
|---|---|---|
| `check_beaming.json` | `031476ca56ab042f2948b72571f491517ede9e3be1f43083e45cbcdeebdc81f6` | `kp_bridge.py` (part A, regression check) |

## Not distributed

- `orlando2018_PDDE_all_electron_LIS_UNVERIFIED.csv`: an unverified transcription of Table 5 of Orlando E., 2018, MNRAS 475, 2724, doi:10.1093/mnras/stx3280, used only by the exploratory E2a branch of `params.py`. Without it `params.py` skips E2a and records the skip under `E2a_input` in `params_results.json`. No number of the paper depends on E2a.
- The timing memo of the harmonic pipeline from which `cost.py` copies the harmonic per-call times (`cost_results.json → build.harmonic_stencil_basis_read_from_survey`). The values were copied by hand, and `cost.py` labels them as read, not re-run.
