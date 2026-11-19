# SyncEM top-level design, version 6
Applied in version 6: the prototype corrections of `prototypes/RESULTS.md` Sec. 3 to 5, the owner decisions of `OWNER_DECISIONS.md` and the native-code decision of `NATIVE.md` (Sec. 0).
Round-3 corrections applied in version 5: r3-01, r3-02, r3-03, r3-04, r3-05, r3-06, r3-08, r3-10, r3-14.
Round-3 findings applied in version 4: r3-01, r3-02, r3-03, r3-04, r3-05, r3-06, r3-07, r3-08, r3-10, r3-14, r3-16, r3-17, r3-20, r3-23, r3-25, r3-26, r3-29, r3-32, r3-33, r3-34, r3-37, r3-40, r3-41.

Earlier rounds: r2-01 to r2-12, r2-14, r2-15, r2-17 to r2-23, r2-26, r2-29, r2-31, r2-36, r2-37, r2-39, r2-42, r2-44 in version 3. r1-01 to r1-19, r1-24, r1-27, r1-30, r1-34, r1-36, r1-37, r1-40, r1-43 in version 2.
Synthesis of proposals P1 to P5 and judge reports J1 to J3, revised after review rounds 1 to 3.
PKG = `.worktrees/syncem` at v0.7.10, PAPER = `main.tex`, ST = `PKG/studies/classical-models`, CTX = the context pack.
`se` means `import syncem as se`. Every name under `se.` is proposed API unless a PKG path is given.
`..` marks a number this document did not compute. Tags such as [0.9] mark parts scheduled after the first release (Sec. 10).
Checks run for this synthesis are in `scratchpad/design/synth/` (`window.py`) and are cited where used. Numbers attributed to a review finding (r1-xx, r2-xx, r3-xx) were computed by its reviewer and verifier.

## 0. Summary

Version 6 applies the corrections of the prototype run (`prototypes/RESULTS.md`), the owner's answers to the open questions of Sec. 14 (`OWNER_DECISIONS.md`) and the native-code decision (`NATIVE.md`). The two accuracy terms are signed, and `predict - direct` equals `representation + outside_lattice` with `outside_lattice` the in-cell minus the full-support emission (Sec. 7.2). Depth powers lie outside the total Taylor degree, and the depth check is the bound `(tau_max h)^(N_d+1) / (N_d+1)! <= depth_tol` (Sec. 4.3, 4.6). A law whose nodes move with a celled parent enters the columns through its derivatives, and frozen nodes are admitted only with density reweighting on a fixed support (Sec. 8.3). The R3a term `conditional_truncation` is bounded per unit parent mass by `E` before any data (Sec. 5.2), and S1 tables come from one S0 table in `ln x` plus lookups (Sec. 8.2). SciPy is a base dependency, `AzimuthFourier(m_max)` ships in 0.8.0, and 0.8 and 0.9 add no native code.

### 0.1 What changes for a user

1. `import syncem as se` gives a front door of four functions: `se.predict`, `se.fit`, `se.plan`, `se.spectral_basis`. `se.problem` and `se.compare` complete it.
2. The user writes one `se.Population`. Each keyword is a variable of the base model. Its value states what is known: a number, a distribution, a function of other variables, a family with `se.Free` parameters, `se.FreeMoments` or `se.NonNegativeCells`.
3. A forward prediction is `se.predict(population, channels)`. A fit is `se.fit(population, data)`. The estimator follows from the declaration, and the result names it.
4. Cells exist on every variable of the kernel except the azimuth, which is resolved by its harmonics: exactly by `AzimuthHarmonic`, and by `AzimuthFourier(m_max)` with a truncation term in the two cases of rule R4a (b) (Sec. 3.3, 5.2). Under the beaming limit of the continuum kernel, cells on the pitch cosine need an inclination law with a density or Legendre degrees (Sec. 4.2). `se.Lattice` states which variables are resolved and how. Without one, a default lattice is derived by numbered rules, and each derived choice is printed with `[default]`.
5. The workflow of PAPER Sec. 5 (19 energy cells, specified field, angle and screen statistics, free moments, non-negative cells, families, extrapolation) is typed in about ten lines. Its numbers are acceptance tests.
6. Dimensional inputs carry units, data included. A field strength, a frequency or a data array without a unit raises.
7. Every result prints the role of each variable, the statistics the basis contains, a measured accuracy number where one can be measured, and the names of the terms that are not bounded.
8. `fit.predict(other_channels)` needs no second object. `se.fit(..., predict=other_channels)` makes the default lattice cover them. A free-moment prediction carries `unconstrained`, the norm fraction of each prediction row outside the retained directions (scaled moments, so it depends on the reference weights `w`).
9. Generality is reached by adding arguments (`lattice=`, `kernel=`, `phase=`) to the same calls.
10. `syncem.model` and the legacy modules are unchanged in the first release, and the v0.7.10 studies keep running.

### 0.2 Three layers

```text
Layer 1  base model         Kernel (declared variables, Stokes blocks, capabilities with preconditions) and PhaseLaw, observed through Channels
                            g_Xj(x) = exp(i s_X phi) K_Xj(gamma, B, mu, eta, depth)                 PAPER eq: channel kernel

Layer 2  lattice            Lattice: one representation per resolved variable, product cells stored in a cell table
         per-cell             Cells with a local polynomial order | Legendre degrees | exact azimuth harmonic | azimuth Fourier rows | point masses
         expansions         every other variable is averaged against a specified distribution (quadrature, or a characteristic function)
                            -> SpectralBasis  X~[j, alpha] and a column map S from lattice rows    expensive, cached, records what it contains

Layer 3  population models  Population: fixed | specified | conditional | family with Free | FreeMoments | NonNegativeCells
                            sums of components, Samples, Ray, Moments
                            -> moments  a[alpha] = L(theta) c                 cheap
         forward            d = X~ S a                                        one matrix product
         fit                min over theta, theta_b, c of | W X~(theta_b) S L(theta) c - d_w |^2
                            estimator read from the form of L, the set of c and the basis parameters theta_b
```

### 0.3 Source of each part

| Part | Taken from | Reason |
|---|---|---|
| object core: representations per variable, product cells in a cell table, `average`, `expand`, `pair`, normal form `L(theta) c` | P2 | the only object model that meets B1 to B7 of the owner's clarification without an exception (J2 Sec. 7.1) |
| engine: stages, cache keys per channel row, column providers, cell table, scale plan, dispatch tests | P4 | its two engine claims were reproduced by all three judges (J3 C2, C3) |
| generic route: a Taylor jet in any number of celled variables of the engine-side averaged kernel (`AveragedKernel`, Sec. 8.1). `_build_core` with the Stirling re-expansion stays as the fast path and oracle for `(gamma, B)` synchrotron cells | P5, generalised in review round 1 (r1-01) | P5's route reproduces the study columns to 2.4e-12 (J2, J3 C1). It differentiates two named variables only, so the general contract is the jet |
| grammar, ordered role rules, plan, keys, fixed caveats | P3 | one reading rule for all declarations (J1, J2, J3) |
| front door surface: defaults table, units, error messages, `se.paper` | P1 | lowest barrier in all three reviews |
| delivery: additive release, guards, one acceptance test per step | P5 | the only plan with measured steps (J1, J2, J3) |

The judges named three different spines (P1, P2, P4) and the same division of parts. The division above is that common ground.

Requirements of digest Sec. 7 and where each is met: 1 (Sec. 3.4, 5.3, 8.2), 2 (Sec. 4.2, 5.2), 3 (Sec. 4.2, 4.7, G11), 4 (Sec. 3.3, 4.1), 5 (Sec. 3.2, 8.1, 8.3, 8.4), 6 (Sec. 6), 7 (Sec. 3.4, 7), 8 (Sec. 9.6, 10.2), 9 (Sec. 1, 2.4, 8.6), 10 (Sec. 9.5, 9.7).

## 1. Principles

1. Each term of `d = X~ S a` has one object: a kernel, a lattice, a spectral basis with its column map, a population, a moment vector.
   Reason: the shortest script and the most general script then run the same code.
2. Each base variable has one role per model: resolved by a representation on the lattice, or averaged against a specified distribution.
   Reason: this split is the only difference between the configurations of the paper (digest Sec. 2).
3. The variables are treated symmetrically, and the kernel declares them. No engine contract names a variable.
   Reason: owner's clarification B1, B2 and B7.
4. An explicit lattice is the one statement of what is resolved. A default lattice is derived by numbered rules.
   Reason: a statement by the user is never changed by a rule (J2 Q2, J3 Q6).
5. Separate keywords are independent. A positional block is a joint statement.
   Reason: one reading rule covers specified and unknown factors (J1 Q2, J2 Q3, J3 Q5).
6. The spectral basis is the expensive, cached object, and it records the specified statistics it contains.
   Reason: a basis with averaged statistics must be rebuilt when they change (PAPER:846, E1M).
7. Every compiled model has the normal form `a = L(theta) c`. Free parameters of averaged statistics, `theta_b`, change the basis and are named as such.
   Reason: every fit then has the same inner linear problem (P2 Sec. 6.1, P4 Sec. 5.3).
8. Derivative columns define the basis spectra.
   Reason: this is PAPER eq: illustrated derivative spectra, and these columns produced the paper's numbers.
9. An assumption is applied only with a printed line and a ledger entry, and a missing error input is `unbounded`.
   Reason: project rule in `AGENTS.md`.
10. A default is allowed when it is printed, stored in `result.defaults` and refused under `strict=True`.
    Reason: defaults decide numbers (J1 Sec. 5.1).
11. Units are converted once, in constructors and in the tabulation of conditionals.
    Reason: JAX strips units silently (CTX 09 Sec. 5.5).
12. Policies are typed objects with defaults.
    Reason: a string that changes the parameter set hides the model (CTX 09 pitfalls).
13. Public classes are immutable `equinox.Module`s or frozen dataclasses. Static fields hold metadata only, never arrays. Expensive work runs in `build` classmethods.
    Reason: a changed static field recompiles, a static array cannot be hashed, and a constructor that computes reruns on every rebuild.
14. Every dispatch between two numerical methods has a test that evaluates both at the dispatch point and at extreme parameter values, with the dispatcher bypassed.
    Reason: failures at a threshold are invisible to point checks through the dispatcher.
15. The first release is additive.
    Reason: the paper cites v0.7.10, and `tests/model` holds 900 tests of the present behaviour.
16. One name per concept, taken from the paper where the paper has a word.
    Reason: the paper is the specification (lead notes Sec. 5).
17. A shortcut (a cached table, an analytic column relation, a half-range quadrature, a parity mask, closed-form depth weights) runs only where its stated precondition holds. The plan checks the precondition against the whole declaration, the parents of every conditional included, records the route in `PlanRecord.routes`, and otherwise selects the general route.
    Reason: a shortcut outside its precondition drops terms without an error (r1-02, r1-09, r2-01 to r2-04).

## 2. Front door

### 2.1 First screen

```python
import syncem as se                      # enables float64, as today. Names load lazily

pop  = se.Population(gamma=se.PowerLaw(2.6), B="5 uG",
                     phi=se.AzimuthFactor(0.4), depth=se.GaussianScreen(mean=8.0, sigma=0.5))
spec = se.predict(pop, ([0.408, 1.4, 2.3, 4.8, 23.0], "GHz"))
print(spec)

data = se.StokesData(obs, stokes=stokes, sigma=sigma, unit="Jy/sr")   # obs: a Channels. stokes (..., n_ch, 3). sigma is a standard deviation
high = se.Channels.at([30, 100, 300], unit="GHz")
fit  = se.fit(pop.replace(gamma=se.FreeMoments()), data, predict=high)   # electrons free. The default lattice also covers high
fit.predict(high)
```

```text
Spectrum     5 channels 0.408-23 GHz, monochromatic, Stokes I Q U
unit         erg s^-1 cm^-2 Hz^-1 sr^-1 for dN/dgamma = 1 cm^-2 at gamma = 1e4                    [default: normalisation]
  nu [GHz]        I           Q           U       |P|/I
     0.408        ..          ..          ..       ..
     ...
roles        gamma  resolved  13 cells in ln gamma on [1.70e3, 7.96e5], order 2, 39 moments        [default: range, cells, order]
             B      averaged  5 uG, fixed
             eta    averaged  isotropic                                                            [default]
             mu     averaged  isotropic                                                            [default]
             phi    averaged  azimuth factor <exp(2i phi)> = 0.40
             depth  averaged  independent Gaussian screen, mean 8, s.d. 0.5 rad m^-2
assumed      B, eta, mu, phi, depth independent of gamma and of each other (separate keywords)
basis        continuum kernel (V not modelled), Faraday rotation tau = 2 (c/nu)^2
             contains B, eta, mu, depth. Rebuild when these change
accuracy     measured   representation: max |dX| / I = ..  (inside the cells, same kernel, 0.408-23 GHz)
             measured   outside_lattice: max |dX| / I = ..  (emission outside the cell range, default range cut)
not bounded  physical_kernel, specified:B, specified:eta, specified:mu, specified:phi, specified:depth
```

The cell range and count are the output of the rule of Sec. 4.6 for these channels (`prototypes/synth/window.py`). The two accuracy terms are signed and add to the difference from direct integration over the declared support. Their reported maxima bound that difference and do not add to it (Sec. 7.2).
Three levels of use share these calls. A newcomer writes a `Population` and calls `predict`. An analyst adds `StokesData`, `fit`, the markers and `lattice=`. An expert builds `Lattice`, `SpectralBasis` and `Moments` directly (Sec. 3) and reads the design matrix.

### 2.2 Signatures

```python
se.predict(population, channels, *, lattice=None, kernel=None, phase=None, basis=None, stokes=None,
           check=True, resolve=(), average=(), allow_mismatch=False, max_columns=20_000, strict=False, cache=True) -> Spectrum
se.fit(population, data, *, lattice=None, kernel=None, phase=None, basis=None, predict=None, rank=None, penalty=None,
       starts=None, solver=None, keep_directions=False, chunk_bytes=2**31, check=True, resolve=(), average=(),
       allow_mismatch=False, max_columns=20_000, strict=False, cache=True, key=None) -> Fit
se.plan(population, channels=None, *, lattice=None, kernel=None, phase=None, stokes=None, predict=None,
        resolve=(), average=(), max_columns=20_000, strict=False) -> Plan   # builds no basis stage. May call cheap kernel capabilities
se.spectral_basis(population, channels, *, lattice=None, kernel=None, phase=None, stokes=None, check=True,
                  resolve=(), average=(), max_columns=20_000, strict=False, cache=True) -> SpectralBasis | BasisSum
se.problem(population, data, *, <the compile options of se.fit>) -> Problem     # se.fit is se.problem(...).solve(...)
se.compare(*fits, withheld=None) -> Comparison
se.convergence(population, channels, *, order=None, refine=None, **predict_options) -> Comparison   # the same prediction at higher order or on refined cells
```

* `channels` is a `Channels` object or a pair `(frequencies, unit)`, which gives monochromatic channels.
* `data` is a `StokesData`, or the `RetainedCombinations` of an earlier free-moment fit (compressed likelihood, Sec. 6).
* `se.predict(pop, ch)` equals the sum of `part.predict(pop)` over the parts of `se.spectral_basis(pop, ch)`, which is one `SpectralBasis` unless the components of a sum differ in an averaged statistic (R7, `BasisSum` in Sec. 3.4). Bases are memoised in the process, so later calls with the same specified statistics, lattice and channels build nothing.
* `predict=` (channels) in a fit makes the default energy range cover the union of the fitted and the predicted channels (Sec. 4.6 rule 1). With `lattice=` it only records the channels for the check of Sec. 6.5.
* `max_columns` caps the basis columns of every lattice, explicit or default. The plan prints the row count, the column count and the moment memory per sightline, and a refusal names `max_columns=` (Sec. 4.6 rule 3).
* `None` for a policy means the default of the problem class (Sec. 6.1). `strict=True` raises on the first default and lists all of them.
* `population` is a `Population`, or one block (`Samples`, `Ray`, `Moments`), which is wrapped.
* `resolve=` acts on the default lattice only (Sec. 5.2). With `lattice=` it is an error, because the lattice already states what is resolved. `average=` drops a declared dependence and is valid in both cases.
* `keep_directions=True` stores the retained directions of every pixel when the noise differs per pixel. By default they are recomputed in chunks when needed. `chunk_bytes` is the memory budget of one chunk of per-pixel SVDs (Sec. 6.4).
* `allow_mismatch=True` applies when `basis=` is given (in `se.problem` through the compile options of `se.fit`). It accepts a basis whose specified statistics differ from the population and records the difference, measured on the host (Sec. 5.2). It never bypasses the lattice-key check, which only `Lattice.with_key` does (Sec. 3.3). `se.spectral_basis` takes no `basis=` and no flag.

### 2.3 Defaults: value, source, place of report

Every applied default is a `[default]` tag in `print(plan)`, `print(spectrum)` and `print(fit)`, and an entry of `result.defaults`.

| Default | Value | Taken from | Also reported in |
|---|---|---|---|
| precision | float64 at import | present behaviour (`PKG/syncem/__init__.py:41-44`) | import docstring |
| channel response | a pair `(frequencies, unit)` is monochromatic (kernels without lines only). `fractional_width=f` gives the unit-area bump of PAPER eq: forward response families with spread `f nu_j`, `half_width=x` the same bump with support half-width `x nu_j` (decision 124) | PAPER:396-403, 1119 | header line |
| channel normalisation | unit area, values per Hz | paper. Reverses `unit_peak` of `Channels.bump` | unit line |
| channel quadrature | 64 nodes per channel, doubled while the `quadrature` term exceeds 1e-8 of I, at most three times | `channels.py:138`. J3 C12: 64 against 800 nodes changes the paper columns by 2.0e-12 | `basis.record`, ledger |
| kernel | per cell, or on the averaged support of the primary variable when it is not celled: continuum where the cell lies at `gamma >= 300` and V is not requested. Otherwise harmonic when `required_m_max <= 16368` (harmonic cells: 0.9). Otherwise `RegimeError`. The energy range is derived first, with the kernel passed or with the continuum window (Sec. 4.6) | PAPER:420, `kernels.py:110`, `_basis_core.py:36`, r2-18 | `basis` line, cells below 300 listed |
| phase law | Faraday rotation, `tau = 2 (c/nu)^2` | PAPER eq: channel kernel | `basis` line |
| Stokes blocks | those the kernel provides | kernel | header. Data in an unmodelled block raise `DataError` |
| resolved variables | the kernel's primary variable (`Kernel.primary`, `gamma` for the synchrotron kernels), each variable that is unknown, and each parent of a resolved variable | rules R2, R3, R3a of Sec. 5.2 | roles table with the rule |
| energy range | the kernel's `EnergyWindow`: electrons outside contribute less than 1e-3 of the band-integrated `I_j` in every channel on each side, for `gamma^-2.6`. The channels are the fitted channels and those of `predict=`. Mapped with the channel-support edges and the bounded support of the field distribution, or its declared truncation (Maxwellian: 8 rms), or the lattice's `B` range when `B` is resolved. Intersected with a bounded declared support. Lower edge at least the kernel limit. A kernel without `EnergyWindow` needs a bounded `gamma=` support or `lattice=` | PAPER:1059. For the paper statistics over 0.1 to 300 GHz, monochromatic channels give an upper edge of 3.0e6 (`prototypes/synth/window.py`) against the paper's single-frequency 2.96e6. Bump channels with `f = 8 %` give 3.3e6 (r2-31) | roles table, `outside_lattice` |
| energy cells | equal in `ln gamma`, half-width at most 0.242, reference points at the centres, order 2 | PAPER:1069. The only width the study tested (CTX 06 D.3) | roles table |
| representation of another resolved variable | positive or real variable: one affine cell over a bounded range, lattice order. An unknown variable takes the range from `support=` on its marker. A declared law with unbounded support takes its `[1e-3, 1 - 1e-3]` quantile range, and the tail mass is a measured `outside_lattice` term. No range: `RoleError`. A depth axis takes the smallest cell count that passes the depth check. Cosine: `Legendre(8)` in a forward call, required in a fit. Azimuth: exact harmonic, or `AzimuthFourier(m_max)` for a resolved azimuth with an averaged child or with a child outside its free block (R4a (b)) | PAPER:1077 (E6), PAPER:1081, PAPER:708, r2-19 | roles table with the rule |
| reference weights `w` | cell masses of `gamma^-2.6` on the lattice range for an axis named `gamma`, summing to one, and factor 1 for other axes, resolved when the lattice is built. A lattice given without `weights=` takes the kernel's `ReferenceDensity` on its axis through the plan, with `[default]` and a new lattice key (Sec. 4.5) | PAPER eq: fit coefficient scales, r2-07 | plan, fit summary |
| `mu`, `eta` undeclared | `Isotropic()` | PAPER:456 (random field directions) | roles table |
| `phi` undeclared | uniform, so `Q = U = 0`. The print says so and names `phi=` and `eta=` | no direction is supplied on the user's behalf (J1, J2, J3) | roles table |
| `depth` undeclared | `NoRotation()` | nothing declared | roles table |
| `gamma`, `B` undeclared | error | no neutral value exists | message |
| electron normalisation | forward: `amplitude = 1 cm^-2` at `pivot = 1e4`, with `amplitude` the density `dN/dgamma` at the pivot. For a family with a cut-off (`AgedPowerLaw`) `amplitude` is the coefficient of the uncut power law at the pivot, which stays defined when the cut-off lies below the pivot. Fit: free linear amplitude. Factors outside an unknown block are probability laws, and the column lives in the block (Sec. 4.5) | PAPER eq: index mixture and the caption of tab: classical parameters. The table's stored values follow the study's coefficient convention (`ST/lib/ensembles.py:13`) and are reached by `paper_coefficient` (Sec. 3.5). J1 Sec. 5.1: a unit column on an automatic range changes values by factors 1.7 to 3.1 with the channel list | unit line, fit summary |
| data unit | none. `StokesData` without `unit=` raises `UnitError`. `unit="cgs"` names the internal unit | Jy/sr data read as cgs change columns and amplitudes by 1e23 without changing `chi2` (r1-30) | fit header, next to `fit.column` |
| quadrature of specified distributions | each distribution's own `rule()`, with a declared range and node count and a `nodes=` override. Uniform field: 16 Gauss-Legendre nodes. Maxwellian: 96 nodes on `[0, 8 rms]`. Normal and LogNormal: 32 nodes in standard units or `ln B` over 6 widths, intersected with the support. Inclination: 12 nodes, Gauss-Legendre in `eta` when the support lies inside `-1 < eta < 1`, and in `theta = arccos(eta)` when it reaches `eta = -1` or `eta = 1` (`K_iso` depends on `sqrt(1 - eta^2)`). With 12 half-range nodes for an isotropic `eta`, the error of I at `gamma = 1e4` is 1.6e-4 in `eta` and 1.6e-6 in `theta`. For `U[-0.5, 0.5]` it is 5e-16 (prototype beaming). A user family with a narrow density takes a log-spaced or peak-centred rule, or is refused when its width is below the node spacing. A `Free` width takes the count of its narrowest admissible bound. Nodes are doubled while the `quadrature` term exceeds 1e-8 of I, at most three times. The half range serves a block only under the joint-reversal precondition of Sec. 4.3, checks (a) to (d) | `ST/lib/config.py:56-69, 129-131` (16 nodes for the uniform field, 96 for the Maxwellian). 16 nodes miss a log-normal of width 0.05 on (0.5, 50) uG by 2.4 per cent (r2-17). V is odd (`_harmonic_taylor.py:335-341`) | `basis.record`, ledger |
| quadrature of other averaged variables (the primary variable when averaged, the pitch cosine of a kernel without `Beaming`, user variables) | the distribution's `rule(n=None)` default, printed. An averaged electron family takes its support from `lo` and `hi`, or from the kernel's `EnergyWindow` used as a quadrature support. Otherwise `RoleError` | r2-18 | roles table, `basis.record` |
| quadrature of resolved densities | 24 nodes per panel, panels split at cell edges, kinks and support edges, Gauss-Jacobi at a declared algebraic end point | `ST/lib/ensembles.py:77-101` | record |
| family moments in a fit | 32 nodes per cell. With a declared cut-off, 64 nodes in `v = ln(u_c - u)` in every cell under a mask | `ST/lib/ensembles.py:520-588` | record |
| accuracy check | direct integration of a fully specified population when its estimated cost is below 5 s. Otherwise the line reads "not measured" with the cost | CTX 04 B.7 | `accuracy` line |
| free-moment fit | whitened SVD in the reference weights, no column equilibration, singular values below 1e-13 of the largest dropped, rank by UPRE up to the numerical rank, degenerate pairs kept whole | PAPER:718, `ST/fits.py:80-229` | fit summary |
| non-negative cells | 17 point masses at Chebyshev-Lobatto points per cell, penalty `alpha |a|^2`, `alpha` from the UPRE of the same fit without positivity | PAPER:1146, `ST/fits_positive_ridge.py:82-132` | fit summary |
| family fit | full linear likelihood, linear unknowns profiled, bounded least squares, starts supplied by the family (3 or 4), 4 around `Free(start)` for a user family | `ST/baseline_models.py:89-121` | fit summary |
| goodness-of-fit flag | probability of a larger `chi2` below 0.001 | PAPER:1146 | fit summary |
| disk cache | kernel tables (stages S0, S1) of fixed specified statistics under the user cache directory (`SYNCEM_CACHE`), at most 2 GB, least recently used entries removed first. Tables for trial values of a `Free` parameter stay in memory. Bases in memory | hash of the evaluated tabulation of each specified statistic, kernel source hash, version (Sec. 5.6, 8.5) | plan prints present and absent stages and the build estimate |

### 2.4 Units

* A dimensional scalar is a string (`"5 uG"`, `"19 Myr"`, `"60 deg"`). A dimensional array is a pair `(array, "uG")`. A constructor with one dimensional kind takes `unit=`. An astropy `Quantity` is accepted through an `isinstance` check.
* Bare numbers are accepted for quantities with one conventional unit: Lorentz factors, cosines, Faraday depths in rad m^-2, columns and amplitudes in cm^-2. A bare field strength, frequency, angle, age or data array raises `UnitError` with the spelling to use.
* Conversion happens once, in constructors, to Hz, G, rad m^-2, rad and cm^-2. Kernels see floats. A value outside the usual range of its variable adds one warning line to the plan.
* `Cells` and every built-in `Distribution` record the unit they were given in the static field `given_unit`. `unit=None` means bare numbers. At placement in `Population(**factors)` or `Lattice(**axes)`, a bare value on a variable with `bare=False` raises `UnitError`, and a given unit is checked for dimension against `Variable.unit`. On a cosine an angle unit is accepted on a point value only, read as its cosine. A range of angles on a cosine is `se.UniformAngle(lo, hi, unit=)`, uniform in the angle with its density on the cosine.
* Inside a conditional each parent arrives as a parent value, `se.Quantity(value, unit)`, a small pytree whose unit is static: canonical, or the unit named in `se.Conditional(fn, units={"B": "uG"})`. Dimensional slots accept a parent value or arithmetic on one, and the unit propagates. A plain float in a dimensional slot is refused. The plan prints the parent units (J1 Q5).
* Output units: `spectrum.to("Jy/sr" | "MJy/sr" | "K_RJ" | "K_CMB")` divides each channel by the band integral of the unit's reference spectrum over the same channel quadrature. `StokesData(unit=...)` is required and applies the same row scaling to the design, so a fit never asks for the internal unit.

### 2.5 Error messages

Value checks run in eager code on Python floats. The unit and range checks of a conditional run once, eagerly, on its tabulation (Sec. 5.6). Traced evaluations reuse the checked structure and run no checks. Compiled cores return result codes that are checked after the call (P4 Sec. 8.5). Each message names the remedy and the call that takes it.

```text
UnitError: B=5.0 has no unit. Write B="5 uG", or se.Uniform(4, 6, unit="uG") for a range.
UnitError: B=se.Cells.linear(2, 8, 3) has no unit. Write se.Cells.linear("2 uG", "8 uG", 3), or pass unit="uG".
UnitError: eta=se.Uniform(10, 50, unit="deg") is a range of angles on a cosine. Write se.UniformAngle(10, 50, unit="deg")
    (uniform in the angle) or se.Uniform(0.64, 0.98) (uniform in the cosine).
UnitError: phi=GaussianScreen(...) is a distribution of Faraday depth. phi is the projected-field azimuth. Write depth=.
UnitError: StokesData has no unit. Pass unit="Jy/sr", "MJy/sr", "K_RJ", "K_CMB", or "cgs" for erg s^-1 cm^-2 Hz^-1 sr^-1.
UnitError: the conditional for gamma returned loss_field=5e-6, a plain float. Use the parent value B, or a string such as "5 uG".
RoleError: gamma is conditional on B, and the lattice does not resolve B.
    Add B=se.Cells.single("5 uG", scale="1 uG") to the lattice, or pass average=("B",) to drop the dependence (recorded as an assumption).
RoleError: mu has Cells on the lattice, the continuum kernel declares Beaming (only mu = eta contributes), and eta is fixed at 0.5.
    The integrand is the density on the diagonal mu = eta, which is not a moment of the population.
    Give eta a distribution with a density, write mu=se.Legendre(8) on the lattice, or remove mu from the lattice and declare a pitch density.
RoleError: eta=lambda phi: ... is conditional on phi, and the lattice resolves phi by AzimuthHarmonic, which holds only <exp(2i phi) ...>.
    Write phi=se.AzimuthFourier(m_max) on the lattice, with m_max the Fourier degree in phi of the averaged kernel (rule R4a (b)), average phi with
    nodes (phi=se.AzimuthDensity(...) or se.Tabulated), put eta=se.Legendre(8) on the lattice with a specified phi law,
    or pass average=("phi",) to drop the dependence (rule R4a). Without lattice= the plan takes AzimuthFourier and prints it.
RoleError: B is unknown (FreeMoments("gamma", "B")) and no lattice is given, so B has no range. Pass
    se.FreeMoments("gamma", "B", support={"B": ("2 uG", "8 uG")}), or lattice=se.Lattice(..., B=se.Cells.linear("2 uG", "8 uG", 3)).
RoleError: depth=se.CumulantScreen(...) gives only a characteristic function, and phi depends on depth.
    Use se.GaussianScreen or se.EmpiricalScreen, which give nodes, or put depth on the lattice with se.Cells.
RoleError: the default lattice needs a range for a, and grain_emission declares no EnergyWindow.
    Pass lattice=se.Lattice(a=se.Cells.log(...)), or declare a distribution of a with bounded support.
RegimeError: cell 0 (gamma 3 to 8) needs harmonics up to 4e8. The harmonic kernel stops at 16368, and the continuum kernel
    is used for gamma >= 300. Start the cells at 300, or pass kernel=se.ContinuumKernel(gamma_min=3) to accept it there.
SpecifiedMismatch: the basis contains B ~ Uniform(4, 6) uG. The population declares B ~ Maxwellian(rms = 5 uG).
    Build a basis for this population, or pass allow_mismatch=True to se.predict (the difference is measured on the host and recorded).
NothingToPredict: gamma=FreeMoments() has no values. Fit first, or predict from a specified population.
DataError: the data hold Q and U, and phi is undeclared (uniform, Q = U = 0). Declare phi=se.AzimuthFactor(c),
    phi=se.FreeMoments() (one shared factor), or the block se.FreeMoments("gamma", "phi") (joint moments).
DepthExpansionError: depth cells of half-width 5 rad m^-2 give tau_max h = 140 at 0.08 GHz, the lowest support frequency, and the depth
    bound (tau_max h)^3/3! = 4.6e5 > depth_tol = 1e-3. Use cells with h <= 0.0065 rad m^-2, raise the depth order, restrict the channels,
    or predict a Ray by direct integration.
SizeError: the lattice has 2 737 152 rows and 1 140 480 basis columns (21.9 MB of moments per sightline), above max_columns=20 000.
    Pass max_columns=1_200_000 to accept, or use fewer cells or a lower order.
```

## 3. Object model

Static fields are marked `S` (metadata, part of the jit key, never arrays). Array leaves are marked `A` (traced, batchable, differentiable when inexact).

### 3.1 Layers and dependencies

```text
units              unit conversion, Quantity
values             Observable, Spectrum, Caveat, the fixed caveat sentences with their PAPER lines, origin labels, the print blocks shared by Spectrum, Plan and Fit, the error classes. Imports only units
base               Variable, Kernel, capabilities, PhaseLaw, Channels
distributions      Distribution, electron families, screens, Conditional, Free
lattice            Cells, CellTable, Legendre, AzimuthHarmonic, AzimuthFourier, PointMasses, FieldScaling, Lattice, MomentIndex, Moments
                   imports no kernel and no population
records            ErrorTerm, Ledger, AssumptionRecord, SpecifiedRecord, BuildRecord
engine           = base + distributions + lattice + records   AveragedKernel, average, expand, direct, providers, ColumnMap, cache ─► SpectralBasis, BasisSum
population       = distributions + lattice + records          Population, markers, Samples, Ray, pair, to_distributions    ─► Moments
planning         = population + lattice + base kernel (cheap capabilities) + engine column map   ─► PlanRecord, Plan, MomentMap, default_lattice
results            Comparison, the Fit and Comparison prints
fitting          = engine + planning + results               StokesData, policies, solvers                                ─► Problem, Fit
front              predict, fit, plan, spectral_basis, problem, compare, convergence
existing           syncem.model (unchanged). The engine calls into it. Nothing in it imports the new layers
```

Every layer may import `units` and `values`. Annotations in `values` that name later types (`Moments`, `Ledger`, `PlanRecord`, `DefaultRecord`, `Channels`) are strings, and imports for type checkers sit under `if TYPE_CHECKING:`, so nothing is imported at run time. Module-level imports follow this order: a layer imports only layers listed above it. A convenience method may delegate to a later layer through a function-local import. Each such method is listed with the free function it calls:

| Method | Calls |
|---|---|
| `basis.moments(population)` | `population.pair` |
| `basis.predict(...)` | `front.predict(..., basis=basis)` |
| `basis.direct(population, within=)` | `population.to_distributions`, then `engine.direct`, which is `engine.expand` with an empty lattice. Under `Beaming`, `mu` samples with no density for `eta` are paired at the basis's `L_mu` (Sec. 4.2) |
| `basis.fit(...)` | `fitting.fit` |
| `Lattice.default(...)` | `planning.default_lattice` |
| `Lattice.single(about=...)` | `population.single_lattice` |
| `SpectralBasis.build(check=True)`, `basis.at(...)` with the check on | `population.pair` and `engine.direct` on the three reference populations (`reference_cases`, Sec. 7.3) |

An import-graph test walks every import statement in the AST, inside functions included. A module-level import must follow the order, with `if TYPE_CHECKING:` blocks exempt. A function-local import of a later layer must sit in a method listed in this table (Sec. 9.7).

The engine receives distributions of the averaged variables, never a `Population`. The planning layer may call the cheap declared capabilities of a base `Kernel` (`EnergyWindow`, `regime`). It builds no engine stage.

### 3.2 Base model

```python
class Variable(eqx.Module):                      # all fields S
    name: str                                    # keyword in Population and Lattice, argument name in conditionals, part of moment labels
    meaning: str                                 # one line, printed by se.plan
    kind: Literal["positive", "real", "cosine", "angle"]      # admissible representations, Sec. 4.2
    unit: str                                    # canonical unit
    bare: bool                                   # a bare number is read in the canonical unit
    usual: tuple[float, float] | None            # usual range, checked at the boundary
    default: str | None                          # name of the neutral declaration when undeclared ("Isotropic"), resolved by
                                                 #   planning, because distributions are a later layer. None: required
    display_unit: str                            # unit used in messages and remedies ("uG" for B)

class Kernel(Protocol):
    name: str
    variables: tuple[Variable, ...]              # S. Every engine contract refers to variables by these names
    primary: str | None                          # S, the variable resolved by default (R2), gamma for the synchrotron kernels. None: only unknown blocks
    stokes: tuple[str, ...]                      # S, subset of ("I", "P", "V"), P = Q + iU
    azimuth: str | None                          # S, variable that enters only as exp(i s_X phi), s_I = s_V = 0, s_P = 2
    depth: str | None                            # S, variable of the propagation phase exp(i tau(nu) depth), spin-2 block only
    traceable: bool                              # S, False: host code, columns through the cell table
    lines: bool                                  # S, Dirac lines: channels must be smooth and of finite width
    capabilities: tuple[Capability, ...]         # S, Sec. 8.1
    def channel(self, channels, point, *, phase, per_node=False) -> Modes     # point: {name: scalar}. I, V (n_ch,), P (n_w, n_ch) complex,
                                                 #   phase inside. phase=False: the phase-free kernel, without exp(i tau depth) (Sec. 4.3).
                                                 #   per_node=True: weighted values at the channel quadrature nodes, a trailing axis n_nu
    def spectral(self, nu, point) -> Modes | None             # per Hz, for kernels without lines. The engine integrates over the channel
    def regime(self, box, channels) -> Regime                 # ok, or refused with the reason, for one cell box
    def physical_error(self, channels) -> ErrorTerm           # unbounded unless declared
    def describe(self) -> KernelRecord                        # S, hashable, strict JSON

# capabilities, each with a precondition that the plan checks (Sec. 8.1)
se.Similarity(variables=("gamma",))   se.FieldScalingRelation(variable="B")     se.Parity(even=("I", "P"), odd=("V",))
se.Beaming(variable="mu", partner="eta")         se.EnergyWindow(window)        # Beaming: the admissible roles of mu follow the role of eta
se.ReferenceDensity(variable, density)           se.Lines()      se.AnalyticTaylor()      se.HostOnly()
# engine hooks: Beaming carries factor 2 and evaluation at mu = eta. A kernel with an azimuth carries a spin-2 factor hook:
#   P times the azimuth factor c at each averaged node, I and V unchanged (R5)

se.ContinuumKernel(*, gamma_min=300.0, x_min=None)   # PAPER eq: directional continuum. I, P. Similarity, FieldScalingRelation, Beaming,
                                                 # Parity, EnergyWindow. x_min=None: set from the channels and the lattice (Sec. 8.3)
se.HarmonicKernel(*, m_max=None)                 # PAPER eq: channel kernel. I, P, V. Lines, Parity, AnalyticTaylor        [cells: 0.9]
se.HighOrderHarmonicKernel()                     # host code of PKG/syncem/model/high_order_harmonic.py. HostOnly          [0.9]
se.HybridKernel(*kernels, switch=None, assign=None, switch_tol=1e-3)   # kernel per product cell, Sec. 8.4. assign: callable(box)
                                                 # -> index. switch={"gamma": 300} fills assign by a threshold   [harmonic entries: 0.9]
se.kernel(*, variables, stokes, azimuth=None, depth=None, capabilities=())   # decorator: fn(nu, **point) -> {"I": ...} becomes a
                                                 # traceable Kernel. Cells on any of its variables through the jet provider
se.FaradayRotation(*, screen=None)               # PhaseLaw with tau(nu) = 2 (c/nu)^2 in m^2
se.PhaseLaw(tau, *, screen=None)                 # any law exp(i tau(nu) depth). screen: independent foreground factor [extension]
se.Modes                                         # NamedTuple I, V, P of PKG/syncem/model/_kernel_helpers.py:23-36

class Channels(eqx.Module):                      # subclass of PKG/syncem/model/channels.py:106, unit-area normalisation
    centres_hz, widths_hz, support, nodes, weights            # A
    family, normalisation, n_nu, unit                         # S
    Channels.log(lo, hi, n, *, unit, fractional_width=None, half_width=None, n_nu=64)   # f: h_j = f nu_j / sigma_rho.
    Channels.at(frequencies, *, unit, fractional_width=None, half_width=None, n_nu=64)  #   half_width=x: h_j = x nu_j
    Channels.from_bandpass(tables, *, unit, n_nu=256)         # one (frequencies, response) table per channel
    channels[rows] -> Channels
```

The engine applies `Beaming`, azimuth factors and the phase through these hooks and the two `channel` options, never through wrapper kernels. The depth Taylor coefficients of an averaged kernel with a depth parent are taken from `phase=False` (Sec. 4.3). The closed-form depth weights act inside the channel integral, so the engine takes `per_node=True` and applies them at each channel node. The prototypes needed six wrapper kernels for these cases. They also turned J channels of T nodes into J T one-node channels, which multiplied the jet's channel count by T (prototype R5, depth, beaming).

The six variables of the synchrotron kernels are the names of `VARS` (`PKG/syncem/model/_factor.py:44`).

| name | meaning | kind | canonical unit | bare number | undeclared |
|---|---|---|---|---|---|
| `gamma` | electron Lorentz factor | positive | none | yes | error |
| `B` | field strength | positive | G | no | error |
| `mu` | pitch cosine | cosine | none | yes | `Isotropic()` |
| `eta` | field-inclination cosine | cosine | none | yes. An angle string is a point value read as its cosine | `Isotropic()` |
| `phi` | projected-field azimuth | angle | rad | no | `UniformAzimuth()` |
| `depth` | Faraday depth from emitter to observer | real | rad m^-2 | yes | `NoRotation()` |

### 3.3 Lattice

```python
class Cells(eqx.Module):                         # a partition of one variable's range. Each cell has its own reference point and order
    chart: Literal["log", "affine"]              # S. u = ln(x / x_b) / h_b, or u = (x - x_b) / h_b
    edges: Array | None                          # A (n + 1,), canonical unit. None: cells without edges (reference points only)
    reference: Array                             # A (n,), x_b. Default: the centre in the chart
    half_width: Array                            # A (n,), h_b in the chart. The scale of a cell without edges
    order: tuple[int, ...]                       # S (n,), local polynomial order N_b
    given_unit: str | None                       # S. Unit given at construction. None: bare numbers, checked at placement (Sec. 2.4)
    Cells.log(lo, hi, n, *, order=2, unit=None, reference=None)
    Cells.linear(lo, hi, n, *, order=2, unit=None, reference=None)
    Cells.from_edges(edges, *, chart="log", order=2, unit=None, reference=None)
    Cells.single(reference, scale, *, order=2, chart="affine", unit=None, support=None)
    Cells.at_points(references, half_widths, *, order=2, chart="affine", unit=None)   # n cells without edges at arbitrary reference
                                                 #   points, for example the probe and reference cells of a check (prototype fieldcells)
    def with_order(self, N) -> Cells
    def shift(self, fraction) -> Cells           # reference points moved by a fraction of the half-width (PAPER:741)
    def refine(self, k) -> Cells                 # every cell split in k

class CellTable(eqx.Module):                     # the stored product cells, one row per product cell
    variables: tuple[str, ...]                   # S, the celled variables
    lo, hi, reference, half_width: Array         # A (n_cells, n_celled), canonical units. For a cell without edges, the ends of the variable's
                                                 #   kind: [0, inf) for a positive variable in either chart (prototype R3a), [-1, 1] for a cosine,
                                                 #   (-inf, inf) for a real variable
    chart: tuple[tuple[str, ...], ...]           # S, per row and variable: "log" or "affine"
    order: tuple[tuple[int, ...], ...]           # S, per row and variable: N_v
    tensor: tuple[int, ...] | None               # S, cell counts per variable when the table is a tensor product. None otherwise

class Legendre(eqx.Module):         degree: int                     # S. Legendre degrees on a cosine: P_l with P_l(1) = 1, l = 0..degree
class AzimuthHarmonic(eqx.Module):  pass                            # the kernel's azimuth variable: complex spin-2 moments, no truncation
class AzimuthFourier(eqx.Module):   m_max: int                      # S. Rows <exp(i k phi) ..>, k = 0 .. m_max + 2, conjugates folded, one set shared by the spin-0 columns (k = |m|)
                                                                    #   and the spin-2 columns (k = |2 + m|), |m| <= m_max, for laws conditional on phi. Rule
                                                                    #   R4a (b) routes to it, with the truncation term azimuth_truncation (Sec. 4.2, 7.2)
class PointMasses(eqx.Module):      cells: Cells, points: int = 17  # S points. Kernel values at Chebyshev-Lobatto points of each cell [extension, 0.9]
class FieldScaling(eqx.Module):                  # push-forward along the kernel's field-scaling relation      [0.9]
    reference: Array                             # A, the field B_0 at which the basis is built
    inclination: bool = False                    # S. True: the scaling variable is B sqrt(1 - eta^2). The axis resolves B and eta, and the
                                                 # basis is evaluated at sin(theta) = 1. Requires an isotropic pitch law or the folded density (Sec. 4.2)

class Lattice(eqx.Module):
    cells: CellTable                             # product cells of the Cells axes
    representations: Mapping[str, Representation]    # keys S. Every resolved variable. A variable without one is averaged
    order: int | None                            # S. Cap on the total Taylor degree in a product cell over the variables other than depth.
                                                 #    Depth powers are outside it (Sec. 4.3). None: the largest axis order. An order above the
                                                 #    sum of the axis orders of the non-depth celled variables raises, since no row reaches it
                                                 #    (prototype partition: Cells of order 2 under a total of 4 silently gave 3 rows per cell)
    weights: Mapping[str, Distribution] | Array | None        # reference densities per axis, or the weights (n_cells,). None: resolved at
                                                 #    construction without a kernel (Sec. 4.5)
    index: MomentIndex                           # S, derived. Unpruned: one row per product of the representations (Sec. 4.4)
    key: str | None                              # S. Content hash of cell table, orders, index and the resolved weights, computed eagerly at
                                                 #    construction. None for a lattice built inside a transformation
    axes: Mapping[str, Representation]           # derived view: one Cells per celled variable and the other representations.
                                                 #    Exists for a tensor table only. Otherwise access raises and names `cells`
    Lattice(*, order=None, weights=None, depth_in_total=False, **axes)     # tensor product of the axes. depth_in_total=True:
                                                 #   depth inside the total degree, the index set of Lattice.single
    Lattice.default(population, channels, *, kernel=None, phase=None, resolve=(), average=(), max_columns=20_000) -> Lattice   # Sec. 4.6
    Lattice.single(*, about, order=2, degree=8) -> Lattice                 # one reference point at weighted means (PAPER Sec. 4.2)
    Lattice.from_cells(table, *, representations=None, order=None, weights=None)       # any table of boxes: overlap checked, no coverage
    def with_key(self, key) -> Lattice           # attaches a key computed before a transformation (Sec. 8.6)
    def with_order(self, N) / replace(self, **axes) / select(self, **cells) / shift(self, **fractions) -> Lattice
    def refine(self, *, where=None, **factors) -> Lattice       # where: a selector. Only the selected product cells are split
    n: int                                       # number of lattice rows (real moments before the column map)
    n_cells: int
    reference_weights: Array                     # (n_cells,), w
    def labels(self) -> tuple[str, ...]
    def column(self, moments) -> Observable      # order-0 sum, N_src. With a FieldScaling axis: the effective column at B_0 (Sec. 4.2)
    def raw(self, moments) -> Observable         # w a, the unscaled integrals in cm^-2
    def normalised(self, moments) -> tuple[Observable, Observable]      # (N_src, M) with M_000 = 1, the convention of PAPER Sec. 4.2
    def recentre(self, moments, to) / coarsen(self, moments, to) -> Moments     # linear maps without approximation, one chart
    def realisable(self, moments) -> Observable  # Hausdorff conditions per cell (ST/fits.py:232-245)
    def conditional(self, moments, variable) -> Observable    # ratios such as the conditional Legendre coefficients per cell
    # every method that takes Moments refuses another lattice key, returns the origin of the moments, attaches the PAPER:718
    # caveat to a minimum-norm input and gives NaN on masked rows. unconstrained is None here: fit.column and fit.observable(c)
    # give the numerical value

class Representation(Protocol):                  # implemented by Cells, Legendre, AzimuthHarmonic, AzimuthFourier, PointMasses, FieldScaling
    def functions(self, x) -> Array              # psi(x), used when a measure is paired with the lattice
    def functionals(self) -> Functionals         # how basis columns are taken from the kernel: derivatives, projections or values
    def signature(self) -> Hashable              # cells with equal signature share one compiled routine

class MomentIndex:                               # frozen dataclass, S. Arithmetic: one row per lattice row, never stored row by row
    counts, monomials, blocks                    # rows per product cell and order pattern, lower sets of exponents, block layout and
                                                 #   Legendre pairs. key: hash of this definition. rows: lazy view of Row(block, cell,
                                                 #   intervals, powers, degrees, pair), Sec. 4.4
    def position(self, **selector) -> int        # integers per variable on a tensor table. Interval or predicate selectors on any table
    def select(self, **cells) -> tuple[int, ...] # the same selectors
    def summary(self) -> str                     # (named summary: the field table holds the CellTable)

class Moments(eqx.Module):                       # an element of the dual of a lattice
    values: Array | SparseRows                   # A (..., n) real, the scaled moments a of PAPER eq: energy cell response, in cm^-2, in
                                                 #   lattice rows (Sec. 4.4). SparseRows: (row, value) pairs per leading index (R8)
    index: MomentIndex                           # S
    key: str | None                              # S, key of the lattice the values refer to. Compared at predict
    masked: tuple[int, ...]                      # S, rows that no column of the fitting basis uses (ColumnMap). Zero by the minimum-norm rule
    origin: Literal["specified", "family", "minimum_norm", "penalised", "nonnegative_penalised"]      # S (Sec. 6.3)
    retained: str | None                         # S, key of the RetainedCombinations of a minimum-norm or penalised vector
    component: str | None                        # S, component key when the moments are one part of a BasisSum (Sec. 3.4)
    Moments(lattice, values, *, origin="specified")
    def table(self) -> str
```

The lattice stores its product cells in `cells`, one row per product cell with the interval, reference point, half-width, chart and order of each celled variable. Tensor constructors are public in the first release, and `axes` is their view. `select` and `refine` produce new tables. Unequal orders, local refinement and subsets are rows of the table (B3, B4). `refine(where=selector, ...)` splits only the selected product cells and is valid by construction. `Lattice.from_cells` accepts any table of axis-aligned boxes: it rejects overlapping boxes by an interval test on `lo` and `hi` per variable (shared faces allowed), requires no coverage, and sets `tensor=None`. Uncovered emission goes to `outside_lattice`, as for `select`. The kernel per product cell comes from the regime rule (`kernel=None`) or from `HybridKernel(assign=)` (Sec. 8.4). It lives on the kernel side and is recorded in `basis.record`, so `CellTable` and `Lattice.key` stay kernel-free and moments compare across kernels. `Moments` holds static keys in place of the lattice and the column map, so a vmapped function that returns `Moments` does not broadcast cell arrays (J3 Q15). `origin` and `masked` travel with the values (Sec. 6.3).

`Lattice.key` is computed eagerly when the lattice is built outside a transformation, hashes the cell-table arrays, the definition of the index (not its rows) and the resolved weights (Sec. 4.5), and `Moments` copies it. A lattice of 2.74e6 rows (G15) therefore costs no per-row object. `jit` and `vmap` over moment values therefore keep a concrete key. A lattice built inside a transformation has `key = None`. `predict` and `fit` refuse to compare a `None` key with a key unless the lattice carries a key from `with_key` (Sec. 8.6). `allow_mismatch` does not bypass this check.

### 3.4 Spectral basis

```python
class ColumnMap:                                 # frozen dataclass, S. Exact selection and summation from lattice rows to basis columns
    rows: Sequence[tuple[int, ...]]              # lattice rows summed into each column (Beaming pairs), a lazy view. Unused rows: no column
    uses: Mapping[str, tuple[int, ...]]          # per Stokes block, the columns where its spectra can be nonzero (parity masks, depth s = 0 where it holds)
    spin0_from: tuple[int, ...]                  # per column, the stored spin-0 spectrum it repeats (depth cells share one, Sec. 4.3)
    reasons: tuple[str, ...]                     # per dropped or merged group: "parity", "beaming", "symmetry", "support", "unused"
    key: str                                     # lattice key + kernel capabilities + Stokes request + pruning record
    n: int                                       # number of basis columns
    def apply(self, values) -> Array             # values in lattice rows -> column space
    def matrix(self) -> Array                    # S as a sparse (n, n_rows) 0/1 matrix, apply(a) = S a. A free fit uses the design D S
    def labels(self, index) -> tuple[str, ...]   # a merged column is labelled as the sum of its rows

class SpectralBasis(eqx.Module):
    lattice: Lattice
    channels: Channels
    columns: ColumnMap                           # S
    spectra0: Array                              # A (n_ch, n_s0, n0) real. I~ (and V~) per stored spin-0 spectrum, per unit unscaled moment
    spectra2: Array | None                       # A (n_ch, n2) complex. P~ per spin-2 column, phase inside. Without azimuth factor
                                                 #   unless the factor depends on an averaged or Legendre-resolved variable (R5)
    specified: tuple[SpecifiedRecord, ...]       # variable, family type, parents and fingerprint S. Parameter values A
    record: BuildRecord                          # S. Kernel per cell, phase law, provider and route per stage, quadrature, versions, times
    ledger: Ledger                               # kernel-side terms measured at build
    key: str                                     # S
    SpectralBasis.build(channels, lattice, *, specified=None, symmetries=(), kernel=None, phase=None, stokes=None, check=True, max_columns=20_000, cache=True)
    SpectralBasis.load(path)
    SpectralBasis.from_single_reference(old_basis)            # a syncem.model basis as a one-cell lattice
    def at(self, channels) -> SpectralBasis      # same lattice and statistics on other channels. Range and reference_cases rechecked (Sec. 6.5)
    def select(self, **cells) -> SpectralBasis   # view. The omitted cells are written to the ledger
    def with_order(self, N) -> SpectralBasis
    def design(self, data=None, *, stokes=None) -> Array      # (n_ch, n_stokes, columns.n) real with w applied. Whitened rows of data when given
    def moments(self, population, *, allow_mismatch=False) -> Moments          # the pairing of a measure with the lattice, lattice rows
    def predict(self, population_or_moments, *, check=True, allow_mismatch=False) -> Spectrum    # applies the column map
    def direct(self, population, *, within="support") -> Spectrum   # empty lattice, capabilities applied. within="cells": cell boxes only
    def fit(self, population, data, *, allow_mismatch=False, **policies) -> Fit    # argument order and policies of se.fit
    def save(self, path) -> None
    reference_cases: tuple[ReferenceCase, ...]   # representation errors measured on reference populations, with their channel span (Sec. 7.2)
    remainder: Array | None                      # (n_ch, n_stokes, n_cells), sampled sup of the order N + 1 term. Ledger only

class BasisSum(eqx.Module):                      # one basis per distinct set of specified statistics of a sum (R7)
    parts: tuple[SpectralBasis, ...]
    components: tuple[str, ...]                  # S, component key of each part
    def design(self, data=None, *, stokes=None) -> Array      # horizontal concatenation of the parts' designs, block layout in record
    def predict(self, population_or_moments) -> Spectrum      # sum of the parts. Moments: a tuple, one per component key
    def at(self, channels) -> BasisSum
```

The three operations of the pairing are free functions. `engine.average(kernel, distributions)` returns an `AveragedKernel` (Sec. 8.1). `engine.expand(averaged, channels, lattice, columns)` takes columns from it. `population.pair(measure, lattice)` gives moments in lattice rows. `SpectralBasis.build` is `expand(average(...))`. `engine.direct` is `expand` with an empty lattice, so it applies the same capabilities as the basis and no truncation, except the `Legendre(L)` pairing of `mu` samples under `Beaming` when `eta` has no density (Sec. 4.2). It evaluates each sample or `Ray` step at its own `eta` on the full cosine range. `se.spectral_basis` returns a `BasisSum` only when the components of a sum differ in an averaged statistic. Free blocks of the same variable in different parts are separate unknowns. Free blocks in components that share one part are tied: only their sum is constrained, and the plan merges them and says so. The column map is computed from static information only: the kernel's capabilities, the Stokes request, `is_even` of the averaged laws, the parities declared by conditionals and azimuth factors, the parent graph of every conditional, `Population.symmetries` and the depth variable (Sec. 4.3). An undeclared parity counts as not even. The plan computes it before any build.

### 3.5 Distributions, markers and populations

```python
class Distribution(Protocol):                    # a distribution of one variable, or of a joint group
    variables: tuple[str, ...]                   # S
    support: tuple
    given_unit: str | None                       # S. Unit given at construction. None: bare numbers (Sec. 2.4)
    def rule(self, n=None, *, parents=None) -> tuple[Array, Array]      # nodes and probability weights on the law's own range and default
                                                 #   count (Sec. 2.3), traceable in parameters and parent values. Absent if characteristic-only
    def density(self, x, *, parents=None) -> Array                      # absent for a characteristic-only law
    def record(self) -> dict                     # strict JSON with a fingerprint
    # optional: characteristic(tau), circular(s), legendre(L), is_even, kinks, cutoff, and fixed_nodes (S): the nodes of rule()
    #           do not move with the parents (Sec. 8.3)

# any variable
se.Delta(value)                                  # a number or a unit string in a slot is shorthand for it
se.Uniform(lo, hi, *, unit=None)      se.Normal(mean, sigma, *, unit=None)      se.LogNormal(median, width, *, unit=None)
# every built-in law takes nodes= to override the count of its rule
se.Tabulated(x, density, *, unit=None)
se.Maxwellian(rms, *, cut=8.0, nodes=96)         # field strength of an isotropic Gaussian random field (PAPER:1057), rule on [0, cut rms]
se.Isotropic()                                   # uniform cosine on [-1, 1]
se.UniformAngle(lo, hi, *, unit)                 # uniform in the angle, density on the cosine. The only range of angles accepted on a cosine
# azimuth
se.UniformAzimuth()                              # <exp(2i phi)> = 0. rule(): equally spaced nodes, n nodes, exact for trigonometric polynomials of degree below n
se.AzimuthFactor(c)                              # specifies <exp(2i phi) | parents> = c. c is a constant, se.Free, or a Conditional of
                                                 #   resolved or averaged variables (routes of R5)
se.AzimuthDensity(amplitude, angle, *, unit=None)             # [1 + A cos 2(phi - angle)] / (2 pi)
# Faraday depth (PKG/syncem/model/phase.py, screens.py)
se.NoRotation()       se.GaussianScreen(mean, sigma)       se.LaplaceScreen(mean, sigma)       se.GammaScreen(mean, sigma, shape)
se.EmpiricalScreen(depths, weights=None)         se.CumulantScreen(cumulants)      # characteristic function only: averaged role, never a parent
# electron families: distributions of gamma that carry a normalisation, column= (cm^-2 on [lo, hi]) or amplitude= (dN/dgamma at
#   pivot, and for a family with a cut-off the coefficient K of the uncut power law at the pivot)
se.PowerLaw(index, *, lo=None, hi=None, column=None, amplitude=None, pivot=1e4)
se.BrokenPowerLaw(index1, index2, gamma_break, *, smoothness=2.0, lo=None, hi=None, column=None, amplitude=None, pivot=1e4)
se.PowerLawMixture.moment_form(order, index0, *, pivot=1e4, lo=None, hi=None)            # linear in A m_n, PAPER eq: index moment expansion
se.PowerLawMixture.cumulant_form(k1, k2=0.0, *, pivot=1e4, lo=None, hi=None, amplitude=None)        # PAPER eq: index cumulant expansion
se.AgedPowerLaw(index, *, gamma_cut=None, age=None, loss_field=None, radiation=None, pitch=None, lo=None, column=None, amplitude=None, pivot=1e4)
                                                 # PAPER eq: reference densities, E4. pitch=None: JP. pitch=mu: KP. N = K (gamma/pivot)^-p
                                                 # (1 - gamma/gamma_c)^(p-2) on [lo, gamma_c), K = amplitude. Note printed if gamma_c <= pivot
family.paper_coefficient(params) -> Observable   # every electron family: the coefficient A of PAPER tab: classical parameters
Family.from_paper_coefficient(A, **params)       #   (the study's convention, ST/lib/ensembles.py:13) and its inverse. sigma by the Jacobian
se.family(*, variable, support, unit=None, cutoff=None, linear=())       # decorator: an unnormalised density fn(x, **params) becomes a family
se.Conditional(fn, *, params=None, units=None, parity=None, name=None)   # fn(<parents>, **params) -> Distribution
                                                 # params: {name: Free or value}, the only place for a Free, named "<variable>.<name>". parity:
                                                 #   {cosine parent: "even" | "odd"} (Sec. 4.3). A bare function: canonical units, no params, no parity
se.Quantity(value, unit)                         # parent value inside a conditional: value A, unit S. Arithmetic propagates the unit

class Free(eqx.Module):                          # stands where a number would. All fields S
    start: float | str | None = None, bounds: tuple | None = None, prior: Distribution | None = None, log: bool = False
    # start and bounds take unit strings for a dimensional parameter ("19 Myr")
    shared: bool = False                         # one value for all leading axes of the data                      [0.9]
    # prior is a declaration record with Python-number parameters. Evaluated priors are array leaves of the Params pytree
class FreeMoments(eqx.Module):                   # the distribution is unknown: its moments on the lattice are the unknowns
    FreeMoments(*variables, cells=None, support=None, prior=None)      # keyword value: gamma=FreeMoments(). Positional block:
                                                 #   FreeMoments("gamma", "phi"). support: {variable: (lo, hi)} without a lattice axis (Sec. 4.6)
class NonNegativeCells(eqx.Module):              # unknown and non-negative: point masses in each cell (PAPER:1146)
    NonNegativeCells(*variables, point_masses=17, cells=None, support=None)

class Population(eqx.Module):
    Population(*blocks, symmetries=(), **factors)
    factors: Mapping[str, Factor]                # keyword factors, independent of each other unless a Conditional links them
    blocks: tuple[Block, ...]                    # joint statements: FreeMoments, NonNegativeCells, Samples, Ray, Moments, a joint Distribution
    components: tuple[Population, ...]           # filled by a + b
    symmetries: tuple[str, ...]                  # S. "field_reversal", "pitch_reversal": statements that make moments zero (_factor.py:66-91)
    def replace(self, **factors) -> Population
    def at(self, params) -> Population           # free leaves filled with values
    def to_distributions(self, lattice=None)     # the averaged-variable distributions the engine receives (Sec. 3.1)
    def to_dict(self) -> dict                    # strict JSON. from_dict in [0.9]
    # a + b adds measures. c * a scales the column

se.Samples(*, weights=None, column=None, **arrays)            # weighted samples of several variables. Arrays (S,) or pairs (array, unit)
se.Ray(*, step, B_vector, thermal_density, electrons, electron_density, pitch=None)      # profiles along the ray, a sum of one
                                                 #   component per step (Sec. 5.1). Leading axes are sightlines, ray[i] selects them
```

Conversion of the amplitude to the paper's coefficient, with `gamma_0` the pivot 1e4 and `N(gamma)` the density. The values of PAPER tab: classical parameters do not change (Sec. 9.6):

| Family | `A_paper` |
|---|---|
| power law, power-law mixture | `gamma_0 A` |
| smoothly broken power law | `gamma_0 N(gamma_0) / [(gamma_0/gamma_br)^-p1 (1 + (gamma_0/gamma_br)^s)^(-(p2-p1)/s)]` |
| aged power law (JP) | `gamma_0 K`, with `K` the uncut coefficient (`amplitude=`). Finite for every `gamma_c` |

### 3.6 Plan and problem

```python
@dataclass(frozen=True)
class PlanRecord:                                # hashable, strict JSON, no arrays. Carried by Spectrum and Fit
    roles: Mapping[str, Role]                    # Role(variable, declared, role, representation, how, rule, default)
    kernel: KernelRecord
    phase: PhaseRecord
    specified: tuple[str, ...]                   # fingerprints of the averaged statistics the basis will contain
    lattice_key: str
    index: MomentIndex
    columns: ColumnMap
    routes: tuple[RouteRecord, ...]              # provider and shortcut per stage, with the precondition checked and its outcome
    kind: str                                    # fixed | linear | non-negative linear | separable | bilinear (two forms) | basis-parametric | nonlinear
    solver: SolverRecord                         # estimator and default policies of the kind
    defaults: tuple[DefaultRecord, ...]
    assumptions: tuple[AssumptionRecord, ...]
    caveats: tuple[Caveat, ...]                  # fixed sentences with their PAPER line (Sec. 7.3)
    cost: CostEstimate                           # columns, stages to run, cache hits, capability calls, seconds
    rules_version: int
    def to_dict(self) -> dict

class Plan(eqx.Module):                          # product of the eager compile step (se.plan)
    record: PlanRecord                           # static field. Attribute access falls through to it. print(plan) is Sec. 5.5
    lattice: Lattice
    moment_map: MomentMap

class MomentMap(eqx.Module):                     # the normal form a = L(theta) c
    params: Params                               # named pytree of the nonlinear parameters: value, lower, upper and priors are A. transform is S
    basis_params: Params                         # free parameters theta_b of averaged statistics. They change the basis (Sec. 5.2, R6)
    n_linear: int                                # S
    constraint: Literal["fixed", "free", "nonnegative"]       # S, the set of c
    def linear_map(self, theta) -> Array         # L(theta) with the column map applied, (columns.n, n_linear), traceable

class Problem(eqx.Module):                       # plan + basis + data
    record: PlanRecord                           # eqx.field(static=True)
    lattice: Lattice
    moment_map: MomentMap
    basis: SpectralBasis | BasisSum              # the design is the horizontal concatenation of the parts' designs
    data: StokesData | RetainedCombinations
    params: Params
    def log_likelihood(self, params) -> Array    # pure JAX when moment_map.basis_params is empty. Host in 0.8 otherwise (Sec. 8.6)
    def log_prob(self, params) -> Array          # log likelihood + log priors. ImproperPosterior for FreeMoments without a prior (Sec. 6.6)
    def profiled(self) -> Callable               # theta -> chi2 with the linear unknowns solved in closed form
    def constrain(self, z) / unconstrain(self, params)
    def solve(self, *, rank=None, penalty=None, starts=None, solver=None, keep_directions=False, chunk_bytes=2**31, key=None) -> Fit
    # problem_a + problem_b concatenates two designs on the same data (a BasisSum). Shared parameters across problems: [0.9]
```

### 3.7 Data, policies and fit

```python
class StokesData(eqx.Module):                    # extends PKG/syncem/model/fit/observation.py:90 (whitening, mask, response)
    StokesData(channels, *, unit, I=None, Q=None, U=None, V=None, stokes=None, sigma=None, cov=None,
               mask=None, response=None, response_uncertainty=None, discrepancy=None)
    # unit is required: "Jy/sr", "MJy/sr", "K_RJ", "K_CMB", or "cgs" (erg s^-1 cm^-2 Hz^-1 sr^-1)
    # discrepancy: a declared bound on the model discrepancy per channel, user input (ledger term discrepancy, Sec. 7.2)
    stokes: Array                                # A (..., n_ch, n_stokes). Leading axes are pixels
    sigma: Array | None                          # A, standard deviation, (n_ch, n_stokes) shared or with leading axes
    unit: str                                    # S
    def withhold(self, rows) -> StokesData       # rows that validate only. They never enter an estimator or a rank choice
    StokesData.simulate(spectrum, *, noise, key) -> StokesData               # sigma = noise * I_j on every component. Unit of the spectrum
    # data[rows] selects channels. data_a + data_b joins two instruments on the same pixels

class RetainedCombinations(eqx.Module):          # the constrained quantities of a free-moment fit, and the data of a compressed fit
    beta: Array                                  # A (..., R_max), masked beyond the rank
    sigma: Array                                 # A (..., R_max), 1 / singular value
    directions: Array | None                     # A (..., R_max, n), rows of V^T in the scaled moments. No pixel axis: shared noise,
                                                 #   broadcast. None: per-pixel noise without keep_directions, recomputed (Sec. 6.4)
    rank: Array                                  # A (...,) integer. Fixed per fit in a basis-parametric fit (Sec. 6.1)
    key: str                                     # S, lattice key including the reference weights and the column-map key

se.UPRE(cap=None)                                # rank rule R* = argmin [chi2(R) + 2R] up to the numerical rank or the cap
se.Rank(R)                                       # fixed rank
se.FullRank()                                    # least squares
se.Penalty(alpha=None)                           # alpha |a|^2 in the scaled moments. None: alpha by the UPRE of the same fit without positivity
se.AccuracyCheck(max_seconds=5.0, providers=False, own_peak_tol=None)   # value of check=. Second provider. own_peak_tol makes the own-peak test of check (a) fatal (Sec. 8.3)

class Fit(eqx.Module):
    kind: str                                    # S
    estimator: str                               # S, for example "noise-whitened SVD in the reference weights, rank by UPRE"
    rank, chi2, dof, p_value: Array              # A (...,)
    retained: RetainedCombinations | None
    moments: Moments | tuple[Moments, ...]       # one per component key of a BasisSum. Free fit: minimum-norm vector in lattice rows (D S),
                                                 #   no uncertainties, rows without a column in moments.masked. Origins by kind: Sec. 6.3
    params: ParamTable | None                    # value, sigma, bounds, at_bound, identified
    population: Population | None               # family or fixed: the declaration with fitted values. Non-negative cells: the
                                                 #   point-mass measure. Free moments: None (fit.retained holds what the data constrain)
    weights, penalty, effective_parameters: Array | None      # non-negative cells
    withheld: Withheld | None                    # chi2, n, chi2_predictive, max_residual
    ledger: Ledger
    plan: PlanRecord
    problem: Problem
    defaults: tuple[DefaultRecord, ...]
    def predict(self, channels=None, *, cov=False) -> Spectrum   # cov=True fills Spectrum.cov, with a printed size (Sec. 6.4)
    def observable(self, c) -> Observable        # any linear functional c of the moments
    column: Observable                           # order-0 sum: N_src, or the effective column at B_0 on a FieldScaling lattice (labelled)
    source_column: Observable | None             # N_src = effective column / <w>, the mean push-forward weight (Sec. 4.2), when the laws in w are specified with fixed
                                                 #   parameters. None otherwise, with the reason in the label
    def electron_spectrum(self, gamma=None)      # family: density. Non-negative cells: point-mass weights. Free moments: raises
    def profile_max(self, channels, *, stokes="I", delta_chi2=1.0) -> Observable   # non-negative cells, refit without the penalty
                                                 #   (PAPER:1148, ST/fits_free_positive.py:117-149). Origin "nonnegative_profile", Sec. 6.3
    def resample(self, n, *, key) -> Fit         # refits n noise realisations, leading axis n
    def derived(self, name) -> Observable        # invariants such as "break_frequency" (PAPER:833), and "paper_coefficient" (Sec. 3.5)
    def save(self, path) -> None
```

### 3.8 Results and records

```python
class Spectrum(eqx.Module):                      # values layer
    stokes: Array                                # A (..., n_ch, n_stokes). Accessors nu, I, Q, U, V, P
    sigma: Array | None                          # A, noise only
    cov: Array | None                            # filled only on request (fit.predict(cov=True)). Per-value sigma otherwise
    unconstrained: Array | None                  # A (..., n_ch, n_stokes), free-moment and penalised fits, Sec. 6.3
    moments: "Moments | tuple[Moments, ...] | None"   # None above 1 GB of dense moments, printed. basis.moments() gives them sparse
    ledger: Ledger
    plan: PlanRecord
    unit: str                                    # S
    channels: "Channels"                         # S, the channels and their quadrature, used by nu and to(unit)
    defaults: tuple[DefaultRecord, ...]
    def to(self, unit) -> Spectrum
    def table(self) -> str
    def through(self, response) -> Spectrum      # a linear response after the line-of-sight model

class Ledger:                                    # records layer. Named tuple of ErrorTerm (PKG/syncem/model/errors.py:66-148, five kinds,
                                                 #   unchanged, plus a static flag declared)
    def __getitem__(self, name) -> ErrorTerm
    def by_kind(self, kind) / sum(self, kind) / unbounded(self) / declared(self)
    def envelope(self) -> Envelope | None        # sum of all valued terms, labelled with the weakest kind present. None while a term is
                                                 #   unbounded. Names the declared terms it depends on (Sec. 7.2)
    def allow(self, name, term, *, note) -> Ledger    # replaces an unbounded term by a declared one and writes an AssumptionRecord

class Observable(NamedTuple):   value: Array     sigma: Array | None     unconstrained: Array | None     origin: str     label: str
                                # values layer
class Comparison:               # results layer. Table of chi2, dof, p, effective parameters and withheld chi2, or of differences per channel
se.ErrorTerm                    # re-export, with the field declared: bool = False
se.UnitError, se.RoleError, se.RegimeError, se.SpecifiedMismatch, se.NothingToPredict, se.DataError, se.DepthExpansionError,
se.SizeError, se.ImproperPosterior
se.paper                        # module: channels(), cells(), specified(), e1() .. e6(), e1m(), extrapolation_channels()
```

## 4. The lattice and per-cell expansions

### 4.1 Cells, coordinates, reference points, orders

```python
cells = se.Cells.log(300, 2.9626e6, 19)                 # equal in ln gamma, u_b = ln(gamma / gamma_b) / h_b, h_b = 0.2420, order 2
se.Cells.linear("2 uG", "8 uG", 3, order=1)             # affine chart, u = (B - B_b) / h_b
se.Cells.log("0.5 uG", "50 uG", 4)                      # field cells in the log chart
se.Cells.from_edges([300, 1e3, 1e4, 3e6], order=(2, 2, 4))      # unequal cells, one order per cell
se.Cells.single("5 uG", scale="1 uG")                   # one cell without edges: x_B = (B - 5 uG) / (1 uG) of E6
se.Cells.single(20.0, scale=20.0)                       # the reference point of PAPER Sec. 4.2
cells.shift(0.5)                                        # reference points moved by a quarter of the cell width (PAPER:741)
se.Lattice(gamma=cells, B=se.Cells.single("5 uG", scale="1 uG"), mu=se.Legendre(8), phi=se.AzimuthHarmonic(), order=2)
```

A cell has an interval, a chart, a reference point, a half-width and an order. The lattice `order` caps the sum of the exponents over the celled variables other than depth (Sec. 4.3), and `order=None` takes the largest order of those axes. A cap above the sum of those axis orders changes no row, so `Lattice` raises `ValueError` with the axis orders and names `Cells.with_order` (prototype partition: cells of order 2 under a total of 4 gave 3 rows per cell). The default reference point is the centre in the chart, because a quarter-width shift raises the `N = 2` error by factors 6 to 18 (CTX 06 D.2). In a bounded cell `|u| <= 1`. A cell without edges is unbounded, and its remainder has no uniform control. The plan prints the largest `|u|` that the declared support reaches in such a cell.

### 4.2 Roles of a variable

| Role | Spelling | Functions `psi` | Variables | Where it acts |
|---|---|---|---|---|
| resolved by cells and local polynomials | `Cells` on the lattice | `1_b(x) u_b^r`, `r <= N_b` | positive, real, cosine. The pitch cosine under `Beaming` when the inclination has a density or `Legendre` degrees | basis columns `(1/r!) d^r/du^r` of the averaged kernel at the reference point (PAPER eq: illustrated derivative spectra), by the jet provider or a shortcut whose precondition holds (Sec. 8.3) |
| resolved by Legendre degrees | `Legendre(L)` | `P_l(x)`, `l <= L` | cosine | basis columns by projection, factor `(2l+1)/2` per cosine (PAPER eq: legendre spectral response) |
| resolved by the exact azimuth harmonic | `AzimuthHarmonic()` | `exp(i s_X phi)` | the kernel's azimuth | no truncation. Splits the lattice rows into a real spin-0 block and a complex spin-2 block. Exact while no averaged law has `phi` as a parent |
| resolved by azimuth Fourier rows | `AzimuthFourier(m_max)` | `exp(i k phi)`, `0 <= k <= m_max + 2`, one row set shared by spin 0 (`k = |m|`) and spin 2 (`k = |2 + m|`), `|m| <= m_max` | the kernel's azimuth | for laws conditional on `phi` (R4a (b)), in 0.8.0. The truncation term is zero when `Kbar(phi)` of every averaged child is a trigonometric polynomial of degree at most `m_max`, and measured otherwise |
| resolved by point masses, population side | `NonNegativeCells(point_masses=17)` on a `Cells` axis | `delta(u - u_q)` paired with `u^r` | variables with cells | moment map `a = M w`, `M[r, q] = u_q^r`, `w >= 0`. Basis unchanged (PAPER:1146) |
| resolved by point masses, basis side [extension, 0.9] | `PointMasses(cells, 17)` | Lagrange functions on the points | positive, real | basis columns are kernel values at the points. This is the kernel table of PAPER:848 |
| resolved through the scaling relation [0.9] | `FieldScaling(reference)` | none added | a variable with `FieldScalingRelation` whose role precondition holds (Sec. 8.1). With `inclination=True` it resolves `B` and `eta` | the measure is pushed forward to energy `gamma (B/B_0)^(1/2)` with weight `B/B_0`. The order-0 sum is the effective column at `B_0` |
| averaged against a specified conditional | no axis, a `Distribution` with fixed parameters | the constant | any | quadrature inside the kernel average. Nodes and weights may depend on resolved or averaged parents (PAPER:644). The columns include the derivatives of the conditional (PAPER:659) |
| averaged through an exact characteristic function | no axis, a depth distribution with `characteristic` | the constant | the kernel's depth, when depth is a parent of no averaged law and of no azimuth factor (PAPER:606) | `C_scr(tau(nu))` at every quadrature frequency or line inside the channel integral (PAPER:625). A screen returned by a conditional is evaluated per parent node or cell node. Otherwise depth is averaged by its `rule()` nodes, with its children nested there (R4) |
| averaged in the beaming limit | no axis, kernel with `Beaming` | the constant, or `P_l` | pitch cosine | the pitch density is evaluated at `mu = eta`, in the basis and in direct integration (PAPER:568, 1081) |

A variable without an axis must have a distribution with fixed parameters, or it becomes a basis parameter (Sec. 5.2, R6). Cells on any variable use the jet provider of Sec. 8.3, with shortcuts where their preconditions hold. The azimuth takes no cells: the kernel depends on it only through the harmonic, unless an averaged law has `phi` as a parent (R4a). An averaged `phi` then takes nested quadrature at its nodes (R4a (a)). The two cases of R4a (b), an averaged child of a `phi` on `AzimuthHarmonic` and a child outside the free block of its `phi`, take `AzimuthFourier(m_max)` rows in 0.8.0.

Beaming. Under `Beaming(variable="mu", partner="eta")` only `mu = eta` contributes, and the admissible roles of `mu` follow the role of `eta`. The isotropic continuum average gives `(1/2) ∫ K_nat dmu = K_iso` (PAPER eq: isotropic continuum average), so in the beaming limit `K_nat -> 2 K_iso(eta) delta(mu - eta)`. Every column below, the folded weight of `FieldScaling` and the `Beaming` row of Sec. 8.1 carry this factor 2.

* `eta` averaged with a density `p_eta` (its parents resolved or averaged, B5). `mu` may be averaged (its density is evaluated at `mu = eta`), resolved by `Legendre(L)` (columns are the eta weights `(2l+1) P_l(eta)`), or resolved by `Cells`, `NonNegativeCells`, `PointMasses` and sample sums. The effective averaged kernel in `mu` is `g(mu) = 2 p_eta(mu | ...) K_iso(..., eta = mu)`, an ordinary function. Its columns are `(1/r!) d^r g / du^r` for cells (jet provider), `g(mu_q)` for point masses and `sum_i g(mu_i)` for samples. Cell edges must include the jump points of `p_eta` (`±0.5` for `U[-0.5, 0.5]`). A default lattice places them and prints them with `[default]`. A given lattice with a jump inside a `mu` cell raises `RoleError` naming `Cells.from_edges` with the jump points (open question 17). The jet does not see such a jump: a density written with `jnp.where` has zero derivative there and is taken at its value at the cell reference point, so this `RoleError` is the guard (prototype beaming). Cells outside the support of `p_eta` get zero columns and are dropped by the column map with the reason "support". The averaged `eta` law is built into these columns, as the averaged laws are in the paper's energy-cell bases (PAPER:846), so the basis is rebuilt when it changes. For `eta ~ U[-0.5, 0.5]`, `p(mu) ∝ e^{2 mu}`, `gamma = 1e4`, the E1 field law, channels and screen, 1, 2 and 4 equal cells on `[-0.5, 0.5]` reproduce direct integration to 2.7e-2, 3.2e-3 and 2.3e-4 at `N = 2` and to 1.3e-3, 5.2e-5 and 1.1e-6 at `N = 4` (max |dX| / I). A jump of `p_eta` at a cell centre gives 1.5. Cells that touch `|mu| = 1` converge slowly, because `K_iso` depends on `sqrt(1 - mu^2)`: two cells on `0.95 <= |mu| <= 1` give 0.49 at `N = 2` and 4.3e-2 at `N = 4`. For E4-KP with 2 and 4 cells on `[-0.5, 0.5]` the forward error is 2.2e-3 and 7.3e-3 at `N = 2` and 7.4e-4 and 6.8e-4 at `N = 4`, against 8.8e-3 and 6.5e-4 with `Legendre(16)` (prototype beaming).
* `eta` resolved by `Legendre(L_eta)`. `Cells` on `mu` take the columns `(1/r!) d^r/du^r [(2k+1) P_k(mu) K_iso(mu)]` at the cell reference point, and the moments are `<1_b(mu) u_b^r P_k(eta)>` with `u_b = (mu - mu_b)/h_b`. `PointMasses` on `mu` take the columns `(2k+1) P_k(mu_q) K_iso(mu_q)`. `NonNegativeCells` use the cell columns. The `mu` cells must cover `[-1, 1]`, because the Legendre partial sum of the `eta` population does not vanish outside its support. They must also resolve `(2k+1) P_k(mu) K_iso(mu)` up to `k = L_eta`, so a larger `L_eta` needs more cells. The representation term is measured. For `q = (3/4)(1 - eta^2)` on `Legendre(2)`, `p(mu) ∝ e^{2 mu}` and `gamma = 1e4`, 1, 2, 4 and 8 equal cells give 3.2, 0.20, 2.1e-2 and 2.1e-3 at `N = 2` and 1.1, 9.9e-2, 4.7e-3 and 4.3e-4 at `N = 4`. For `eta ~ U[-0.5, 0.5]` on `Legendre(16)` at `N = 4`, 8, 16, 32 and 64 cells give 4.7e-2, 1.8e-3, 2.3e-4 and 1.9e-5. A jump law of `eta` is better averaged, with `mu` cell edges at its jumps (bullet 1, prototype beaming).
* `eta` fixed, `eta` resolved by `Cells` or `PointMasses`, product cells over `(mu, eta)`, or samples with joint `(mu, eta)` values. Only averaged `mu` and `Legendre(L)` are admitted, and other roles raise `RoleError`: the integrand is the population density on the diagonal `mu = eta`, which is not a moment of the population.

Without a lattice axis on `mu`, samples of `mu` are summed through `g` when `eta` has a density and paired through `Legendre(L)` (default 8) otherwise. `basis.direct` follows the same rule at the `L` of the basis's `mu` axis, or at 8 when the basis has no `mu` axis, so the same pairing enters both sides of `representation` and `outside_lattice`. The pairing smooths the samples on `mu = eta` and is written as an `AssumptionRecord` ("samples smoothed by Legendre(L) on mu = eta"). It has no truncation-free reference: as `L` grows the pairing kernel `sum_l (2l+1) P_l(mu_i) P_l(eta)` tends to `2 delta(mu_i - eta)`, so the sum over a finite sample does not converge, and its sampling variance grows in proportion to `L` (r3-05 verifier: 0.4096, 0.4170, 0.4297, 0.4391 at `L` = 8, 16, 32, 64 against 0.4211). The angular part of the representation term is then unbounded (Sec. 7.2).

FieldScaling and the column. The push-forward gives cell masses that sum to `<w> N_src`, not `N_src`, where `w` is the push-forward weight: `B/B_0`, or `2 f_mu(eta) B sqrt(1 - eta^2)/B_0` with `inclination=True`. On a lattice with a `FieldScaling` axis, `lattice.column` and `fit.column` therefore return the effective column at `B_0`, labelled so, with the caveat of PAPER:562-564 and 833. When the field law is specified with fixed parameters (with `inclination=True`, also the inclination law and the folded pitch density), `fit.source_column` divides by the known `<w>`: `<B/B_0>`, or `<2 f_mu(eta) B sqrt(1 - eta^2)/B_0>` with `inclination=True`, which is `<B sqrt(1 - eta^2)/B_0>` for an isotropic pitch. Otherwise it is `None` and the print says that the source column is not separately constrained. `inclination=True` requires an isotropic pitch law, or a pitch density `f_mu` folded into the push-forward weight as `2 f_mu(eta) B_perp/B_0` (`2 f_mu = 1` for an isotropic pitch). The plan checks this when it assigns the role. With `inclination=True` the axis resolves `B` and `eta` through the single scaling variable `B_perp = B sqrt(1 - eta^2)`. The basis kernel is evaluated at `sin(theta) = 1`, and the roles table prints `eta: resolved via FieldScaling`. A separate declaration of `eta` raises `RoleError` with two remedies: drop it, or pass `inclination=False`. For a `Ray`, each step's `eta` comes from `B_vector` and enters only through `B_perp`. The role also needs the precondition of Sec. 8.1, which includes an azimuth factor independent of `gamma` and `B`.

### 4.3 Truncation rule and column map

* Taylor exponents of a product cell form the lower set `r_v <= N_v` for every celled variable and `sum_v r_v <= order` over the variables other than depth. Depth powers are outside the total degree: every row of the other variables carries `s = 0..N_d` (`N_d` = the order of the depth cells, Sec. 3.3). This is the index set of PAPER eq: channel basis expansion with per-variable caps. With depth inside the total, the rows of energy power `r` keep only `s <= N - r`, and the depth error grows linearly in `tau h` (6.2e-3 of I at `tau_max h = 0.1`, `N = 2`, against 3.8e-5 with depth outside the total, prototype depth). The depth rows cost no spectra (closed-form weights), only moments. The lattice therefore applies the total per variable: the cell table and the index mark depth as excluded from it, and `order` and its `None` default refer to the other axes (Sec. 4.1). `Lattice.single` keeps depth inside the total, the index set of the Sec. 4.2 benchmark (PAPER:708, Sec. 4.7), so the depth bound of Sec. 4.6 rule 4 does not hold for it.
* Depth cells and depth powers belong to every product cell. The real block holds `<1_{b,c} u^r d^s>`. When no averaged law, conditional or azimuth factor has depth as a parent, the bare I and V kernels do not depend on depth (PAPER:535). Their columns are then zero for `s > 0` and equal across depth cells, I and V use the columns with `s = 0`, which repeat one stored spin-0 spectrum per depth-free row (`ColumnMap.spin0_from`), and depth enters P through the closed-form Taylor weights of the phase law. This is a fixed summation over depth cells written on the column side. The unknowns stay the real block.
* When an averaged law, conditional or azimuth factor has depth as a parent, the averaged kernel carries the dependence (PAPER:644, 659). The engine keeps the closed-form phase weights and forms a per-cell Cauchy product with the depth Taylor coefficients of the phase-free averaged kernel (the kernel without `exp(i tau depth)`), taken by the jet with the law's nodes and weights moving with depth, or by frozen-node reweighting where its precondition of Sec. 8.3 holds (a law with a density and fixed support). A `Delta` law or a law with moving bounds takes the jet: on frozen nodes their `s > 0` columns are zero (Sec. 8.3, the pattern of R5 (ii)). I and V then have nonzero `s > 0` columns and one spectrum per depth cell, and the column map records the reason. In a depth cell `[6, 10]` with `B = 5 uG (1 + 0.05 (depth - 8))`, the `s = 1` column of I is 0.067 of the `s = 0` column where `x = nu / nu_c` is small (prototype rules/depth, first and last E1 cells), and larger in the exponential tail.
* Parity masks and the half-range quadrature share one precondition: the full averaged integrand of the block is invariant under the joint reversal `(mu, eta) -> (-mu, -eta)` (PAPER:577-582) with the block's sign. The plan checks that (a) every averaged cosine has an even law, a `Delta` at a nonzero value counting as not even, (b) every averaged conditional or azimuth factor whose parents include a cosine declares its parity in that cosine (`se.Conditional(..., parity=)`) and is even for I and P, (c) under `Beaming`, when `mu` is averaged, its density at `mu = eta` is even, and (d) the partner of a cosine that is taken on the half range or carries a parity mask is averaged with an even law over its full range `[-1, 1]` or in the beaming limit (a `Delta` only at 0), resolved by `Legendre(L)` over `[-1, 1]` with the parity mask, or absent from the kernel (`FieldScaling(inclination=True)` evaluates at `eta = 0`). A `mu` resolved by `Legendre(L)` puts the pitch density into the moments and not into the averaged integrand, so it needs no check (c). With `mu = Legendre(L)`, an even `eta` law and a pitch population with an odd part, the half range with the odd-`l` mask equals the full range to 1.1e-15. The mask on `Legendre(L)` degrees of `eta` with an averaged asymmetric pitch law loses 12 per cent of I (prototype beaming). At most one cosine of a block is taken on the half range, and its partner keeps the full range. A user function, `Conditional` or `Tabulated` law without a declaration counts as not even. The block then gets the full range and no mask, and `PlanRecord.routes` records why. For the harmonic kernel with `eta` fixed and `mu` averaged, (a) fails. `Cells`, `NonNegativeCells`, `PointMasses`, sample values and `Ray` steps on the partner cosine fail (d), with the reason "partner cosine resolved pointwise or by cells". The reason: `∫_{-1}^0 p(mu) K(mu, eta) dmu = ± ∫_0^1 p(mu) K(mu, -eta) dmu` with the block's sign, which differs from `± ∫_0^1 p(mu) K(mu, eta) dmu` unless `eta = 0`, and the odd-`l` Legendre columns of I at a fixed `eta ≠ 0` are not zero. Halving both cosines when both are averaged drops the quadrants of opposite sign (55 per cent on a test kernel). The rule is the same with the cosines exchanged (`mu` on `Cells`, `eta` averaged).
* Legendre pairs are all `(l, k)` with `l <= L_mu`, `k <= L_eta`. When both cosines are resolved by `Legendre`, the precondition holds and the kernel declares `Parity`, I and P use `l + k` even and V uses `l + k` odd (PAPER eq: angular parity). A `Legendre` cosine whose partner is on `Cells`, `NonNegativeCells`, `PointMasses`, samples or `Ray` steps gets no mask, by check (d).
* One cosine resolved and the other averaged, with the precondition met and a kernel that declares `Parity`: the I and P columns vanish at odd `l`, and the V columns at even `l`. This is a per-block mask, as in `_harmonic_taylor.py:336-341`. For `eta ~ U[0, 1/2]`, which is not even, the columns with `l = 1, 3` are 0.73 and 2.0 times the order-0 column (J2 finding 1). With `B = 5 uG (1 + 0.5 eta)` averaged, the half range changes I by 15 per cent. With `c(eta) = 0.4 (1 + 0.8 eta)` under route (iv), `eta ~ U[0, 1/2]` and `L = 8`, the even mask lowers `|P|` by 16 per cent (prototype R5, r2-02). There is no user flag for parity.
* A declared population symmetry (`pitch_reversal`) makes the odd-`l` moments zero. Those rows are dropped for every block.
* With `Beaming`, the columns of `(l, k)` and `(k, l)` are equal (PAPER:568). The column map keeps one column per unordered pair, and its moment is `M_lk + M_kl` (J2 finding 3).
* A lattice row has no column only when every requested Stokes block that uses it has a zero column there. Odd-`l` spin-0 rows therefore have columns whenever V is requested. Every drop and merge is recorded in the column map with its reason. A fit keeps every lattice row and lists the rows without a column in `Moments.masked`.

### 4.4 Index, column map and labels

`Lattice.index` has one row per lattice row: `block`, `cell` (the flat product-cell number and, per celled variable, the interval `(lo, hi)` in canonical units, with the cell number per variable as a convenience on tensor tables), `powers`, `degrees`, and `pair` (the partner of a `Re` or `Im` row). The blocks follow the representations alone. With `AzimuthHarmonic` the index has `M0`, `Re M2` and `Im M2` over the same rows. Without it there is one real block. Depth cells and depth powers are rows of every block. The order is block, then cell, then Legendre pair, then powers in graded lexicographic order. Cells are numbered from zero, so the paper's cells 4 to 19 are `slice(3, None)`.

The index is unpruned and independent of the kernel, so `Moments` from a population need no kernel. The basis and the plan own the `ColumnMap` (Sec. 3.4) from lattice rows to basis columns. It is exact. `basis.predict(moments)` applies it. Free-moment fits are parametrised in lattice rows through the design `D S`, with `S = columns.matrix()`, so every `Moments` is in lattice rows and every `Lattice` method applies to it. Merged columns are labelled as sums of rows.

```python
lat = se.Lattice(gamma=se.Cells.log(300, 2.9626e6, 19), phi=se.AzimuthHarmonic())
lat.n                        # 171 lattice rows
lat.labels()[:4]             # ('M0[gamma=0 | u^0]', 'M0[gamma=0 | u^1]', 'M0[gamma=0 | u^2]', 'M0[gamma=1 | u^0]')
lat.labels()[57]             # 'Re M2[gamma=0 | u^0]'
lat.index.position(block="Im M2", gamma=7, powers={"gamma": 1})
lat.index.select(gamma=(1e4, 1e5))          # interval selector: the cells inside [1e4, 1e5]
```

For one celled variable the position inside a block is `b (N + 1) + r`, the layout of `ST/lib/basis.py:241-263`. With depth cells and `phi` averaged, the real block holds the unknowns of every Stokes block. For example 19 energy cells, 3 depth cells and the 9 exponents `(r, s)` with `r <= 2`, `s <= 2` (depth outside the total, Sec. 4.3) give 513 rows, all used by P as `c` times `spectra2` (Sec. 5.2, R5). I uses the 171 columns with `s = 0`, which repeat 57 stored spin-0 spectra when no averaged law depends on depth.

### 4.5 Moment conventions and scales

* Moments carry the electron column: `a_alpha = (1 / w_c) * integral of psi_alpha dN` over product cell `c`, with `dN` the column measure in cm^-2 (PAPER eq: energy cell response). `N_src` is the sum over cells of `w_c` times the order-0 spin-0 moment. On a lattice with a `FieldScaling` axis this sum is the effective column at `B_0` (Sec. 4.2). There is no unit moment and no separate amplitude. This reverses `m[0] = 1` of `PKG/syncem/model/moments.py:261-263`.
* The reference weights `w_c` are fixed before any fit and resolved before the lattice key is formed. The weight of a product cell is the product over its axes of the cell mass of that axis's reference density: `gamma^-2.6` normalised on the lattice range for an axis named `gamma`, and 1 otherwise. When the lattice is given without `weights=` and the kernel declares `ReferenceDensity`, the plan replaces the factor on that axis, prints `[default]`, and the resulting lattice has its own key, so moments paired with the kernel-free weights do not pass the key check. A one-cell lattice has `w = 1`.
* A prediction depends on `w_c a` only. The singular-value decomposition, the minimum-norm vector and the penalty depend on `w` (PAPER:659). The stored spectra are unscaled, so a change of `w` re-runs assembly (S3) only. `basis.design()` applies `w`.
* Legendre polynomials have `P_l(1) = 1`. The factor `(2l+1)(2k+1)/4` sits in the basis spectra.
* Spin-0 moments are real and serve I and V. Spin-2 moments are complex and enter as paired `Re`, `Im` columns.
* Factors outside an unknown block enter as probability laws with unit mass, and the column lives in the unknown block. When no block is unknown, the column lives in the family's `column=` or `amplitude=`. A free amplitude on a specified factor that multiplies a `FreeMoments` block is absorbed into the block and not fitted (r2-18).
* Merged Beaming pairs: the minimum-norm vector in lattice rows splits `c = M_lk + M_kl` equally, `M_lk = M_kl = c/2`. The paper's fits have no merged columns, so no paper number depends on this convention.
* `lattice.recentre` (same cells, other reference points) and `lattice.coarsen` (unions of cells) are exact linear maps for cells of one chart, checked to 2e-13 (J2 Sec. 4). Refinement from moments alone is not defined.

### 4.6 Default lattice and validity checks

1. Energy range. The kernel's `EnergyWindow` returns the window in its scaled variable for the averaged statistics, the density `gamma^-2.6` and the tolerance 1e-3 of the band-integrated `I_j` in every channel on each side. The channels are those of the call and, in a fit, those of `predict=`. The window is mapped to `gamma` with the channel-support edges, a conservative bound because each response vanishes outside its support, and with the bounded support of the declared field distribution. When `B` is resolved, its support is the lattice's `B` range or the `support=` of its marker. A field law with unbounded support supplies its truncation (Maxwellian: 8 rms), and the clamp at the kernel limit applies. A field law without a truncation raises `RoleError` naming `lattice=`. For the continuum kernel this is the eta-averaged closed-form window of `prototypes/synth/window.py`, which needs no S0 or S1 table. The range is intersected with a bounded declared support. With `kernel=None` the continuum window gives the range, and rule 5 then chooses the kernel per cell. A kernel without `EnergyWindow` needs a bounded `gamma=` support or `lattice=`. Otherwise `RoleError`. With channel centres in place of support edges the lower tail at 0.1 GHz is 1.21e-3, above the tolerance (r2-31).
2. Cells. Equal in `ln gamma`, `n = ceil(ln(gamma_hi / gamma_lo) / 0.484)`, order 2, reference points at the centres.
3. Other resolved variables take the representations of Sec. 2.3. Without `lattice=` a resolved non-primary variable needs a bounded range: `support=` on its marker when it is unknown, the `[1e-3, 1 - 1e-3]` quantile range of its law when that law is unbounded, with the tail mass as a measured `outside_lattice` term. A missing range raises `RoleError` naming `lattice=` and `support=`. A depth axis takes the smallest cell count that passes rule 4.
   Size. `max_columns` (default 20 000, an argument of every front-door call, of `Lattice.default` and of `SpectralBasis.build`) applies to every lattice, explicit or default. The plan prints the lattice rows, the basis columns and the dense moment memory per sightline. Above the cap it raises `SizeError` with these numbers and the argument. When the depth count of rule 4 needs more columns than the cap, `DepthExpansionError` names both remedies.
4. Depth expansion. For a depth axis the compile step computes `x = tau_max h`, with `tau_max = 2 lambda^2` at the lowest frequency of the channel supports and `h` the largest depth half-width, and the depth bound `x^(N_d+1) / (N_d+1)!`, where `N_d` is the depth cap of the cells. The depth bound limits the depth part of `max |dP| / I` for any population inside the depth cells, because `|K_P| <= K_I`, `|azimuth factor| <= 1` and `|exp(i a) - sum_{s <= N_d} (i a)^s / s!| <= |a|^(N_d+1) / (N_d+1)!`. The compile step records it as the ledger term `depth_truncation` of kind bound and raises `DepthExpansionError` when it exceeds `depth_tol` (default 1e-3, an argument of every front-door call). The admitted `x` is 0.001, 0.045, 0.182, 0.394 and 0.654 for `N_d` = 0 to 4. The bound needs depth outside the total degree (Sec. 4.3). For bump channels at 0.1 GHz with `f = 8 %` the lowest support frequency is 0.08 GHz and `tau = 28 m^2`, so `N_d = 2` needs `h <= 0.0065 rad m^-2` there. Measured worst cases (a point mass at a depth-cell edge) reach 1e-3 at `x = 0.27` for `N_d = 2`, and the old threshold `x = 1` admits 0.61, 0.22, 0.052, 0.0097 and 0.0015 of I for `N_d` = 0 to 4 (prototype depth).
5. Regime and kernel per cell. `kernel.regime` is called on every cell box, and with `kernel=None` the kernel per cell is chosen by the rule of Sec. 2.3. When the primary variable is averaged, both use its averaged support. A refusal raises `RegimeError` before any build. The plan lists the cells below `gamma = 300` and the kernel each gets.

The plan layer implements these rules in `planning.default_lattice`. It calls `EnergyWindow` and `regime` on the base kernel and no engine stage. No study varied the number of cells (CTX 06 D.3). `se.convergence(pop, channels, refine=2)` and `se.convergence(pop, channels, order=4)` give a measured comparison for a given population.

### 4.7 The paper's configurations as lattices

Basis columns depend on the kernel, the Stokes request and the averaged laws (Sec. 4.3). Lattice rows do not.

| Case | Lattice | Averaged (`basis.specified`) | Lattice rows | Basis columns |
|---|---|---|---|---|
| E1, E2, E3, E4-JP, E1M | `gamma=Cells.log(300, 2.9626e6, 19)` | B, eta, mu, phi (factor), depth (screen) | 57 | 57 |
| E5 joint | the same and `phi=AzimuthHarmonic()` | B, eta, mu, depth | 171 | 171 |
| E5 one factor | `gamma` cells, `phi` free and independent | B, eta, mu, depth | 57 | 57 and one complex factor |
| E6 | `gamma` cells, `B=Cells.single("5 uG", scale="1 uG")`, `order=2` | eta, mu, phi, depth | 114 | 114 |
| E4-KP | `gamma` cells, `mu=Legendre(8)` | B, eta, phi, depth | 9 x 57 = 513 | 5 x 57 = 285 (continuum kernel: no V, even eta law) |
| energy and depth (Sec. 4.4) | `gamma` cells, 3 depth cells, `order=2` | B, eta, mu, phi | 513 | 513 |
| Sec. 4.2 benchmark | `Lattice.single(about=samples, order=2, degree=8)` | none | 3 x 10 x 81 = 2430 | 486 + 2 x 410 = 1306 (HarmonicKernel, IQUV) |

The benchmark rows are three blocks of 10 monomials in `(gamma, B, depth)` times 81 Legendre pairs. The 1306 columns keep 6 depth-free monomials times 81 pairs in `M0` (I on even pairs, V on odd pairs) and 10 monomials times 41 even pairs in each spin-2 block (PAPER:708).

### 4.8 Limits of the finite-element reading

The lattice is a discontinuous piecewise-polynomial space for the kernel, and the moments are its dual degrees of freedom. The documentation states where the analogy stops (P2 Sec. 3.4). The approximated function is the known kernel. The moments are functionals of the measure, and many densities share them. The channels constrain fewer combinations than the lattice has dimensions: the 21 fitted channels of E1 constrain 26 of the 57 (28 with all 41). The approximation is a Taylor jet at one point per cell, so the reference point matters. There is no variational problem and no convergence guarantee (PAPER:566).

## 5. Population models

### 5.1 Declaration grammar

```text
Population(*blocks, symmetries=(), **factors)

factor := number | "5 uG" | Quantity                       fixed value
        | Distribution                                     specified
        | Distribution with Free(...) parameters           family
        | FreeMoments(cells=None, prior=None)              unknown, its moments are the unknowns
        | NonNegativeCells(point_masses=17, cells=None)    unknown and non-negative
        | function(parents) -> factor  |  Conditional      conditional on other variables
block  := FreeMoments(v1, v2, ...) | NonNegativeCells(v1, v2, ..., point_masses=(17, 9))
        | Samples(...) | Ray(...) | Moments(lattice, values) | a joint Distribution
sum    := population + population  |  number * population
```

Reading rules. A keyword is a variable of the kernel. Separate keywords are independent unless one is a function of another. A positional block is a joint statement about its variables. Each variable is declared once. Order and degree belong to the lattice, so one declaration runs at `N = 2` and `N = 4`.

| Statement | Spelling | Compiles to |
|---|---|---|
| fixed value | `B="5 uG"`, `eta=0.5` | averaged: the kernel at the value. Resolved: a point mass |
| specified distribution | `B=se.Uniform(4, 6, unit="uG")`, `mu=se.Isotropic()`, `depth=se.GaussianScreen(8, 0.5)` | averaged into the kernel and recorded in `basis.specified`. Resolved: moments by quadrature |
| independence | separate keywords | product of factors. Printed as the "assumed" row under the roles table, with each conditional named as an exception |
| conditional `P(B given gamma)`, `P(mu given gamma)` | `B=lambda gamma: se.LogNormal(...)` | averaged with a rule that depends on the resolved parent. The columns include the derivatives of the conditional (PAPER:659), through the per-node mode of the stack or the jet provider (Sec. 8.3). Field-scaling columns are not used (Sec. 8.1) |
| conditional of an averaged variable on an averaged variable | `phi=lambda eta: se.AzimuthFactor(0.6 * (1 - eta**2))`, `B=lambda eta: se.LogNormal(...)`, `depth=lambda B: se.GaussianScreen(...)` | nested quadrature inside the kernel average (R4, R5). `B` given an averaged or `Legendre`-resolved `eta`: one-node S1 tables per field node, read from the shared S0 table (Sec. 8.2, decision 144). `eta`, `mu`, depth or `phi` given `B`: the field average in S2 at the field nodes (for `phi`, in the spin-2 block only). `eta`, `mu`, `B` or depth given an averaged `phi`: nested quadrature at the `phi` nodes (R4a). Moving nodes: frozen-node reweighting or the jet with nested quadrature (Sec. 8.3). The plan prints the node product and the route |
| conditional of a resolved variable, `P(gamma given B)` | `gamma=lambda B: se.AgedPowerLaw(..., loss_field=B, amplitude=A)` | both resolved and the parent specified: joint moments by nested quadrature. `B` arrives as a parent value. The returned family carries `column=` or `amplitude=`, so both normalisations of E6 and E4-KP are expressible |
| conditional of a resolved variable on an unknown parent | `gamma=se.FreeMoments(), B=lambda gamma: se.LogNormal(...)` with `B` on the lattice | rule R3a: a fixed linear node (shifted correlation) from the parent's free moments. Linear |
| free parameters inside a conditional | `se.Conditional(fn, params=dict(index=se.Free(2.6)), units={"B": "G"})` | parameters named `<variable>.<param>`. Resolved child: family, separable or nonlinear kind. Averaged child: basis parameters (R6) |
| sum of components | `zone_a + zone_b`, `0.9 * a + 0.1 * b` | moments add |
| weighted samples | `se.Samples(gamma=g, B=(b, "G"), ..., weights=w, column=N)` | moments are weighted sums per cell. Class fixed |
| profiles along the ray | `se.Ray(step=..., B_vector=..., thermal_density=..., electrons=..., electron_density=...)` | a sum of components, one per step: `B`, `eta` and `phi` from `B_vector` and `depth(s)` from the thermal electrons (`PKG/syncem/rm.py:57`), each a `se.Delta`, the declared family in `gamma`, column `electron_density * step` (R8). On a lattice, the step variables must be on axes |
| family on a resolved variable | `gamma=se.BrokenPowerLaw(index1=se.Free(2.0, bounds=(0, 7)), ...)` | `theta -> a(theta)` by a fixed rule per cell. Separable when the normalisation is free |
| family on an averaged variable | `B=se.LogNormal("5 uG", width=se.Free(0.3))` | rule R6. Kind basis-parametric while the variable stays averaged |
| user family from a function | `@se.family(variable="B", unit="uG", support=(0.5, 50))` | as a built-in family. Parameters are the keyword arguments. `linear=("c0", "c1")` declares linear parameters and is verified by superposition at two random points and at the corners of the bounds |
| free moments of one variable | `gamma=se.FreeMoments()` | `L` = identity on its columns. Linear |
| free joint moments of a group | `se.FreeMoments("gamma", "phi")`, `se.FreeMoments("gamma", "depth")`, `se.FreeMoments("gamma", "eta", "phi")` | joint columns. Linear |
| fully joint free moments | `se.FreeMoments("gamma", "B", "mu", "eta", "phi", "depth")` | every variable on the lattice. Linear |
| two unknown groups, independent | `gamma=se.FreeMoments(), phi=se.FreeMoments()` | product of the two moment sets. Bilinear |
| free moments and an independent family on a resolved variable | `gamma=se.FreeMoments(), B=lognormal_field(width=se.Free(0.3))` with `B` cells | product of the free moments and the family's moments. Bilinear (parametrised factor) |
| non-negativity | `gamma=se.NonNegativeCells(point_masses=17)` | `a = M w`, `w >= 0`. Non-negative linear |
| known only as moments | `se.Moments(lattice, values)` | used as is, with its `origin`. The statistics of the basis become assumptions of the result |
| prior | `se.Free(2.6, prior=se.Normal(2.6, 0.3))`, `se.FreeMoments(prior=se.Normal(0, s))` | extra whitened rows. A Gaussian prior on scaled moments equals a penalty `alpha = 1 / s^2`, and the moments get origin `penalised` |

A forward call on a declaration with `Free` markers uses the start values and says so. With `FreeMoments` it raises `NothingToPredict`.

### 5.2 Role rules (rules version 4)

The compile step is eager Python. It builds no basis stage (S0 to S2). It may call the declared cheap capabilities of a base kernel (`EnergyWindow`, `regime`) and prints their cost. It applies the rules in order and stores the deciding rule per variable.

* R0. An undeclared variable takes the neutral declaration of its `Variable.default` with the tag `[default]`.
* R1. With `lattice=`, the variables on its axes are resolved and all others are averaged. A representation may resolve more than one variable: `FieldScaling(inclination=True)` resolves `B` and `eta`. No rule changes this. A declaration that conflicts with it raises `RoleError` with the remedies. `resolve=` is then an error.
* R2. Without `lattice=`, the resolved set is the kernel's primary variable (`gamma` for the synchrotron kernels, none for a kernel without one), every variable of a `FreeMoments`, `NonNegativeCells`, `Samples` or `Moments` block, and every variable in `resolve=`.
* R3. The resolved set is closed under parents. A specified `P(x given parents)` can be averaged for any population of the parents. The reverse would put the unknown distribution of the child into the basis. Without `lattice=` the parent is promoted and the rule is printed. With or without one, `average=("B",)` replaces the conditional by its marginal, records `independence:gamma|B` as an assumption, and the check measures it when the population is specified.
* R3a. A resolved child with a specified conditional on resolved parents whose moments are unknown (`FreeMoments`, `NonNegativeCells`) compiles to a fixed linear node. Let `g[c, s] = E[1_c(x) u_{x,c}^s | parents]` for child cell `c`. It is computed in closed form or by a quadrature on nodes fixed in the child cell's chart, with the conditional density as the weight (frozen-node reweighting). The conditional's own `rule()` nodes times the indicator of `c` are not used: the indicator has zero derivative, so the coefficients lose the flux through the child-cell edges (relative errors 1e-3 to 1.8 in `h_1`, `h_2`). They are admitted only for a child cell without edges. A child law without a density raises `RoleError`, naming `average=` and finer parent cells. Let `h[b, c, s, k] = (1/k!) d^k g[c, s] / du^k` at the reference point of parent product cell `b`, with `u` in that cell's chart and `k` a multi-index over the celled parents, taken by Taylor-mode or nested forward-mode differentiation of `g`. Separate integrals for each `h_k` (a Hermite formula) amplify roundoff by `zeta_b^k / k!`, with `zeta_b` the steepness of the law defined below: 1.4e-6 against 2.2e-15 by differentiation at `zeta_b = 161` and `N = 6` (prototype R3a). With `t` the exponents of the other variables of the parents' block, the node is `a[(b, c), (r, t, s)] = (w_b / w_{b,c}) sum over k with (r + k, t) in the lower set of the parents' block (Sec. 4.3) of h[b, c, s, k] a_parent[b, (r + k, t)]`. For one parent the sum is over `r + k <= N_b`. Since `<1_b u^r g(u)> = sum_k h_k <1_b u^(r+k)>`, the node is a shifted correlation, the moment-side adjoint of the kernel-side Cauchy product of R5 (ii). The weight ratio converts the scaled moments of Sec. 4.5 and is 1 when the child's axes have weight factor 1 (every axis not named `gamma` by default). Under `NonNegativeCells` the node evaluates `g` itself at the point masses, `a[(b, c), (r, s)] = (w_b / w_{b,c}) sum_q u_q^r g[c, s](u_q) w_q`, which is exact, and the kind is non-negative linear. Otherwise the kind is linear. A parent resolved by `Legendre(L)` uses the Legendre product expansion in place of the shift, `<P_l g> = sum_k g_k sum_{j <= L} G(l, k, j) <P_j>` with `g_k` the Legendre coefficients of `g` and `G` the linearisation coefficients of `P_l P_k`. The parent moments outside the lower set (or above degree `L`) are not unknowns, and the dropped terms are the ledger term `conditional_truncation:<child>|<parent>`. A specified parent never uses the node: its joint moments come by nested quadrature (Sec. 5.1). The truncated function `u^r g[c, s](u)` is known, so for every positive parent measure the term in channel `j` is at most `sum_b m_b E[b, j]`, where `m_b` is the parent mass in cell `b`, `E[b, j] = max over |u| <= 1 of |sum over c, (r, s) of col_j[(b, c), (r, s)] u^r (T_{N_b - r} g[c, s] - g[c, s])(u)|`, and `T_n` is the Taylor polynomial of degree `n`. The plan computes `E` before any data, from one evaluation of `g` on a grid per parent cell, and prints `sum_b m_b E[b, j] / I_j` at the reference weights. The bound held on all 2544 grid records, on narrow-edge cases and on four parent shapes, and at 12 to 19 times the actual on E1 (prototype R3a). When `g[c, s]` is a polynomial of degree `K_s` in the parents, the node is exact on every row whose shifts stay in the lower set. For one parent these are the rows with `r + K_s <= N_b`. A lognormal whose log-median is affine in `ln gamma`, on a log-chart child cell without edges, has `K_s = s`. A child edge within reach of the law makes `g` non-polynomial. The own error of rows near `r = N_b` does not fall as `N_b` grows (about 1e-3 of the parent mass at `N_b = 2, 4, 6` for `zeta = 0.08`). The prediction error does fall while `zeta_b = |d mu / du| / sigma <= 1`, where `mu` and `sigma` are the location and width of the law in its log chart: 8.8e-5 at `N = 2` and 2.4e-6 at `N = 4` for the continuum kernel. For `zeta_b > 2` with a child edge in reach, the error grows with `N_b` (to 1.6e8 of the parent mass at `N = 6`). The plan prints the cells where the bound exceeds 1e-3 of `I` and names finer parent cells there. Averaging the child does not remove the term: for a kernel polynomial in the lattice, the averaged-child prediction equals the node prediction to 1e-14, because both routes truncate the same function of `u` at order `N_b`. Averaging removes only the child-direction representation error.
* R4. An averaged variable needs a distribution with fixed parameters whose parents are resolved or averaged. The route follows the kernel, the distribution and the parent graph. Depth: the characteristic function when depth is a parent of no averaged law and of no azimuth factor (the independent-screen precondition, PAPER:606). A screen returned by a conditional is evaluated per parent node or cell node. Otherwise depth is averaged by its `rule()` nodes with its children nested at them, and a characteristic-only law (`CumulantScreen`) as a parent raises `RoleError`. Pitch cosine under `Beaming`: the beaming limit. Otherwise quadrature, nested when a parent is averaged, with the node product printed. Under `Beaming` the admissible roles of the pitch cosine follow the role of the inclination (Sec. 4.2).
* R4a. A law with the azimuth as a parent (`gamma`, `eta`, `mu`, `B` or depth given `phi`). A resolved child of a `phi` that is resolved by `AzimuthHarmonic` with a specified law needs no extra rule. Its rows are the joint moments `<exp(i s_X phi) psi(child) ...>`, formed by nested quadrature over the `phi` law (PAPER:497-499), and they are exact up to the child's Legendre or Taylor truncation. For an averaged child, the averaged kernel is `exp(i s_X phi) Kbar(phi)` in every block, which is not a single-harmonic moment (PAPER:164-166, 480-486).
  (a) `phi` averaged needs a law with `rule()` nodes (`AzimuthDensity`, `Tabulated`, `UniformAzimuth`). The conditional chain becomes route R5 (iii): the kernel, `exp(i s_X phi_q)` included, is evaluated inside the nested quadrature at the `phi` nodes, and the fingerprint of the `phi` law joins the S2 key. Routes R5 (i) and (ii) are barred while `phi` has averaged children. `AzimuthFactor(c)` as a parent raises `RoleError` naming `AzimuthDensity`, `Tabulated` and a joint `Distribution` block.
  (b) Two cases route through `phi=se.AzimuthFourier(m_max)` on the lattice in place of `AzimuthHarmonic` (Sec. 4.2): an averaged child of a resolved `phi`, and any child of a `phi` held in a `FreeMoments` or `NonNegativeCells` block when the child is outside that block. Their rows need `<exp(i k phi) ...>` for `k` other than 0 and 2, and R3a has no Taylor coefficients in an azimuth parent. `AzimuthFourier(m_max)` holds the rows `<exp(i k phi) ...>` for `0 <= k <= m_max + 2`, one set shared by the spin-0 columns (`k = |m|`) and the spin-2 columns (`k = |2 + m|`), and the child's dependence on `phi` enters through its Fourier coefficients of degree `|m| <= m_max`. The truncation term `azimuth_truncation` (Sec. 7.2) is zero when `Kbar(phi)` of every averaged child is a trigonometric polynomial of degree at most `m_max`, and measured otherwise. The plan prints `m_max` and the term. Without `lattice=` the plan takes `AzimuthFourier(m_max)` for `phi`. A lattice that resolves `phi` by `AzimuthHarmonic` is not changed (R1) and raises `RoleError` naming `AzimuthFourier(m_max)` and the alternatives (Sec. 2.5). The alternatives are route (a), resolving the child on the lattice when `phi` is specified, moving the child into the free block when `phi` is free, and `average=("phi",)`, which drops the dependence and records `independence:<child>|phi`.
* R5. The azimuth factor `<exp(2i phi) | parents>` follows its parents.
  (i) A constant `c` multiplies the spin-2 spectra at assembly (S3) and is in no cache key.
  (ii) Parents resolved by `Cells` only: the per-cell Cauchy product `P~'[b, n] = sum over r + k = n of P~[b, r] c[b, k]` over the multi-indices of all celled variables, depth included, with `c[b, k]` the Taylor coefficients of `c` in cell `b`. This equals averaging the factor into the kernel to order `N` (PAPER:644, 659, J2 finding 2).
  (iii) Any averaged parent, or a joint `Distribution` block that holds `phi` and averaged variables: the factor is a weight inside the averaged spin-2 kernel at the parents' quadrature nodes. Its fingerprint joins the S2 key and `basis.specified`. Its parity in a cosine parent must be declared for the half range (Sec. 4.3). With an averaged `B` among the parents, or `B` in the joint block, the spin-2 block cannot use the cached S1, which has already integrated over the field nodes. The field average of that block is taken in S2 at the field nodes, each node `q` weighted by `c(B_q)` (for a joint block, by `E[exp(2i phi) | B_q]` from its `rule()`) (Sec. 8.1, 8.3). When the factor also has parents in `Cells`, `c` is evaluated at each averaged node as a function of the celled parents. Its Taylor coefficients in them combine with those of the kernel at that node, `P~'[b, n] = sum_q w_q sum over r + k = n of P~_q[b, r] c_q[b, k]` (the per-node mode of Sec. 8.3, or the jet of the weighted average). Freezing `c` at the reference point of the cell, or applying route (ii) to the field mean `<c>_B`, drops terms: 1.2 and 7.6 per cent of `|P|` on E1 at `N = 2`, against 0.12 per cent for the per-node product (prototype R5). The spin-0 blocks keep the cached S1. For `B ~ U[4, 6] uG`, `c(B) = 0.4 [1 + 0.5 (B - 5)]` and `gamma ~ PowerLaw(2.6)`, the difference `<c K_P> - <c><K_P>` is 5.9 per cent of `<c><K_P>` and 5.6 per cent of `<c K_P>` in every channel. For one electron energy it is 2.2 per cent of `<c><K_P>` at `x = 1e-3` and 36 per cent at `x = 20` (2.2 and 26 per cent of `<c K_P>`), with `x = nu / nu_c` at 5 uG (r3-02, prototype R5). With depth as a parent of the factor, depth is averaged by its `rule()` nodes in the basis and in `direct`, so the measured representation term does not contain the depth quadrature error. Where the screen depolarises, the nodes do not resolve the characteristic function. For depth ~ N(8, 0.5) rad m^-2 and 60 Gauss-Hermite nodes, `|P|` in the 0.1 GHz channel is 1e6 to 1e11 times the exact value, 1.6e-7 to 8.5e-7 of `I` (prototype R5). This error is a ledger term of its own, `quadrature:depth`, and is not part of `representation`.
  (iv) A cosine parent resolved by `Legendre(L)`: the factor multiplies the kernel inside the engine's Legendre projection, so column `l` holds `(2l+1)/2 ∫ c(x) K(x) P_l(x) dx`. Its fingerprint joins the S2 key.
  Routes (ii) and (iv) put `c` into the basis spectra. The spin-2 representation term is then the truncation of `c K_P`, not of `K_P`, and it grows with the steepness of `c`. Route (iv) is exact only when `L` reaches the degree of `K_P` plus that of `c`. With `phi` resolved by `AzimuthHarmonic` and the same specified law, `c` moves into the spin-2 moments (`<c psi>` by quadrature) and only `K_P` is truncated. On E1, `c = sigmoid(4 ln(gamma / 1e4))` gives 1.4e-3 of `|P|` by route (ii) and 5.2e-4 on the moment side at `N = 2`. `c(eta) = sigmoid(20 eta)` with `eta ~ U[-0.2, 0.5]` gives 2.6e-2 by route (iv) and 1.1e-3 on the moment side at `L = 8`. The plan prints the moment-side alternative with the route.
  A free azimuth independent of the other unknowns is one free complex factor and needs no axis. When the lattice gives no route, `RoleError` names `average=` (dependence dropped and recorded) and `Cells` on the parent. A factor whose parents include the primary variable or a `FieldScaling` variable fails the `FieldScaling` precondition (Sec. 8.1). The plan then prints the fallback: `B` stays averaged and the fit is basis-parametric, or `B` goes on `Cells`.
* R6. A `Free` parameter in the distribution of an averaged variable is a basis parameter `theta_b`. Without `lattice=`, when the kernel declares `FieldScalingRelation` and its role precondition holds (Sec. 8.1, azimuth factor included), the variable takes the role `FieldScaling` and nothing is rebuilt [0.9. In 0.8 the next sentence applies]. Otherwise the variable stays averaged, the kind is basis-parametric (Sec. 6.1), and the plan prints the rebuild cost per evaluation. `resolve=` moves the variable onto the lattice, so the family acts through moments on a fixed basis.
* R7. In a sum, factors equal in all components are common. Components that differ in a resolved variable add in the moments. Components that differ in an averaged azimuth need `phi` resolved: promoted without `lattice=`, `RoleError` with one. When that azimuth has averaged children (R4a (a)), each component gets its own basis in a `BasisSum` instead, and `phi` is not promoted. Components that differ in another averaged variable get one basis per distinct set of specified statistics, held in a `BasisSum` (Sec. 3.4). The predictions add, and a fit concatenates the designs.
* R8. A `Ray` without `lattice=` is predicted by direct integration over its steps, with the depth of each step exact. With a lattice its moments are sums over steps, accumulated per step into sparse `(row, value)` pairs, because each step is a point in `B` and `depth` and touches one `(B, depth)` cell, and on the energy axis only the cells that its electron law reaches. The depth check of Sec. 4.6 applies. A lattice that leaves `B`, `eta`, `phi` or `depth` averaged raises `RoleError` naming the missing axes and the direct route, since each step would need its own basis. A `Ray` on a lattice with a `FieldScaling` axis is [0.9].

After the rules the plan checks the precondition of every shortcut it would use and records the outcome in `PlanRecord.routes`: the cached S1 and the variables each provider differentiates (Sec. 8.1, 8.3), field-derivative columns, the `FieldScaling` role with or without `inclination=True`, the half range and parity masks (joint reversal, Sec. 4.3), the characteristic-function route (R4), the closed-form depth weights with the spin-0 repetition (Sec. 4.3), the node motion of each conditional (`jax.jacfwd` of its nodes in the parents, Sec. 8.3), and the size cap (Sec. 4.6). The parent graph of every conditional, the azimuth included (R4a), is an input. A failed precondition selects the general route, and the plan prints why. The plan also computes the column map (Sec. 4.3).

A basis accepts a population only when the distributions of the averaged variables match `basis.specified` by fingerprint. Otherwise `SpecifiedMismatch` names the difference and the call that takes the flag. `allow_mismatch=True` runs it and writes `specified:<variable>` as a measured term, computed eagerly on the host. Inside a transformation the term is recorded as unbounded with the note "unmeasured (traced)". The flag never bypasses the lattice-key check (Sec. 3.3).

### 5.3 The map to moments and the class of the problem

Each independent group contributes one factor to the moment vector. All classes share the form `a = L(theta) c`, and the data model is `d = X~(theta_b) S L(theta) c`, with `theta_b` empty unless an averaged law has free parameters.

| Declaration on the resolved variables | `L(theta)` | `c` | Kind |
|---|---|---|---|
| everything specified | the moment vector from `pair` | 1 | fixed |
| `FreeMoments` on one joint group | identity on the selected columns | moments | linear |
| moment form of the power-law mixture, a user family with `linear=` | cell moments of each term, computed once | `A m_n` | linear |
| `NonNegativeCells` | `kron(I, M)`, `M[r, q] = u_q^r`, product points for a joint group | point-mass weights `>= 0` | non-negative linear |
| specified conditional of a resolved child on a free parent (R3a) | per-cell shifted correlation `(w_b / w_{b,c}) sum_k h[b, c, s, k] a_parent[b, r + k]` over the lower set of the parents' block, fixed per basis. Under `NonNegativeCells`: `g` at the point masses. Cells with `zeta_b > 2` and a child edge in reach are printed by the plan (Sec. 5.2) | parent moments, or point-mass weights `>= 0` under `NonNegativeCells` | linear, or non-negative linear |
| family with a free normalisation, sums of such components | unit-normalisation moments of each component | amplitudes | separable |
| free moments times one shared azimuth factor | `a -> [a, Re(c_2 a), Im(c_2 a)]` | moments | bilinear |
| free energy moments times shared unknown `Theta_lk` (PAPER:599) | Kronecker structure | moments | bilinear |
| free moments times a parametrised independent factor on resolved variables (a family with fixed normalisation, or a tie) | `kron(I, a_factor(theta))` | moments | bilinear (parametrised factor) |
| any of the rows above with free parameters `theta_b` of an averaged law | as in that row, with the basis `X~(theta_b)` | as in that row | basis-parametric |
| anything else, for example a family with a fixed normalisation | `a(theta)` | 1 | nonlinear |

The kind is read from the structure of the map without numerical probing. Internally the map is a small tree of nodes (constant, columns, point masses, derived, family, tie, conditional correlation, factorised product, sum). The factorised node is a port of the gather algebra of `ParameterMap` (`PKG/syncem/model/assumptions.py:240-248`) with cell labels. The nodes are not public.

### 5.4 Quadrature of resolved densities

* Forward pairing of a density with the cells uses Gauss-Legendre panels split at cell edges, declared kinks and support edges, with a Gauss-Jacobi panel at a declared algebraic end point (port of `ST/lib/ensembles.py:77-101`). Node doubling changes the study's direct spectra by at most 3e-14 of `I_j` (PAPER:1079).
* Family moments in a fit use a fixed rule per cell, so shapes are static and `da/dtheta` comes from `jax.jacfwd`. A smooth family uses 32 Gauss-Legendre nodes in `u`.
* A family that declares a cut-off uses one rule in every cell: `u` on `[u_lo, min(u_hi, u_c)]` is mapped to `v = ln(u_c - u)`, with 64 nodes, the lower limit of `v` truncated where the integrand has fallen by `e^-40`, and cells above the cut-off masked to zero with safe arguments under the mask (port of `jp_moments`, `ST/lib/ensembles.py:559-588`). The rule is chosen by the family type and never by a parameter value, so a cut-off that crosses a cell edge during a fit changes no shape and no branch. This answers the open risk of J3 Sec. 8. Test T7 of Sec. 8.8 sweeps the cut-off across an edge.

### 5.5 What the plan prints

```text
>>> print(se.plan(e6, obs))                     # E6 of Sec. 11 G8, no lattice given
Plan  41 channels 0.1-10 GHz (bump, fractional width 8 %), Stokes I Q U, continuum kernel, Faraday rotation       rules v4
 variable  declared                              role      representation and route                           rule
 gamma     AgedPowerLaw(2.6, cut-off from B) | B  resolved  15 cells in ln gamma on [6.7e2, 6.0e5], order 2     R2  [default]
 B         Uniform(4, 6) uG                       resolved  1 cell at 5 uG, scale 1 uG, affine, order 2, |u| <= 1   R3: gamma depends on B  [default]
 eta       Uniform(-0.5, 0.5)                     averaged  12 Gauss-Legendre nodes on [0, 0.5]: joint reversal holds  R4
 mu        Isotropic()                            averaged  beaming limit, mu = eta                             R4
 phi       AzimuthFactor(0.40 exp(0.50i))         averaged  constant factor on the spin-2 spectra at assembly   R5 (i)
 depth     GaussianScreen(8, 0.5) rad m^-2        averaged  characteristic function: depth is no parent         R4
 assumed   eta, mu, phi, depth independent of gamma, B and each other (separate keywords). gamma given B (conditional)
 range     EnergyWindow of the continuum kernel, field support [4, 6] uG, channel-support edges: 0.01 s
 basis     90 lattice rows, 90 columns = 15 cells x 6 monomials (u_gamma, u_B, total degree 2)
           routes: energy columns from the stack with the cached S1 at 5 uG (no averaged law has a parent)
                   B columns from the field-scaling relation (eta, mu, depth laws independent of gamma and B)
           cache: S1 present (5 uG). S2 absent, estimate 0.2 s
 moments   fixed. 16 nodes in B x 24 nodes per energy panel, Gauss-Jacobi at the cut-off
 problem   fixed (forward)
 note      continuum kernel with fixed geometry, a constant azimuth factor and an independent screen: in a fit the field
           moments are not constrained separately from the energy moments (PAPER:562, 833)
```

The cell count and range follow the support-edge rule of Sec. 4.6 (r2-31 verifier: 15 cells on [671.5, 6.00e5]). The "assumed" row prints `PlanRecord.assumptions` (Sec. 7.4).

Identifiability is reported at two levels. Structural: a capability such as `FieldScalingRelation` gives column relations (generator in `PKG/syncem/model/fit/_relations.py:3-36`), and the plan names the parameters they tie before any data are seen. A bilinear plan states the scaling freedom between its two blocks. Numerical: numerical rank of the whitened design, selected rank, `sigma` of each retained combination. Free parameters of a field distribution together with free energy moments are fitted and reported as not separately constrained. They are not refused (J1 Q20, J2 Q19, J3).

For `(gamma, B)` product cells where the precondition of Sec. 8.1 holds, column (r, s) of a cell is `L sum_n T[(r, s), n] G_n` over the energy coefficients `G_0 .. G_{r+s}` at the shifted reference. The cell therefore spans rank(T) directions, and the plan prints `n_rows - sum_c rank(T_c)` structural null combinations: 114 of 285 rows for 19 x 3 cells at caps (2, 1), and 2 of 6 per cell at total degree 2. With caps (1, 1) at total degree 2 it is 1 of 4 per cell. Two product cells whose shifted references `gamma_g B_b^(1/2)` coincide, with equal energy half-widths and field charts that are equal as functions of `B/B_b`, have proportional columns. Log field cells with `h_B = 2 m h_gamma` for an integer m are an example. This tie between cells is not among the per-cell relations of `_relations.py`, and the plan lists it as well (prototype fieldcells).

### 5.6 Serialisation and fingerprints

* `Population`, `Lattice`, `PlanRecord`, `Spectrum` and `Fit` have `to_dict()` through a tag registry and `strict_json` (`PKG/syncem/model/_provenance.py:24-36`).
* Every averaged statistic, registered or not, is keyed by a hash of its evaluated tabulation: nodes and weights at its parameter values, at the reference point, both edges and the quadrature nodes of every resolved parent cell, and at the quadrature nodes of every averaged parent. S1 and S2 depend on the function only through these weights, which also fix the derivatives to the built order, so the tabulation is the complete key of the memo and of the disk cache. Forming it costs one evaluation of the rule per parent node (16 field or 12 inclination nodes). The unit and range checks of the conditional run on it (Sec. 2.5). This hash is the fingerprint of the `SpecifiedRecord`.
* A function registered with `@se.family` or wrapped by `se.Conditional(fn, name=...)` also carries its import path and source hash as a readable label. Reloading a basis and calling `basis.at` needs the function again. For that path the fingerprint hashes the transitive closure of the functions it references from user modules with their numeric captured state, records the versions of referenced installed packages, and hashes dict, list, tuple and dataclass globals by strict serialisation. Any other referenced global prints `not cached: <reason>` for the reload path. An unnamed conditional is tabulated with the tag `"frozen"`, and the jet at each reference point through `jax.jacfwd` is the alternative to the tabulation.
* Free parameters of a conditional enter through `se.Conditional(fn, params=...)` and are not part of the function's identity. "Frozen" freezes the function, not its parameters. A tabulation that finds a `Free` leaf not supplied by `params=` raises and names `params=`. When no tabulation can be formed, the build skips the memo and the disk cache, and the plan prints `not cached: <reason>`. Source text alone is never a key.

## 6. Fits

### 6.1 One reduction and the dispatch table

With the whitened design `W = Sigma^(-1/2) C`, every fit is `min over theta, theta_b, c of |W(theta_b) L(theta) c - d_w|^2`, plus a penalty and constraints on `c`. `W` is fixed unless the plan has basis parameters `theta_b`. The estimator follows the kind of the plan, which follows the declaration. `rank=`, `penalty=` and `solver=` override it. A policy that does not apply to the kind raises.

| Kind | Inner solve for `c` | Outer solve | Default policy | Parameters counted | First release |
|---|---|---|---|---|---|
| linear, free moments | whitened SVD of `W` in the reference weights, minimum-norm vector | none | `rank=se.UPRE()` | `R` | JAX, `jit`, `vmap` |
| linear, derived basis (moment form) | least squares on `W L` | none | `rank=se.FullRank()` | columns | JAX |
| linear with penalty | the same SVD with filter factors `s^2 / (s^2 + alpha)` | none | `alpha` by UPRE on 321 values of `alpha / s_max^2` from `1e-30` to `1e2` | `sum f_i` | JAX |
| non-negative linear | NNLS on `[W L, sqrt(alpha) L] c = [d_w, 0]` (stacked rows), columns normalised | none | `penalty=se.Penalty()` | effective parameters of the active set | host SciPy `nnls`. Batches in a process pool with BLAS pinned to one thread (Sec. 6.4) |
| separable | least squares for `c` in closed form | bounded least squares on `theta`, exact Jacobian of `a(theta)`, multistart | bounds from `Free`, starts from the family | all | host SciPy `least_squares(method="trf")`. Batches: jitted `vmap` Levenberg-Marquardt with `trf` as oracle (Sec. 6.4) [0.9] |
| bilinear | SVD and rank policy at each trial of the factor | factor of at most two real dimensions (E5): grid start, then refinement. Larger factors (two free groups, `gamma` and `B` cells): start from the leading singular pair of the reshaped minimum-norm joint solution, then alternating least squares or variable projection with a few random starts. The plan prints the scaling freedom and, for the continuum kernel, the PAPER:562 tie | `rank=se.UPRE()` counting `R + dim theta` | `R + dim theta` | host |
| bilinear (parametrised factor) | free-moment SVD at each `theta` | variable projection. `R*` chosen once by minimising `chi2(R | theta) + 2(R + dim theta)` on a coarse `theta` grid, then fixed (as for basis-parametric). Jacobian in `theta` exact by `jacfwd`, starts from the family, tied directions by profile `chi2` | `rank=se.UPRE()` | `R + dim theta` | host |
| basis-parametric | the inner solve of the underlying kind at each `theta_b` | bounded least squares over `(theta, theta_b)`. Jacobian exact in `theta`. In `theta_b` a central finite difference of the S1 and S2 rebuild (S1 by lookups in the S0 table, Sec. 8.2) with the step in the record, or the jet provider's derivative for a traceable kernel on request | starts of the family. With a data-dependent rank in the inner solve (UPRE, or a penalty chosen by UPRE), `R*` is chosen once by minimising `chi2(R | theta_b) + 2(R + dim theta_b)` on a coarse grid of `theta_b`, as the bilinear kind does, and the refinement keeps `R` fixed. `R*` and the grid are in the record. The plan prints rebuilds per evaluation times seconds | all | host |
| nonlinear | none | bounded least squares on the full likelihood | bounds and priors | all | host |
| any model with `data = fit.retained` | none | the same outer solver on `D_R [beta_hat - V_R^T a(theta)]` (PAPER eq: compressed likelihood) | `R = max(R*, number of parameters)` | all | host |

The compressed likelihood takes the retained combinations as the data argument, in the words of PAPER:720. `StokesData` and `RetainedCombinations` both supply a whitened design and a whitened data vector, which is all a solver reads.

### 6.2 Rank and penalty policies

* `se.UPRE(cap=None)`: `R* = argmin [chi2(R) + 2R]` over ranks up to the numerical rank (`s > 1e-13 s_max`) or the cap. A truncation never splits a pair of equal singular values of a complex block (`PAIR_RTOL = 1e-9`), and the pair is rotated to a canonical basis (`ST/fits.py:122-229`).
* `se.Rank(R)` and `se.FullRank()` fix the rank.
* Free-moment fits work in lattice rows with the design `D S` (Sec. 4.4). Rows without a column contribute nothing and get zero. Merged Beaming pairs split equally (Sec. 4.5).
* `se.Penalty(alpha=None)`: the norm is `|a|^2` in the scaled moments, in lattice rows. `None` selects `alpha` by the UPRE of the same fit without positivity (PAPER:1146). `se.Penalty(0.0)` removes it.
* Maps: UPRE per pixel without a cap is the default. The summary prints the rank histogram and, with a cap, the number of pixels whose selected rank equals it and the cap's source (`[default]` or user). The paper's intensity-only fits cap the search at 14, because 2 to 6 per cent of fits otherwise retain 15 or more combinations. Its I, Q, U fits have no cap (PAPER:1077). That sentence is attached to intensity-only fits only.
* Non-negative cells at any order: the build computes every point-mass column on the requested channels. The summary reports how many are negative. With none negative the intensity is non-negative on those channels (PAPER:803). There is no order restriction.

### 6.3 Uncertainties and what the data constrain

| Estimator | Reported | Not included |
|---|---|---|
| whitened SVD | `fit.retained`: `beta_hat`, `sigma = 1 / s`, directions with their largest labelled moments. Prediction covariance `C' V_R D_R^-2 V_R^T C'^T` at any channels | the discarded combinations |
| whitened SVD with a penalty or a Gaussian prior (origin `penalised`) | as above with filter factors `f_i = s^2 / (s^2 + alpha)` | the shrinkage bias set by `alpha` or `s`, which `sigma` excludes |
| NNLS with penalty | weights, moments, effective parameters by the active-space rule (`ST/fits_positive_ridge.py:109-118`). On request `fit.profile_max(...)` and `fit.resample(n, key=...)` | a covariance, which is undefined at active constraints |
| separable, basis-parametric, nonlinear, compressed | Gauss-Newton covariance from `jax.jacfwd` (finite differences in `theta_b`), `at_bound` flags within `1e-5` of a bound, number of starts that reached the minimum. A `theta_b` direction tied by a structural relation, or whose Jacobian column is below the numerical-rank threshold, is reported by its profile `chi2(theta_b)` on the grid in place of a Gauss-Newton sigma | modes outside the starts |
| bilinear | covariance of the linear block at the fitted factor, curvature of the profiled `chi2` for the factor | the joint covariance of the two blocks |

* A free-moment fit constrains the `R*` retained combinations. `fit.moments` is the minimum-norm vector in lattice rows with `origin = "minimum_norm"`, and it carries no uncertainty (J2 Q12). Rows without a column in this basis are listed in `moments.masked` with the reasons of the column map, for example "parity: no column with this kernel". The flag travels with the vector. `basis.predict` of such moments records `unconstrained` as an unbounded ledger term with the PAPER:718 caveat. Every `Lattice` method that takes `Moments` (`column`, `raw`, `normalised`, `conditional`, `realisable`) returns `Observable`s with the origin and the caveat, and NaN on masked rows. `fit.predict`, `fit.column` and `fit.observable` give the numerical `unconstrained`.
* Origins by kind: free moments `minimum_norm`, with a penalty or a Gaussian prior `penalised`, bilinear fits and R3a children of `FreeMoments` parents `minimum_norm`, R3a children of `NonNegativeCells` parents `nonnegative_penalised`, moment-form fits `family`, basis-parametric fits the origin of the inner kind, non-negative cells `nonnegative_penalised`. `penalised` has the remainder rule, the caveat and the `unconstrained` field of `minimum_norm`.
* `fit.profile_max(channels)` refits the non-negative cells without the penalty and returns the largest prediction with `chi2 <= chi2_min + delta_chi2`, an `Observable` with origin `nonnegative_profile` and the noise-only label (PAPER:1148-1149). It is not a ledger bound.
* `Spectrum.unconstrained` is `|c - V_R V_R^T c| / |c|` for the design row `c` of each predicted value in the scaled moments (`ST/fits_extrap_noisefree.py:46`), so its stored values are tests. For a penalised fit it is `|c - V F V^T c| / |c|` with `F = diag(f_i)`, which reduces to the first form when every `f_i` is 0 or 1. It is the norm fraction of the prediction row outside the retained directions, and it depends on `w`. 0 means the data set the prediction. Values near 1 mean the prediction depends mainly on the minimum-norm convention. The number is not a fraction of the predicted value and cannot be turned into an error without assumptions about the discarded moments. On known truth the study also computes the share of the true value carried by the discarded directions (`ST/fits_extrap_noisefree.py:45`): 0.42 for E2 at 1 per cent noise, where the row fraction is 0.88.
* `fit.observable(c)` returns value, noise standard deviation and `unconstrained` for any linear functional. `fit.column` is the order-0 sum in that form, labelled as the effective column on a `FieldScaling` lattice (Sec. 4.2). `fit.electron_spectrum()` raises for free moments and points to families and non-negative cells (PAPER:718, 856).
* `fit.population` is `None` for free moments, the point-mass measure for non-negative cells, and the declaration with fitted values for families.
* Family parameters whose Jacobian column lies in the span of the others are listed with `identified = False`.
* All intervals come from the noise alone (PAPER:1148).

### 6.4 Batches of spectra

* Shared noise. `stokes` of shape `(n_pix, n_ch, n_stokes)` with shared `sigma` uses one SVD, then one product, one cumulative sum and one `argmin` per pixel under `jit` and `vmap`, in chunks of 1e5 pixels. The rank is a traced integer used as a mask. `directions` has no pixel axis.
* Per-pixel noise or masks. Each pixel has its own SVD. `StokesData.simulate` gives `sigma = noise * I_j` per pixel, so this is the common case. The directions are not stored by default. `fit.predict(channels)` and `fit.observable(c)` recompute the whitened SVD from `problem.data.sigma` (16.7 us per 63 x 57 SVD and 51 us per 63 x 171 SVD under JAX `vmap` on all cores, 177 and 611 us on one core, Sec. 8.7), contract `V_R` with the new design rows, and return `sigma` and `unconstrained` without holding `V`. `keep_directions=True` stores `directions` with a pixel axis at the memory given in Sec. 8.7.
* Chunks of per-pixel SVDs, in the fit and in these recomputations, are sized from `chunk_bytes` (default 2 GiB): `n_chunk = floor(chunk_bytes / (8 B (m n + m k + k n) 1.5))` for an `m x n` design with `k = min(m, n)`, the factor 1.5 for workspace. This gives 1.7e4 pixels at 63 x 57 and 7.0e3 at 63 x 171. A fixed chunk of 1e5 pixels would hold 8.3 and 20.4 GB (r3-41). The plan prints the chunk size.
* A result with a pixel axis carries per-value `sigma`. `Spectrum.cov` is filled only on request (`fit.predict(channels, cov=True)`), with the size printed first. It is 1.05 MB per pixel for 121 channels and 3 Stokes.
* Non-negative fits run on the host in the first release, in a process pool with BLAS pinned to one thread (`OMP_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`). On 111 x 272 this gives 0.51 to 0.71 ms per spectrum against 2.96 ms serial, and default BLAS threads oversubscribe the pool at 1.05 ms per spectrum (prototype profile, Sec. 8.7). Separable family fits on a batch of spectra run as a jitted `vmap` Levenberg-Marquardt [0.9]: 60 us per spectrum (40 iterations) against 2.39 ms for `trf`, with parameters within 4.3e-7 of `trf` (prototype profile). `trf` stays the path for one spectrum and the oracle in the tests. In the first release every family fit loops over pixels on the host (Sec. 10.1). No native code is added for these solvers (NATIVE.md). Pixels with different channel sets are grouped by set and use `basis.at`. Shared family parameters with one amplitude per pixel reduce to the Gram matrix `S = D_w^T D_w` (P4 Sec. 8.6) [0.9].

### 6.5 Prediction at other channels

`fit.predict(channels)` calls `basis.at(channels)`. The second basis shares the lattice, the column map, the reference weights and the specified statistics by construction, and stages S0 and S1 are reused. The result carries `sigma` (noise only) and, for a free or penalised fit, `unconstrained`. The caveat on extrapolation (Sec. 7.3) is attached when a predicted channel lies outside the fitted band.

A fitted lattice cannot be widened afterwards, because it sets the minimum-norm vector. The range is therefore set at fit time: `se.fit(..., predict=channels)` derives the default energy range from the union of fitted and predicted channels, as the paper's extrapolation lattice covers 300 GHz (PAPER:1146). `basis.at` re-runs the closed-form `EnergyWindow` of Sec. 4.6 rule 1 on the new channel supports, which needs no S0 or S1 table. Where a window edge falls outside the lattice range, each such channel gets an `outside_lattice` entry with the share carried outside by the reference density `gamma^-2.6`, labelled "estimate for the reference density, not of the source", and the remedy: refit with `predict=` or with a `lattice=` covering these channels. For the first screen of Sec. 2.1 without `predict=`, the share is 1.0e-3 at 23 GHz, 1.3e-3 at 30, 5.2e-3 at 100 and 1.8e-2 at 300 GHz (r3-23). For the 1 to 100.6 GHz default lattice (`[2455, 1.502e6]`, 14 cells) the estimate at 300 GHz is 3.6e-3 of I. The measured shares are 2.85e-3 for the bounded E1 power law and 5.2e-4 for `exp(-gamma/1e6) gamma^-2.6`, so the estimate is not a bound on another source (prototype partition). In a free fit the ledger term stays unbounded. For a specified population the measured `outside_lattice` term applies. With the check on, `basis.at` also re-measures `reference_cases` on the new channels, and every measured accuracy number is printed with its channel span. A kernel without `EnergyWindow` prints "range not checked on these channels".

### 6.6 Sampler hook

```python
prob = se.problem(population, data, lattice=lattice)
prob.params                                      # named pytree: labels such as "gamma.index1", "phi.factor", "a"
logp = eqx.filter_jit(prob.log_prob)             # pure JAX in the params pytree. grad and vmap apply
z    = prob.unconstrain(prob.params)             # unconstrained coordinates with the log-Jacobian in log_prob
chi2 = prob.profiled()                           # theta -> chi2 with the linear unknowns solved in closed form
```

`prob` is an `eqx.Module` whose plan record is static and whose lattice, moment map, basis, data and parameters are array leaves. New bounds, starts or data values therefore do not recompile (test in Sec. 8.8). With basis parameters the log likelihood runs on the host in 0.8 (Sec. 8.6). A `FreeMoments` block without a prior on a rank-deficient design gives an improper posterior along the null space (rank 26 of 57 columns for E1), so `log_prob` raises `ImproperPosterior` and names `prior=` and the compressed likelihood on `fit.retained`. No sampler is a dependency. For NumPyro the three lines are a `numpyro.sample` per parameter prior and one `numpyro.factor("syncem", prob.log_likelihood(params))`.

## 7. Results and honest defaults

### 7.1 Default printouts

A prediction prints the block of Sec. 2.1: header, unit, values, roles, assumed independence, basis, accuracy, terms that are not bounded. A fit prints (E1 values from PAPER tab: classical fit summary):

```text
Fit          free moments of gamma on 19 cells, order 2 (57 columns). 21 of 41 channels, I Q U, sigma = 0.01 I
data unit    erg s^-1 cm^-2 Hz^-1 sr^-1
basis        continuum kernel (V not modelled), Faraday rotation tau = 2 (c/nu)^2, bump channels of fractional width 8 %
             (h_j = 0.2 nu_j)                                                                          [default: kernel, phase law]
estimator    noise-whitened SVD in reference weights w = gamma^-2.6, rank by UPRE, cut 1e-13 s_max  [default: w, rank, cut]
fit          R* = 7 of numerical rank 26 (63 fitted rows). chi2 = 52.1 for 56 dof, p = 0.62
withheld     20 channels: chi2 = .. for 60 values, predictive chi2 = ..
constrained  7 retained combinations of the 57 moments (fit.retained). Discarded combinations are set to zero
             by convention. These zeros are not inferred values (PAPER:718). fit.moments has origin minimum_norm.
             w sets the minimum-norm vector and every extrapolation
specified    B ~ U[4, 6] uG, eta ~ U[-0.5, 0.5], mu isotropic, <exp(2i phi)> = 0.40 exp(0.50i), depth ~ N(8, 0.5^2) rad m^-2
             An acceptable chi2 does not confirm these statistics. Some misspecifications leave chi2 unchanged
             (field-strength law in the continuum limit, PAPER:833). A rejection tests them only jointly with the
             other assumptions (PAPER:836, 852)
assumed      B, eta, mu, phi, depth independent of gamma and of each other (separate keywords)
accuracy     measured on reference cases of this basis, 0.1-10 GHz: 8.6e-4 of I (power law 2.6), 3.6e-3 (cut-off inside a cell)
             Finite cases. They do not transfer to the source
not bounded  representation (no population declared), unconstrained, outside_lattice, physical_kernel, discrepancy, specified:*
caveats      A small residual shows that the basis approximates the spectrum in the fitted band.
             It does not establish the physical moments of the source (PAPER:856). Also attached (fit.plan.caveats):
             noise-only intervals (PAPER:1148), UPRE coverage 56-58 % (PAPER:1083), field scaling (PAPER:833)
```

This print is the reference rendering for the rules of Sec. 2.3, 7.3 and 7.5. A test checks that every entry of `PlanRecord.defaults` and `PlanRecord.caveats` appears in `str(fit)`.

### 7.2 Typed error terms

`Ledger` is a named tuple of `ErrorTerm`s with the five kinds of `PKG/syncem/model/errors.py`: bound, measured, estimate, unbounded, not_applicable. Each term also carries a static flag `declared`, an extension of `errors.py` in the records layer.

* `ledger.sum(kind)` sums within one kind. The default print groups terms by kind. Under `ErrorBudget.total()` every default output read "unbounded", and the print no longer leads with it.
* `ledger.envelope()` returns the sum of all valued terms, labelled with the weakest kind present (bound < measured < estimate), as `ErrorBudget.envelope()` does today. The computed terms are defined so that none overlaps another (a declared `discrepancy` envelope is user input and may overlap them): the signed `representation` and `outside_lattice` add to the difference from direct integration over the declared support, their reported maxima bound it, and no term holds that full difference. It returns `None` while any term is unbounded. It names the declared terms it depends on, and the print reads "conditional on declared: physical_kernel, ...".
* `ledger.allow(name, term, note=...)` requires a note, sets `declared=True` and writes an `AssumptionRecord` to the result. Declared terms print in their own group, "declared (user input, not computed)", and never in the bound group. `ErrorTerm.declared_zero` sets the same flag.

| Term | Default source, no user input | Kind |
|---|---|---|
| `representation`, forward | `basis.predict(pop) - basis.direct(pop, within="cells")` for a fully specified population, same kernel: the direct integral restricted to the cell boxes. Reported as `max |dX| / I_j` with its channel span | measured |
| `representation:angular` | the `Legendre(L)` pairing of `mu` samples under `Beaming` when `eta` has no density (Sec. 4.2). The rest of `representation` stays measured, basis against direct at the same `L` | unbounded, "samples smoothed by Legendre(L) on mu = eta" |
| `representation`, family fit | the same at the best-fit population | measured |
| `representation`, free-moment or non-negative fit | no population exists. `basis.reference_cases` are printed as finite cases | unbounded |
| `remainder` | by `Moments.origin`. Specified or family: remainder table times `|a|` of the retained even moment of each cell, using `|u| <= 1` (CTX 04 B.1). Non-negative penalised: the same on the fitted point-mass measure, labelled "of the fitted measure, not of the source". Minimum-norm or penalised: unbounded, "needs the true even moments of each cell. A free fit constrains only the retained combinations (PAPER:718)". Ledger only | estimate, or unbounded |
| `conditional_truncation:<child>|<parent>` | R3a: the shifted terms with `r + k` outside the parent lower set, bounded by `sum_b m_b E[b, j]` (Sec. 5.2). `E` is known before data | bound per unit parent cell mass, `estimate` with the fitted cell masses plugged in (labelled as for `remainder`), not_applicable for a specified parent and under `NonNegativeCells` |
| `depth_truncation` | rule 4 of Sec. 4.6: `B = x^(N_d+1) / (N_d+1)!` with `x = tau_max h`, `tau_max = 2 lambda^2` at the lowest frequency of the channel supports, `h` the largest depth half-width and `N_d` the depth cap of the cells. `B` bounds the depth part of `max |dP| / I` for any population inside the depth cells. The compile step raises `DepthExpansionError` when `B > depth_tol` (default 1e-3, an argument of every front-door call) | bound, or not_applicable without a depth axis |
| `quadrature` | node doubling of the kernel averages and of the channel quadrature on three probe cells (lowest, middle, highest), and of the moment quadrature | estimate |
| `quadrature:depth` | with depth as a parent of the azimuth factor (R5), depth is averaged by its `rule()` nodes in the basis and in `direct`, so `representation` does not contain this error. Where the screen depolarises, the nodes do not resolve the characteristic function. For depth ~ N(8, 0.5) rad m^-2 and 60 Gauss-Hermite nodes, `|P|` in the 0.1 GHz channel is 1e6 to 1e11 times the exact value, 1.6e-7 to 8.5e-7 of `I` (prototype R5) | estimate |
| `small_x` | accuracy of the continuum `F`, `G` evaluation at the `x_min` the adapter sets, and of the small-x asymptote that `basis.direct` uses above `gamma_t` (Sec. 8.3) | estimate |
| `provider_agreement` | default column provider against a second provider where both exist (Sec. 8.3) | measured |
| `outside_lattice` | `basis.direct(pop, within="cells") - basis.direct(pop)`, the negative of the emission of the declared population over its support outside the cell boxes, with the base kernel on the averaged statistics. It is computed by its own quadrature on the uncovered support (complement of the boxes, interior holes included) and not as a difference of two full-support integrals. With this sign `representation + outside_lattice = predict - basis.direct(pop)`. The print gives its magnitude as the uncovered share `|outside_lattice| / I_j`. `kernel.regime` is applied to the uncovered support, and where it reaches below the kernel limit the term prints the regime caveat. In a fit: unbounded until declared, with the reference-density share per channel at new channels (Sec. 6.5) | measured, or unbounded |
| `harmonic_truncation` | `required_m_max` per cell box | bound, or not_applicable |
| `azimuth_truncation` | `AzimuthFourier(m_max)` (R4a (b)): the Fourier orders `|m| > m_max` of `Kbar(phi)` of the averaged children that the rows drop, against direct integration over `phi` | measured, or not_applicable when `Kbar(phi)` of every averaged child is a trigonometric polynomial of degree at most `m_max` |
| `kernel_switch` | both kernels at each switch (Sec. 8.4) | measured |
| `noise` | noise propagated through the estimator | estimate |
| `unconstrained` | discarded combinations of a free fit. `Spectrum.unconstrained` gives the row fraction per value | unbounded |
| `specified:<variable>` | one per averaged statistic. It covers the declared law with its stated parent set, so the independence of the variable from every non-parent belongs to it | unbounded, or measured after `allow_mismatch=True` on the host. Inside a transformation: unbounded, "unmeasured (traced)" |
| `independence:<a>|<b>` | one per dropped dependence and per declared independence between unknown groups | unbounded, or measured for a specified population |
| `physical_kernel` | never computed | unbounded unless declared |
| `response` | `|K| |delta R| |S_hat|` with the plug-in `S_hat` (the fitted channel Stokes) and `K` the estimator of the fit, `declared=True` because `delta R` is user input. It is the response part of `delta` in `fit/_linear_core.py:80-120`. For a free-moment fit `K = S_R^-1 U_R^T L^-1` is the estimator of the `R*` retained combinations, and the term is the bias of `fit.retained`. The null-space part belongs to `unconstrained`. The existing code truncates at the `tol` rank and prints only a maximum, so this value is new code | estimate when `response_uncertainty` is nonzero. Declared zero when it is all zero. Unbounded when `response=` is set without it. not_applicable without `response=` |
| `discrepancy` | `|K| (|R| + |delta R|) E`, with `E` the Stokes-space envelope of `StokesData(discrepancy=)`, or `|K| E` for a data-space envelope. `K` is the same as for `response`. It holds the cross term `|delta R| E`, so `response + discrepancy` equals the combined `|K| |delta|` of the existing code. `E` is user input and may include errors that other terms also cover (`representation`, `remainder`, `physical_kernel`) | the kind of the declared `ErrorTerm`, with `declared=True`. Unbounded when none is declared |

Rules for the measured representation term.
* `predict - basis.direct(pop)` over the full support equals `representation + outside_lattice` channel by channel and per Stokes parameter, with `outside_lattice` signed as in the table. The identity holds to the agreement of the two quadratures (measured 1.7e-15 to 3.0e-15 of `I_j` with independent panels, 1.2e-12 on one cell of half-width 4.60). Every reported value of both terms and of the envelope is max over channels and I, Q, U of `|dX| / I_j` with `I_j` the full-support direct intensity, never the in-cell intensity, which is zero in channels that the cells do not reach. The reported maxima bound the total and do not add to it: their sum was 1.00 to 26.7 times `max |predict - direct| / I_j`, because the two terms can have opposite signs. `envelope()` sums the two maxima and never the full difference. A test checks the identity to 1e-11 of `I_j` for an unbounded `PowerLaw(2.6)`, a cut-off power law whose support extends beyond the lattice on both sides, a lattice with an interior hole, and a lattice that does not cover the channels (prototype partition).
* `within="cells"` integrates on quadrature panels that break at every cell face that meets the support. Masking the nodes of a face-blind quadrature is not allowed: on the E1 configuration it errs by 6.6e-7 (19 cells), 1.8e-4 (cells 2 to 16) and 2.0e-2 (a hole at cell 9) of `I_j`, more than the N = 2 representation term at the hole (prototype partition).
* The reference shares no truncation with the basis (J2 finding 5), except the `Legendre(L)` pairing of `mu` samples under `Beaming`, which both sides apply at the same `L` and which is the unbounded `representation:angular`. For cell variables and averaged variables it is the cell table restricted to the cell boxes (Sec. 8.3) when the table passes the check of Sec. 8.3 on the probe cells. With 17 nodes it reproduces kernel-call direct spectra to 4.7e-16 of `I_j` at half-width 0.242 and 1.9e-8 at 0.92, and misses by 5.0e-2 at 4.60 (prototype partition). Otherwise the reference is kernel calls on cell-face panels.
* For a population with a field distribution and a kernel with `FieldScalingRelation` whose role precondition holds (azimuth factor independent of `gamma` and `B` included), the reference is the fixed-field energy table at `B_0` combined with `Kbar(gamma, B) = (B/B_0) Kbar(gamma (B/B_0)^(1/2), B_0)` at each field node. It needs no new kernel calls, takes milliseconds, and needs the table range widened by `(B_max/B_min)^(1/2)` (tests T5, T13). The E6 term of G8 is measured this way.
* For Legendre and depth axes the reference is `basis.direct`, the empty-lattice construction, which applies the kernel's capabilities (the pitch density at `mu = eta` under `Beaming`). Under `Beaming` this holds for populations whose `mu` or `eta` has a density. Samples of `mu` with no density for `eta` have no truncation-free reference, and the angular part is `representation:angular`. An `L` against `2L` difference is not printed as an estimate, because the paired sum does not converge.
* The line states its case: "this population, same kernel". The same kernel is on both sides, so the term says nothing about `physical_kernel`.
* The sampled remainder stays out of the default print. For E1 to E4-JP it propagates to 9 to 20 noise standard deviations where the actual remainder gives 0.16 to 0.66 (PAPER:897, disabled block).
* `basis.reference_cases` holds the representation error for three reference populations on this basis: the power law of the reference weights, a smoothly broken power law, and an aged power law with its cut-off inside a cell. They are measured when the check is on and labelled as finite cases.

### 7.3 Fixed caveats

Caveats are fixed sentences attached to the plan by kind and capability, each with its source line.

| Attached to | Sentence | Source |
|---|---|---|
| free-moment fit (origin `minimum_norm` or `penalised`), and every result computed from its moments | Discarded combinations are set to zero by convention. These zeros are not inferred values. | PAPER:718 |
| intensity-only map fit with a rank cap | The paper capped intensity-only fits at rank 14. Its I, Q, U fits had no cap. | PAPER:1077 |
| every fit | A small residual shows that the basis approximates the spectrum in the fitted band. It does not establish the physical moments of the source. | PAPER:856 |
| prediction outside the fitted band | An acceptable in-band fit does not validate extrapolation, which depends on the assumed electron spectrum at energies the band does not constrain. | PAPER:813, 850 |
| every fit | Intervals come from the noise alone and exclude uncertainty in the specified statistics and in the electron family. | PAPER:1148 |
| rank by UPRE | The rank is selected from the data, which lowers the coverage of nominal 68 per cent regions (56 to 58 per cent in the paper's tests). | PAPER:1083 |
| bilinear | With an unknown shared factor either set of unknowns is determined only with the other held fixed. | PAPER:599 |
| non-negative cells | Non-negative cells give a non-negative intensity only where every point-mass column is non-negative. | PAPER:803 |
| family fit | Parameter values hold under the named family and the named specified statistics. | PAPER:827, 854 |
| `FieldScalingRelation` with a specified field and an azimuth factor independent of `gamma` and `B` | A common rescaling of field strengths is offset by rescaling electron energies, amplitude and energy range. The spectrum constrains only the invariant combinations. | PAPER:833 |
| `FieldScaling` axis | The column is the effective column at the reference field. Changing the reference field rescales the inferred energies and column. | PAPER:562-564 |
| an envelope with declared terms | The envelope holds only under the declared terms. | PAPER:892 |
| every fit | An acceptable chi2 does not confirm the specified statistics. Some misspecifications leave chi2 unchanged, and a rejection tests them only jointly with the other assumptions. | PAPER:833, 836, 852 |

### 7.4 How assumptions are recorded

* The basis: `basis.specified` holds one record per averaged statistic (variable, family type, parents, fingerprint, parameter values as array leaves). `basis.record` holds kernel per cell, phase law, provider and route per stage, quadrature and versions.
* The plan: `PlanRecord.roles` holds the declaration, the role and the deciding rule per variable. `PlanRecord.routes` holds each shortcut with its precondition. `PlanRecord.assumptions` holds every independence statement and every dropped dependence as an `AssumptionRecord` (`PKG/syncem/model/_provenance.py`), printed as the "assumed" row of every roles table and fit print. `PlanRecord.defaults` holds every applied default.
* The result: `Spectrum` and `Fit` carry the `PlanRecord` and the ledger. A prediction from bare `Moments` lists the statistics of the basis as assumptions of the result, and keeps the origin of the moments. `ledger.allow` adds an `AssumptionRecord`.

### 7.5 The project rule without user inputs

Reference model: kernel, phase law, channel response and unit, in the header. Simplified or excluded: the roles table, and the kernel notes (no absorption, no conversion, no beam, V not modelled by the continuum kernel). Validity range: cell ranges, orders, the regime and depth checks, the shortcut preconditions. Bounds, finite measurements, estimates, declared terms and unbounded terms appear in separate groups. Parts this package does not cover have named interfaces: `StokesData(response=, response_uncertainty=, discrepancy=)` for the observing response and the model discrepancy, `ledger.allow(name, term, note=)` for assumption terms, `physical_error` on the kernel.

## 8. Engine

### 8.1 Kernel protocol, averaged kernel and capabilities

The protocol is in Sec. 3.2. It is the existing `KernelModel` (`PKG/syncem/model/kernels.py:59-100`) with the variable list made explicit, the Legendre projection moved to the engine, and shortcuts declared as typed capabilities with preconditions that the plan checks and the engine tests. The existing kernels are adapted behind it, and `KernelModel` stays as an adapter.

`AveragedKernel`. `engine.average(kernel, distributions)` returns the function `z -> Modes` that every provider differentiates or tabulates.
* `z` holds the local chart coordinates of the `d` celled variables of one product cell, each in its own chart (log or affine) around the cell's reference point. `d` is any number.
* The function maps `z` to a point, evaluates `kernel.channel` there (or `kernel.spectral` with the engine's channel quadrature), averages the averaged variables by quadrature with conditionals allowed (nested where a parent is averaged), applies the phase law, and projects `Legendre` axes onto their degrees.
* Variable names come from `Kernel.variables`. The only regime check is `kernel.regime`. No variable is assumed to exist, and no reference value is required to exceed a limit.
* A depth axis enters through the closed-form Taylor weights of the phase law (`PKG/syncem/model/phase.py:122-170`) in place of differentiation when no averaged law has depth as a parent. Otherwise the weights are combined with the depth Taylor coefficients of the averaged kernel (Sec. 4.3).
* The outermost average is mapped in batches (`outer_batch`, a provider setting), and the inner levels are vmapped. `vmap` at every level was killed by the OS at `N = 4` with 32 `eta` and 16 `B` nodes, and `outer_batch=1` runs (prototype beaming).

Kernel requirements of the engine.
* A per-node output mode returns spectra at the channel quadrature nodes before the channel integral. The closed-form depth weights act inside the channel integral, and turning `J` channels of `T` nodes into `J T` one-node channels multiplies the jet's channel count (prototype depth).
* A kernel option `phase=False` gives the phase-free kernel of Sec. 4.3. Capability hooks carry a factor on the spin-2 block and `Beaming` (factor 2 and evaluation at `mu = eta`). The prototypes needed six wrapper kernels for these (prototype R5, depth, beaming).
* A kernel that the jet differentiates has a jet-compatible path (Sec. 8.3, decision (a)). The cubic Hermite `F`, `G` table serves forward evaluation only and is unsuitable for jet derivatives beyond second order (prototype profile).

| Capability | Declared by | Use | Precondition checked by the plan |
|---|---|---|---|
| `Similarity(variables=("gamma",))` | continuum | the kernel is `B_perp W(nu / (a_1 gamma^2 B_perp))`. One table of `W` and `(-x d/dx)^r W` serves all cells, and energy derivatives are analytic (`ST/lib/kernels.py:48-99`, `ST/lib/basis.py:45-113`) | cached S1, checked per Stokes block: the field law has no parents and shares no joint `Distribution` block with another averaged variable. No averaged law, conditional or azimuth factor that enters the block has an averaged `B` as a parent. An azimuth factor, and a joint block holding `phi`, enter the spin-2 block only. With an averaged `B` as a parent or in the block, the spin-2 block takes the field average in S2 at the field nodes (R5 (iii)), and the spin-0 blocks keep the cached S1. An azimuth factor whose parents are constant, in `Cells` (R5 i, ii) or in `eta`, `mu` or depth may stay. The inclination, pitch and depth laws have no parents among averaged or `Legendre`-resolved variables, the `Beaming` tie `mu = eta` excepted. Parents in `Cells` go to the per-node mode. Every `Similarity` variable is celled in the product cell (an averaged `gamma` goes to the jet in 0.8.0 and to a per-node mode over its nodes in 0.9). Every celled variable of the product cell is in `variables` or has a declared column route: `FieldScalingRelation` for `B`, phase weights for depth, `Legendre` and `Beaming` projections inside the `eta` average. Otherwise the per-node mode over averaged parents, or the jet for the whole product cell (Sec. 8.3) |
| `FieldScalingRelation` | continuum | `Kbar(gamma, L B) = L Kbar(L^(1/2) gamma, B)` (PAPER eq: field scaling): field-derivative columns from energy columns (`ST/ensemble_e6.py:97-129`), the `FieldScaling` role, the scaling-table reference of Sec. 7.2, the structural identifiability note | field-derivative columns on `B` cells: energy cells in the same product cell, pitch, inclination and depth laws independent of `gamma` and `B`, and no averaged variable a parent of the field law (PAPER:562, 661). A field law conditional on resolved variables (`B=lambda gamma: ...`, or `P(gamma given B)` of E6) changes the moments only and keeps these columns. Column (r, s) of a cell needs the energy columns at `gamma_b (B_b/B_0)^(1/2)` to order r + s, so the energy columns are evaluated to the largest r + s of the cell lower set, which exceeds the energy cap when `N_gamma < order`. A resolved azimuth factor `c(gamma, B)` multiplies these columns by the per-cell Cauchy product (R5 (ii)). The azimuth factor enters at S3 after these columns. The `FieldScaling` role, the scaling-table reference, the column relations of `_relations.py` and the PAPER:833 caveat also need the azimuth factor `<exp(2i phi) | parents>` independent of `gamma` and `B` (PAPER:562, "field direction"). Otherwise `B` columns come from the jet provider and the role falls back (R5) |
| `Beaming(variable, partner)` | continuum | `K_nat -> 2 K_iso(eta) delta(mu - eta)` (Sec. 4.2). Pitch-Legendre columns are `eta` weights `(2l+1) P_l(eta)` (`ST/lib/basis.py:186-191`). With `eta` averaged by a density, `Cells` on `mu` take columns from `g(mu) = 2 p_eta(mu) K_iso(eta = mu)`. With `eta` in `Legendre(L)`, from `(2k+1) P_k(mu) K_iso(mu)`. Replaces the forced `L_mu = 0` of `_continuum.py:89-90` | the role of `mu` is admitted by the role of `eta` (Sec. 4.2) |
| `Parity(even, odd)` | harmonic, continuum | per-block masks of Legendre pairs and degrees (Sec. 4.3). Half-range quadrature of an even averaged cosine for the even blocks (Sec. 2.3) | the full averaged integrand is invariant under the joint reversal, checks (a) to (d) of Sec. 4.3: every averaged cosine has an even law, every conditional or azimuth factor with a cosine parent declares even parity, under `Beaming` an averaged pitch density is even at `mu = eta`, and the partner of a halved or masked cosine is even-averaged over its full range or in the beaming limit, `Legendre`-resolved or absent. At most one cosine per block is halved. `Cells`, point masses, samples and `Ray` steps on the partner force the full range and no mask |
| `EnergyWindow(window)` | continuum | default energy range without S0 or S1 (Sec. 4.6). `window(averaged, channels, *, density, eps) -> (lo, hi)` in the kernel's scaled variable | the field support is bounded or the field law declares a truncation |
| `ReferenceDensity(variable, density)` | optional | default reference weights on that axis (Sec. 4.5) | none |
| `Lines` | harmonic | channels must be smooth to order `N + 1`. Harmonic cutoff from the cell box | none |
| `AnalyticTaylor` | harmonic | `HarmonicKernel.angular_taylor` (`_harmonic_taylor.py:190-344`) | none |
| `HostOnly` | high-order harmonic route, external codes | not traceable. Columns through the cell table only | none |

The phase law is a separate object of the base model. Any law `exp(i tau(nu) depth)` keeps the closed-form Taylor weights and the characteristic-function route, each under its precondition (Sec. 4.3, R4). An independent foreground screen multiplies the weights of a resolved depth [extension] (J2 Q28).

### 8.2 Stages and cache keys

| Stage | Product | Depends on (cache key) | Measured cost |
|---|---|---|---|
| S0 primitive | `W^(r)(ln x)`, `r = 0..6`, uniform grid | kernel source hash, grid, order | 0.38 s once (NumPy, prototype profile) |
| S1 field average | `T^(r)(ln y) = <B W^(r)(y / B)>_B`, by lookups in the S0 table at `ln y - ln B_q` for the field law's nodes `B_q`, summed with its weights, in place of a quadrature per field node | S0, the S1 method (`lookup` or `quadrature`, decision 140), the field distribution, per Stokes block, when it has no parents, shares no joint block with another averaged variable, and no averaged law, conditional or azimuth factor of the block has an averaged `B` as a parent (an azimuth factor with constant, `Cells`, `eta`, `mu` or depth parents may stay, and one with an averaged `B` parent removes the S1 table from the spin-2 block only). With `B` given an averaged or `Legendre`-resolved `eta`, or any other parent: one-node tables keyed by method and field node, read from the shared S0 (decision 144). Entries for a trial value of a `Free` parameter stay in a memory LRU (Sec. 8.5) | about 6 ms per field node by S0 lookups: 4.5 ms at 1 node, 48 ms at 8 (prototype profile). Step 5a at load 19: 7.3 ms at 1 node, 34 ms at 16, 0.17 s at 96 (1.8 to 2.1 ms per node above one node), plus one S0 build of 0.49 s per memo. The per-node quadrature it replaces costs 0.32 to 0.34 s per node in NumPy (0.35 s at 1 node, 2.54 s at 8) and 36 ms (1 node) and 248 ms (8) in a JAX port. The lookup table equals the quadrature table to 1.7e-13 in `ln T` and to 3.3e-10 of the peak of `|T^(r)|` at `r = 6` (step 5a, one node at 4 uG). Both tables carry the S1 spline error in `ln y`, 3.4e-9 in `ln T` between knots and below 7e-10 of the peak at every order to `r = 6` for 1 to 16 nodes. Pointwise relative errors of `T^(r)` near its sign changes measure the distance to the root, not the table (decision 140 of step 5a corrects the prototype figures of 3.7e-8 and 1.8e-7, which were of that kind). The quadrature S1 stays as the oracle path (`s1="quadrature"`) for the 1e-10 acceptance tests of Sec. 9.6. The plan multiplies by the node count and by `2^k` for `k` doublings. Then disk |
| S2 channel columns, one row per channel | columns per cell to the built order, or the cell table | S1 or kernel, content hash of that channel's response, phase law, depth distribution, the other specified statistics, conditionals of averaged variables on celled or averaged variables (per-node mode, `phi` given `B` included), an azimuth factor of route R5 (iii) or (iv), cell geometry, provider settings | stack, 19 cells x 41 channels: 0.15 s (`N = 2`) and 0.21 s (`N = 4`) at 400 channel nodes, 0.026 and 0.037 s at 64. Jet at 64 nodes: 6.7 s and 58 s (profile p2, quiet machine). 1.5 to 1.8 s for the 17-node table |
| S3 assembly | reference weights, azimuth factor of route R5 (i) or Cauchy product (ii), field-derivative columns where the precondition holds, repetition of spin-0 spectra across depth cells where its precondition holds, component sums, cell selection, Stokes layout, unit row scaling | none | microseconds |
| S4 factorisation | whitened SVD of the selected rows | S3 choice, noise pattern (memo) | below 1 ms |

In no key: the reference weights, an azimuth factor of route R5 (i) or (ii), the electron column, the resolved part of the population, the data, a cell selection, the output unit, and an order at or below the built order. A higher order runs S2 for the new columns only. Adding channels computes the new rows only. A subset of channels is a row selection, so the 21 fitted and 20 withheld channels of the paper share one basis.

### 8.3 Column providers

| Provider | Applies to | How | Cost on the paper configuration | First release |
|---|---|---|---|---|
| jet | any traceable kernel, any number `d` of celled variables, and every product cell with a celled variable outside a shortcut's declared set | nested forward mode over the lower set of multi-indices on the `AveragedKernel`, or `jax.experimental.jet`. The `taylor_tensors` loop of `_basis_taylor.py` works for any `d` once seeded with `jnp.zeros(d)`. `multi_indices`, `lower_closure`, `_DIRECTIONS` of `point_taylor` and `columns` are generalised from `(r, s)` to `d`-tuples | about `prod_i (1 + |D_i|)` primals per cell over the nesting levels `i` (decision 134), with an E1 averaged-kernel primal of 0.19 to 0.24 s after jit. `d = 1` (E1, 19 cells): 0.35 s per cell at `N = 2` and 3.0 s at `N = 4` (6.7 s and 58 s in all, compile 0.3 s and 2.4 s), 260 and 1570 times the stack at the same 64 channel nodes, on about 15 cores against one. `d = 2` (`gamma`, `B`, 57 product cells, 12 `eta` nodes, no `B` average): 0.048 s per cell at `N = 2`, 1.37 s at `N = 4`. `d = 3` (`gamma`, `B`, depth celled, `eta` averaged on 12 nodes, total 2, 8 rows): 0.245 s per cell plus 1.04 s compile (prototype core, fieldcells, profile) | JAX. The general route and the oracle of every shortcut |
| core | `(gamma, B)` cells of the synchrotron kernels | the existing jitted `_build_core` per cell (nested forward mode, or `angular_taylor`), then the lower-triangular re-expansion `T[n, r] = h^n r! S(n, r) / n!` from the affine to the log coordinate (P5 Sec. 4) | 1.05 to 1.15 s for the first cell, then 0.39 to 0.5 s per cell at 64 channel nodes, about 2.5 s at 400 nodes. One compilation for all cells | adapter around unchanged code. Fast path and oracle of the jet at `d <= 2` |
| stack, cached S1 | the continuum kernel (`ContinuumKernel` itself, decision 144) where the cached-S1 precondition of Sec. 8.1 holds, every `Similarity` variable is celled, and every celled variable is in `Similarity.variables` or has a declared column route | analytic derivative stack of S0 and S1 at each reference point | 19 kernel points. 0.15 s at `N = 2`, 0.21 s at `N = 4` at 400 channel nodes, 0.026 and 0.037 s at 64 | host NumPy port of `ST/lib`. JAX port in 0.9 |
| stack, per-node conditional | the continuum kernel (decision 144) with an averaged law that depends on a celled variable or on another averaged variable, on frozen nodes | parent in `Cells`: the nodes are frozen at the cell's reference point, `x_q = rule(parents=ref_b).nodes`, and the weights reweighted, `w_q(parent) = w_q(ref_b) density(x_q | parent) / density(x_q | ref_b)`. Then `Kbar^(n)[b] = sum_q sum_{r+k=n} K_q^(r) w_q^(k)[b]`, with `K_q^(r)` the stack column at node `q` and `w_q^(k)[b]` the Taylor coefficients of the reweighted weights (`jax.jacfwd`). The plan checks that the reweighted weights reproduce the rule's own weight jet node by node when the nodes are fixed, and the Taylor coefficients of its moving-node moments `sum_q p_q t_q^m`, `m = 0..3`, to 1e-8 of `max(1, sum_q |w_q^(k)[b]| |t_q|^m)` when they move. It refuses weights amplified beyond `max_k sum_q |w_q^(k)[b]| = 1e4`, and otherwise takes the jet (decision 139). Laws with parent-dependent bounds (`Uniform(lo(g), hi(g))`, `Maxwellian(rms(g))`) go to the jet unless they declare `fixed_nodes`. `B` given an averaged or `Legendre`-resolved `eta`: one-node S1 tables per field node from the shared S0, with the `eta` or Legendre weights applied after the field average (decision 144). `eta`, `mu`, depth or `phi` given `B`: no S1 table for the affected blocks, the field average is taken in S2 at the field nodes, each node weighted by the conditional's weights there (by `c(B_q)` in the spin-2 block for `phi`. When `c` also has celled parents, its Taylor coefficients in them at each node enter the same sum as `w_q^(k)[b]`). No kernel autodiff | one leaf per node pair of the walk. Step 5a at load 19, `B` (16 nodes) given `eta` (12 nodes), 19 cells at `N = 4`, 41 channels of 400 nodes: 192 one-node tables in 1.2 s once per memo by S0 lookups (42 s by quadrature), then 2.2 to 2.4 s per cell for 192 leaves, against 26 ms per cell in cached-S1 mode (decision 144) | host NumPy |
| table | bounded cells with one or two table variables. Required for `HostOnly` kernels and `PointMasses` | averaged kernel on 17 Chebyshev-Lobatto nodes per cell. Taylor columns by differentiating the interpolant | 323 points, 1.5 to 1.8 s. Columns equal the stack to 1.2e-13 (`r = 2`), 1.4e-12 (`r = 4`), 1.7e-11 (`r = 6`) at half-width 0.242. Against the jet at the same channel quadrature: 3.0e-14 to 2.9e-13 (`r = 2`), 1.3e-12 to 3.3e-12 (`r = 4`) and 2.8e-11 to 3.1e-11 (`r = 6`) of the channel column maximum. 9.0e-4 at half-width 0.92 (prototype partition) | host. Built on first use by the accuracy check, then cached |

Decisions.
* The default source of basis spectra is the stack where its precondition holds (cached S1 or per-node mode) and every celled variable is in its declared set, the jet for other traceable kernels and for every product cell with a celled variable outside that set, and the table for host kernels. Providers are not mixed within the monomials of one product cell. For `B` columns of `Similarity` kernels the field-scaling relation is used where its precondition holds, with energy references shifted by `(B_b/B_0)^(1/2)` and weight `B_b/B_0` for every field cell `b`. This needs energy cells in the product cell and energy columns to the largest r + s of its lower set. Field cells with `gamma` averaged go to the jet with `gamma` on nested quadrature in 0.8.0, recorded as "gamma averaged: Similarity variable not celled -> jet". In 0.9 a per-node stack mode computes `(d/d ln B)^n K` at `(gamma_q, B_b)` as `(1 + (1/2) d/d ln gamma)^n` applied to the analytic stack at each `gamma` node and sums with the node weights. This needs no derivative of the electron law when its nodes and weights do not depend on `B`. A `gamma` law with `B` as a parent (a cut-off from the local field) takes the jet, because the derivatives of its nodes and weights enter the columns. Frozen at `B_b`, the per-node sum is off by 1.2e-2 at n = 1 in the lowest of four log field cells on [0.5, 50] uG and by 1.8 in the highest. The averaged kernel is not a power law in `B`, so `L^((s+1)/2)` scaling of the averaged columns is not a route: it is off by 4.0e-3 on [300, 2.96e6] and by 0.15 on [300, 3e4] (prototype fieldcells). The record and `PlanRecord.routes` name the provider, the precondition and the reason, for example "eta celled: outside Similarity.variables -> jet".
* The table is a second cached product. It supplies the direct integral for the measured representation term, the scaling-table reference (Sec. 7.2), the remainder table, `reference_cases` and the `PointMasses` columns. Provider comparisons use one channel quadrature on both sides. At 64 against 400 channel nodes P differs by 1.4e-8 of the channel maximum for any provider. Two checks apply at r <= 2. (a) Derivative providers (stack, jet, core) against each other: the build fails above 1e-10 of the channel column maximum, the maximum over all cells and orders of that channel (measured at most 4.0e-12 for half-widths 0.023 to 4.60), and it warns above 1e-8 of the cell's own peak at the same order (measured at most 2.3e-10 for r <= 4 and 1.0e-9 at r = 6). (b) The 17-node table against the default provider, against the channel column maximum only: the build fails above 1e-10 (measured 2.9e-13 at half-width 0.242 and 2.3e-14 at 0.023, 9.0e-4 at 0.92 and 0.41 at 4.60). A table that fails (b) on the probe cells is not used as the representation reference or for the remainder table, and the record says so. No own-peak test applies to the table: in E1 cell 1, an exponential tail at every 0.1-10 GHz channel, the table misses its own peak by 3.9e-5 (I) and 1.2e2 (P) at r = 1 and by 4.0e-6 (I) and 6.6 (P) at r = 2. `AccuracyCheck(own_peak_tol=...)` applies to check (a) (prototype partition).
* In an exponential-tail cell the 17-node interpolant of the averaged kernel gives Taylor columns at r = 1, 2 within 3.9e-5 and 4.0e-6 of the cell's own I peak and 1.2e2 and 6.6 of its own P peak (E1 cell 1). The r = 0 column is a node value and is exact. The core route is at 7.5e-9 of cell 1's own peak (J3 C1). Tail cells carry little weight relative to the channel maximum. Fewer than 17 nodes are not offered in the first release.
* The jet and core providers are about 260 times slower than the stack at `N = 2` at equal channel quadrature, and the jet is about 1570 times slower at `N = 4` (profile p2 on a quiet machine. J3 C1 estimated 350 times). They are not the default build of the continuum kernel. They run in the test suite on all cells and under `check=se.AccuracyCheck(providers=True)` on three probe cells.
* No contract limits the number of celled variables. The jet seeds `d` directions, and its cost grows with the size of the lower set, which the plan prints.
* The continuum adapter sets `x_min` of `_continuum.py` from the lowest channel-support node, the top lattice edge and the largest `B_perp` of the field support, records it in the kernel record, and writes the small-`x` accuracy of `F` and `G` at that value to the ledger term `small_x`. The default `x_min = 1e-6` raises in the last paper cell at the 0.1 GHz channel, where `x` is about 5.9e-7 (r2-36). `basis.direct(pop)` over a support above the gamma at which x reaches x_min at the lowest channel-support node and the largest B_perp uses the small-x asymptote F, G proportional to x^(1/3), Kbar(gamma) = Kbar(gamma_t) (gamma/gamma_t)^(-2/3), above gamma_t. The kernel raises below x_min, so an unbounded support cannot be integrated by kernel calls alone. For the E1 statistics with x_min = 1e-9, gamma_t = 3e7. The support above it carries 8.9e-8 of I_j at 10 GHz and 4.2e-6 at 300 GHz, and the asymptote agrees with kernel calls on [1e7, 3e7] to 9.6e-11 (0.1-10 GHz), 6.1e-9 (1-100 GHz) and 4.4e-8 (300 GHz) of I_j. The difference is written to `small_x`. The lower support below the kernel limit is integrated with the base kernel and prints the regime caveat (prototype partition).
* Direct integration (`basis.direct`, also `within="cells"`) integrates on quadrature panels that break at every cell face that meets the support (Sec. 7.2). Masking the nodes of a face-blind quadrature errs by 6.6e-7 to 2.0e-2 of I (prototype partition). It offers per-node output and a node mask without retracing, and it evaluates the kernel once per distinct non-`phi` coordinate, since nodes that differ only in `phi` share the kernel value (prototype fieldcells, R5).
* (a) The jet provider takes Taylor coefficients of one-parent conditionals (in step 7, decision 137) by a Taylor-mode path where the kernel allows it. Nested `jacfwd` compile time grows about 3 times per order (2.7 to 101 s at orders 1 to 4, order 6 unfinished after 10 minutes), and Taylor mode gives the same `h` in 3.4 to 17.8 s at order 3. `jax.experimental.jet` does not run on syncem's `F`, `G` today (`eqx.error_if` and a broadcast `jnp.where`), so a jet-compatible kernel path is a precondition. The production jet also uses the cap-aware nesting of `_harmonic_taylor.level_plan`, which the prototype jet did not. (b) Orders above 4 under the jet need a cost estimate in the plan: an N = 6 build took 3187 s with a 14 min 46 s compile, and an N = 8 build was stopped unfinished. (c) A smooth tabulated `F`, `G` (piecewise Chebyshev in `ln x`) is the candidate to cut the jet primal. The cubic Hermite table is adequate for forward evaluation (1.1e-11, 45 times faster) and not for derivatives beyond second order (prototype R3a, partition, beaming, profile).
* In production a `Ray` takes the basis route on a lattice (R8): 3.7 ms per 300-step ray under `vmap` over 1000 rays, plus the columns once. Its direct reference (R8) evaluates the tabulated `F`, `G`: 2.5 s per ray, against 98 s with quadrature `F`, `G`, extrapolated from 30 steps (prototype profile).

### 8.4 Kernel per cell and the check at the switch

`kernel=None` applies the regime rule of Sec. 2.3 per cell. `se.HybridKernel(*kernels, assign=...)` states the assignment per product cell, and `switch={"gamma": 300}` is a threshold rule that fills `assign`. The plan resolves `assign` eagerly into a static tuple with one kernel index per product cell, stored in `BuildRecord`, in the basis key and in `PlanRecord.routes`. The cell table and the lattice key do not hold it, so moments compare across kernels. The kernels must declare the same variables, and the Stokes blocks are those every kernel provides. On every pair of product cells that share a face and differ in kernel, and on the lowest and highest cells where both kernels are valid, the engine evaluates both kernels directly, bypassing the assignment, for every channel and block. The largest relative difference is the measured term `kernel_switch`. Below `1e-3` passes, below `1e-1` warns, above refuses, with `switch_tol` as the argument. The paper's comparison on twelve cells with `gamma >= 300` gives 1.8e-7 of `I_j` (CTX 06 F.1).

At microgauss fields `required_m_max` is 4.1e8 for `gamma <= 20` at 0.12 GHz, against the limit 16 368 of the JAX harmonic kernel (J3 C15). A hybrid by energy with harmonic cells therefore serves strong fields or low frequencies only, and harmonic entries are deferred to 0.9. `assign` among traceable non-harmonic kernels (the continuum kernel, `@se.kernel` kernels) needs no traced mask and is in 0.8.0. A harmonic lattice recompiles per cell today, because `HarmonicKernel.for_support` writes static fields per support (`harmonic.py:277-324`). The stated change is a traced mask over harmonics up to a block maximum, with cells bucketed by powers of two of `required_m_max` (P4 Sec. 8.4).

### 8.5 Caching and serialisation

* Builds are memoised in the process on a content key: kernel record, phase law, lattice structure and geometry, column-map key, the tabulation hashes of the specified statistics (Sec. 5.6), provider settings, package version. The reference weights are not in this key, so a change of `w` re-runs assembly (S3) only. Channel rows are keyed one by one.
* The disk cache holds stages S0 and S1 of fixed specified statistics as one `.npz` and one JSON description per key (the tabulation hash of Sec. 5.6), written to a temporary file and moved (pattern of `ST/lib/basis.py:116-145`). It lives under the user cache directory, moved by `SYNCEM_CACHE`, disabled by `cache=False`. Its size is at most 2 GB by default (`SYNCEM_CACHE_MAX`), and the least recently used entries are removed first.
* S1 tables for a trial value of a `Free` parameter (kind basis-parametric) are not written to disk. They stay in a memory LRU of 8 entries, about 6 MB each, as `ST/lib/basis.py` does with `lru_cache(maxsize=8)`. A build without a tabulation skips the memo and the disk cache (Sec. 5.6). There is no other module-level state.
* `basis.save(path)` writes arrays as `.npz` and the records as strict JSON. `SpectralBasis.load` checks the key and the package version. A loaded basis predicts and fits. `basis.at(channels)` after a load needs the functions of its conditionals again unless they are registered.
* Cells with the same representation signature form one block that is evaluated with `lax.map` in padded batches, so the cell count is not compiled in.

### 8.6 JAX properties

| Operation | jit | vmap | grad |
|---|---|---|---|
| `se.plan`, defaults, unit and regime checks, `EnergyWindow` | eager | | |
| `SpectralBasis.build` | eager shell, jitted jet per signature. Host code for the stack, the per-node mode and the table in 0.8 | over cells | jet provider: in reference points and in parameters of specified statistics (array leaves of `SpecifiedRecord`) |
| family moments `a(theta)` | yes | over parameter batches and pixels | yes, forward mode |
| `basis.predict(moments)` | yes | yes | yes |
| free-moment and penalised fits | yes, rank as a mask | over pixels | no (the rank is an integer) |
| non-negative, separable, bilinear (both kinds), basis-parametric fits | host in 0.8, residual and Jacobian jitted | process pool with BLAS pinned to one thread per process (`OMP_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`). Default BLAS threads oversubscribe | Jacobian by `jacfwd` (finite differences in `theta_b`) |
| batched family fits (separable kind over many spectra) [0.9] | jitted Levenberg-Marquardt in transformed coordinates after the agreement test T12, with host `trf` as the oracle | over spectra | Jacobian by `jacfwd` |
| `problem.log_prob` | yes, with `eqx.filter_jit`, when `basis_params` is empty. Host in 0.8 with basis parameters. JAX in 0.9 with the JAX stack port | yes, same condition | yes, same condition |

Static: variable names, charts, orders, cell counts, index, column map, kernel configuration, plan record, markers, keys. Traced: cell-table arrays, weights, specified parameter values, channel arrays, basis arrays, moments, data, parameter values and bounds. A changed bound or data value does not recompile, and a changed marker structure does (J3 C6). `Lattice.key` is static and hashes the geometry and the resolved weights, so a new geometry or a new `w` retraces jitted functions once. Bases are memoised per key, so this happens once per basis. A lattice built inside a transformation has `key = None`, and the comparison rule of Sec. 3.3 applies: `predict` and `fit` refuse a `None` key against a key unless `Lattice.with_key(key)` supplied one.

Pairing of `Samples` and `Ray` steps with the cell table locates nodes by `searchsorted` per axis and adds by `segment_sum`. A (nodes x cells x d) boolean array takes several GB for 19 x 843 cells and 1e5 nodes, and a Python loop over cells takes 2.05 s per 300-step ray against 3.7 ms for a JAX `segment_sum` (prototype depth, profile). Pairing and moment functions are jit-compatible and return traced arrays inside fits and rays, not dense host arrays computed eagerly (prototype core).

### 8.7 Costs

Times are measurements unless the row says estimate. Sizes and counts are computed, not measured. J, CTX and r sources are study measurements, and prototype sources are the prototype run.

| Item | Value | Source |
|---|---|---|
| `import syncem` with the lazy front door | JAX and Equinox 0.32 s, package 0.01 s | J3 C10, not re-measured |
| basis of the paper configuration after the S1 table (stack, 19 cells) | 0.15 s (`N = 2`) and 0.21 s (`N = 4`) at 400 channel nodes, 0.026 and 0.037 s at 64. The study gives 0.15 to 0.18 s | prototype core, CTX 06 A.5, J1 |
| S0 table | 0.38 s once (NumPy) | prototype profile |
| S1 build | about 6 ms per field node by lookups in the S0 table at `ln y - ln B_q`: 4.5 ms (1 node) to 48 ms (8). The per-node quadrature it replaces: NumPy 0.32 to 0.34 s per node (0.35 s at 1 node, 2.54 s at 8), JAX port 36 ms (1 node) and 248 ms (8). Step 5a, load 19: 7.3 ms (1 node), 34 ms (16), 0.17 s (96), S0 once in 0.49 s. The lookup table equals the quadrature table to 1.7e-13 in `ln T` and 3.3e-10 of the peak at `r = 6` (decision 140) | prototype profile, step 5a (loaded) |
| basis-parametric fit | one S1 rebuild per trial value of `theta_b`, plus one per finite-difference step: about 0.2 s per trial with S0 lookups | estimate from the per-node lookup cost, prototype profile |
| `B` given an averaged `eta` | one-node S1 tables per field node, once per memo: 1.2 s for 192 nodes by S0 lookups, 42 s by quadrature. The columns then cost 2.2 to 2.4 s per cell (192 leaves at 16 by 12 nodes, `N = 4`), the cost item of this case | step 5a, load 19 |
| E1 averaged-kernel primal | 0.24 s after jit (41 channels x 64 nodes, 16 `B` x 12 `eta`, 128 `F` nodes), cpu/wall 15. 1.4 s with 400 channel nodes. 0.12 s with 64 `F` nodes (values change by 6e-15) | prototype core |
| jet provider at `d = 1`, E1 | `N = 0`: 0.26 s per cell (loaded machine). `N = 2`: 0.35 s (6.7 s for 19 cells, compile 0.3 s). `N = 4`: 3.0 s (58 s, compile 2.4 s). 260 times (`N = 2`) and 1570 times (`N = 4`) the stack at 64 channel nodes. The first prototype runs, under load, gave 0.55 s and 3.7 s per cell. The core provider takes 0.39 to 0.5 s per cell (Sec. 8.3) | prototype core, profile |
| jet provider at `d = 2` | (`gamma`, `B`), 57 product cells, 12 `eta` nodes and no `B` average: 0.048 s per cell at `N = 2`, 1.37 s at `N = 4`. Field-scaling route at the `d = 1` cost: 0.029 and 0.212 s | prototype fieldcells |
| jet provider at `d = 3` | 0.245 s per cell run plus 1.04 s compile, 4.4 primals of 56 ms (`gamma`, `B`, depth celled, `eta` averaged on 12 nodes, total 2, 8 rows) | prototype profile |
| jet provider above `N = 4` | `N = 6`: 3187 s with a 14 min 46 s compile. `N = 8`: stopped unfinished | prototype partition, beaming |
| accuracy check | 1.5 to 1.8 s once for the table, then 0.6 to 1.6 ms per population. By kernel calls in the study 1.0 to 2.4 s, 13.6 s for E6. By kernel calls on the quadrature kernel, the cost of the reference without the study's table: 0.23 s per node, 104.5 s for 456 E1 nodes, 183 s for E6 over 1728 nodes. E6 through the scaling-table reference: milliseconds, estimate | J3 C2, CTX 06 E.2, prototype core, fieldcells |
| moments of a family on 19 cells | 26 to 58 microseconds | CTX 06 E.2 |
| prediction from moments, 123 x 57 | 1 microsecond | CTX 06 E.2 |
| free-moment fit, shared noise | JAX 0.21 microseconds (63 x 57) and 0.15 microseconds (63 x 171) per spectrum on 6 to 11 cores. NumPy on one core 0.83 and 0.76 microseconds. Ranks agree | prototype profile |
| free-moment fit, noise per pixel | per SVD at 63 x 57 and 63 x 171: one core 177 and 611 microseconds, NumPy on 28 threads 13.6 and 39 microseconds, JAX `vmap` 16.7 and 51 microseconds. The same cost applies per pixel and call of `fit.predict` when directions are recomputed. The study's 16 to 39 microseconds were multi-core | prototype profile, J1, J3 C14 |
| NNLS on 111 x 272 | 2.96 ms serial, median 13 active atoms. Process pool with BLAS pinned to one thread: 0.51 to 0.71 ms per spectrum throughput (4 to 28 processes, under load). Default BLAS threads oversubscribe (1.05 ms). JAX FISTA is slower and inaccurate at condition number 1.5e18 | prototype profile |
| family fit per spectrum | `trf` 2.39 ms (12 evaluations, NumPy residual). Batched JAX Levenberg-Marquardt under `vmap`, 40 iterations: 60 microseconds per spectrum, parameters within 4.3e-7 of `trf`. `trf` stays the oracle | prototype profile |
| `Ray`, 300 steps | basis route 3.7 ms per ray (`vmap` over 1000 rays) plus columns once (608 cells x 0.245 s). Direct with tabulated `F`, `G`: 2.5 s. Direct with quadrature `F`, `G`: 98 s, extrapolated from 30 steps | prototype profile |
| G15 `lat8` (`gamma` 22 x `B` 4 x depth 48 cells of depth order 3, `Legendre(8)`, 3 blocks) | 2 737 152 lattice rows, 1 140 480 columns, 21.9 MB of dense moments per sightline. Sparse `Ray` pairing: 14 256 values per step (648 rows per product cell in each of the 22 energy cells that the power law reaches). Depth powers are outside the total degree (Sec. 4.3). Depth order 2 on the same cells gives 2 052 864 rows and 887 040 columns and fails the depth check (bound 7.2e-3) | r3-08, recounted under Sec. 4.3, not measured |
| per-pixel SVD chunk (transient) | `8 B (m n + m k + k n) 1.5` per pixel: 125 kB at 63 x 57, 306 kB at 63 x 171. Chunks sized by `chunk_bytes` (Sec. 6.4) | r3-41 |
| harmonic cell, `L = 8`, `N = 2` | 6.5 s and 0.8 GB, one compilation per cell. The host harmonic route costs 2.2 to 2.9 ms per point for 3 channels, about 90 per cent per-call overhead | CTX 02 B.5, not re-measured. Host route: prototype profile |
| interval certification | 1.30 to 1.43 s per call under load, 73 per cent in pure-Python `mpmath`. 0.33 s on a quiet machine in the profile guide | prototype profile |
| 1e6 pixels, 63 data, 26 retained directions, shared noise | data 504 MB, projections 208 MB, moments 456 MB | P4 Sec. 8.6 |
| 1e6 pixels, 171-column moments | 1.37 GB | r1-34 |
| 1e6 pixels, per-pixel noise, stored directions (`keep_directions=True`) | `1e6 x R_max x n x 8 B`: 6.4 GB at cap 14 and `n = 57`, 19 GB at cap 14 and `n = 171`, 11.9 GB at `R_max = 26` and `n = 57` | r1-34 |

A JAX port of the S1 quadrature is measured (S1 row). A JAX port of the full stack provider is not measured, and the other stage costs are NumPy measurements. The prototype ran on an M3 Ultra (28 cores, CPU, float64) shared with other jobs, so its wall times are upper bounds. Profile times taken under load may be 2 to 3 times slow, and the rule-family wall times (prototype R3a, R5, beaming, fieldcells, partition, depth) ran at load averages 60 to 220 and may be 3 to 8 times slow. Ratios timed back to back hold.

### 8.8 Dispatch tests

Each dispatch has a parametrised test under `tests/unit/<module>/test_<api>_boundaries.py` that evaluates both sides directly and asserts finiteness and agreement.

| # | Dispatch | Both sides compared | Extremes included |
|---|---|---|---|
| T1 | kernel per cell | both kernels on every pair of face-sharing product cells with different kernels, for `switch` and for `assign` | `gamma` 30 to 1000, `B` 0.1 uG to 10 mG, `nu` 30 MHz to 100 GHz, `eta` 0.05 to 0.95 |
| T2 | column provider, and cached S1 or per-node mode | stack (both modes), jet and table at every cell, relative to the channel maximum and to the cell's own peak. Conditionals of `B`, `eta` (under `Beaming`), `mu` and depth on `gamma`, of averaged variables on each other (`B` given `eta`, `eta` given `B`, depth given `B`, `B` given `Legendre(eta)`), and of `B` on a resolved depth over 3 depth cells, against direct integration with slopes from 0 to large. A log-normal `B` whose median moves with `gamma`, steep and narrow, against the jet, and the naive frozen-weight result (zero `k > 0` columns) pinned as a regression case. Provider choice pinned for `gamma x eta` and `gamma x B x eta` lattices on the continuum kernel. Own-peak agreement between derivative providers in cell 1 at the lowest channel and in the top cell at the highest channel (measured 1.1e-10 to 2.4e-10 and at most 1.1e-10). The table against the jet only against the channel column maximum, with half-widths 0.023, 0.242 and 0.92, the last pinned as a failing case | `N` 0 to 6, half-width 0.02 to 1.0, 1, 19 and 200 cells, first and last cell, extreme `eta` cells, conditional widths 0.01 to 1.5, steep dependence |
| T3 | primitive table range and `x_min` | table against direct `F`, `G` quadrature at both ends. The last cell and the top cell edge at the lowest channel-support node with the adapter's `x_min`. The small-`x` asymptote above `gamma_t` against kernel calls on `[1e7, 3e7]` at the lowest and highest channels | `x` from 1e-9 to 700, pins from mpmath |
| T4 | tabulated or inline field average, and the default rule of each law | both for fixed, uniform, Maxwellian (96 nodes on `[0, 8 rms]`), log-normal of width 0.05, 0.01 and 1.5, and a user family of width 0.05, against an adaptive reference | 0.1 uG and 1 mG |
| T5 | scaling-relation columns or jet columns in `B` | both against direct integration for E6 and for 3 B cells x 19 energy cells. Field cells alone (`Cells.log` 0.5 to 50 uG, 4 cells) with `gamma ~ PowerLaw(2.6)` averaged on `[300, 3e6]` take the jet and agree with direct integration over `(gamma, B)` at both extreme field cells and the lowest and highest channels. A `B`-dependent pitch conditional exercises the fallback, and the scaling identity is shown to fail there. An azimuth factor `c(gamma)` with a free field law fails the `FieldScaling` precondition, its fallback agrees with direct integration over `(gamma, B)`, and a forced push-forward does not. Scaling columns equal jet columns to 1e-8 relative at the first and last energy cells, for `B_0` at each field cell reference and at both ends of the field range, at orders 2 and 4. With `gamma` averaged, the per-node identity equals the jet for a `B`-independent electron law. It is pinned to fail for a law whose cut-off depends on `B`, whose jet columns are checked against five-point differences. Field cells wide against a field-dependent cut-off (3 log cells on [0.5, 50] uG) give a representation error of order 1 at 10 GHz, and it is pinned with its refinement (9 cells: 0.10) | field range ratio 1.01 and 100 |
| T6 | half-range or full-range quadrature, under the joint-reversal precondition | I, P and V on `HarmonicKernel` with `eta = Isotropic()` and `Uniform(-0.5, 0.5)`, with `B` given `eta`, with `eta` fixed and `mu` averaged, with `eta = Cells` on `[0.2, 0.5]` and `mu` averaged, with `mu = Cells` and `eta = Isotropic()`, and with samples whose `eta` values lie on `eta` cells, with `PointMasses` and `Ray` steps on `eta`, and with `mu` and `eta` both isotropic (one cosine halved, against both halved pinned as a regression), each against a full-range reference. V is zero to quadrature accuracy on the full range | `eta` support edges 0.05 and 1 |
| T7 | smooth rule or cut-off rule for family moments, and the cut-off against the pivot | cut-off swept across a cell edge at distances 1e-6 and 1e-3, against adaptive quadrature. `gamma_c` on both sides of the pivot: forward default, a G5 fit started with `gamma_cut` below 1e4, `paper_coefficient` finite, an E6 conditional on a Maxwellian field with nodes above 9.4 uG | cut-off in the first and the last cell |
| T8 | samples at cell edges and outside the lattice | a sample on an edge, conservation of the column | all samples outside |
| T9 | rank cut `1e-13 s_max`, pair rule `1e-9` | perturbed pairs | 1 and 171 columns |
| T10 | shared-noise path or per-pixel path, stored or recomputed directions | the same data through all paths | 1 and 1e5 pixels |
| T11 | host NNLS or a later JAX solver | a sample of pixels through both | `alpha` 0 and the top of the grid |
| T12 | host `trf` or a later JAX solver in transformed coordinates | interior solutions and solutions at a bound | starts at both bounds |
| T13 | accuracy check by table, by the scaling-table reference or by kernel calls | E1, E4-JP, E6, and the E4-KP direct spectrum against the stored study truth | cut-off inside the lowest cell |
| T14 | cache hit or rebuild | bitwise equality. The key changes with every specified statistic, including an edited helper function, an edited global dict read only by the helper, and a helper reached through a factory closure | an unnamed conditional, a named closure built in a factory, a conditional on an averaged parent. A resolved-child conditional does not change the key |
| T15 | azimuth-factor routes R5 (i) to (iv), the azimuth as a parent (R4a), and depth as a parent or child | each route against direct integration over `(eta, phi)`, `(mu, phi)`, `(depth, phi)`, `(B, phi)` with `B` averaged and as a joint `Distribution` block over `(B, phi)`, with the I block checked to use the cached S1 and the spin-2 block the S2 field average, and `(depth, B)` with depth averaged, and depth as a child of averaged `B`. `eta` given `phi` and `B` given `phi` with `phi ~ AzimuthDensity`, against direct integration over `(eta, phi)` and `(B, phi)`. A characteristic-only screen or `AzimuthFactor` as a parent raises. The two cases of R4a (b), an averaged law given a resolved `phi` and a child of a `phi` in a `FreeMoments` block outside that block, route to `AzimuthFourier(m_max)` rows instead of raising (T20) | `c` near 0 and near 1, steep `c(eta)` and `c(B)`, field range ratio 100 |
| T16 | jet or core provider, and jet or table | columns at `d = 1, 2, 3`. A polynomial kernel in three variables against closed-form derivatives. G16 columns against finite differences and the table. Taylor-mode and nested `jacfwd` coefficients of a one-parent conditional (Sec. 8.3, decision (a), in step 7 by decision 137) | orders 0 to 4, total-degree and per-variable caps, first and last cell |
| T17 | parity mask per block | harmonic kernel, `mu = Legendre(L)`, `eta` isotropic, V requested, pitch density with an odd part: V nonzero and equal to the direct sum. An odd `c(eta)` under route (iv), and an asymmetric averaged pitch law under `Beaming` with `eta = Legendre(L)`, get no mask. An asymmetric pitch population on `mu = Legendre(L)` with an even averaged `eta` law keeps the mask and equals the full range to quadrature accuracy. `mu = Legendre(L)` with `eta` on `Cells`: odd-`l` I columns nonzero and no mask | `L` 0 and 8 |
| T18 | fixed or re-selected rank in a basis-parametric free-moment fit | start and optimum on opposite sides of a UPRE rank change: the reported rank is fixed and the fit is the same from every start | width at both bounds |
| T19 | cells on `mu` under `Beaming` | `Cells` on `mu` with `eta ~ U[-0.5, 0.5]` against direct integration with 1, 2 and 4 cells and edges at ±0.5. `eta = Legendre(4)` with 2, 4 and 8 `mu` cells on `[-1, 1]` and an isotropic and an anisotropic pitch density, against `basis.direct`, with the error pinned per cell count (one or two cells give errors of order one at `N = 2`). `eta = Legendre(16)` with 8 and 32 cells shows the cell count that the degree needs. `eta` fixed raises `RoleError`. Joint `(mu, eta)` samples give an unbounded `representation:angular`, never a measured zero. `mu` samples with `eta` fixed on a lattice with `mu=se.Legendre(4)`: `basis.direct` pairs at `L = 4`, and the measured `representation` equals that of the same population with the `mu` axis at `L = 8` to quadrature accuracy | anisotropic pitch density, a jump of `p_eta` inside a cell |
| T20 | `AzimuthFourier(m_max)` rows or nested quadrature at the `phi` nodes (R4a (b) or (a)) | the prediction from `AzimuthFourier` rows and its truncation term against direct integration over `(child, phi)`, for `eta` given `phi` and `B` given `phi` with `phi` on `AzimuthFourier`, and for a child outside a `FreeMoments` block that holds `phi`, with the block's moments from a specified population. The truncation term is zero when `Kbar(phi)` of every averaged child is a trigonometric polynomial of degree at most `m_max`, and it equals the measured difference otherwise | `m_max` 0 and 2, a steep conditional in `phi` |

Property tests: a prediction does not depend on `w`. Components add. `basis.at(channels)` equals a fresh build. `recentre` equals a rebuild at the new reference points. `vmap` over pixels equals a loop. Gradients agree with finite differences. `eqx.filter_jit` of `log_prob` does not recompile when bounds, starts or data values change, and `grad` and `vmap` apply. `jit` and `vmap` over `Moments` values keep the concrete key, a `None` key against a key raises, and `allow_mismatch=True` does not accept it. An anisotropic pitch density through `Legendre` equals `basis.direct`, and `Cells` on `mu` with `eta` fixed raise `RoleError`. Every `Lattice` method that takes `Moments` keeps a `minimum_norm` origin. A minimum-norm vector with a negative `a_b2` gives an unbounded `remainder` and leaves `ledger.sum("estimate")` unchanged. R3a with a conditional whose `g[c, s]` is a polynomial of degree `K_s` in the parent is exact for `r + K_s <= N_b`, and under `NonNegativeCells` it is exact for every row. For a kernel polynomial in the lattice, the R3a prediction equals the averaged-child prediction. The bound `E` of Sec. 5.2 holds against nested quadrature for flat, steep, broken and edge-concentrated parents, at `sigma` from 0.003 to 3 and with the median on a child edge. `predict - direct` over the full support equals `representation + outside_lattice` with outside_lattice signed as in Sec. 7.2 (within cells minus full support). The unsigned sum misses by 2 |outside_lattice| and is pinned as a regression case. The node-mask implementation of within="cells" is pinned as a failing case. Two components with different averaged `B` laws: the `BasisSum` prediction equals the sum of the parts, and a fit concatenates the designs. A fit with `se.Penalty()` has origin `penalised` and a finite `unconstrained`. The import graph follows Sec. 3.1. Unit pins: Hz against GHz and gauss against microgauss raise, `B=se.Uniform(4, 6)`, `B=se.Cells.linear(2, 8, 3)` and `eta=se.Uniform(10, 50, unit="deg")` raise `UnitError`, and a conditional that returns a plain float for a field slot raises `UnitError` at plan time.

### 8.9 Native code

SyncEM 0.8 and 0.9 add no Rust or other new native code. No profiled hot path is bound by the interpreter in a way that native code would fix better than the algorithmic changes above: S0 lookups for S1 (Sec. 8.2), tabulated `F`, `G` for forward kernels and rays (Sec. 8.3), batched family fits and pinned NNLS pools (Sec. 8.6). The jet compile cost is a JAX compile cost, and a port would not change it. Three seams stay open, each behind an existing boundary.
* Host-only kernels sit behind the kernel protocol with the `HostOnly` capability and reach the basis through the cell table only (Sec. 8.1, 8.3).
* Interval certification sits behind the interval context of the certified routines (`PKG/syncem/model/_certified_dense_harmonic.py`), pure-Python `mpmath` today. `gmpy2` (2 to 5 times, estimate) or `python-flint`/Arb (10 to 100 times, estimate) enter through the certification extra before any new native crate.
* The harmonic host driver sits behind `high_order_channel_modes` (`PKG/syncem/model/high_order_harmonic.py`) and the `HighOrderHarmonicKernel` adapter [0.9]. It is vectorised in NumPy first.

The decision is revisited when interval certification becomes routine, when a vectorised harmonic driver still exceeds about 0.3 ms per channel point, when a host path takes more than half the wall time of an end-to-end fit or build, or when a caller outside Python needs the kernels. The decision record is NATIVE.md.

## 9. Migration

### 9.1 Package layout

```text
syncem/__init__.py     float64 as today. Lazy front door through module __getattr__. Legacy imports unchanged in 0.8
syncem/constants.py    unchanged (AST-pinned by ST/lib/syncem_compat.py)
syncem/units.py        boundary conversion, Quantity
syncem/values/         observable.py  spectrum.py  caveats.py  blocks.py  errors.py
syncem/base/           variables.py  kernel.py  capabilities.py  continuum.py  harmonic.py  hybrid.py  user.py  phase.py  channels.py
syncem/distributions/  protocol.py  common.py  azimuth.py  screens.py  electrons.py  family.py  conditional.py
syncem/lattice/        cells.py  table.py  representations.py  lattice.py  index.py  moments.py  maps.py
syncem/records/        ledger.py  records.py
syncem/engine/         average.py  expand.py  direct.py  columns.py  primitive.py  field_average.py  provider_jet.py  provider_core.py
                       provider_stack.py  _stack_series.py
                       provider_table.py  assemble.py  cache.py  basis.py  basis_sum.py
syncem/population/     population.py  markers.py  samples.py  ray.py  pair.py  quadrature.py
syncem/planning/       roles.py  defaults.py  preconditions.py  moment_map.py  classify.py  plan.py  report.py
syncem/results/        comparison.py  printing.py   (Fit and Comparison prints. Shared blocks come from values/blocks.py)
syncem/fitting/        data.py  policies.py  linear.py  penalty.py  nonnegative.py  separable.py  bilinear.py  factor.py
                       basis_parametric.py  compressed.py  problem.py  fit.py
syncem/paper.py        the objects of PAPER Sec. 5 and App. B
syncem/model/          unchanged
pyproject.toml         base dependencies gain "scipy>=1.16,<2" next to JAX, Equinox and NumPy. SciPy is imported lazily (Sec. 9.7)
```

No subpackage shares a name with a front-door function (`fitting` for `se.fit`, `planning` for `se.plan`). Files stay under 400 lines. `syncem.SpectralBasis` and `syncem.StokesData` are new classes. `syncem.model.SpectralBasis` and `syncem.model.fit.StokesData` keep their meaning, and `import syncem` does not load `syncem.model`.

### 9.2 Reused without change of behaviour

| Module under `PKG/syncem/model` | Use |
|---|---|
| `channels.py`, `_channel_shapes.py`, `bandpass.py` (`fit_bandpass`), `_basis_combine.py` | base of `se.Channels`. Line kernels with a measured bandpass go through `fit_bandpass` |
| `phase.py`, `screens.py` | screens as depth distributions with `characteristic`, `TaylorPhase` weights for depth cells |
| `kernels.py` (`KernelModel`, `required_m_max`), `_continuum.py`, `harmonic.py`, `_harmonic_taylor.py`, `_bessel_recurrence.py` | kernels behind the new protocol, regime rule. The continuum adapter sets `x_min` from the channels and the lattice (Sec. 8.3) |
| `basis.py` (`_build_core`), `_basis_taylor.py`, `_basis_core.py` | core provider for `(gamma, B)` cells. The `taylor_tensors` loop is called by the jet provider with `d` directions |
| `high_order_harmonic.py` (unchanged, AST-pinned), `high_order_basis.py` | host kernel behind the table provider |
| `_direct.py` (`direct_channel_average`), `moments.py` (chunked sums) | direct averages and moments of `Samples` |
| `errors.py` (`ErrorTerm`), `_provenance.py` (`strict_json`, `AssumptionRecord`), `_remainder_route.py` (`lower_set_margin`) | records layer (the `declared` flag is added in the new layer), remainder of capped index sets |
| `fit/observation.py`, `fit/_combination_core.py` (`find_clusters`), `fit/_transform.py`, `fit/_relations.py` | whitening, mask and response of `StokesData`, pair clusters, bounded parameters, scaling relations |
| `fit/_linear_core.py` | `solve` and `linearised_bias` are reused. The file is not edited (Sec. 9.7 guard). `fitting/linear.py` computes the `response` and `discrepancy` terms of Sec. 7.2 that replace `bias_bound`. The response term is computed without a declared discrepancy, and a rank-deficient free fit is evaluated at `R*` on the retained combinations in place of returning unbounded |
| top level: `ultrarel.py`, `cumulants.py` (`raw_moments_from_cumulants`), `rm.py` (`faraday_depth_practical`) | primitive values, cumulant form, depth along a `Ray` |

### 9.3 Ported from the study scripts

| Study code | New home | Language in 0.8 |
|---|---|---|
| `ST/lib/kernels.py` (derivative recursion, log-safe form) | `engine/primitive.py` | NumPy |
| `ST/lib/basis.py` (`KernelTable`, `channel_kernels`, `kbar`, `derivative_spectra`, `bin_derivative_grid`) | `engine/field_average.py` (`KernelTable`), `engine/provider_stack.py` (`channel_kernels`), `engine/cache.py` | NumPy |
| `ST/ensemble_e6.py:97-129` (`field_coefficients`) | `engine/assemble.py`, and the scaling-table reference | NumPy |
| `ST/lib/ensembles.py` (`Measure`, `measure_1d`, families, `jp_moments`, the coefficient convention) | `population/quadrature.py`, `distributions/electrons.py` | JAX |
| `ST/fits.py` (`svd_signed`, `canonical_pairs`, `LinearFit`, `FreeFactorFit`) | `fitting/linear.py`, `fitting/bilinear.py` | JAX, host for bilinear |
| `ST/fits_positive_ridge.py`, `ST/fits_free_positive.py` | `fitting/penalty.py`, `fitting/nonnegative.py` | JAX, host for NNLS |
| `ST/paramlib.py`, `ST/baseline_models.py`, `ST/mixture_forms.py` | `fitting/separable.py`, `fitting/compressed.py`, `distributions/electrons.py` | host solver, JAX residual |
| `ST/sensitivity.py:109-121` (`shifted_moments`) | `lattice/maps.py` (`recentre`) | JAX |

The NumPy versions are kept, hash-checked, under `tests/reference/classical_models/` as the oracle (pattern of `tests/model/reference/validation`).

### 9.4 Recorded decisions that the new layer reverses

Each is rewritten in `docs/DESIGN.md` and the changelog. None changes behaviour inside `syncem.model`.
1. `docs/DESIGN.md` Sec. 7.5 and CHANGELOG 0.3.0 reject a field-averaged kernel. The new layer averages specified conditionals into the kernel, as PAPER:644 states and Sec. 5 uses. Each averaged statistic is a named ledger term.
2. Unit moment `m[0] = 1` with a scalar amplitude. Moments carry the column (PAPER:653-659).
3. Column equilibration and a relative tolerance in `fit_linear` (`fit/linear.py:8-10`). The default free fit uses the fixed reference weights and UPRE. `syncem.model.fit.fit_linear` remains for the single-reference layer.
4. `unit_peak` as the channel default. New constructors use unit area.
5. `noise=` as a variance. The new `StokesData` takes `sigma`.
6. `ContinuumKernel` forcing `L_mu = 0`. The adapter declares `Beaming`.
7. `ErrorBudget.total()` as the headline. The ledger reports by kind.
8. Affine displacements only. Each `Cells` has a chart.

### 9.5 Version plan

| Version | Content |
|---|---|
| 0.8.0 | additive. `git diff v0.7.10 -- syncem` shows new files and `syncem/__init__.py` only, and `pyproject.toml` adds SciPy to the base dependencies. Scope of Sec. 10.1, `AzimuthFourier(m_max)` included |
| 0.8.1 | `SpectralBasis.from_single_reference`, and the harmonic single-reference basis of `Samples` on `Lattice.single` through the existing builder (first half of G11) |
| 0.9.0 | JAX port of the stack (all modes, with the per-node mode over `gamma` nodes for field cells with `gamma` averaged) and JAX gradients in basis parameters. `FieldScaling`, and `Ray` on a `FieldScaling` lattice. Harmonic cells with the traced mask, harmonic entries of `HybridKernel`, host kernels through the table, `PointMasses`. JAX solvers after the agreement tests T11 and T12. Per-pixel family fits at scale, shared parameters across problems. `from_dict`. Blocks over variables whose representations arrive in 0.9.0 arrive with those representations |
| 1.0.0 | legacy top-level modules move to `syncem.legacy` with import shims. `docs/DESIGN.md` is rewritten around the three layers. `syncem.model` stays as the single-reference engine with its tests and is not rewired |

### 9.6 Acceptance tests from the paper's numbers

Values come from PAPER tables and the stored `ST/*_results.json`, copied with hashes under `tests/acceptance/data`. Seeded fits use the study's stored noise draws (seed 20260928, NumPy generator).

| Test | Expected | Tolerance |
|---|---|---|
| design shape and numerical rank | design `(41, 3, 57)`. Rank of the noise-whitened design (`sigma = 0.01 I_j`, threshold `s > 1e-13 s_max`): 26 on the 21 fitted channels (63 rows), 28 on all 41. E5 joint `(41, 3, 171)`: 61 on the fitted rows, 77 on all rows | exact |
| columns against `ST/lib/basis.derivative_spectra` | all cells and channels | 1e-10 of the largest column value per channel with S1 by quadrature (`s1="quadrature"`, the oracle path). 1e-7 with the default S1 by S0 lookups |
| forward error at `N = 0, 2, 4` | E1 2.56e-2, 8.58e-4, 8.76e-5. E2 3.83e-2, 9.07e-4, 1.16e-4. E3 2.74e-2, 1.02e-3, 9.64e-5. E4-JP 1.85e-1, 3.64e-3, 4.47e-4 (`forward_results.json`, read for this synthesis) | 1 per cent of each value |
| E5 | joint 1.76e-3 and 1.39e-4. One factor 2.90e-1 | 1 per cent |
| E1M (96 field nodes on `[0, 8 rms]`, the study's rule) | own basis 6.18e-4. `U[4, 6]` basis 3.84e-2 at `N = 2`, 3.76e-2 at `N = 4` | 1 per cent |
| E6 | joint 7.8e-3 (`cost_results.json`), field averaged 8.4e-2 | 1 per cent |
| E4-KP, pitch-Legendre to `l = 8` | 8.8e-3 at `N = 2`, 6.6e-4 at `N = 4` (PAPER:1081). `basis.direct` equals the stored study truth | 2 per cent. Direct spectrum 1e-10 of `I_j` |
| reference points shifted by a quarter cell | intensity error of E1, E2, E4-JP at most 6.4 per cent (`N = 2`) and 0.92 per cent (`N = 4`) (PAPER:741) | stored values, 1 per cent |
| free-moment fits, seeded | `R* = 7, 7, 7, 9, 9, 17, 7` and `chi2`/dof 52.1/56, 49.9/56, 52.5/56, 49.0/54, 48.4/54, 43.2/46, 52.1/56 for E1, E2, E3, E4-JP, E4-KP, E5 joint, E1M | exact rank, `chi2` to 1e-6 |
| E5 one factor | `R* = 7 + 2`, `chi2 = 270.9 / 54`, factor modulus 0.286, largest error 6.1 per cent | 1e-3 |
| E2 over 1 to 100 GHz, median 300 GHz error at 0.01 and 1 per cent noise | free 1.58 and 15.7. Penalty only 1.66 and 15.8. Non-negative cells 1.26 and 14.9. Without penalty 4.37 and 25.6 per cent | fast tier: a subset against stored per-realisation values. Slow tier: 500 realisations |
| families | eSBPL for E2: 0.080 and 0.93 per cent (1 to 100 GHz), 2.57 per cent (1 to 40 GHz). Cumulant form for E3: 0.015 and 0.8 per cent | as stored |
| `paper_coefficient` of the truth populations | `ln A_paper` = -5.14049 (E1), -2.96604 (E2), -5.31605 (E3), -5.11590 (E4-JP): the truths of `ST/verify_params_results.json` (E2 line 480, E3 line 629, E4-JP line 643) and PAPER tab: classical parameters. `ST/lib/ensembles.py` defines the convention | 1e-5 |
| parameters, compressed and full | PAPER tab: classical parameters within one standard deviation after `paper_coefficient`, with the uncertainties propagated through its Jacobian. Uncertainties within 0.3 per cent. Coverage 65.7 to 69.1 per cent | slow tier |
| `unconstrained` at 300 GHz | `row300_norm_fraction_outside_VR` of `fits_extrap_noisefree_results.json` | 1e-8 |
| Sec. 4.2 benchmark [0.8.1] | `tests/model/reference/validation/full_response_results.json` | tolerance of the existing test |

### 9.7 Infrastructure constraints

* Guards. A test holds the two AST digests of `ST/lib/syncem_compat.py`, the five names the studies import from `syncem.model` (`Channels`, `Reference`, `Support`, `Truncation`, `build_high_order_basis`), and the v0.7.10 hash of every existing file under `syncem/` except `__init__.py`. An import-graph test (AST walk of every import statement, function-local ones included, `if TYPE_CHECKING:` blocks exempt) enforces the layer order for module-level imports and the delegation table of Sec. 3.1 for function-local imports of later layers.
* Docstrings. `tests/model/test_docstrings.py:50-57` scans `syncem.model` and `syncem.model.fit` only. The new subpackages join `_public_modules` and the wording scans. Each public module states its `LABEL`, units, shapes and what is not certified, uses "validated" for finite checks, and has no semicolons.
* Labels. New docstrings cite only labels present in `main.tex` today. Everything else is `[extension]`. A new test compares cited labels with a stored list of the manuscript's labels.
* Dependencies. Base: JAX, Equinox, NumPy and SciPy (`scipy>=1.16,<2`). SciPy is imported inside the non-negative and bounded solvers and the host kernels, following `high_order_harmonic.py:138-146`, so `import syncem` does not load it. Import time is a measured cost (Sec. 8.7). optimistix is not a dependency.
* Native code. 0.8 and 0.9 add no Rust or other new native code, and the build needs no Rust toolchain (`NATIVE.md`, prototype profile). The seams for later native code are the host-only kernels, the certification arithmetic backend and the harmonic host driver. An optional `certification` extra may add `gmpy2` or `python-flint` (Arb) to speed up the interval arithmetic of `syncem.model._certified_*`, which runs on the pure-Python backend of `mpmath` today. Their gains, 2 to 5 times and 10 to 100 times, are estimates and were not measured. The harmonic host driver is vectorised in NumPy first, and Rust is revisited only if a vectorised driver still exceeds about 0.3 ms per channel point.
* Docs and release. `docs/api/index.rst` gains the new subpackages, and the build stays free of warnings. `publish.yml:70` gains the new subpackages in the smoke import. The smoke environment installs SciPy with the wheel as a base dependency, and the smoke test also imports `scipy.optimize` there. `tests/reference` joins the Ruff and Black excludes. There is no test CI today, so the fast tier must run locally in a few minutes.
* Tests come first. Each step of Sec. 10 starts by writing its acceptance test and seeing it fail.

## 10. First-release scope and implementation plan

### 10.1 Scope of 0.8.0

Built: units, `Quantity`, `Variable`, `Channels` constructors. `Cells`, `CellTable`, `Legendre`, `AzimuthHarmonic`, `AzimuthFourier(m_max)` with its truncation term, `Lattice` with product constructors, `Lattice.from_cells` and local `refine(where=)`, `MomentIndex` as an arithmetic index with interval selectors, `Moments` with `origin`, `masked` and sparse values, the size cap. The kernel protocol with capabilities and preconditions, the continuum adapter with `x_min`, `@se.kernel` for traceable kernels, `HybridKernel(assign=)` over traceable non-harmonic kernels. `AveragedKernel`, the jet provider in any number of celled variables, the core provider, the stack with S1 by lookups in a cached S0 table in `ln x` and per-node mode (celled and averaged parents, frozen-node reweighting), the table provider, the column map, `BasisSum`, the cache with size bound. Distributions with their own rules and `given_unit`, electron families with `paper_coefficient`, `@se.family`, `Conditional` with parent values, `params=` and `parity=`, `Population` with keyword factors and blocks, sums, `Samples` and `Ray` on any 0.8.0 lattice that resolves their variables, pairing, direct integration. The plan with rules R0 to R8, R3a and R4a (a) and (b). All fit kinds of Sec. 6.1 including basis-parametric and parametrised-factor bilinear (host), batches with shared and per-pixel noise, prediction at other channels with `predict=`, `se.problem` and sums of problems on the same data, `se.compare`. `Spectrum`, `Fit`, `Ledger` with declared terms, prints, unit conversion, `basis.save`, `se.paper`.

Representations in 0.8.0, which fix the scope of every block:
* `Cells` on any variable of a traceable kernel other than the harmonic kernels: energy, field strength (one or several cells), Faraday depth (one or several cells), the inclination cosine, and the pitch cosine (under `Beaming` when the inclination has a density or `Legendre` degrees). Field cells use the scaling relation where its precondition holds (energy cells included) and the jet otherwise, so field cells with `gamma` averaged take the jet. A product cell with a celled variable outside the stack's declared set goes to the jet.
* `Legendre` on either cosine. `AzimuthHarmonic`, and `AzimuthFourier(m_max)` for laws conditional on the azimuth (R4a (b)).
* `FreeMoments` blocks over any group of variables with these representations, for example `(gamma, B)`, `(gamma, phi)`, `(gamma, mu)`, `(gamma, depth)`, `(gamma, eta)` with `Legendre`, `(gamma, eta, phi)`. `NonNegativeCells` blocks over any group of variables with `Cells`.

Deferred, with the reason:
* Harmonic cells: they need the traced mask over harmonics first (Sec. 8.4).
* Harmonic entries of `HybridKernel`: they serve strong fields or low frequencies only (Sec. 8.4) and need harmonic cells. Per-cell choice among traceable non-harmonic kernels ships in 0.8.0.
* `HostOnly` kernels and `PointMasses`: they need the table provider as a column source with its own acceptance tests. 0.8.0 builds the table as a reference.
* `FieldScaling`: it changes the meaning of the order-0 sum (Sec. 4.2), and field cells serve the same fits in 0.8.0.
* `Ray` on a lattice with a `FieldScaling` axis: it needs the per-step push-forward of `FieldScaling`.
* JAX port of the stack, and its per-node mode over `gamma` nodes for field cells with `gamma` averaged: no timing of a ported stack provider exists yet (only the S1 quadrature port is timed, 36 ms at one node, prototype profile), and the jet covers that case in 0.8.0. JAX solvers: they need agreement with the host solvers that produced the paper's numbers.
* Per-pixel family fits at scale and shared parameters across problems: the Gram-matrix route has no prototype. A batched JAX Levenberg-Marquardt under `vmap` takes 60 us per spectrum with parameters within 4.3e-7 of `trf` (prototype profile), and it is the starting point of 0.9 with `trf` as the oracle. Sums of problems on the same data are a concatenation and ship in 0.8.0.

Outside this design: absorption, Faraday conversion, beams that vary inside a channel, a maximum-entropy density from free moments, a lattice chosen from an accuracy target, plotting.

### 10.2 Ordered steps

Day counts for steps that P5 also lists are P5's. The others are estimates. `prototypes/core` (723 lines) has a product table, an averaged kernel with conditional laws and Legendre axes, jet columns, pairing, prediction and direct integration, and it is a starting point for steps 2, 3, 4 and 6. The rule prototypes under `prototypes/rules` are starting points for the field cells of step 5, the partition and provider checks of step 6, and R3a, R5, `Beaming` and depth in step 7. The day counts do not assume them.

| # | Step | Acceptance test | Days |
|---|---|---|---|
| 0 | guards of Sec. 9.7 | passes before any other change | 0.5 |
| 1 | `units`, `Quantity`, `Variable`, `Channels` constructors, error classes | `B=5.0`, `B=se.Uniform(4, 6)`, `B=se.Cells.linear(2, 8, 3)` and `eta=se.Uniform(10, 50, unit="deg")` raise. Hz against GHz pins. `Channels.log(0.1, 10, 41, unit="GHz", half_width=0.2, n_nu=400)` equals `Config.chan_nu` and `chan_w` to 1e-14. `fractional_width=f` gives the spread `f nu_j` of PAPER eq: forward response families. The `Uniform` and `Cells` cases use stand-ins with the placement protocol until those classes exist | 2 |
| 2 | `Cells`, `CellTable`, `Legendre`, `AzimuthHarmonic`, `Lattice`, `from_cells`, `refine(where=)`, `MomentIndex`, `Moments`, `recentre`, `coarsen` | lattice rows 57, 171, 114, 513 (E4-KP), 513 (energy and depth, Sec. 4.4), 2430. Depth powers `s = 0..N_d` on every row of the other variables, outside the total degree (Sec. 4.3). Position `b (N + 1) + r`. Interval selectors on a refined table. `recentre` and `coarsen` exact to 1e-12 on a density with a cut-off inside a cell. A `(gamma, B)` table with one corner product cell split 2 x 2: row count, index positions, `coarsen` back to the parent table to 1e-12, and moments equal to those of the uniformly refined tensor table on the shared cells (the prediction check runs in step 5). Overlapping boxes raise. The G15 lattice `lat8` (2 737 152 rows) builds its arithmetic index and key without per-row objects, and the size cap refuses it at the default with the numbers of Sec. 2.5 | 6 |
| 3 | kernel protocol, capabilities, continuum adapter, `AveragedKernel`, core provider, re-expansion | columns equal the oracle in cells 1, 10, 19 at `N = 0..4` to 1e-6 of the order-0 peak (measured 2.4e-12 in cells 10 and 19, 7.5e-9 in cell 1, with `x_min = 1e-12`). The adapter's `x_min` admits the last cell at the lowest channel node (T3). Joint energy-field columns equal the scaling-relation columns to 1e-10. `EnergyWindow` reproduces the edges of `prototypes/synth/window.py` | 4 |
| 4 | jet provider in `d` directions | T16: a polynomial kernel in three variables against closed-form derivatives to 1e-12. The jet equals the core provider at `d = 2`. G16 columns, and cells on `eta` with the continuum kernel routed to the jet for the whole product cell (Sec. 12 row (n)), against finite differences of the averaged kernel. The table comparisons of T16 run in step 6 | 4 |
| 5 | stack provider with S1 by lookups in a cached S0 table and per-node mode (frozen-node reweighting. `phi` given `B` in S2 moves to step 7, decision 142), the quadrature S1 as the oracle path, field-derivative columns with their precondition, column map, `SpectralBasis.build`, `BasisSum`, `at`, cache with channel rows and size bound | all cells and channels equal the oracle to 1e-10 with the quadrature S1 and to 1e-7 with S1 by lookups. Shapes and ranks of Sec. 9.6. T2 with averaged-parent and moving-median conditionals (depth-parent in step 7, decision 142), T5 with field cells alone, T14. S1 by S0 lookups against the quadrature table from 0.1 to 100 uG at orders 0 to 6 (measured 1.7e-13 in `ln T` and 3.3e-10 of the peak at `r = 6`). S1 builds at 1, 16 and 96 nodes against the per-node lookup cost of Sec. 8.7 | 7 |
| 6 | distributions, families with `paper_coefficient`, `Population`, pairing with the cut-off rule, `direct` with the kernel's capabilities, `within=` on cell-face panels and the small-x asymptote above `gamma_t` (Sec. 8.3), table provider, scaling-table reference, `predict`, ledger with declared terms | forward errors of E1, E2, E3, E4-JP, E1M at `N = 0, 2, 4`. T4. The 17-node table against kernel-call direct spectra to 1e-12 of `I_j` at half-width 0.242, with half-width 4.60 pinned as a failing case (Sec. 7.2). Own-peak differences between derivative providers in cell 1 and the top cell. `paper_coefficient` row of Sec. 9.6. `predict - direct = representation + outside_lattice` (signed, Sec. 7.2) to 1e-11 of `I_j` for an unbounded power law, a cut-off power law beyond both lattice ends, a lattice with a hole and the 1-100 GHz default lattice at 300 GHz. Provider checks (a) and (b) of Sec. 8.3. T7, T13 | 6 |
| 7 | role rules, sums, conditionals with parent values and `params=`, R3a with its bound `E`, R4a (a) and (b), `AzimuthFourier(m_max)` with its index rows and truncation term, R5 routes, `Beaming` roles including cells on `mu`, the joint-reversal precondition with check (d), the depth-parent route, the characteristic-function precondition, ranges of unknown variables, joint blocks, field and depth cells, `Ray`, `se.plan`, `se.predict` | E5, E6 and E4-KP rows of Sec. 9.6. The role tables of the cases of Sec. 4.7. Basis columns 285 (E4-KP) and 1306 (benchmark, `HarmonicKernel`, IQUV) from the column map. Each `RoleError` case. T6, T15, T17, T19. `gamma x Legendre(eta)` and `gamma x Legendre(eta) x AzimuthHarmonic` against direct integration at `L_eta = 0` and 8 and at the first and last energy cells. 3 field cells x 19 energy cells against direct integration for a Maxwellian field at range ratios 1.01 and 100. `(gamma, depth)` moments of a two-zone population with different screens against direct integration at the extreme channels and near the depth bound of Sec. 4.6. `FreeMoments("gamma", "B")` without a lattice, with and without `support=`. R3a: a specified parent by nested quadrature against direct integration, exact on the rows with `r + K_s <= N_b` for a polynomial conditional, and the bound `E` of Sec. 5.2 against nested quadrature. `AzimuthFourier(m_max)` in both cases of R4a (b), an averaged `eta` or `B` law conditional on a resolved azimuth and a law conditional on a free azimuth outside its block, against direct integration over `(eta, phi)` and `(B, phi)` at `m_max = 0, 2, 4`: rows `exp(i k phi)` for `k = 0 .. m_max + 2`, one set shared by spin 0 and spin 2, a truncation term zero to quadrature accuracy when every averaged `Kbar(phi)` is a trigonometric polynomial of degree at most `m_max`, and measured otherwise. Two components with different averaged `B` laws through a `BasisSum`. A `Ray` by direct integration and on a 0.8.0 lattice, and `RoleError` on a lattice that leaves its variables averaged. The Taylor-mode check of T16 for one-parent conditionals (decision 137) | 18 |
| 8 | `StokesData` with required unit, free-moment fit, withheld diagnostics, batches with stored or recomputed directions and chunks sized by `chunk_bytes` | seeded ranks and `chi2` of Sec. 9.6. `vmap` equals a loop. T9, T10 | 4 |
| 9 | penalty with origin `penalised`, non-negative cells, prediction at other channels with `predict=` and the range check of Sec. 6.5, `unconstrained`, `profile_max`, `resample` | E2 extrapolation medians of Sec. 9.6. The `unconstrained` row. The first screen's 300 GHz `outside_lattice` share without `predict=` | 4 |
| 10 | separable fits, moment and cumulant forms, compressed likelihood on `fit.retained`, basis-parametric and parametrised-factor bilinear fits, `se.problem` with `PlanRecord` and sums of problems, `response` and `discrepancy` terms | family and parameter rows of Sec. 9.6. `log_prob` gradients against finite differences, including `jax.grad` through the nested quadrature of a conditional with `params=`. `ImproperPosterior` without a prior. T18. `eqx.filter_jit` of `log_prob` without recompilation. G14 with `B` averaged, its free-moment variant, and `fit_r` with `gamma=se.FreeMoments()` on `lat_b` | 9 |
| 11 | bilinear shared factor, and larger factors | E5 one-factor row. `gamma` and `B` both `FreeMoments` on `lat_b` from the singular-pair start | 3 |
| 12 | prints, defaults table, `to(unit)`, `save` and `load`, `se.paper`, `se.compare`, docs, changelog, `DESIGN.md` Sec. 7.5, version 0.8.0 | gallery G1 to G10, G12, G13, G14 (without `lat_s`), G15 (without the `FieldScaling` lattice) and G16 run as tests. The fit print of Sec. 7.1 against `PlanRecord.defaults` and `caveats`. Docs build without warnings. Docstring and label tests | 6 |
| 13 | 0.8.1 bridge: `from_single_reference`, the harmonic single-reference basis of `Samples` on `Lattice.single` | Sec. 4.2 benchmark row | 3 Amended by decision 124 |

About 76 working days. P5 estimated 38 for its steps 0 to 9 on the fixed layout. The general container, the jet provider, the per-node stack mode, the plan with its preconditions, the column map, the table provider, `from_cells`, `Ray`, R3a, R4a, `BasisSum`, the parametrised-factor kind, cells on `mu` under `Beaming` and `AzimuthFourier(m_max)` (about 4 days in step 7) account for the difference.

## 11. Script gallery

Shared setup for G1 and G3 to G10 (PAPER tab: classical ensembles). `se.paper` returns the same objects.

```python
import numpy as np, jax, jax.numpy as jnp
import syncem as se

obs     = se.Channels.log(0.1, 10, 41, unit="GHz", half_width=0.2)        # h_j = 0.2 nu_j
known   = dict(B=se.Uniform(4, 6, unit="uG"), eta=se.Uniform(-0.5, 0.5), mu=se.Isotropic(),
               phi=se.AzimuthFactor(0.4 * np.exp(0.5j)), depth=se.GaussianScreen(mean=8.0, sigma=0.5))
GR      = dict(lo=300, hi=2.9626e6)
lattice = se.Lattice(gamma=se.Cells.log(300, 2.9626e6, 19, order=2))             # 19 energy cells, 57 moments
e1      = se.Population(gamma=se.PowerLaw(2.6, column=1.0, **GR), **known)
free    = e1.replace(gamma=se.FreeMoments())
```

**G1. Paper E1 forward**
```python
spec  = se.predict(e1, obs, lattice=lattice)
print(spec)                                             # accuracy line: representation 8.6e-4 of I, measured
spec.I, spec.Q, spec.U                                  # (41,) each
spec.ledger["representation"]                           # ErrorTerm(kind="measured"), value (41, 3)
spec4 = se.predict(e1, obs, lattice=lattice.with_order(4))                       # 8.8e-5
basis = se.spectral_basis(e1, obs, lattice=lattice)     # the memoised object behind both calls
truth = basis.direct(e1)                                # direct integration, a Spectrum
basis.design().shape                                    # (41, 3, 57)
# built: S1 table for U[4, 6] uG (disk), 57 columns (stack with the cached S1: no averaged law has a parent),
#        the cell table for the check. Order 4 adds columns only
# solver: none. Moments by 24 nodes per cell, then one matrix product
# returns: Spectrum with the measured representation error against direct integration with the same kernel
```

**G2. Simplest forward call**
```python
pop  = se.Population(gamma=se.PowerLaw(2.6), B="5 uG", depth=se.GaussianScreen(mean=8.0, sigma=0.5))
spec = se.predict(pop, ([0.408, 1.4, 2.3, 4.8, 23.0], "GHz"))
print(spec)
spec.defaults
se.plan(pop, ([0.408, 1.4, 2.3, 4.8, 23.0], "GHz"), strict=True)                 # raises and lists every default
# defaults applied, each tagged [default]: monochromatic channels, per Hz. Energy range from the EnergyWindow of the continuum
#   kernel, then the continuum kernel per cell by the regime rule. Faraday rotation. 13 cells of half-width at most 0.242,
#   order 2, centred reference points, weights from gamma^-2.6. Amplitude 1 cm^-2 at gamma = 1e4. mu and eta isotropic.
#   phi uniform, so Q = U = 0 and the screen has no effect: the print names phi= and eta=. Accuracy check on
# built: S1 table for a fixed 5 uG, 39 columns. solver: none. returns: Spectrum
```

**G3. Free-moment fit of 21 of the 41 channels**
```python
data = se.StokesData.simulate(truth, noise=0.01, key=jax.random.key(0))          # sigma = 0.01 I_j on I, Q, U. Unit of truth
data = data.withhold(np.arange(1, 41, 2))                                        # 20 channels validate only
fit  = se.fit(free, data, lattice=lattice)              # rank=se.UPRE() is the default of this kind
print(fit)                                              # Sec. 7.1
fit.rank, fit.chi2, fit.dof, fit.p_value
fit.retained                                            # the 7 constrained combinations
fit.withheld.chi2, fit.withheld.n
pred = fit.predict()                                    # all 41 channels with pred.sigma and pred.unconstrained
# built: nothing new. The fitted and withheld channels are rows of the G1 basis
# solver: noise-whitened SVD in the reference weights, rank by UPRE over the numerical rank (26 on the 63 fitted rows)
# returns: Fit with retained combinations, the minimum-norm moments (origin "minimum_norm"), chi2 in fitted and withheld channels
```

**G4. Non-negative cells with the penalty, cells 4 to 19**
```python
nn     = e1.replace(gamma=se.NonNegativeCells(point_masses=17, cells=slice(3, None)))
fit_nn = se.fit(nn, data, lattice=lattice)              # penalty=se.Penalty() is the default: alpha by UPRE
fit_nn.weights                                          # (16, 17), weights >= 0 at Chebyshev-Lobatto points
fit_nn.moments, fit_nn.penalty, fit_nn.effective_parameters
fit_nn.profile_max(obs[-1:], stokes="I", delta_chi2=1.0)   # largest 10 GHz intensity among unpenalised non-negative fits
                                                        #   within chi2_min + 1. An Observable, noise only
# built: nothing new. The (48, 272) map from point masses to moments. Cells 1 to 3 are fixed to zero and the summary says so
# solver: alpha from the UPRE of the penalised fit without positivity, then NNLS on the stacked system (host)
# returns: Fit with weights, moments (origin "nonnegative_penalised"), effective parameters, negative-column count. No covariance
```

**G5. Electron families through the same basis**
```python
sbpl  = se.BrokenPowerLaw(index1=se.Free(2.0, bounds=(0, 7)), index2=se.Free(3.2, bounds=(0, 7)),
                          gamma_break=se.Free(7800, bounds=(100, 3e5), log=True), smoothness=2.0, **GR)
fit_a = se.fit(e1.replace(gamma=sbpl), data, lattice=lattice)                    # (a) amplitude free and profiled
mom   = se.PowerLawMixture.moment_form(order=2, index0=2.6, pivot=1e4, **GR)
cum   = se.PowerLawMixture.cumulant_form(k1=se.Free(2.6, bounds=(0.5, 6)), k2=se.Free(0.0, bounds=(-1, 1)), pivot=1e4, **GR)
fit_m = se.fit(e1.replace(gamma=mom), data, lattice=lattice)                     # (b) linear in A m_0, A m_1, A m_2
fit_c = se.fit(e1.replace(gamma=cum), data, lattice=lattice)                     # (b) separable
jp    = se.AgedPowerLaw(index=se.Free(2.6, bounds=(1.5, 4)), gamma_cut=se.Free(2e4, bounds=(1e3, 1e6), log=True), lo=300)
                                                        # the amplitude is the uncut coefficient, defined for gamma_cut below 1e4
fit_j = se.fit(e1.replace(gamma=jp), fit.retained, lattice=lattice)              # (c) compressed likelihood on the combinations of G3
fit_f = se.fit(e1.replace(gamma=jp), data, lattice=lattice)                      #     full linear likelihood
print(se.compare(fit_a, fit_c, fit_j, fit_f))
fit_j.params, fit_j.derived("break_frequency")
fit_f.derived("paper_coefficient")                      # A of PAPER tab: classical parameters, sigma through the Jacobian
# built: nothing new. Each family supplies a(theta) by a fixed rule per cell (the cut-off rule for jp)
# solver: (a), cumulant form, (c): linear unknowns in closed form, bounded least squares outside. Moment form: least squares
# returns: Fit with params (value, sigma, at_bound, identified), the measured representation error at the best fit, chi2
```

**G6. Extrapolation from 26 channels to 300 GHz**
```python
band  = se.Channels.log(1, 40 ** (31 / 20), 32, unit="GHz", half_width=0.2)      # ratio 40^(1/20)
dense = se.Channels.log(1, 300, 121, unit="GHz", half_width=0.2)
e2    = e1.replace(gamma=se.BrokenPowerLaw(2.0, 3.2, 7800, smoothness=2.0, column=1.0, **GR))
b32   = se.spectral_basis(e2, band, lattice=lattice)
d26   = se.StokesData.simulate(b32.direct(e2), noise=0.01, key=jax.random.key(1))[:26]  # 1 to 100.6 GHz
high  = slice(3, None)                                                                   # cells 4 to 19
treatments = {"free moments": se.FreeMoments(cells=high), "non-negative cells": se.NonNegativeCells(cells=high), "eSBPL": sbpl}
fits  = {name: se.fit(e2.replace(gamma=g), d26, lattice=lattice) for name, g in treatments.items()}
for name, f in fits.items():
    p = f.predict(dense)                                # 121 channels on the same cells and weights
    print(name, p.I[-1], p.sigma, p.unconstrained)      # sigma is None for non-negative cells, unconstrained is None for a family
refits = fits["non-negative cells"].resample(200, key=jax.random.key(2)).predict(dense)
np.percentile(refits.I, [16, 50, 84], axis=0)
print(se.compare(*fits.values()))
# built: 32 channel rows and 121 prediction rows on one lattice. S0 and S1 shared
# solver: SVD with UPRE, penalised NNLS, bounded least squares with profiled amplitude
# returns: Spectra with noise-only sigma, unconstrained for the free fit (it rises above the band), the extrapolation caveat
```

**G7. Paper E5: two zones behind a common screen**
```python
rest   = {k: known[k] for k in ("B", "eta", "mu", "depth")}
zone   = lambda index, angle, column: se.Population(gamma=se.PowerLaw(index, column=column, **GR),
                                                    phi=se.AzimuthDensity(0.8, angle, unit="rad"), **rest)
e5     = zone(3.1, 0.25, 0.938) + zone(2.1, 0.25 + np.pi / 4, 0.062)
lat5   = lattice.replace(phi=se.AzimuthHarmonic())      # 171 lattice rows
spec5  = se.predict(e5, obs, lattice=lat5)              # 1.8e-3 of I at N = 2
truth5 = se.spectral_basis(e5, obs, lattice=lat5).direct(e5)
d5     = se.StokesData.simulate(truth5, noise=0.01, key=jax.random.key(3)).withhold(np.arange(1, 41, 2))
joint  = se.Population(se.FreeMoments("gamma", "phi"), **rest)                   # joint energy-azimuth moments
one    = se.Population(gamma=se.FreeMoments(), phi=se.FreeMoments(), **rest)     # independent: one shared complex factor
fit_joint = se.fit(joint, d5, lattice=lat5)             # linear. Paper: R* = 17, chi2 = 43.2 / 46
fit_one   = se.fit(one, d5, lattice=lat5)               # bilinear. Paper: R* = 7 + 2, chi2 = 270.9 / 54
print(se.compare(fit_joint, fit_one))
fit_one.params["phi.factor"]
# built: one basis. Its spin-0 and spin-2 blocks serve the forward call and both fits
# solver: SVD with UPRE that keeps (Re, Im) pairs whole. Variable projection over the complex factor, UPRE counts R + 2
# returns: two Fits to compare by chi2. The one-factor print carries the bilinear caveat. Without phi on an explicit
#          lattice, se.predict(e5, ...) raises RoleError (rule R7)
```

**G8. Paper E6: the energy cut-off depends on the local field**
```python
aged    = lambda B: se.AgedPowerLaw(index=2.6, age="19 Myr", loss_field=B, radiation="1 eV cm-3", lo=300, amplitude=A6)
e6      = e1.replace(gamma=se.Conditional(aged, units={"B": "G"}))               # P(gamma given B). B arrives as se.Quantity in G
lat6    = lattice.replace(B=se.Cells.single("5 uG", scale="1 uG", order=2))      # x_B = (B - 5 uG) / (1 uG): 114 columns
print(se.plan(e6, obs))                                 # no lattice: rule R3 resolves B and prints it (Sec. 5.5)
s_joint = se.predict(e6, obs, lattice=lat6)             # 7.8e-3
s_avg   = se.predict(e6, obs, lattice=lattice, average=("B",))                 # field averaged over its marginal, dependence dropped: 8.4e-2
s_avg.ledger["independence:gamma|B"]                    # measured through the scaling-table reference (Sec. 7.2)
aged_f  = se.Conditional(lambda B, *, index, age: se.AgedPowerLaw(index=index, age=age, loss_field=B,
                                                                  radiation="1 eV cm-3", lo=300),
                         params=dict(index=se.Free(2.6, bounds=(1.5, 4)), age=se.Free("19 Myr", bounds=("1 Myr", "1 Gyr"), log=True)),
                         units={"B": "G"})
d6      = se.StokesData.simulate(s_joint, noise=0.01, key=jax.random.key(4))
fit6    = se.fit(e6.replace(gamma=aged_f), d6, lattice=lat6)                     # parameters gamma.index, gamma.age. Amplitude profiled
# built: a fixed-field S1 table at 5 uG. B columns from the field-scaling relation, no further kernel calls. The precondition
#        holds: the eta, mu and depth laws do not depend on gamma or B. fit6 reuses this basis, because gamma is resolved
# solver: none for the predictions. Joint moments by 16 nodes in B times the cut-off rule in energy. fit6: separable, bounded
#         least squares over gamma.index and gamma.age, amplitude in closed form
# returns: two Spectra with measured representation terms (scaling-table reference). The second records the dropped dependence
#          with a measured term. se.predict(e6, obs, lattice=lattice) without average= raises RoleError (rule R1)
```

**G9. Paper E4-KP: the pitch-angle density depends on energy**
```python
kp    = lambda mu: se.AgedPowerLaw(index=2.6, age="19 Myr", loss_field="5.03 uG", radiation="1 eV cm-3", pitch=mu,
                                   lo=300, amplitude=A_kp)                       # P(gamma given mu), b(mu) of PAPER:1025
e4kp  = e1.replace(gamma=kp)                            # mu stays Isotropic() from `known`
lat9  = lattice.replace(mu=se.Legendre(8))              # 513 lattice rows
s9    = se.predict(e4kp, obs, lattice=lat9)             # 8.8e-3 at N = 2, 6.6e-4 with lat9.with_order(4)
s9.plan.roles["mu"]                                     # resolved, degrees 0, 2, .., 8, evaluated at mu = eta (Beaming)
s9.plan.columns.n                                       # 285
# built: 57 columns per even degree, as eta weights (2l + 1) P_l(eta). Odd degrees are dropped: the eta law U[-0.5, 0.5] is
#        even, Parity makes the I and P columns vanish there, and the continuum kernel has no V (Sec. 4.3). Reference: basis.direct.
#        lattice.replace(mu=se.Cells.from_edges([-1, -0.5, 0, 0.5, 1], chart="affine")) is also admitted: eta has a density,
#        and the edges hold its jumps at ±0.5 (Sec. 4.2). With eta=0.5 fixed it would raise RoleError
# solver: none. Moments <P_l(mu) u^r> per cell, mu panels split where the cut-off crosses a cell edge
# returns: Spectrum with the measured representation error
```

**G10. Paper E1M: another field-strength distribution**
```python
e1m    = e1.replace(B=se.Maxwellian(rms="5 uG"))        # the only change
spec_m = se.predict(e1m, obs, lattice=lattice)          # a new S1 table: 6.2e-4
basis.predict(e1m)                                      # raises SpecifiedMismatch: B is Uniform(4, 6) uG in the basis
wrong  = basis.predict(e1m, allow_mismatch=True)        # 3.8e-2 at every order
wrong.ledger["specified:B"]                             # measured
# built: a second S1 table (96 nodes by lookups in the shared S0 table, about 6 ms per node, then disk), keyed by the field
#        law (Sec. 8.2). Channels, cells, weights unchanged
# solver: none. returns: Spectrum. A fit cannot detect this mismatch in the continuum limit (PAPER:833), and its print says so
```

**G11. Full joint expansion in the harmonic regime, with V** [0.8.1 for one reference point, 0.9 for cells]
```python
sim  = se.Samples(gamma=g, B=(B_gauss, "G"), mu=mu, eta=eta, phi=(phi, "rad"), depth=psi, weights=w, column=N_src)
ch   = se.Channels.at(nu_hz, unit="Hz", fractional_width=0.26)                   # smooth channels of finite width
one  = se.Lattice.single(about=sim, order=2, degree=8)
one.n                                                   # 2430 lattice rows
s1   = se.predict(sim, ch, lattice=one, kernel=se.HarmonicKernel(), stokes="IQUV")
s1.plan.columns.n                                       # 1306 basis columns (HarmonicKernel, IQUV)
many = one.replace(gamma=se.Cells.log(5, 100, 6, order=2))
s2   = se.predict(sim, ch, lattice=many, kernel=se.HarmonicKernel(), stokes="IQUV")
# built: the full joint basis through the existing analytic Taylor route. It depends on kernel, channels and reference
#        points only. About 6.5 s and 0.8 GB per cell, one compilation per cell until the traced mask exists
# solver: none. Moments are weighted sums over the samples, split by cell
# returns: Spectra with V and the measured error against the direct sum over the samples (PKG/syncem/model/_direct.py)
```

**G12. Per-pixel fits and 300 GHz maps**
```python
bands = se.Channels.at(band_ghz, unit="GHz", half_width=0.2)
maps  = se.StokesData(bands, stokes=stokes, sigma=sigma, unit="K_CMB")           # stokes (n_pix, n_ch, 3), sigma (n_ch, 3) or per pixel
sky   = se.Population(se.FreeMoments("gamma", "phi"), B="5 uG", eta=se.Uniform(-0.5, 0.5))
ch300 = se.Channels.at([300], unit="GHz", half_width=0.2)
fit   = se.fit(sky, maps, predict=ch300)                # default lattice covering 300 GHz. UPRE per pixel, no cap (I, Q, U data)
m300  = fit.predict(ch300).to("K_CMB")
m300.I, m300.sigma, m300.unconstrained                  # (n_pix, 1), (n_pix, 1, 3), (n_pix, 1, 3)
fit.rank                                                # (n_pix,). The summary prints the rank histogram
sky_I = se.Population(gamma=se.FreeMoments(), B="5 uG", eta=se.Uniform(-0.5, 0.5))      # maps_I: StokesData with I only
fit_i = se.fit(sky_I, maps_I, predict=ch300, rank=se.UPRE(cap=14))   # intensity only: the paper's cap (PAPER:1077), pixels at cap printed
# built: one basis for all pixels and one row at 300 GHz on the same lattice
# solver: shared sigma: one SVD, then a product, a cumulative sum and an argmin per pixel under vmap. Per-pixel sigma: one SVD
#         per pixel in chunks sized by chunk_bytes (7.0e3 pixels of 63 x 171 at 2 GiB). fit.predict recomputes the directions
#         in such chunks, because storing them takes 19 GB at 1e6 pixels (Sec. 8.7). Spectrum.cov is not filled (Sec. 6.4)
# returns: maps with noise-only sigma and, per value, the row fraction outside the retained directions
```

**G13. Measured bandpasses and brightness temperature**
```python
bp   = se.Channels.from_bandpass([(nu_k, R_k) for nu_k, R_k in tables], unit="GHz")
spec = se.predict(e1.replace(gamma=se.PowerLaw(2.6, column=1e12, **GR)), bp, lattice=lattice)
spec.to("K_RJ"), spec.to("K_CMB"), spec.to("MJy/sr")
# built: one S2 row per bandpass, keyed by the content hash of each table. A kernel without lines needs no smooth response.
#        A line kernel goes through fit_bandpass onto smooth reference responses, and the residual enters the ledger
# solver: none. returns: band-integrated intensity divided by the band integral of the unit's reference spectrum
```

**G14. A user-defined family on any variable**
```python
@se.family(variable="B", unit="uG", support=(0.5, 50.0))
def lognormal_field(B, median=5.0, width=0.3):          # unnormalised density, normalised by quadrature
    return jnp.exp(-0.5 * (jnp.log(B / median) / width) ** 2) / B

model = e1.replace(gamma=sbpl, B=lognormal_field(width=se.Free(0.3, bounds=(0.05, 1.0))))
print(se.plan(model, obs, lattice=lattice))             # B averaged: "B.width" is a basis parameter, kind basis-parametric
fit_p = se.fit(model, data, lattice=lattice)            # B averaged: S1 rebuilt per trial width
lat_b = lattice.replace(B=se.Cells.log("0.5 uG", "50 uG", 4, order=2))           # four bounded field cells: 456 columns
fit_r = se.fit(model, data, lattice=lat_b)              # B resolved: the family acts through field moments, one basis
fit_rf = se.fit(model.replace(gamma=se.FreeMoments()), data, lattice=lat_b)      # bilinear (parametrised factor)
lat_s = lattice.replace(B=se.FieldScaling(reference="5 uG"))
fit_s = se.fit(model, data, lattice=lat_s)              # [0.9] push-forward onto the energy cells, one basis
fit_s.column, fit_s.source_column                       # effective column at 5 uG. None: the width is free (Sec. 4.2)
# built: fit_p: one S1 table per trial width by lookups in the cached S0 table, about 6 ms per field node (about 0.2 s per
#        trial at 32 nodes, the count of the narrowest width 0.05, estimate, more after doublings), in memory only.
#        fit_r, fit_rf: one basis, field columns from the scaling relation
# solver: fit_p: bounded least squares over (index1, index2, gamma_break, width), amplitude profiled, finite-difference
#         Jacobian in width, rebuild cost printed. fit_r: separable, field moments of the family per field cell. fit_rf:
#         variable projection over the width with the free-moment SVD inside
# returns: Fit. With free energy moments on lattice (basis-parametric) or on lat_b (fit_rf) the rank is chosen once on a width
#          grid and kept fixed, and the width, not constrained separately (PAPER:562), is reported by its profile chi2
```

**G15. A line of sight given as profiles** [`FieldScaling` lattice: 0.9]
```python
ray  = se.Ray(step=(ds, "pc"), B_vector=(Bxyz, "uG"), thermal_density=(ne, "cm-3"), electron_density=(ncr, "cm-3"),
              electrons=se.PowerLaw(p_s, lo=300, hi=1e7))                        # arrays (n_ray, n_s, ...), third field axis along the ray
spec = se.predict(ray[0], obs)                          # one sightline: direct integration over the steps, depth exact (rule R8)
hi   = se.Channels.log(1, 40, 21, unit="GHz", half_width=0.2)
lat8 = se.Lattice(gamma=se.Cells.log(300, 1e7, 22), B=se.Cells.log("1 uG", "30 uG", 4), eta=se.Legendre(8),
                  phi=se.AzimuthHarmonic(), depth=se.Cells.linear(-60, 60, 48, order=3), order=2)
print(se.plan(ray, hi, lattice=lat8, max_columns=1_200_000))   # tau_max h = 0.35 at 0.8 GHz, the lowest support frequency,
                                                        #   depth bound (tau_max h)^4/4! = 6.3e-4 <= depth_tol = 1e-3: passes.
                                                        #   2 737 152 rows, 1 140 480 columns, 21.9 MB dense per sightline
maps8 = se.predict(ray, hi, lattice=lat8, max_columns=1_200_000)   # (n_ray, 21, 3). Moments: sparse sums over steps
lat  = se.Lattice(gamma=se.Cells.log(300, 1e7, 22), B=se.FieldScaling(reference="5 uG", inclination=True),
                  phi=se.AzimuthHarmonic(), depth=se.Cells.linear(-60, 60, 48, order=3), order=2)      # [0.9]
b    = se.SpectralBasis.build(hi, lat, specified=dict(mu=se.Isotropic()))   # B and eta resolved by FieldScaling, basis at sin(theta) = 1
maps = b.predict(ray)                                   # the pitch law is isotropic (pitch=None), so inclination=True passes its check
# built: lat8: one basis for every sightline. Energy columns from the stack, B columns from the field-scaling relation, depth
#        powers s = 0..3 outside the total degree (Sec. 4.3) by closed-form phase weights, eta by Legendre projection.
#        lat: one basis at the reference B_perp
# solver: none. Each step is a component with Delta factors in B, eta, phi and depth, all on lattice axes (R8). Its moments
#         add to the lattice rows as 14 256 (row, value) pairs per step: 648 rows in each of the 22 energy cells of its (B, depth) cell.
#         maps8.moments is None for many sightlines (Sec. 3.8)
# returns: Spectra per sightline. On lat the order-0 sums are effective columns at B_perp = 5 uG. The electrons start at the
#          continuum limit gamma = 300. Starting them at 100 needs kernel=se.ContinuumKernel(gamma_min=100), with the regime
#          caveat printed. With order=2 on the depth cells the depth bound is (tau_max h)^3/3! = 7.2e-3 and the plan raises
#          DepthExpansionError (2 052 864 rows and 887 040 columns). With the 0.1 to 10 GHz
#          channels the same depth cells give tau_max h = 35, a depth bound of 6.3e4, and the plan refuses
```

**G16. A custom kernel**
```python
@se.kernel(variables=[se.Variable("a", kind="positive", unit="cm", meaning="grain radius"),
                      se.Variable("n_H", kind="positive", unit="cm-3", meaning="gas density")], stokes=("I",))
def grain_emission(nu, a, n_H):                         # emissivity per grain and per Hz, traceable by JAX
    nu_peak = 30e9 * (a / 5e-8) ** -2.0 * (n_H / 30.0) ** 0.1
    return {"I": 1e-40 * (nu / nu_peak) ** 2 * jnp.exp(1.0 - (nu / nu_peak) ** 2)}

grains = se.Population(a=se.FreeMoments(), n_H=se.LogNormal(median=30.0, width=0.5, unit="cm-3"))
lat_a  = se.Lattice(a=se.Cells.log(3.5e-8, 1e-6, 8, order=2, unit="cm"))
fit16  = se.fit(grains, data_I, lattice=lat_a, kernel=grain_emission)            # data_I: StokesData with I only and a unit
both   = (se.problem(free, data_I, lattice=lattice)
          + se.problem(grains, data_I, lattice=lat_a, kernel=grain_emission)).solve()    # two mechanisms, one SVD
# built: jet provider with d = 1 (a in its log chart), quadrature over n_H. Reference weights 1 (no gamma axis, no
#        ReferenceDensity). both: a BasisSum, designs concatenated, moments returned per component key
# solver: SVD with UPRE. returns: the same Fit and Spectrum objects. Nothing in lattice, population or fitting refers to synchrotron
```

## 12. Generality stress table

| Case | Expression in this design |
|---|---|
| (a) cells in `B` and in Faraday depth | `se.Lattice(gamma=se.Cells.log(300, 3e6, 19), B=se.Cells.linear("2 uG", "8 uG", 3), depth=se.Cells.linear(-20, 20, 8), order=2)`. The product cells are rows of the cell table. The real block holds `<1_{b,c} u^r x_B^s d^t>` over all product cells, with `t = 0..N_d` for every `(r, s)` of the `(gamma, B)` lower set (depth outside the total degree), and I and V repeat their depth-free spectra across depth cells when no averaged law depends on depth (Sec. 4.3). `B` columns come from the scaling relation when the `eta`, `mu` and depth laws do not depend on `gamma` or `B`, and from the jet provider otherwise. The plan prints the column count, the routes, `tau_max h` and the depth bound, which passes at order 2 when the lowest channel-support frequency is above 1.57 GHz with these depth cells (h = 2.5 rad m^-2). `B` cells without energy cells (`gamma` averaged) take the jet in 0.8.0 (Sec. 8.3, T5) |
| (b) two Taylor variables, joint monomials, per-variable cap | `se.Lattice(gamma=se.Cells.log(300, 3e6, 19, order=2), B=se.Cells.single("5 uG", scale="1 uG", order=1), order=2)` keeps `u^r x_B^s` with `r <= 2`, `s <= 1`, `r + s <= 2` |
| (c) unknown angular statistics coupled to energy | `se.Population(se.FreeMoments("gamma", "mu"), ...)` on `se.Lattice(gamma=..., mu=se.Legendre(8))`. Linear. Moments are in lattice rows (Sec. 4.4), so `lattice.conditional(moments, "mu")` returns the conditional Legendre coefficients per cell as an `Observable`. After a free fit they come from the minimum-norm vector and carry its origin and the PAPER:718 caveat. With the continuum kernel and an even `eta` law the odd-`l` coefficients have no column: they are not observable from I, Q and U, are listed in `moments.masked`, and are returned as NaN. `fit.observable(c)` gives constrained linear functionals. Separate keywords give shared coefficients and a bilinear fit |
| (d) emission-depth correlation | `depth=se.Cells.single(ref, scale, order=2)` or several depth cells, with `se.FreeMoments("gamma", "depth")` or `se.Samples`. The unknowns are the real block `<1_{b,c} u^r d^s>` (Sec. 4.4). With a known azimuth factor, P is `c` times `spectra2` on those rows, and I uses the depth sums at `s = 0`. Depth powers are closed-form weights. A law that depends on depth takes the Cauchy-product route of Sec. 4.3. The depth check of Sec. 4.6 applies. Depth powers are outside the total degree (Sec. 4.3). `se.Ray` on any 0.8.0 lattice that resolves `B`, `eta`, `phi` and `depth` (G15), with `max_columns` raised for large lattices. Otherwise `RoleError` names the missing axes and the direct route (R8) |
| (e) another phase law. Absorption or conversion | `phase=se.PhaseLaw(tau=fn)` keeps both depth routes. Absorption and conversion are out of scope: they depend on the order of sites along the ray (PAPER:298), so the data are no longer a linear functional of the measure. The hook is the per-step evaluation of `se.Ray`, whose emissivities a transfer solver (`PKG/syncem/transfer.py`) can consume |
| (f) a population known only as moments | `basis.predict(se.Moments(lattice, values))`. The lattice key must match, and the basis applies its column map. `lattice.recentre` and `lattice.coarsen` convert within one chart. The statistics of the basis become assumptions of the result, and `representation` is unbounded |
| (g) priors and a sampler hook | `se.Free(2.6, prior=se.Normal(2.6, 0.3))`, `se.FreeMoments(prior=se.Normal(0, s))` (origin `penalised`), `eqx.filter_jit(se.problem(...).log_prob)` (Sec. 6.6). `FreeMoments` without a prior raises `ImproperPosterior` there |
| (h) channels or noise that differ per pixel | `sigma` with a pixel axis: one SVD per pixel, directions recomputed in chunks or stored with `keep_directions=True` (Sec. 6.4). Missing channels: `mask`. Disjoint channel sets: pixels grouped by set, one `basis.at(channels_k)` per group. Two instruments on the same pixels: `data_a + data_b` |
| (i) family parameters shared across pixels | `se.Free(..., shared=True)` on the family, amplitude per pixel. The pixels enter through one Gram matrix [0.9]. In 0.8 through `se.problem(...).log_prob` |
| (j) hybrid kernels with a check at the switch, kernel per product cell | `kernel=se.HybridKernel(se.HarmonicKernel(), se.ContinuumKernel(), switch={"gamma": 300})` [0.9], or `se.HybridKernel(k_a, k_b, assign=fn)` with `fn(box) -> 0 or 1` over traceable non-harmonic kernels [0.8.0]. Both kernels are evaluated on every face shared by cells of different kernels and at the extreme cells with the assignment bypassed. The difference is the measured term `kernel_switch`, and the build refuses above `switch_tol`. The assignment is kernel-side, so moments compare across kernels |
| (k) uncertain specified statistics | `se.Free` inside the distribution of an averaged variable: rule R6 and the kind basis-parametric, an outer solve with a rebuild per evaluation (host in 0.8). Other routes: `se.Cells` on the lattice (moments of the family, one basis), `se.FieldScaling` [0.9]. With free moments the inner rank is chosen once on a grid of `theta_b` and kept fixed (Sec. 6.1). Free moments of `gamma` with the family on `B` cells: kind bilinear (parametrised factor), variable projection (G14 `fit_rf`). The plan names the route, the cost and the structural degeneracies (G14) |
| (l) a second observing response | `se.StokesData(..., unit=..., response=R, response_uncertainty=dR, discrepancy=e)`: the design is `R C`. The `response` term is a declared estimate from the plug-in `S_hat`, and `discrepancy` is unbounded unless declared (Sec. 7.2). `spectrum.through(R)` for a forward call. A beam that varies inside a channel must enter the channel response of each sightline (PAPER:865) and is out of scope |
| (m) azimuth statistics that depend on the inclination | `phi=lambda eta: se.AzimuthFactor(0.6 * (1 - eta**2))` with `eta=se.Uniform(-0.5, 0.5)` and a lattice on `gamma` only. Rule R5 (iii): the factor is a weight inside the averaged spin-2 kernel at the `eta` nodes, and its fingerprint joins the S2 key. With `eta=se.Legendre(L)` on the lattice, route (iv). The lambda declares no parity, so `eta` takes the full range. `se.Conditional(fn, parity={"eta": "even"})` restores the half range |
| (n) cells on three variables, one a cosine | `se.Lattice(gamma=..., B=se.Cells.linear("2 uG", "8 uG", 3), eta=se.Cells.linear(0, 0.5, 2))` with the continuum kernel: `eta` is outside `Similarity.variables`, so every product cell goes to the jet provider in `(u_gamma, u_B, u_eta)`, and `PlanRecord.routes` says why. `mu=se.Cells.from_edges([-1, -0.5, 0, 0.5, 1], chart="affine")` with `eta=se.Uniform(-0.5, 0.5)` takes columns from `g(mu) = 2 p_eta(mu) K_iso(eta = mu)`, so `se.NonNegativeCells("gamma", "mu")` has a route. With `eta=0.5` fixed it raises `RoleError`: the density on the diagonal `mu = eta` is not a moment |
| (o) averaged variables that depend on each other, depth as a parent | `B=lambda eta: se.LogNormal(...)`, `depth=lambda B: se.GaussianScreen(...)`, `phi=lambda depth: se.AzimuthFactor(...)`. The cached S1 does not apply. `B` given `eta` builds one-node S1 tables per field node (decision 144). Depth given `B` is averaged in S2 at the field nodes, with the characteristic function of the returned screen per node. Depth as the parent of `phi` is averaged by its `rule()` nodes, and a `CumulantScreen` there raises `RoleError`. `phi=lambda B: se.AzimuthFactor(...)` with `B` averaged: the spin-0 blocks keep the cached S1, and the spin-2 field average is taken in S2 at the field nodes weighted by `c(B_q)`. Each route is in `PlanRecord.routes` and tested against direct integration (T2, T15, including steep `c(B)` at field range ratio 100) |
| (p) a celled variable conditional on a free parent, free parameters inside a conditional | `se.Population(gamma=se.FreeMoments(), B=lambda gamma: se.LogNormal(...))` on a lattice with `B` cells: rule R3a, a fixed linear node (shifted correlation) from the free energy moments, with the term `conditional_truncation:B|gamma` (bounded per unit parent mass by `E`, Sec. 5.2). `se.Conditional(fn, params=dict(index=se.Free(2.6)))` puts free parameters inside a conditional (G8) |
| (q) an averaged law conditional on the azimuth | `eta=lambda phi: se.Uniform(...)` or `B=lambda phi: se.LogNormal(...)` with `phi=se.AzimuthDensity(0.8, 0.25, unit="rad")`: rule R4a (a), nested quadrature at the `phi` nodes with `exp(i s_X phi_q)` inside the average (T15). With `phi` resolved by `AzimuthHarmonic`, or in a `FreeMoments` block that leaves the child outside: rule R4a (b), `phi` on `AzimuthFourier(m_max)` with rows `<exp(i k phi) ...>` for `0 <= k <= m_max + 2` (Sec. 4.2). The truncation term is zero when `Kbar(phi)` of every averaged child is a trigonometric polynomial of degree at most `m_max`, and measured otherwise. A resolved child of a specified `phi` on `AzimuthHarmonic` takes the joint moments (PAPER:497-499) |
| (r) components that differ in an averaged statistic | `zone_a + zone_b` with `B=se.Uniform(4, 6, unit="uG")` and `B=se.Maxwellian(rms="5 uG")`, each with `gamma=se.FreeMoments()`: a `BasisSum` of two bases. Forward prediction sums the parts, and a fit concatenates the designs with separate free blocks per part (R7). Two mechanisms on the same data: `problem_a + problem_b` (G16) |

## 13. Decision log

| # | Decision | Alternatives | Reason |
|---|---|---|---|
| 1 | Object core of P2, engine of P4, grammar of P3, surface of P1, delivery of P5 | one proposal as a whole | the three judges named different spines and the same division. P2 alone meets B1 to B7 (J2 Sec. 5.3) |
| 2 | No public `Model` class. The compile product is a `Plan`. The expensive object is an explicit `SpectralBasis`. `Fit` keeps the plan record and the basis | `Model = Population + Lattice + Kernel` (P1, J1). `se.compile -> Model` (P3) | two of three judges. `fit.predict(other channels)` still needs no second object, which was J1's reason |
| 3 | An explicit lattice states the roles. A default lattice follows rules R0 to R8 with R3a and R4a | roles from value types alone (P1). Roles in fit keywords (P5) | J2 Q2. One statement, visible, and T3 holds |
| 4 | Separate unknown keywords are independent (bilinear). A positional block is joint | joint reading (P1). `Joint(..., by=)` (P4). `Independent()` (P5) | all three judges. The only reading under which partitions into joint groups can be stated |
| 5 | Variable names are the six names of `VARS`: `gamma, B, mu, eta, phi, depth`. One spelling | words (P1, P5, J1). `Psi` (P2, P4). Aliases (J3) | two of three judges for symbols. Names extend to other kernels and match the package. All judges reject two spellings. J1's concern about `phi` is met by a typed error message (Sec. 2.5). Open question 1 |
| 6 | A resolved variable that depends on an averaged one: the parent is promoted under a default lattice, `RoleError` under an explicit one. `average=` drops the dependence as a recorded assumption | always promote (P1, P3). Always refuse (P2 text, P4). Silent marginal (P5) | J3 Q6 and J2 Q10. A user's lattice is never changed. The silent marginal assumed independence without a statement |
| 7 | Undeclared direction: `eta` isotropic, `phi` uniform, `Q = U = 0` stated. A fit of `Q`, `U` data with `phi` undeclared raises | inclination 90 deg, azimuth 0 (P1, P3) | all three judges. A 73 per cent polarised default reads as a result (J1 Sec. 5.1) |
| 8 | Bare numbers refused for field strength, frequency, angles, ages and data. Accepted for Lorentz factors, cosines, depths, columns | canonical floats with a range guard (P2, P3). Converters (P5). Suffix keywords (P4) | all three judges. A magnitude guard passes 0.5 as 0.5 G |
| 9 | Conditionals receive parents as `se.Quantity` parent values with a static unit, canonical unless `se.Conditional(fn, units=)` says otherwise. Plain floats in dimensional slots are refused. Checks run once on the tabulation | unit of the declaring slot (P1, P4). Bare canonical floats (v1) | J1 Q5. r1-11: a bare float contradicted the unit rule of Sec. 2.4, and traced parents cannot be checked as Python floats |
| 10 | Undeclared normalisation: amplitude 1 at `gamma = 1e4` in a forward call, free in a fit. `amplitude=` is `dN/dgamma` at the pivot. Families convert to the paper table's coefficient by `paper_coefficient` | unit column on the automatic range (P1, P2, P4). `column=se.Free()` required (P4). The study coefficient as the public convention (r1-10 reviewer) | J1 Q7 and J2 Q24. The paper states `dN/dgamma` at the pivot (eq: index mixture, caption). The table's values use the study coefficient, which v1 did not say (r1-10). Open question 15 |
| 11 | Default energy range by the tail criterion on the band-integrated `I_j`, through the kernel's `EnergyWindow` capability, mapped with the channel-support edges | fixed window in `x` (P1, P4, P5). A quadrature on the S1 table (v1). Channel centres (v2 numbers) | J1 Q8, J3 Q21. It is the paper's rule (PAPER:1059). Monochromatic channels reproduce the paper's single-frequency upper edge (3.0e6 against 2.96e6), and bump channels with `f = 8 %` widen it to 3.3e6. r1-12: the plan builds no basis stage. r2-31: with centres the lower tail at 0.1 GHz is 1.21e-3, above the stated tolerance |
| 12 | Default half-width at most 0.242 | 0.25 (P1, P2, P3, P5) | J3 Q22. The only tested value |
| 13 | `fractional_width=` for channels | `half_width=0.2` (P2, P3) | J1 Q9. It is the paper's `f`, and no fraction sits beside a unit |
| 14 | `se.Lattice(gamma=se.Cells.log(...), mu=se.Legendre(8), phi=se.AzimuthHarmonic())`, stored as a cell table of product cells. `Cells` carries the order | `Space` of `Taylor(cells, order)` families (P2). `Cells` of `Axis` with string kinds (P4). Three fixed axes (P5) | J1 Q10, J2 Q25, J3 Q16. The owner's word for the collection, the paper's word for the parts. `Family` and `Harmonic` collide with the electron family and the harmonic kernel |
| 15 | The azimuth harmonic is an explicit representation, `AzimuthHarmonic()` | no object (J1) | under decision 3 the lattice must be able to state that `phi` is resolved |
| 16 | `weights=` for the reference weights `w` | `scales=` (P2), `ReferenceWeights(slope=)` (P4), `reference_slope=` (P5) | J1 Q11. PAPER:653 and 1069 call `w_b` reference weights |
| 17 | Cell selection in the marker (`FreeMoments(cells=...)`). `select` on lattice and basis as views. Omitted cells go to the ledger | basis view only (P2, P4, J2). Fit keyword (P5) | J1 Q12 and J3. The selection is part of what is unknown, and fits with different selections share one basis. J2's ledger entry is kept |
| 18 | `data.withhold(rows)`, slicing also allowed | `withheld=` on fit (P3). `channels=` on fit (P5) | J1 Q13. Withheld channels cannot enter an estimator (PAPER:718) |
| 19 | Compressed likelihood: the retained combinations are the data argument | `likelihood=se.Compressed(fit)` (P1, P3, P4, J1) | J2 Q16 and J3. It is the wording of PAPER:720, and both data types supply a whitened design and vector. J1's concern about the argument type is met by that common protocol |
| 20 | The flag of a free-moment prediction is `unconstrained`, with the study's definition (`ST/fits_extrap_noisefree.py:46`), described as the norm fraction of the prediction row outside the retained directions | `unresolved` (P1, J1). `resolved_fraction` (P2, P3). `constrained` (P4) | the study's definition makes stored values tests (J1 Q15). "Resolved" already names a role. r1-17: v1 described it as a share of the value, which it is not |
| 21 | The sampled remainder stays in the ledger. The fit print shows measured reference cases | estimate in the print (P3, P4) | all three judges. It overstates by one to two orders (PAPER:897) |
| 22 | The reference of the measured representation term shares no truncation with the basis | the same table for every axis (P4) | J2 finding 5 |
| 23 | Derivative columns define the basis spectra. The cell table is a second cached product, built on first use by the check | table as default source (P4, J1) | J2 Q7 and J3 Q2. Paper definition, and the tail-cell result: in E1 cell 1 the 17-node interpolant of the averaged kernel gives Taylor columns at r = 1, 2 within 3.9e-5 and 4.0e-6 of the cell's own I peak and 1.2e2 and 6.6 of its own P peak (prototype partition). The r = 0 column is a node value and is exact. Open question 5 |
| 24 | The stack provider (cached S1 and per-node mode) is host NumPy in 0.8.0 and JAX in 0.9.0 | JAX port in the first release (P1 to P4, J1) | J3 Q3. No JAX timing of the full stack provider exists. A JAX port of the S1 quadrature takes 36 ms at one field node, and S0 plus lookups 4.5 ms (prototype profile, decision 117). The build is eager and cached, and the host port reproduces the paper numbers by construction. J1 rejected a host engine as the permanent production engine, which this plan does not adopt |
| 25 | Providers are compared at build only where both already exist. The jet comparison runs in tests and on request | generic comparison at every build (P2) | J3: 1.5 s plus compilation on a 0.15 s build. Measured on E1 at N = 2: the jet takes 6.7 s for 19 cells plus a 0.3 s compile, the stack 0.026 s at the same 64 channel nodes and 0.15 s at 400 (profile p2) |
| 26 | The azimuth factor follows its parents (R5): constant and `Cells` parents at assembly, averaged parents inside the S2 average, `Legendre` parents inside the projection | assembly only (v1). Factor always inside the spin-2 spectra (P2). Constant tie only (P3, P4, P5) | J2 Q8 and finding 2, r1-03. PAPER:644 averages a joint conditional of angles and azimuth, which routes (i) to (iv) cover together. Factors of routes (i) and (ii) stay out of every key |
| 27 | Parity pruning is per Stokes block. With one cosine averaged the law must be even and the kernel must declare `Parity`. A row is dropped only when no requested block uses it. No user flag | one mask for all blocks (v1). `parity="even"` (P2, P4). "Parity trait" (P3) | J2 finding 1. r1-08: V lives on the odd degrees that v1 dropped |
| 28 | With `Beaming`, one column per unordered pair with the moment `M_lk + M_kl` | keep one of the two (P3) | J2 finding 3 |
| 29 | Depth expansion is checked at compile time by the bound `x^(N_d+1)/(N_d+1)! <= depth_tol` with `x = tau_max h` (Sec. 4.6, decision 114). A `Ray` defaults to direct integration | no check (P1, P3, P4, P5). `tau_max h <= 1` (v5) | J2 Q11. P3's G15 lattice had `tau h = 90`. The threshold `x = 1` admits 0.052 of I at `N_d = 2` (prototype depth) |
| 30 | A free fit returns retained combinations. `fit.moments` has no covariance. `electron_spectrum()` raises for free moments. `fit.observable(c)` for functionals | covariance on moments (P2). Cell columns as a spectrum (P1, P5). Column with noise sigma in the print (P3) | J2 Q12 and finding 7 (PAPER:718, 856) |
| 31 | The estimator of a linear problem follows the declaration | choice by condition number (P3) | J2 Q15 |
| 32 | Host SciPy `nnls` and `least_squares(trf)` in 0.8.0. SciPy is a base dependency, imported lazily (decision 121). JAX solvers after agreement tests | in-house JAX NNLS first (P2). LM by projection (P1). SciPy under an extra (v5) | all three judges. The installed stack has no bounded or non-negative solver (J3 C13), and the paper's numbers came from SciPy. Open question 4 |
| 33 | Non-negative cells at any order with a build-time column check. Exact kernel values at the points are the `PointMasses` representation [extension] | `OrderError` (P1). `columns="table"` flag (P4) | J1 Q19, J2 Q21. PAPER:803 states only that non-negativity is no longer ensured |
| 34 | Free field-distribution parameters with free energy moments are fitted and reported as not separately constrained | refusal (P4) | all three judges. The degeneracy needs more than the kernel symmetry (PAPER:562) |
| 35 | Basis parameters: the route is a lattice statement (`FieldScaling`, `Cells`) or the basis-parametric kind. The default lattice picks the push-forward when its precondition holds | `hyper=` strings (P2). Hand-written loop (P5) | J2 Q18, J3 (typed routes). Under decision 3 no extra keyword is needed |
| 36 | A population that differs from the basis raises. `allow_mismatch=True`, on every call that takes `basis=`, measures on the host. It covers specified statistics only | print and continue (P5). `mismatch="allow"` (P2, P4). A separate `trust_key=` flag for the lattice key (r2-08 reviewer) | all three judges. A boolean in place of a string. r2-08: the flag was missing from the front door and also bypassed the key check, which `with_key` already covers |
| 37 | `unit=` on `StokesData` is required, with `"cgs"` for the internal unit. Conversion by the band integral inside the channel quadrature | no data unit (P2, P4, P5). Optional unit (v1). Effective frequency (P4) | J1 Q23. r1-30: Jy/sr data read as cgs shift columns and amplitudes by 1e23 and leave `chi2` unchanged |
| 38 | Kernel protocol: `channel(channels, point, *, phase)`, declared `variables`, `azimuth`, `depth`, `traceable`, typed capabilities with preconditions. `@se.kernel` for functions. The existing protocol stays as an adapter | trait struct (P3). Registry keyed by kernel type (P5) | J2 Q22, J3 Q19 |
| 39 | The phase law is an object of the base model. Phase factors multiply | argument of a screen (P1). Phase-weights class (P5) | J2 Q23, Q28 |
| 40 | `Moments` holds values in lattice rows, a static index, a static lattice key, its `origin`, `retained` and `masked`. Geometry methods live on `Lattice` | the lattice as a pytree inside `Moments` (P2, P4). A column-space vector with a column-map key (v2) | J3 Q15. Cell arrays are not broadcast under `vmap`. r1-37: a minimum-norm vector must keep its flag outside the `Fit`. r2-09: lattice methods were undefined in column space |
| 41 | `se.Free` is a static marker. Parameters and evaluated priors live in a named pytree with array leaves | array leaves in the declaration (P2). String-keyed dictionary only (P3) | J3 Q24. New starts and bounds do not recompile |
| 42 | Cache key of every averaged statistic: a hash of its evaluated tabulation. Registered functions also carry import path and source hash as a label, and a transitive-closure hash for the reload path. No tabulation: no caching, printed | path, source hash and captured numeric state (v2). Path and source hash only (v1). Source text (P4). None (P1, P5) | J3 Q13. r1-16: a factory closure gave every parameter value the same key. r2-14: helpers and non-numeric globals were outside the v2 key, while S1 and S2 depend on the function only through the tabulated weights |
| 43 | Disk cache on for kernel tables of fixed statistics, at most 2 GB with least-recently-used removal. Tables for trial values of `Free` parameters in a memory LRU | on for all (P1, P2, P4). Optional (P3, P5). Unbounded (v1) | J3 Q14. r1-24: a multistart would write about 1 GB of S1 tables. Open question 8 |
| 44 | Maps: UPRE per pixel, `se.UPRE(cap=)` and `se.Rank(R)` offered, rank histogram printed | common rank | J1 Q17, J3 Q23. Open question 7 |
| 45 | Intervals of non-negative cells on request | 200 refits by default (P1) | J1 Q18. Cost per pixel |
| 46 | Names `Spectrum`, `StokesData`, `sigma`, `NonNegativeCells(point_masses=17)`, `se.Free`, `AgedPowerLaw(pitch=)`, `PowerLawMixture.moment_form` | `Prediction`, `Data`, `Observations`, `sd`, `se.free`, `joint=se.KPAged`, `form=` strings | J1 Q27, J3. Existing class names and the paper's words. One conditional mechanism for E6 and E4-KP |
| 47 | The top-level class is `SpectralBasis` | `Basis` (P4, J3) | two of three judges and the paper's term. `import syncem` does not load `syncem.model`. Open question 6 |
| 48 | `se.Penalty(alpha=None)` selects `alpha` by UPRE. `se.UPRE` is a rank rule only. The penalty is a fit policy | `Penalty(se.UPRE())` (P2). `penalty` inside the marker (P1, P2, P4) | J1 on G4 and on the double use of `UPRE` |
| 49 | 0.8.0 is additive with the scope of Sec. 10.1. `syncem.model` is never rewired | rewire in 0.9 (P1). Forward only first (P2 table). Whole compiler first (P3) | all three judges. 900 tests fix the present behaviour, and a bridge object gives access to old bases |
| 50 | Moment-map nodes are internal. The public form is `a = L(theta) c` | eight public node types (P3) | J3 |
| 51 | `check=True` or `se.AccuracyCheck(...)`, the same default on functions and methods | `check=False` on the method and `"auto"` on the function (P3) | J1 Sec. 5.1 |
| 52 | Ledger: named tuple of `ErrorTerm` with a static `declared` flag. Sums per kind. `envelope()` sums all valued terms with the weakest-kind label, is `None` with an unbounded term, and names its declared terms. `allow()` requires a note and records an assumption | nine fixed slots (P5). `total()` with the old rule as headline (P3). A sixth kind "declared" (r1-40 reviewer) | J3 Q11. r1-40: v1 left the envelope rule undefined and printed declared terms as bounds. The five kinds of `errors.py` stay unchanged |
| 53 | Corrected facts carried into this document | | PAPER:813 does hold the extrapolation sentence (J2 right, J1's objection wrong, both lines 813 and 850 are cited). The factor 6 to 18 for shifted reference points is from CTX 06, and PAPER:741 gives absolute errors (J3 C16). The per-cell autodiff cost is 0.39 to 0.5 s, and "below 0.01 s" held for the bare kernel only (J2, J3 C1). The jet on E1 measures 0.35 s (N = 2) and 3.0 s (N = 4) per cell on a quiet machine, 260 and 1570 times the stack at N = 2 and 4 at equal channel quadrature (profile p2). PAPER:803 states that non-negativity is no longer ensured at higher order, which is weaker than a guarantee at `N = 2`. The ranks 26 and 61 belong to the 21 fitted channels. All 41 channels give 28 and 77 (r1-43) |
| 54 | The engine contract is a Taylor jet of the `AveragedKernel` in any number of celled variables. `_build_core` is the `(gamma, B)` fast path and oracle | the unchanged `_build_core` as the generic route (v1) | r1-01: `_build_core` differentiates two named variables in affine charts, calls `angular_projection`, which the new protocol lacks, and requires `gamma0 > 1` |
| 55 | The jet provider is in 0.8.0, and cells on any variable of a traceable kernel ship with it | three or more celled variables in 0.9 (r1-18 reviewer, and the second option of r1-01) | owner's rule: generality wins and the implementation is staged. The existing `taylor_tensors` loop already handles any `d` (r1-01 verifier). This settles the conflict between the r1-01 and r1-18 fixes |
| 56 | Shortcut capabilities carry preconditions that the plan checks against the whole parent graph: cached S1, the variables each provider differentiates, field-derivative columns, the `FieldScaling` role (azimuth factor included), half range and parity masks (joint reversal), `FieldScaling(inclination=True)`, the characteristic-function route, closed-form depth weights | the shortcut whenever the kernel declares the capability (v1). Preconditions on independence from celled variables only (v2) | r1-02, r1-09, r1-27, r2-01 to r2-04, r2-06, r2-11. Outside its precondition a shortcut drops terms without an error |
| 57 | With an averaged law that depends on a celled or an averaged variable the stack runs per node: per-node Taylor weights for celled parents, one S1 table per `eta` node for `B` given `eta`, the field average in S2 for laws given `B` | always fall back to autodiff (r1-02 reviewer) | r1-02 verifier: the per-node mode keeps the stack cost and needs no S1 table. r2-01: the cached S1 replaced averaged-on-averaged conditionals by the product of marginals. r3-20: moving nodes are frozen and reweighted, and the jet remains for moving bounds and for laws without a density (decision 115) |
| 58 | Under `Beaming` the roles of `mu` follow the role of `eta`. With a density on `eta`, cells, point masses and samples on `mu` are admitted through `g(mu) = 2 p_eta(mu) K_iso(mu)`. With `Legendre` degrees on `eta`, cells and point masses on `mu` are admitted through `(2k+1) P_k(mu) K_iso(mu)`, paired with `<1_b u_b^r P_k(eta)>` (Sec. 4.2). These `mu` cells must cover `[-1, 1]` and resolve `(2k+1) P_k(mu) K_iso(mu)` up to `k = L_eta`. With `eta` fixed, celled or in point masses, only averaged `mu` and `Legendre` are admitted | averaged or `Legendre` only (v2). A local Legendre projection inside `Cells` (r1-04 reviewer) | r2-05: v2's reason holds only for a fixed or point-resolved `eta`. With a density the effective kernel in `mu` is an ordinary function, and cell columns converge. For `eta ~ U[-0.5, 0.5]` and `p(mu) ∝ e^{2 mu}`, 1, 2 and 4 cells give 2.7e-2, 3.2e-3 and 2.3e-4 at `N = 2` and 1.3e-3, 5.2e-5 and 1.1e-6 at `N = 4`. With `eta` on `Legendre(2)`, 1, 2, 4 and 8 cells on `[-1, 1]` give 3.2, 0.20, 2.1e-2 and 2.1e-3 at `N = 2` (prototype beaming) |
| 59 | `Lattice.index` is unpruned. The plan and the basis hold a static `ColumnMap` with its own key | a pruned index on the lattice (v1) | r1-06: the lattice imports no kernel, and the pruning needs kernel capabilities and averaged laws |
| 60 | Depth cells and depth powers are rows of every block. I and V repeat depth-free spectra across depth cells | depth exponents in the spin-2 block only (v1) | r1-05: a known azimuth factor with depth cells left the unknowns without rows |
| 61 | `Lattice` stores a `CellTable`. `axes` is a view of a tensor table. Selectors accept intervals. `from_cells` and `refine(where=)` are in 0.8.0 | axes only (v1) | r1-07: B3 asks for local refinement and subsets, and per-variable cell numbers are ill-defined after a local refinement. r2-20: `select` already makes non-covering tables, overlap of boxes is an interval test, and the rest of the stack handles general rows |
| 62 | `PlanRecord` (static, no arrays) travels on `Spectrum` and `Fit`. `Plan` and `Problem` hold the lattice and the moment map as array leaves | a hashable `Plan` that holds arrays (v1) | r1-14: hashing fails under `jit`, measured with jax 0.10.0 and equinox 0.13.7 |
| 63 | Specified parameter values are array leaves. Keys are computed eagerly and hash the resolved weights. A `None` key against a key raises unless `with_key` supplied one | static values and a silent skip (v1). A side registry (r1-15 reviewer). A key split into structure, geometry and weights parts (r2-07 reviewer) | r1-15. A registry cannot hash tracers either. r2-07 verifier: the weights belong in the compatibility check, and the defect was `weights=None` standing for two different `w`. r2-08 |
| 64 | Free parameters of averaged laws give the kind basis-parametric, with a host outer solve in 0.8 | refusal in 0.8 with a `RoleError` (r1-24 alternative) | owner's rule: generality wins. The kind adds no object, and `basis_params` already existed |
| 65 | On a `FieldScaling` lattice the order-0 sum is the effective column at `B_0`. `source_column` divides by `<B/B_0>` when the field law is fixed | the electron column (v1) | r1-27, PAPER:562-564. For a log-normal of width 1.0 the factor is 1.65 |
| 66 | Per-pixel noise: directions recomputed in chunks by default. `keep_directions=True` stores them | stored per pixel | r1-34: 6 to 83 GB at 1e6 pixels |
| 67 | `fit.population` is `None` for free moments and the point-mass measure for non-negative cells. `electron_spectrum` keeps its behaviour | the declaration plus `fit.retained`. Rename to `point_mass_weights` (r1-37 reviewer) | r1-37 verifier: `electron_spectrum` already raises for free moments and labels point-mass weights |
| 68 | Module imports follow the layer order, enforced by an import-graph test. Convenience methods late-bind and are listed with their functions. `Ledger` and `ErrorTerm` live in a records layer above the engine. `Observable`, `Spectrum` and `Caveat` live in a leaf `values` layer. `direct` is an engine function | the v1 rule without exceptions. `Observable` in the records layer (r3-14 reviewer) | r1-13. r3-14: lattice methods, `paper_coefficient` and `population.direct` imported later layers. Records sits after lattice. A leaf layer removes the upward imports without reordering records |
| 69 | The 0.8.0 scope is stated by representation | a list of four blocks (v1) | r1-18, r1-19. Field cells, depth cells and joint blocks over any group of 0.8.0 representations use code paths that already exist |
| 70 | The E6 reference uses the fixed-field energy table and the scaling relation | print "not measured" for E6 (r1-36 alternative) | r1-36: the default check gives a measured term in milliseconds without new kernel calls, while kernel calls cost 13.6 s. The prototype direct on the quadrature kernel took 183 s over the 1728 E6 nodes (prototype fieldcells) |
| 71 | Free-moment fits are parametrised in lattice rows through `D S`. `Moments` is always in lattice rows, and rows without a column are masked | column-space vectors with a lift `S^+` (r2-09 alternative) | r2-09: lattice methods were undefined on column-space vectors. The paper's fits have no merged columns, so no paper number changes. Merged Beaming pairs split equally |
| 72 | Every `Lattice` method that takes `Moments` returns `Observable`s with the origin | bare arrays from `column` and `raw` (v2) | r2-42. `Lattice` holds only the `retained` key, so the numerical `unconstrained` comes from `fit.column` and `fit.observable` |
| 73 | Providers declare the variables they differentiate. A product cell with a celled variable outside the set goes wholly to the jet | the stack wherever its law precondition holds (v2). Chain-rule columns in `eta` through `B_perp` (r2-06 verifier, optional) | r2-06. The chain rule is an optimisation with its own acceptance test (open question 18). No route depends on it |
| 74 | Closed-form depth weights and the I and V spin-0 repetition only when no averaged law has depth as a parent. Otherwise a per-cell Cauchy product with the depth Taylor coefficients of the phase-free averaged kernel, taken by the jet or by frozen-node reweighting where decision 115 admits it | full jet differentiation in depth (r2-03 reviewer) | r2-03 verifier: the Cauchy product keeps the closed-form phase weights and reuses the R5 (ii) machinery. The Cauchy product equals the jet to 2.2e-16, and frozen Delta nodes give zero `s = 1` columns (prototype depth) |
| 75 | The characteristic-function route needs depth to be a parent of no averaged law and of no azimuth factor. A characteristic-only screen may not be a parent | route from the depth law alone (v2) | r2-04, PAPER:606 |
| 76 | One precondition for the half range and the parity masks: invariance of the full averaged integrand under the joint reversal. An undeclared parity counts as not even | the marginal law and the kernel's `Parity` (v2) | r2-02: under the v2 rule legal declarations change I by 15 per cent and the magnitude of P by 16 per cent for `eta ~ U[0, 1/2]` and `L = 8` (prototype R5) |
| 77 | `FieldScaling(inclination=True)` resolves `B` and `eta`. G15 starts its cells at the continuum limit | accept the continuum kernel below 300 by `gamma_min=100` (r2-10 alternative) | r2-10. Starting at 300 needs no regime caveat. The alternative is stated in G15 |
| 78 | The `FieldScaling` role, the scaling-table reference and the PAPER:833 caveat need the azimuth factor independent of `gamma` and `B`. Field-derivative columns on `B` cells do not | one condition for every use (r2-11 reviewer) | r2-11 verifier: on `B` cells the factor enters at S3 after the columns |
| 79 | Every distribution owns its quadrature rule, with node doubling as for channels | 16 Gauss-Legendre nodes on the support for every field law (v2) | r2-17: the Maxwellian has no bounded support, the study used 96 nodes, and 16 nodes miss a narrow log-normal by 2 per cent |
| 80 | The default resolved set uses `Kernel.primary`. Factors outside an unknown block are probability laws. An averaged primary variable needs a bounded or normalisable support | the literal `gamma` (v2) | r2-18, B1, B7 |
| 81 | A resolved non-primary variable without a lattice needs a bounded range from `support=` or from its law's quantiles | one cell over a declared support that unknowns lack (v2). An edgeless cell at the median (r2-19 reviewer) | r2-19 verifier: an edgeless cell has no remainder control |
| 82 | `se.Ray` is in 0.8.0 as a sum of per-step components. Only a `Ray` on a `FieldScaling` lattice waits for 0.9 | `Ray` in 0.9 (v2) | r2-21, B6. The direct route and 0.8.0 lattices use parts that ship in 0.8.0 |
| 83 | A specified conditional of a resolved child on a free parent compiles to a fixed linear node (R3a) in 0.8.0 | `RoleError` in 0.8 with the node in 0.9 (r2-22 alternative) | owner's rule: generality wins. r3-01: the node is the shifted correlation `sum_k h_k a_parent[r + k]`, the moment-side adjoint of the R5 (ii) Cauchy product. v3 wrote a Cauchy product |
| 84 | Free parameters inside a conditional go through `se.Conditional(fn, params=)` | `Free` markers found by calling the function (r2-23 alternative reading) | r2-23. Markers created inside a function body are not leaves of the declaration, and the v2 "frozen" rule blocked re-parametrisation |
| 85 | `Cells` and distributions record the unit they were given, and placement checks it. Angle ranges on a cosine need `se.UniformAngle` | conversion in constructors without a record (v2) | r2-26: `se.Uniform(4, 6)` on `B` was read as 4 to 6 G |
| 86 | A basis-parametric fit with a data-dependent inner rank fixes the rank chosen on a grid of `theta_b`. Tied directions are reported by profile `chi2` | UPRE re-selected at every trial (v2) | r2-29: the outer objective jumps at rank changes |
| 87 | `specified:<variable>` covers the declared law with its parent set, and the plan prints an "assumed" row | an `independence:` term per pair of averaged and resolved variables (r2-44 alternative) | r2-44. One term per statistic keeps the print short and states the independence |
| 88 | The `remainder` term follows `Moments.origin` | the same rule for every origin (v2) | r2-39: the inequality needs a non-negative measure and true moments |
| 89 | The continuum adapter sets `x_min` from channels and lattice | the default `1e-6` (v2) | r2-36: the default raises in the last paper cell at 0.1 GHz |
| 90 | The E4-JP acceptance value of `paper_coefficient` is the truth -5.11590 | -5.11631 (v2), a noise-free fit value | r2-12 |
| 91 | The "specified" line of a fit print states what chi2 can and cannot test | "This fit tests none of them" (v2) | r2-37: PAPER:836 rejects the E5 one-factor fit by chi2 |
| 92 | The fit print of Sec. 7.1 is the reference rendering of the defaults, caveats and reference-model rules, checked by a test | an example print (v2) | r2-15 |
| 93 | R3a keeps the weight ratio `w_b / w_{b,c}` | no ratio (r3-01 verifier) | the verifier read the moments as unnormalised. Sec. 4.5 scales them by `w`, and the ratio is 1 under the default weights, so both readings agree there |
| 94 | Cached S1 is decided per Stokes block. It needs no averaged law, conditional or azimuth factor of that block with an averaged `B` parent, and `B` in no joint block with another averaged variable. `phi` given an averaged `B`, or a joint block holding `phi` and `B`, takes the spin-2 field average in S2 at the field nodes. The spin-0 blocks keep the cached S1 | a `c`-weighted S1 keyed by `c`'s fingerprint (r3-02 alternative) | r3-02: v3 admitted `c(B)` and returned `<c><K_P>`. The difference `<c K_P> - <c><K_P>` is 5.9 per cent of `<c><K_P>` (5.6 per cent of `<c K_P>`) for `gamma ~ PowerLaw(2.6)`, and 2.2 to 36 per cent of `<c><K_P>` for one electron energy (prototype R5). The S2 route reuses the mode of laws given `B` and needs no new table |
| 95 | Under `Beaming`, `K_nat -> 2 K_iso delta(mu - eta)` is stated once, and every column and the folded `FieldScaling` weight cite it | per-bullet normalisations (v3) | r3-03: `Cells` on `mu` with `eta` in `Legendre` gave half the emission |
| 96 | Half-range and parity-mask check (d) on the partner cosine's representation. One halved cosine per block | the law of the partner alone (v3) | r3-04. Cells, point masses, samples and `Ray` steps on the partner are pointwise evaluations, and `∫_{-1}^0 p K(mu, eta) dmu = ± ∫_0^1 p K(mu, -eta) dmu` |
| 97 | Samples of `mu` paired through `Legendre(L)` under `Beaming` give an unbounded `representation:angular` and an assumption record | `L` against `2L` printed as an estimate (r3-05 reviewer) | r3-05 verifier: the paired sum does not converge as `L` grows |
| 98 | R4a: an averaged law conditional on `phi` needs `phi` averaged with nodes in 0.8.0. A resolved child of a specified `phi` on `AzimuthHarmonic` takes the joint moments (PAPER:497-499). `AzimuthFourier(m_max)` for the other cases of R4a (b) ships in 0.8.0 (decision 122). `UniformAzimuth` gets a uniform rule and may be a parent | `UniformAzimuth` as a parent raises (r3-06 verifier). `AzimuthFourier` staged to 0.9 (v5) | r3-06, B5, PAPER:164-166. A uniform law has a density, and owner's rule keeps the more general reading. Open question 19 |
| 99 | Field cells with `gamma` averaged take the jet in 0.8.0. A per-node stack mode over `gamma` nodes comes with the 0.9 stack port | the per-node mode in 0.8.0 (r3-07 verifier, first option) | r3-07: the jet covers the case with no new code. The identity `d/d ln B = 1 + (1/2) d/d ln gamma` makes the 0.9 mode need no derivative of the electron law when its nodes and weights do not depend on `B`. A `gamma` law with `B` as a parent takes the jet: frozen at `B_b`, the per-node sum is off by 1.2e-2 to 1.8 (prototype fieldcells) |
| 100 | `max_columns` on every front-door call, on `Lattice.default` and on `SpectralBasis.build`, applied to every lattice. `MomentIndex` is arithmetic. `Ray` and `Samples` pair into sparse `(row, value)` pairs, and `Spectrum.moments` may be `None` | cap on default lattices only. A tuple of rows | r3-08: G15 had 1 140 480 rows and 549 120 columns with depth inside the total degree. With depth outside it and depth order 3 it has 2 737 152 rows and 1 140 480 columns (Sec. 4.3, decision 114). A tuple of the 1.14e6 rows of v5 costs 3.6 s and 0.5 GB |
| 101 | `response` is an estimate marked declared. A separate `discrepancy` term, unbounded unless declared | `response` as a bound (v3) | r3-10: `fit/_linear_core.py` returns an estimate from a plug-in `S_hat`, and unbounded without a discrepancy |
| 102 | Per-cell kernel choice lives on the kernel side (`HybridKernel(assign=)`), in 0.8.0 for traceable non-harmonic kernels | `CellTable.kernel` (r3-16 reviewer). All per-cell choice in 0.9 (v3) | r3-16 verifier: a kernel in the cell table would enter `Lattice.key` and make moments incomparable across kernels. Only harmonic entries need the traced mask |
| 103 | `BasisSum` for sums whose components need different bases. Sums of problems on the same data ship in 0.8.0 | `RoleError` for multi-basis fits (r3-17 minimal fix) | r3-17. The composite basis keeps the generality of R7, and a same-data sum is a concatenation. The Gram route is needed only for per-pixel family fits and shared parameters |
| 104 | Per-node mode freezes nodes at the cell reference point and reweights by the density ratio, with a coverage check. This needs a law with a density on a fixed support, and other laws take the jet (decision 115) | jet for every location-scale conditional (r3-20 reviewer) | r3-20 verifier: `jacfwd` of a location-scale rule's weights at frozen nodes is zero, and reweighting matches the true derivative to 5e-8 |
| 105 | `se.fit(..., predict=)` sets the default range from fitted and predicted channels. `basis.at` re-runs the `EnergyWindow` check | re-run at predict time only (r3-23 reviewer) | r3-23 verifier: a fitted lattice cannot be widened, because it sets the minimum-norm vector |
| 106 | `representation` against the direct integral inside the cell boxes, `outside_lattice` outside them with the sign of decision 113, both printed | one whole-support reference (v3 table) | r3-25: v3 defined the reference two ways and either double counted or hid a term of the size of the expansion error. The unsigned v5 term left a residual of 2 `abs(outside_lattice)` (prototype partition) |
| 107 | Origin `penalised`, `unconstrained` with filter factors, `ImproperPosterior` without a prior | separate `prior` and `nonnegative` origins and a `fill:` term (r3-26 reviewer) | r3-26 verifier: the existing unbounded `unconstrained` term carries the shrinkage once defined for filtered fits |
| 108 | `fit.profile_max` returns an `Observable`. No `bound` alias | `fit.bound -> Array` (v3) | r3-29: the refit drops the penalty (PAPER:1148), and "bound" is a ledger kind. Nothing has shipped, so no alias is kept |
| 109 | `amplitude=` of a family with a cut-off is the uncut coefficient at the pivot | density at the pivot with a `ValueError` below the cut-off (r3-32 fallback) | r3-32: the density at the pivot is zero for `gamma_c <= 1e4`, and `paper_coefficient` was NaN. The study uses the uncut coefficient (`ST/lib/ensembles.py:12-13`), so no stored value changes |
| 110 | G12 uses UPRE without a cap for I, Q, U maps and shows the cap only for intensity-only maps | `cap=14` for all maps (v3 G12) | r3-33, PAPER:1077 |
| 111 | Kind bilinear (parametrised factor) by variable projection. Larger bilinear factors start from the leading singular pair | nonlinear kind without a rank rule. Grid start in any dimension (v3) | r3-34 |
| 112 | S1 cost quoted per field node, by S0 plus lookups (decision 117). Provider agreement by checks (a) and (b) of decision 118: fail at 1e-10 of the channel column maximum, warn at 1e-8 of the cell's own peak between derivative providers only. Per-pixel SVD chunks from a memory budget | 2.1 to 3.8 s per table. An unstated own-peak threshold. Chunks of 1e5 pixels (v3). A 1e-5 own-peak warning for every provider pair (v5) | r3-37, r3-40, r3-41. The 1e-5 warning fires on every paper build through the table (prototype partition) |
| 113 | `outside_lattice` is signed, `basis.direct(pop, within="cells") - basis.direct(pop)`, and computed by its own quadrature on the uncovered support. Then `representation + outside_lattice = predict - basis.direct(pop)` channel by channel and per Stokes parameter. Every reported share uses the full-support `I_j`. The print gives the magnitude `abs(outside_lattice) / I_j`, and `envelope()` sums the two maxima. `within="cells"` integrates on panels that break at every cell face | the unsigned term `direct - direct(within="cells")` (v5). A node mask on a face-blind quadrature. Shares of the in-cell intensity | prototype partition: with the old sign the residual is 2 `abs(outside_lattice)`, 3.4e-5 to 2.0 of `I_j`. The signed identity holds to 1.7e-15 to 3.0e-15 (1.2e-12 on one cell of half-width 4.60). The node mask errs by 6.6e-7 to 2.0e-2 of `I_j`. The reported maxima bound the total by factors 1.00 to 26.7, because the two terms can have opposite signs |
| 114 | Depth powers are outside the total degree: every row of the other variables carries `s = 0..N_d`. The depth check computes `x = tau_max h` at the lowest channel-support frequency and the bound `x^(N_d+1)/(N_d+1)!`, records it as `depth_truncation` of kind bound, and raises `DepthExpansionError` above `depth_tol` (default 1e-3). The admitted `x` is 0.001, 0.045, 0.182, 0.394 and 0.654 for `N_d = 0` to 4 (Sec. 4.3, 4.6) | depth inside the total degree with `tau_max h <= 1` (v5) | prototype depth: the bound holds in all 80 cells (ratio at most 0.62). The old threshold `x = 1` admits 0.61, 0.22, 0.052, 0.0097 and 0.0015 of I for `N_d = 0` to 4. Depth inside the total degree gives 6.2e-3 of I at `x = 0.1`, `N = 2`, against 3.8e-5 outside it, and the bound needs depth outside the total. G15 moves to depth order 3 (bound 6.3e-4) |
| 115 | Frozen nodes are used only with density reweighting on a fixed support: the per-node mode, R3a's `g[c, s]` on nodes fixed in the child cell's chart, and depth Taylor coefficients. In the per-node mode and for depth coefficients, a law without a density (a Delta law) or with parent-dependent bounds takes the jet. In R3a a child law without a density raises `RoleError`. The own `rule()` nodes of a conditional times a cell indicator are used only for a child cell without edges | frozen nodes without reweighting, or a conditional's own nodes times a cell indicator (v5 text of Sec. 4.3 and 5.2) | prototype core, R3a, fieldcells and depth: when the derivatives of moving nodes do not enter the columns, the error is 1.0e-1 (core), 1.3e-3 to 1.8 in `h_1`, `h_2` (R3a), 1.2e-2 to 1.8 (per-node identity with a `B`-dependent cut-off), and a frozen Delta law gives zero `s > 0` columns (depth). Reweighting on a fixed support equals the jet to 7.5e-16 |
| 116 | R3a: the `conditional_truncation` term of a child and its parent is bounded by `sum_b m_b E[b, j]`, with `E` computed before any data from one evaluation of `g` on a grid per parent cell. Its kind is bound per unit parent cell mass, `estimate` with the fitted cell masses plugged in, and not_applicable for a specified parent (nested quadrature) and under `NonNegativeCells`. The plan prints the cells where the bound exceeds 1e-3 of `I` and the cells with `zeta_b > 2` and a child edge in reach | unbounded for a free parent (v5). Averaging the child as the remedy (v5) | prototype R3a: the bound holds on all 2544 grid records, on narrow-edge cases and on four parent shapes, at 12 to 19 times the actual on E1. The E1 continuum truncation is 8.8e-5 (`N = 2`) and 2.4e-6 (`N = 4`). The averaged-child prediction equals the node prediction to 1.1e-14, so averaging removes only the child-direction representation error |
| 117 | S1 by an S0 table in `ln x`, built once, and lookups at `ln y - ln B_q` per field node, in place of a per-node S1 quadrature | per-node S1 quadrature (v5). A JAX port of that quadrature | prototype profile: S0 once in 0.38 s, then about 6 ms per node (4.5 ms at 1 node, 48 ms at 8), against 0.32 to 0.34 s per node in NumPy and 36 ms (1 node) and 248 ms (8) for a JAX port. Agreement with the study quadrature: 3.4e-13 (`ln T`) to 3.7e-8 (r = 6), pointwise near sign changes (decision 140 restates it as 3.3e-10 of the peak). A basis-parametric trial drops from about 8 s to about 0.2 s, and `B` given an averaged `eta` from about 65 s to about 1.2 s (estimates). The effect on a fit was not measured |
| 118 | Provider checks at `r <= 2` use one channel quadrature on both sides. (a) Derivative providers (stack, jet, core) against each other: the build fails above 1e-10 of the channel column maximum and warns above 1e-8 of the cell's own peak, and `AccuracyCheck(own_peak_tol=...)` applies to (a). (b) The 17-node table against the default provider, against the channel column maximum only: the build fails above 1e-10. A table that fails (b) on the probe cells is not used as the representation reference or for the remainder table, and the record says so. No own-peak test applies to the table | a 1e-5 own-peak warning for every provider pair, the table included (v5) | prototype partition: (a) at most 4.0e-12 of the channel column maximum for half-widths 0.023 to 4.60, own peak at most 2.3e-10 for r <= 4 and 1.0e-9 at r = 6. (b) 2.9e-13 at half-width 0.242 and 2.3e-14 at 0.023, 9.0e-4 at 0.92 and 0.41 at 4.60. In E1 cell 1 the table misses its own P peak by 1.2e2 at r = 1. At 64 against 400 channel nodes P differs by 1.4e-8 of the channel maximum for any provider |
| 119 | Batched family fits run as a jitted `vmap` Levenberg-Marquardt in 0.9.0 (Sec. 10.1), with SciPy `trf` as the oracle. NNLS stays on the host and batches through a process pool with BLAS pinned to one thread (`OMP_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`) | `trf` per spectrum. JAX FISTA for NNLS. Default BLAS threads in the pool | prototype profile: `trf` 2.39 ms per spectrum (12 evaluations), batched LM 60 us per spectrum with parameters within 4.3e-7 of `trf`. NNLS on 111 x 272: 2.96 ms serial, 0.51 to 0.71 ms per spectrum in the pinned pool, 1.05 ms with default BLAS threads. FISTA is slower and inaccurate at cond 1.5e18 |
| 120 | No Rust or other new native code in 0.8 and 0.9. Named seams: host-only kernels, the certification arithmetic backend (`gmpy2` or `python-flint`/Arb in the certification extra) and the harmonic host driver. The harmonic driver is vectorised in NumPy first, and Rust is revisited only if a vectorised driver still exceeds about 0.3 ms per channel point. A smooth tabulated `F`, `G` (piecewise Chebyshev in `ln x`) is the candidate to cut the jet primal. The `Ray` direct reference uses tabulated `F`, `G`, and the basis route on a lattice is the production path (R8). Record: NATIVE.md | a Rust core for the stack, the jet, the S1 build or the ray pairing | prototype profile: no hot path is interpreter-bound in a way that Rust would fix better than an algorithmic change or batching. The largest factors are algorithmic or come from batching: S0 plus lookups per field node, tabulated `F`, `G` (45 times at 1.1e-11 of I), the basis route for rays, batched LM (40 times) and per-pixel SVD on all cores (13 times). NATIVE.md Sec. 3 gives the S0, ray and harmonic factors as re-derived from the profile files. The jet compile cost is a JAX cost. Certification spends 73 per cent in pure-Python `mpmath`, and the harmonic host route (2.2 to 2.9 ms per point) is mostly per-call overhead |
| 121 | SciPy is a base dependency. The non-negative and bounded solvers import it lazily, following `high_order_harmonic.py:138-146`, and the publish smoke test installs it | an extra `syncem[fit]` (v5 recommendation) | owner, open question 4 (OWNER_DECISIONS.md). Import time matters, so the lazy import stays |
| 122 | `AzimuthFourier(m_max)` ships in 0.8.0, and R4a (b) routes through it instead of raising. An averaged child of a `phi` resolved by `AzimuthHarmonic`, and a child of a `phi` in a `FreeMoments` or `NonNegativeCells` block outside that block, take the rows `<exp(i k phi) ...>`, `0 <= k <= m_max + 2`. Its truncation term and acceptance tests are in step 7 of Sec. 10.2, and 0.8.0 grows to about 76 days | `RoleError` in 0.8.0 naming route (a) and `average=`, with `AzimuthFourier` in 0.9 (v5 recommendation) | owner, open question 19 (OWNER_DECISIONS.md), about 4 more days. The truncation term is zero when `Kbar(phi)` of every averaged child is a trigonometric polynomial of degree at most `m_max`, and measured otherwise (Sec. 4.2) |
| 123 | `fit/_linear_core.py` is not edited. The new fitting layer reuses `solve` and `linearised_bias` and computes the split `response` and `discrepancy` terms in `fitting/linear.py` | editing `bias_bound` in place (v6 Sec. 9.2) | the Sec. 9.7 guard pins the v0.7.10 hash of every existing file, `syncem.model` stays unchanged in 0.8, and the manuscript cites v0.7.10 |
| 124 | `Channels.log` and `Channels.at` take `half_width=x` (`h_j = x nu_j`) beside `fractional_width=f` (`h_j = f nu_j / sigma_rho`). The paper's configurations are written with `half_width=0.2` | `fractional_width=` only (decision 13) | `sigma_rho = 0.39764`, so `f = 0.08` gives `h_j = 0.2012 nu_j`. PAPER:1069 states `h_j = 0.2 nu_j` and labels it `f = 8.0%` after rounding. With `half_width=0.2` the channels equal the study's to 3e-15, with `f = 0.08` they differ by 0.6 per cent |
| 125 | `SyncEMError` and `UnitError` are defined in `syncem.units` and re-exported by `values.errors`. Declaration objects meet `Variable.place` through a placement protocol (`given_unit`, `is_point`, optional `carries_values` for declarations that hold no values of the variable, `as_cosine` for a point given as an angle on a cosine, `unit_remedy`, `angle_remedy`, `noun`), so `base` imports no distribution or lattice class. `Variable.default` is the name of the neutral declaration | `UnitError` in `values`, `Variable.default` as a `Distribution` | `units` precedes `values` in the layer order of Sec. 3.1 and raises `UnitError` itself, and `base` precedes `distributions` |
| 126 | `Lattice(..., depth_in_total=True)` puts depth inside the total degree, the index set of `Lattice.single` and of the Sec. 4.2 benchmark (2430 rows). The default keeps depth outside (Sec. 4.3) | a private flag of `Lattice.single` | the benchmark lattice can be written and tested before the population layer exists, and the choice is visible in the key |
| 127 | The paper's energy range is `[300, 2962642.632683605]`, the study's `Config.gamma_hi`. `2.9626e6` in the examples of this document is a rounded display value. `se.paper` and the acceptance tests use the study value | `2.9626e6` | the rounded end moves the edges by 1.4e-5 relative and the reference weights accordingly, above the 1e-13 agreement the tests pin |
| 128 | `Lattice.representations` holds the non-cell representations. Cells live in `Lattice.cells`, and `Lattice.axes` gives both on a tensor table. `select` recomputes the default weights on the range of the selected cells | celled variables in `representations` | a non-tensor table has no single `Cells` per variable. The default weights are normalised on the lattice range (Sec. 4.5) |
| 129 | `AveragedKernel(kernel, averaged, legendre=, fixed=, phase=, outer_batch=)` takes one quadrature rule per averaged variable, a function of the point that returns nodes and probabilities, so laws may depend on resolved variables and on averaged variables listed before them. The distributions layer supplies the rules, and `engine.average(kernel, distributions)` wraps them | distributions as the only input | the engine and its providers are testable before the distributions layer, and the rule is the one contract the providers need |
| 130 | The continuum adapter's `resolved(channels, gamma_max, b_perp_max)` sets `x_min` to half of `x` at the lowest channel-support node, the top lattice edge and the largest `B_perp`. `x_min=None` on a direct call takes the model default 1e-6 | a fixed `x_min` | the default refuses the last paper cell at 0.1 GHz (x = 3.6e-7 at 6 uG on 64 channel nodes), and a factor 2 below the smallest `x` met keeps the guard active for points outside the lattice |
| 131 | The core provider takes `gamma` and optionally `B` cells and no `Legendre` axes. Legendre axes go to the jet (step 4) | Legendre pairs through the model index | the model's pair layout differs from the lattice index, and the jet covers every Legendre case |
| 132 | The kernel protocol's `channel` takes `phase` as a `PhaseLaw`, `False` (phase-free) or `None` (no phase), and returns `P` with one weight row. Depth Taylor weights act per channel node in the engine through `per_node=True` | the model's `PhaseWeights` with several weight rows | one phase contract for every kernel, and the closed-form depth weights act inside the channel integral (Sec. 4.3) |
| 133 | `EnergyWindow.window(averaged, channels, *, density, eps)` takes the averaged laws as `{name: (nodes, probabilities)}`. The continuum window reads `averaged["eta"]`, isotropic when absent, adds the closed-form tail `c y^a`, `a = (p - 1)/2 + 1/3`, below its grid, and refuses `density <= 1/3` | the averaged kernel object as the first argument; no tail below the grid | the window needs only the laws of the averaged variables. Without the tail the lower edge is 5 per cent high at `p = 1.5` and a factor 57 at `p = 1.0`, which makes `gamma_hi` too low (step-3 review) |
| 134 | The jet nests forward mode level by level and differentiates at each level only the directions that some exponent needs there, with the derivatives of the smallest-cap variables taken innermost: the `level_plan` of `_harmonic_taylor.py` generalised to `d` directions. Cost per cell `prod_i (1 + |D_i|)` primals (24 for caps (4, 1) at total 4, against 81 for full nesting) | nested `jacfwd` in every active direction at every level (prototype) | decision (a) of Sec. 8.3 asks for the cap-aware nesting. The jet equals the core provider to 1.5e-16 at `d = 2` |
| 135 | `Variable(name, meaning=, kind=, unit=, bare=None, usual=None, default=None, display_unit=None)` by keyword. The unit is stored canonical, `bare=None` follows the dimension (`syncem.units.BARE`), `display_unit=None` takes the unit given | positional fields only | the G16 spelling `se.Variable("a", kind="positive", unit="cm", meaning=...)` |
| 136 | `@se.kernel` returns a `UserKernel` whose `primary` is its first variable and whose function receives point values in canonical units, the azimuth and the depth excluded: the depth enters `P` through the phase law only, so Sec. 4.3 holds. `describe` carries a hash of the function's code and closure values. `physical_error` is unbounded | a primary of `None` | a user kernel is resolved by default in its first variable, as G16 does with `a` |
| 137 | The Taylor-mode path for one-parent conditionals (decision (a) of Sec. 8.3, T16) moves to step 7, which tests conditionals with parent values (Sec. 10.2). The jet of step 4 uses nested forward mode only | Taylor mode in step 4 | `jax.experimental.jet` does not run on the `F`, `G` of `syncem.ultrarel` (`eqx.error_if`, broadcast `jnp.where`), and no step-4 acceptance item depends on it |
| 138 | The engine rule is `NodeRule(fn, parents=(), density=None)`: `fn(point) -> (nodes, probs)`, the declared parents (`None` for undeclared) and the density `density(x, point)` on the rule's support. `fixed_rule` and `uniform_rule` return it, with a density only when the ends are constant. Bare callables still work and route the product cell to the jet. The distributions of step 6 compile to it | a density on the `AveragedKernel` level. Parents found by tracing the rule | per-node mode needs the parents (which laws feed which) and a density (frozen-node reweighting, decision 115). A traced parent set misses parents read through Python control flow |
| 139 | The frozen-node check compares the reweighted weights with a jet of the rule itself. When the rule's nodes do not move with `gamma`, the reweighted series equals the rule's weight jet node by node to 1e-8 of `max(1, sum_q |w_q^(k)|)`. When they move, for `t = (x - centre) / spread` on the reference nodes and `m = 0..3`, `sum_q w_q^(k) t_q^m` equals the Taylor coefficients in the cell chart of `sum_q p_q(gamma) t(x_q(gamma))^m` to 1e-8 of `max(1, sum_q |w_q^(k)| |t_q|^m)`. Weights amplified beyond `max_k sum_q |w_q^(k)| / sum_q p_q = 1e4` refuse. Non-finite values, a density that is not positive and finite at a node with positive probability, and a rule or density that JAX cannot trace in the parent also refuse. The check bounds moments. A law whose nodes do not resolve the kernel gives a quadrature error shared with the jet, reported by the tests and not certified as jet agreement. Node shifts across the cell do not enter | the reweighted mass at the cell ends within 1e-8 of one and the node shift below half a node spacing (v6 text of Sec. 8.3). The normalisation identity `sum_q w_q^(k) = delta_k0` alone. The four moments alone for fixed nodes. A tolerance scaled by the amplification without a bound | the columns are Taylor coefficients at the reference point. Step-5a probe with a log-normal `B` whose median moves with `gamma`: reweighting equals the jet to 3.5e-11 with nodes moved by 19 of their gaps, which the shift check refuses. The normalisation identity passes a uniform law that translates at constant width (column errors 8 to 21 per cent) and a density whose parent dependence differs from the rule's, and the `m = 1` test fails both. Re-review: on 8 fixed Gauss-Legendre nodes a density error along `P_4` passes the four moments and gives 4.0e-4 to 2.4e-3 against the jet, and the node-by-node identity refuses it. A law of width 0.01 across a cell of `h = 4.6` at order 4 gives amplifications 928, 5.8e5 and 3.6e8 at slopes 0.02, 0.1 and 0.5 with stack-jet errors 1.7e-11, 8.5e-9 and 9.8e-6, so the error is 1.5e-14 to 5.4e-14 of the amplification, and the bound keeps it below about 5e-10. The accepted test cases reach 2.8e3. Where the law's nodes do not resolve the kernel's tail, both methods carry the law quadrature: at width 0.8 on 16 Gauss-Hermite nodes the stack is 1.2e-3 from the 16-node jet, and the 16-node jet is 9.4e-3 from the 64-node jet |
| 140 | The S1 key names the method, `lookup` or `quadrature`. S1 accuracy is stated against the quadrature table and against the peak of `|T^(r)|` | one key for both methods. Pointwise relative errors | the two tables differ at 1e-13 to 1e-10, so a shared key would make the oracle tests read lookup tables. High orders of `T^(r)` change sign, and a pointwise relative error near a root measures the distance to the root |
| 141 | Step 6 takes the quarter-cell-shift row of Sec. 9.6 (it needs `predict`) and the comparisons of T2 and T5 against direct integration (it needs `direct`). Step 5 compares the stack against the jet and against the study oracle. T2 at `N = 5, 6` compares the lookup stack against the quadrature stack | step 5 builds a direct integrator of its own | `direct` with the kernel's capabilities is step 6. The jet of step 4 caps its cost at `N = 4` on these lattices |
| 142 | Step 7 takes `phi` given `B` in the spin-2 block of S2 (it needs the azimuth factor `c(B_q)` of R5), conditionals under `Beaming` (`eta` and `mu` given `gamma`) and `B` given a resolved depth (it needs depth cells). Step 5 refuses celled variables other than `gamma` and `B` with `StackRefused`, and the caller takes the jet | these routes in step 5 | `AzimuthFactor`, `Beaming` and depth cells are built in step 7 (Sec. 10.2) |
| 143 | Private modules `engine/field_average.py` (S1) and `engine/_stack_series.py` (per-node walk, field cells, `stack_cell`, `stack_lattice`) join the Sec. 9.1 layout | all in `provider_stack.py` | the 400-line file limit |
| 144 | The stack takes only `ContinuumKernel` itself (exact type), whose `F` and `G` S0 holds. It reads the physical `F` and `G` and not the kernel's numerical settings (`n_nodes_F`, `tail_cutoff`, `x_min`). One S0 table per memo serves every S1 table, rebuilt when a law needs a wider `ln x` range, with 10 e-folds beyond each short end. Per-node mode builds one-node S1 tables keyed by method and field node, for any parent structure. Field nodes at `B = 0` and zero weights are dropped, and inclination nodes at `|eta| = 1` add zero (`B W(y / B)` and `s B W(y / (s B))` vanish there). A law that reads a celled variable it does not declare refuses, by an order-1 jet of its nodes and weights once per rule and cell. The S1 grid extends upward on the study spacing to `ln(700 B_max)` when the strongest node exceeds about 1430 uG. Orders above 6 and energies above the grid's reach at the lowest channel raise `StackRefused`. The per-node walk evaluates one leaf per node pair, and batching the leaves of one level over its field nodes is left to the cost work of step 5b | a `Similarity` declaration, or `isinstance` of `ContinuumKernel`, as the kernel check. S0 built per S1 table. S0 rebuilt over the exact union. One S1 table per `eta` node for `B` given `eta` (v6 text of Sec. 8.3). A fixed upper edge `ln 1e6` | the first review of step 5a: a user kernel declaring `Similarity` received continuum numbers (relative error 1.0), and the re-review: a subclass of `ContinuumKernel` that doubles the kernel received the plain continuum. Building S0 per table made lookups cost as much as quadrature in per-node mode (0.64 s against 0.62 s per node, loaded machine). The exact union rebuilt S0 once per cell for six `B` cells from 2 mG to 2 G (3.4 s against 0.5 s), and the headroom builds it once. One-node tables per field node need no grouping of the walk. Their cost (step 5a, load 19, 16 by 12 nodes, 19 cells at `N = 4`): 1.2 s once per memo for 192 tables, then 2.2 to 2.4 s per cell for 192 leaves, against 26 ms per cell in cached-S1 mode. A field node at 0 raised `OverflowError`, an `eta` node at 1 a plain `ValueError`, and a law reading `gamma` under `parents=()` was frozen silently (1.7e-2 from the jet). With a fixed upper edge, a 1 G field gave zero rows at 100 and 1000 GHz where the jet does not |

## 14. Owner decisions on the open questions

The owner answered all 20 questions on 2026-11-17. OWNER_DECISIONS.md holds the record, and the last column gives each answer. "Rec." marks the recommended option. Two answers differ from the recommendation: question 4 makes SciPy a base dependency (decision 121), and question 19 ships `AzimuthFourier(m_max)` in 0.8.0 (decision 122). The other answers are the recommended options, which the rest of this document assumes.

| # | Question | Options | Recommendation | Decision (2026-11-17) |
|---|---|---|---|---|
| 1 | Keywords of `Population` | (a) the names of `VARS`: `gamma, B, mu, eta, phi, depth`. (b) words: `electrons, field, pitch, inclination, azimuth, faraday`. (c) (a) with `Psi` for the depth | (a). One spelling everywhere, valid for other kernels. J1 prefers (b) for newcomers, because `phi` is the Faraday depth of rotation-measure synthesis | (a) (Rec.) |
| 2 | Direction of an undeclared field | (a) isotropic inclination and uniform azimuth, `Q = U = 0` stated. (b) ordered field in the plane of the sky. (c) required arguments | (a) | (a) (Rec.) |
| 3 | Undeclared electron normalisation | (a) amplitude 1 at `gamma = 1e4`, free in a fit. (b) unit column on the cell range. (c) required | (a) | (a) (Rec.) |
| 4 | SciPy for the non-negative and bounded solvers | (a) an extra `syncem[fit]`, imported lazily. (b) a base dependency | (a) in 0.8.0. Decide (b) together with optimistix when the JAX solvers land | **(b)** base dependency. The lazy import stays, and the publish smoke test installs SciPy (decision 121) |
| 5 | Source of the basis spectra of a continuum cell | (a) derivative columns define them, the table is a second product. (b) the table is the default source | (a). Revisit after the 0.9 acceptance tests if order changes without a rebuild matter in practice | (a) (Rec.) |
| 6 | Name of the top-level basis class | (a) `SpectralBasis`, while `syncem.model.SpectralBasis` keeps its meaning. (b) `Basis` | (a) | (a) (Rec.) |
| 7 | Rank rule for maps | (a) UPRE per pixel with the histogram. (b) UPRE with a default cap. (c) one common rank | (a) without a cap, and a documented cap for intensity-only maps (the paper used 14 there and none for I, Q, U) | (a) without a cap, and the cap 14 documented for intensity-only maps (Rec.) |
| 8 | Disk cache | (a) on for kernel tables of fixed statistics under the user cache directory, at most 2 GB, least recently used removed first. (b) opt-in. (c) (a) with another size bound | (a), with the path and the size printed by the plan | (a) (Rec.) |
| 9 | Cell numbering | (a) from zero, as Python slices. (b) from one, as "cells 4 to 19" in the paper | (a). Labels and prints state the convention | (a) (Rec.) |
| 10 | `PointMasses` (exact kernel values at the point masses) | (a) offered in 0.9 with the tag `[extension]`. (b) left out | (a). It departs from the recipe of PAPER App. B and gives a non-negative intensity at any order | (a), in 0.9 with the tag `[extension]` (Rec.) |
| 11 | Normalisation of E4-KP in `se.paper` | PAPER:1025 says unit column at each pitch angle. `ST/lib/ensembles.py:302-304` normalises the joint density with one common amplitude | follow the study code, so that the stored numbers are reproduced, and state it in the docstring. The owner confirms which is meant | follow the study code and state it in the docstring (Rec.) |
| 12 | Two conventions until 1.0 (normalised moments, variances and `unit_peak` in `syncem.model`, the paper's conventions in the new layer) | (a) accept, with a README pointer. (b) teach `syncem.model.predict` the new moments now | (a) | (a) (Rec.) |
| 13 | Link between the paper and the new API | (a) the paper keeps citing v0.7.10, and the acceptance tests are the link. (b) the studies are re-expressed on the new API for a later citation | (a). This task does not change the paper | (a) (Rec.) |
| 14 | Default reference weights for every free fit | (a) `gamma^-2.6` with a printed line. (b) a required argument | (a). They set the minimum-norm vector and therefore the extrapolation, and the print says so | (a) (Rec.) |
| 15 | Amplitude convention of PAPER tab: classical parameters. The caption calls `A` the density amplitude at `gamma = 1e4`. The stored values are the study's coefficient: `gamma_0` times the density for E1 and E3, and parameter-dependent factors for E2 and E4-JP (r1-10) | (a) the package keeps `dN/dgamma` at the pivot, documents `paper_coefficient`, and leaves the paper as it is. (b) the package adopts the study coefficient as its public convention | (a). For a family with a cut-off, `amplitude=` is the uncut coefficient at the pivot, the study's convention, so it stays defined below the cut-off and `A_paper = gamma_0 K` (decision 109). This design proposes no manuscript change. The owner may want to know that the caption wording does not match the stored values for E2 and E4-JP | (a). No manuscript change (Rec.) |
| 16 | Size of 0.8.0 | (a) about 72 days, with the jet provider, cells on field strength, depth, inclination and the pitch cosine under `Beaming`, the per-node stack mode with averaged parents, the basis-parametric and parametrised-factor kinds, `from_cells`, `Ray`, R3a, R4a (a) and `BasisSum`. (b) about 56 days, with these in a 0.8.x point release | (a). Under the owner's rule these parts close B2, B3, B5, B6 and T1, T2 cases in the first release. (b) is possible because none of them changes a public signature | (a) (Rec.), plus about 4 days for `AzimuthFourier` (question 19): about 76 days |
| 17 | A jump of the inclination density inside a user's `mu` cell under `Beaming` | (a) `RoleError` naming `Cells.from_edges` with the jump points. (b) accept, list the cells in the plan, and rely on the measured representation term | (a). A Taylor representation across a jump does not converge in the cell, and the remedy is one argument | (a) (Rec.) |
| 18 | Chain-rule columns in `eta` for the continuum kernel, which depends on `eta` only through `B_perp` | (a) a declared `Similarity` extension in 0.9 with its own acceptance test. (b) the jet only | (a) if `gamma x eta` cell lattices are slow on the jet. Correctness does not depend on it | (a), only if `gamma x eta` cell lattices are slow on the jet (Rec.) |
| 19 | An averaged law conditional on a resolved azimuth, or a law conditional on a free azimuth outside its block (R4a (b)) | (a) `RoleError` in 0.8.0 naming route (a) and `average=`, `AzimuthFourier(m_max)` in 0.9. (b) `AzimuthFourier` in 0.8.0 | (a). The paper's configurations do not use it, and the new blocks and truncation term need their own acceptance tests. (b) adds about 4 days | **(b)** `AzimuthFourier(m_max)` in 0.8.0. R4a (b) routes instead of raising, with the truncation term and acceptance tests in step 7 (decision 122) |
| 20 | Default of `max_columns` | (a) 20 000 with a refusal that prints the size and the argument. (b) no cap, a printed size only. (c) a cap in memory, e.g. 1 GB of dense moments per sightline | (a). Every paper configuration fits under it, and G15 shows the explicit raise. (c) is closer to the cost but harder to read in a plan | (a) (Rec.) |
