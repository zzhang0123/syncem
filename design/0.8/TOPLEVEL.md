# SyncEM redesign: top-level overview

This overview summarises the specification `SPEC.md` (version 6) (cited as Spec Sec. N), which is the authority on every name, signature and number below.
It changes nothing in the paper. The paper's configurations are acceptance tests of the package, and they are not its limit.

## 1. The idea

The channel Stokes vector is a linear functional of the population measure: `X_j = N_src < exp(i s_X phi) K_Xj(gamma, B, mu, eta, depth) >`, with spin weight `s_X = 0` for I and V and 2 for `P = Q + iU`.
Every construction of the formalism is a choice, made per variable, between two operations.
A variable is resolved when a function family on a lattice (cells with local polynomials, Legendre degrees, the azimuth harmonic, point masses) is paired with the measure, which gives moments.
A variable is averaged when the kernel is integrated against a specified distribution of it, which puts its statistics into the basis spectra.
The paper's cases E1 to E6, E1M, E4-KP and the Sec. 4.2 benchmark differ only in this choice (Spec Sec. 4.7).
The package therefore has one object per term of `d = X~ S a`, treats the six variables symmetrically, and lets the kernel declare its variables, so that another emission mechanism uses the same lattice, populations and fits (Spec Sec. 1, 3.2).

## 2. Three layers

```text
Layer 1  base model         Kernel (declared variables, Stokes blocks, capabilities with preconditions), PhaseLaw, Channels
                            g_Xj(x) = exp(i s_X phi) K_Xj(gamma, B, mu, eta, depth)
Layer 2  lattice            Lattice: one representation per resolved variable, product cells in a CellTable
         per-cell           Cells (own chart, reference point, order) | Legendre | AzimuthHarmonic | point masses
         expansions         other variables averaged against specified distributions
                            -> SpectralBasis X~ and ColumnMap S              expensive, cached, records its statistics
Layer 3  population models  Population: fixed | specified | conditional | family with Free | FreeMoments | NonNegativeCells
                            sums, Samples, Ray, Moments
                            -> moments a = L(theta) c                       cheap
         forward            d = X~ S a                                      one matrix product
         fit                min over theta, theta_b, c of |W X~(theta_b) S L(theta) c - d_w|^2
```

Normal form `d = X~ S L(theta) c` (Spec Sec. 0.2, 5.3):
* `X~` is the spectral basis, one column per basis spectrum, built from derivatives, projections or values of the averaged kernel on each cell.
* `S` is the column map, an exact 0/1 selection and summation from lattice rows to basis columns (parity masks, `Beaming` pairs, unused rows). It is computed from static information before any build (Spec Sec. 3.4, 4.3).
* `L(theta) c` is the moment map. `c` holds the linear unknowns and `theta` the nonlinear parameters of families.
* `theta_b` holds free parameters of averaged laws. They change the basis and are named as basis parameters (Spec Sec. 5.2 R6).

The estimator follows from the form of `L`, the set of `c` and whether `theta_b` is empty (Spec Sec. 6.1).

## 3. The lattice

A `Lattice` states which variables are resolved and how (Spec Sec. 3.3, 4).
* Cells exist on every variable of a traceable kernel except the azimuth: energy, field strength, Faraday depth and both cosines.
* Each cell has a chart (`log`: `u = ln(x/x_b)/h_b`, or `affine`: `u = (x - x_b)/h_b`), its own reference point (default the centre in the chart) and its own order.
* Product cells over any subset of variables are rows of a `CellTable`. Inside a product cell the Taylor exponents form the lower set `r_v <= N_v` per variable with `sum_v r_v <= order` over the variables other than depth (total degree and per-variable caps). Depth powers lie outside the total degree, and a depth axis passes the bound `(tau_max h)^(N_d+1)/(N_d+1)! <= depth_tol` (Spec Sec. 4.3, 4.6).
* A tensor grid is one constructor. `Lattice.from_cells`, `select` and `refine(where=...)` give general tables with local refinement and subsets. Overlapping boxes raise. Uncovered emission is the ledger term `outside_lattice`.
* The azimuth is resolved by the exact harmonic `exp(i s_X phi)`, which splits the rows into a real spin-0 block and a complex spin-2 block.
* The kernel per product cell lives on the kernel side (`HybridKernel(assign=)`), so `Lattice.key` stays kernel-free and moments compare across kernels.

| Role | Spelling | Functions | Variables | Basis columns |
|---|---|---|---|---|
| cells, local polynomials | `se.Cells` on the lattice | `1_b(x) u_b^r`, `r <= N_b` | positive, real, cosine (pitch cosine under `Beaming` when `eta` has a density or `Legendre` degrees) | `(1/r!) d^r/du^r` of the averaged kernel at the reference point |
| Legendre degrees | `se.Legendre(L)` | `P_l(x)`, `l <= L` | cosine | projection, factor `(2l+1)/2` per cosine |
| exact azimuth harmonic | `se.AzimuthHarmonic()` | `exp(i s_X phi)` | the kernel's azimuth | no truncation |
| azimuth Fourier rows | `se.AzimuthFourier(m_max)` | `exp(i k phi)`, `0 <= k <= m_max + 2` | the kernel's azimuth | for laws conditional on `phi` (R4a (b)), truncation term zero for a trigonometric polynomial of degree at most `m_max` |
| point masses, population side | `se.NonNegativeCells(point_masses=17)` | `delta(u - u_q)` | variables with cells | basis unchanged, `a = M w`, `w >= 0` |
| point masses, basis side [0.9] | `se.PointMasses(cells, 17)` | Lagrange functions | positive, real | kernel values at the points |
| field-scaling push-forward [0.9] | `se.FieldScaling(reference)` | none added | `B` (and `eta` with `inclination=True`) | energy columns at `B_0` |
| averaged, specified conditional | a `Distribution`, no axis | constant | any | quadrature inside the kernel average, derivatives of the conditional included |
| averaged, characteristic function | a screen with `characteristic` | constant | depth, when it is a parent of nothing | `C_scr(tau(nu))` inside the channel integral |
| averaged, beaming limit | kernel with `Beaming` | constant or `P_l` | pitch cosine | pitch density evaluated at `mu = eta` |

Averaged laws may depend on resolved variables and on each other (Spec Sec. 5.2 R4, R4a, R5). The plan routes each dependence and prints the route.

Example with cells in both `gamma` and `B` (Spec Sec. 12 (a), (b)):

```python
lat = se.Lattice(gamma=se.Cells.log(300, 2.9626e6, 19, order=2),
                 B=se.Cells.linear("2 uG", "8 uG", 3, order=1), order=2)
lat.n            # 285 lattice rows: 57 product cells x 5 monomials u^r x_B^s with r <= 2, s <= 1, r + s <= 2
lat.index.select(gamma=(1e4, 1e5))           # interval selector: the cells inside [1e4, 1e5]
```

With the continuum kernel the `B` columns come from the field-scaling relation when the `eta`, `mu` and depth laws do not depend on `gamma` or `B`, and from the jet provider otherwise (Spec Sec. 8.1, 8.3).

## 4. Population declarations

Each keyword of `se.Population(*blocks, symmetries=(), **factors)` is a variable of the kernel. Separate keywords are independent unless one is a function of another. A positional block is a joint statement (Spec Sec. 5.1). Order and degree belong to the lattice, so one declaration runs at any order.

| Statement | Spelling | Compiles to | Kind and estimator |
|---|---|---|---|
| fixed | `B="5 uG"`, `eta=0.5` | kernel at the value, or a point mass when resolved | fixed, none |
| specified | `B=se.Uniform(4, 6, unit="uG")`, `depth=se.GaussianScreen(8, 0.5)` | averaged into the basis and recorded in `basis.specified`, or moments by quadrature when resolved | fixed, none |
| conditional, averaged child | `B=lambda eta: se.LogNormal(...)`, `phi=lambda eta: se.AzimuthFactor(...)` | nested quadrature in the kernel average, route printed | fixed, none |
| conditional, resolved child, specified parent | `gamma=se.Conditional(aged, units={"B": "G"})` with `B` on the lattice | joint moments by nested quadrature | fixed, none |
| conditional, resolved child, free parent | `gamma=se.FreeMoments(), B=lambda gamma: se.LogNormal(...)` | fixed linear node (shifted correlation, R3a) | linear, SVD with UPRE |
| family with `Free` on a resolved variable | `gamma=se.BrokenPowerLaw(index1=se.Free(2.0, bounds=(0, 7)), ...)` | `a(theta)` by a fixed rule per cell | separable: amplitude in closed form, bounded least squares (`trf`), multistart |
| family with `Free` on an averaged variable | `B=se.LogNormal("5 uG", width=se.Free(0.3))` | basis parameter `theta_b` (R6) | basis-parametric: inner solve per `theta_b`, finite-difference Jacobian in `theta_b` |
| user family from a function | `@se.family(variable="B", unit="uG", support=(0.5, 50))` | as a built-in family. `linear=` declares linear parameters | as the built-in case |
| linear family forms | `se.PowerLawMixture.moment_form(order=2, index0=2.6)` | cell moments of each term | linear, least squares (`se.FullRank()`) |
| free moments of one variable | `gamma=se.FreeMoments()` | `L` = identity | linear, whitened SVD in reference weights, rank by `se.UPRE()` |
| free moments of a joint group | `se.FreeMoments("gamma", "phi")`, up to all six variables | joint columns | linear, same estimator |
| two independent unknown groups | `gamma=se.FreeMoments(), phi=se.FreeMoments()` | product of moment sets | bilinear, variable projection, UPRE counts `R + dim theta` |
| free moments times a family factor | `gamma=se.FreeMoments()`, `B=` a family with `Free` on `B` cells | `kron(I, a_factor(theta))` | bilinear (parametrised factor) |
| non-negative cells | `gamma=se.NonNegativeCells(point_masses=17)`, or a joint block | `a = M w`, `w >= 0` | non-negative linear, NNLS with penalty `alpha` by UPRE (host SciPy) |
| sums | `zone_a + zone_b`, `0.9 * a + 0.1 * b` | moments add. Different averaged laws give a `BasisSum` (R7) | kind of the parts |
| samples | `se.Samples(gamma=g, B=(b, "G"), ..., weights=w)` | weighted sums per cell, sparse | fixed |
| profiles along the ray | `se.Ray(step=..., B_vector=..., ...)` | direct integration without a lattice, sparse sums over steps with one (R8) | fixed |
| known moments | `se.Moments(lattice, values)` | used with its `origin`. Basis statistics become assumptions | fixed |
| priors | `se.Free(2.6, prior=se.Normal(2.6, 0.3))`, `se.FreeMoments(prior=se.Normal(0, s))` | extra whitened rows. A Gaussian prior on moments equals a penalty | moments get origin `penalised` |

A fit with `data=fit.retained` uses the compressed likelihood on the retained combinations (Spec Sec. 6.1). Without a lattice, rules R0 to R8 derive a default one, and each choice is printed with `[default]` (Spec Sec. 5.2).

## 5. The front door

Signatures are those of Spec Sec. 2.2. Defaults are tabulated in Spec Sec. 2.3.

(a) Newcomer forward call (Spec Sec. 2.1, G2):

```python
import syncem as se
pop  = se.Population(gamma=se.PowerLaw(2.6), B="5 uG",
                     phi=se.AzimuthFactor(0.4), depth=se.GaussianScreen(mean=8.0, sigma=0.5))
spec = se.predict(pop, ([0.408, 1.4, 2.3, 4.8, 23.0], "GHz"))
print(spec)   # 13 cells in ln gamma, order 2, 39 moments, eta and mu isotropic: each tagged [default]
se.plan(pop, ([0.408, 1.4, 2.3, 4.8, 23.0], "GHz"), strict=True)   # raises and lists every default
```

(b) The paper's Sec. 5 E1 forward with 19 cells (Spec G1):

```python
import numpy as np, jax
obs     = se.Channels.log(0.1, 10, 41, unit="GHz", half_width=0.2)
known   = dict(B=se.Uniform(4, 6, unit="uG"), eta=se.Uniform(-0.5, 0.5), mu=se.Isotropic(),
               phi=se.AzimuthFactor(0.4 * np.exp(0.5j)), depth=se.GaussianScreen(mean=8.0, sigma=0.5))
GR      = dict(lo=300, hi=2.9626e6)
lattice = se.Lattice(gamma=se.Cells.log(300, 2.9626e6, 19, order=2))        # 57 moments
e1      = se.Population(gamma=se.PowerLaw(2.6, column=1.0, **GR), **known)
spec    = se.predict(e1, obs, lattice=lattice)
spec.ledger["representation"]          # measured against direct integration: 8.6e-4 of I at N = 2
se.predict(e1, obs, lattice=lattice.with_order(4))                         # 8.8e-5 at N = 4
```

The acceptance values are 2.56e-2, 8.58e-4 and 8.76e-5 at `N = 0, 2, 4` (Spec Sec. 9.6).

(c) Free-moment fit with UPRE and prediction at 300 GHz (Spec G3, G6):

```python
band = se.Channels.log(1, 40 ** (31 / 20), 32, unit="GHz", half_width=0.2)
e2   = e1.replace(gamma=se.BrokenPowerLaw(2.0, 3.2, 7800, smoothness=2.0, column=1.0, **GR))
truth = se.spectral_basis(e2, band, lattice=lattice).direct(e2)
d26  = se.StokesData.simulate(truth, noise=0.01, key=jax.random.key(1))[:26]      # 1 to 100.6 GHz
fit  = se.fit(e2.replace(gamma=se.FreeMoments(cells=slice(3, None))), d26, lattice=lattice)  # rank=se.UPRE()
p300 = fit.predict(se.Channels.at([300], unit="GHz", half_width=0.2))
p300.I, p300.sigma, p300.unconstrained   # noise-only sigma, row fraction outside the retained directions
```

The lattice covers 300 GHz. With a default lattice, `se.fit(..., predict=channels)` makes the energy range cover the predicted channels (Spec Sec. 6.5).

(d) A family fit and non-negative cells by replacing one slot (Spec G4, G5):

```python
sbpl   = se.BrokenPowerLaw(index1=se.Free(2.0, bounds=(0, 7)), index2=se.Free(3.2, bounds=(0, 7)),
                           gamma_break=se.Free(7800, bounds=(100, 3e5), log=True), smoothness=2.0, **GR)
fit_f  = se.fit(e2.replace(gamma=sbpl), d26, lattice=lattice)                       # separable
fit_nn = se.fit(e2.replace(gamma=se.NonNegativeCells(point_masses=17, cells=slice(3, None))),
                d26, lattice=lattice)                                               # non-negative linear
print(se.compare(fit, fit_f, fit_nn))
```

All three fits reuse one basis. Each print names its kind and estimator.

(e) A joint lattice with cells in `gamma` and `B` and free joint moments (Spec Sec. 5.1, 10.1, 12):

```python
lat_gb = lattice.replace(B=se.Cells.linear("2 uG", "8 uG", 3))                     # 342 lattice rows
joint  = se.Population(se.FreeMoments("gamma", "B"),
                       **{k: known[k] for k in ("eta", "mu", "phi", "depth")})
print(se.plan(joint, band, lattice=lat_gb))   # routes, column count, and the field-scaling tie on (gamma, B)
fit_gb = se.fit(joint, d26, lattice=lat_gb)                                         # linear, SVD with UPRE
```

With the continuum kernel the field-scaling relation ties the `B` columns to the energy columns, so the data constrain only invariant combinations of the joint moments. The plan states this structural tie before any data are seen (Spec Sec. 5.5). The fully joint block is `se.FreeMoments("gamma", "B", "mu", "eta", "phi", "depth")` on a lattice that resolves all six.

## 6. Honest defaults

What every result prints (Spec Sec. 2.1, 7.1):
* header (channels, response, Stokes blocks) and unit line, with the normalisation default tagged.
* roles table: per variable, declared statistic, role, representation, route and the deciding rule.
* "assumed" row: every independence from separate keywords, with conditionals named as exceptions.
* basis line: kernel, phase law, the statistics the basis contains and when to rebuild.
* accuracy line: measured numbers with their channel span, or "not measured" with the cost.
* "not bounded" line: names of the terms with no value.
* a fit also prints the estimator, rank, `chi2`, withheld diagnostics, what is constrained, and the caveats.

Every applied default is a `[default]` tag and an entry of `result.defaults`. `strict=True` raises on the first default and lists all of them (Spec Sec. 2.3).

Measured representation error (Spec Sec. 7.2):
* `representation` is `basis.predict(pop) - basis.direct(pop, within="cells")` for a fully specified population, same kernel, reported as `max |dX| / I_j` with `I_j` the full-support direct intensity.
* `outside_lattice` is the negative of the emission of the declared population outside the cell boxes, computed by its own quadrature on the uncovered support.
* The two signed terms add to `predict - direct` over the full support. Their reported maxima bound it. A test checks the identity for an unbounded power law, a cut-off spectrum beyond both lattice ends, a lattice with a hole and a lattice that does not cover the channels.
* For a free-moment or non-negative fit no population exists, so `representation` is unbounded and `basis.reference_cases` (power law, broken power law, cut-off inside a cell) are printed as finite cases.

Typed ledger: the five kinds of `errors.py` (bound, measured, estimate, unbounded, not_applicable) plus a static `declared` flag for user input. `ledger.sum(kind)` sums within a kind. `ledger.envelope()` is `None` while any term is unbounded and names the declared terms it depends on. `ledger.allow(name, term, note=)` records an assumption. `physical_kernel` and every `specified:<variable>` term are unbounded unless declared.

Fixed caveats are attached by kind with their PAPER line (Spec Sec. 7.3). Examples: discarded combinations are set to zero by convention and are not inferred values (PAPER:718). A small residual does not establish the physical moments (PAPER:856). Intervals come from the noise alone (PAPER:1148). UPRE lowers coverage to 56 to 58 per cent (PAPER:1083). An acceptable `chi2` does not confirm the specified statistics (PAPER:833, 836, 852).

## 7. Engine in brief

Kernel protocol (Spec Sec. 3.2, 8.1): `variables`, `primary`, `stokes`, `azimuth`, `depth`, `traceable`, `lines`, `capabilities`, and the methods `channel`, `spectral`, `regime`, `physical_error`, `describe`. No engine contract names a variable. `@se.kernel` turns a JAX function into a traceable kernel. The existing `KernelModel` stays as an adapter.

Capabilities are shortcuts with preconditions that the plan checks against the whole declaration, parents of every conditional included. A failed precondition selects the general route, and `PlanRecord.routes` records why (Spec Sec. 1 principle 17). The capabilities are `Similarity`, `FieldScalingRelation`, `Beaming`, `Parity`, `EnergyWindow`, `ReferenceDensity`, `Lines`, `AnalyticTaylor` and `HostOnly`.

Column providers (Spec Sec. 8.3):
* jet: the general contract. Nested forward mode on the `AveragedKernel` in any number `d` of celled variables. It handles every product cell that a shortcut does not cover, and it is the oracle of the shortcuts. On E1 it costs 260 times the stack at `N = 2` and 1570 times at `N = 4` at equal channel quadrature, and its nested `jacfwd` compile time grows about 3 times per order, so the plan prints a cost estimate above order 4 (prototype core, profile). A smooth tabulated `F`, `G` (piecewise Chebyshev in `ln x`) is the candidate to cut its primal.
* core: the existing jitted `_build_core` for `(gamma, B)` cells, with a re-expansion from the affine to the log chart. Fast path and oracle at `d <= 2`.
* stack: the analytic derivative stack of the study, with a cached field-average table (S1) or a per-node mode for conditionals. Default for `Similarity` kernels where its precondition holds. Host NumPy in 0.8.0.
* table: 17 Chebyshev-Lobatto nodes per cell. Required for host-only kernels and point masses. In 0.8.0 it serves the accuracy check and the provider comparison.

Stages and caching (Spec Sec. 8.2, 8.5): S0 primitive table in `ln x`, S1 field average by lookups in the S0 table at each field node `B_q` in place of a quadrature per node (prototype profile), S2 channel columns per channel row, S3 assembly (weights, constant azimuth factor, column map), S4 factorisation. Builds are memoised on content keys that include a hash of the evaluated tabulation of every specified statistic. S0 and S1 of fixed statistics go to a disk cache of at most 2 GB. Reference weights are in no key, so a change of `w` re-runs assembly only.

JAX properties (Spec Sec. 8.6): planning is eager. `basis.predict`, family moments and free-moment fits are `jit`, `vmap` and (where defined) `grad` compatible. `problem.log_prob` is pure JAX when there are no basis parameters. Non-negative, separable, bilinear and basis-parametric fits run on the host in 0.8 with jitted residuals. Batches of NNLS run in a host process pool with BLAS pinned to one thread. In 0.9, batches of separable family fits run as a jitted `vmap` Levenberg-Marquardt with SciPy `trf` as the oracle (Spec Sec. 6.4, 10.1). Static fields hold metadata only, so new bounds, starts or data do not recompile.

Native code (NATIVE.md): 0.8 and 0.9 add no Rust or other new native code, because the largest measured factors come from algorithmic changes and batching. The named seams are host-only kernels, the certification arithmetic backend (`gmpy2` or `python-flint`/Arb in the certification extra) and the harmonic host driver, which is vectorised in NumPy first and revisited for Rust only if it still exceeds about 0.3 ms per channel point.

| Cost item | Value |
|---|---|
| `import syncem` | 0.33 s, almost all JAX and Equinox |
| S1 table | per-node quadrature 0.32 to 0.34 s per field node (NumPy). S0 table once (0.38 s), then S1 by lookups at about 6 ms per node, 3.3e-10 of the peak from the quadrature table at order 6 (SPEC decision 140) |
| E1 basis after S1 | 0.15 s (`N = 2`) and 0.21 s (`N = 4`) at 400 channel nodes |
| accuracy check | 1.5 to 1.8 s once for the table, then about 1 ms per population |
| core and jet providers | core 0.39 to 0.5 s per cell. Jet on E1 0.35 s per cell at `N = 2` and 3.0 s at `N = 4`, 260 and 1570 times the stack at equal channel quadrature. Jet at `d = 3`: 0.245 s per cell plus 1.04 s compile |
| family moments on 19 cells | 26 to 58 microseconds |
| free-moment fit | 0.15 to 0.21 microseconds per spectrum with shared noise (JAX). Per-pixel noise: 16.7 and 51 microseconds per SVD at 63 x 57 and 63 x 171 under JAX `vmap`, 177 and 611 microseconds on one core |
| family fit per spectrum | `trf` 2.39 ms. Batched JAX Levenberg-Marquardt 60 microseconds per spectrum [0.9] |
| NNLS on 111 x 272 | 2.96 ms serial, 0.51 to 0.71 ms per spectrum in a process pool with BLAS pinned to one thread |
| basis-parametric fit | one S1 by lookups per trial value, about 0.2 s for a 32-node log-normal (estimate) |
| not measured | a JAX port of the full stack provider, JAX sharding of the jet, one S1 table per `eta` node (about 1.2 s with lookups, estimate) |

## 8. Migration and release plan

0.8.0 is additive (Spec Sec. 9). `git diff v0.7.10 -- syncem` shows new files and `syncem/__init__.py` only. `syncem.model` and its 900 tests are unchanged, and `import syncem` does not load it. The studies keep running: a guard test holds the AST digests of `ST/lib/syncem_compat.py`, the five names the studies import from `syncem.model`, and the v0.7.10 hash of every existing file. The paper keeps citing v0.7.10, and the acceptance tests of Spec Sec. 9.6 are the link between paper and code. Study code is ported (stack to NumPy, fits and family moments to JAX), and the NumPy originals stay hash-checked as oracles.

| # | Step | Acceptance test | Days |
|---|---|---|---|
| 0 | guards | pass before any change | 0.5 |
| 1 | units, `Variable`, `Channels`, errors | bare `B=5.0` and unit-less ranges raise. Channels equal the study to 1e-14 | 2 |
| 2 | lattice, cell table, index, `Moments`, `recentre`, `coarsen` | row counts 57, 171, 114, 513 (E4-KP), 513 (energy and depth), 2430. Local refinement coarsens back to 1e-12 | 6 |
| 3 | kernel protocol, continuum adapter, core provider | columns equal the oracle in cells 1, 10, 19 at `N = 0..4`. `EnergyWindow` edges | 4 |
| 4 | jet provider in `d` directions | three-variable polynomial kernel to 1e-12. Jet equals core at `d = 2` | 4 |
| 5 | stack (cached S1, per-node mode), column map, `SpectralBasis`, `BasisSum`, cache | all cells to 1e-10 of the oracle. Ranks 26 and 28. T2, T5, T14 | 7 |
| 6 | distributions, families, pairing, `direct`, table, `predict`, ledger | forward errors of E1 to E4-JP and E1M at `N = 0, 2, 4`. Partition identity | 6 |
| 7 | role rules, conditionals, R3a, R4a, R5, `Beaming`, parity, `Ray`, `AzimuthFourier(m_max)`, `se.plan` | E5, E6, E4-KP rows. Column counts 285 and 1306. T6, T15, T17, T19. `AzimuthFourier` rows and truncation term | 18 |
| 8 | `StokesData`, free-moment fit, batches | seeded ranks `R* = 7, 7, 7, 9, 9, 17, 7` and `chi2` to 1e-6 | 4 |
| 9 | penalty, non-negative cells, prediction at other channels | E2 medians at 300 GHz. `unconstrained` row to 1e-8 | 4 |
| 10 | separable, compressed, basis-parametric, parametrised-factor fits, `se.problem` | family and parameter rows. `log_prob` gradients. T18 | 9 |
| 11 | bilinear shared factor | E5 one-factor row: `R* = 7 + 2`, `chi2 = 270.9 / 54` | 3 |
| 12 | prints, `se.paper`, docs, release | gallery G1 to G16 run as tests, except G11 and the 0.9 lattices. Docs build without warnings | 6 |
| 13 | 0.8.1 bridge to single-reference bases | Sec. 4.2 benchmark | 3 |

Each step starts by writing its acceptance test and seeing it fail. The total is about 76 working days, 4 of them for `AzimuthFourier`. The prototypes tested the rules behind steps 4 to 7 but not the column map, the fits beyond timings or the front door, so the total is an estimate. P5 estimated 38 days for the subset it covered (Spec Sec. 10.2). 0.9.0 adds the JAX stack, `FieldScaling`, harmonic cells, `PointMasses` and JAX solvers. 1.0.0 moves the legacy top-level modules to `syncem.legacy` (Spec Sec. 9.5).

## 9. Owner decisions

The owner answered all 20 open questions of Spec Sec. 14 on 2026-11-17, recorded in OWNER_DECISIONS.md. Eighteen answers are the recommended options, which the specification already assumed. E4-KP in `se.paper` follows the study code and says so in its docstring (Q11). Two answers differ from the recommendation. SciPy is a base dependency (Q4), the lazy import stays, and the publish smoke test installs SciPy. `AzimuthFourier(m_max)` ships in 0.8.0 (Q19), so R4a (b) routes through it instead of raising, its truncation term and acceptance tests belong to step 7, and the total grows from about 72 to about 76 days (Q16).

## 10. Where the design is least certain

* The detailed rules grew through three review rounds, and each round introduced new defects, for example a cached field table that returned `<c><K_P>` for an azimuth factor depending on `B`, 5.9 per cent off relative to `<c><K_P>` (5.6 per cent of `<c K_P>`, r3-02). Prototypes of the core and of six rule families (R3a, R5, beaming, fieldcells, partition, depth) then compared every rule with direct integration at moderate and extreme parameters, and their 905 tests passed (prototypes/RESULTS.md). The core showed no correctness bug. The partition identity was wrong as written and holds with a signed `outside_lattice`. The other five families were correct with changes, among them frozen-node reweighting in R3a, `mu` cells covering `[-1, 1]` under `eta` on Legendre degrees, depth powers outside the total degree, and the depth bound `x^(N_d+1)/(N_d+1)!` in place of the threshold `tau_max h <= 1` (prototype depth). Not prototyped are the column map, units, the ledger, the front door, the fits beyond timings, the harmonic route, `Ray`, `PointMasses` and depth outside the total degree in a production lattice. The r2-02 and r2-03 figures were not reproduced (RESULTS Sec. 7). The G15 counts under the new depth rule are recounted in Spec Sec. 8.7 and not measured.
* The shortcut preconditions (cached S1 per Stokes block, joint-reversal checks (a) to (d), characteristic-function route, closed-form depth weights) are intricate. A wrong precondition drops terms without an error. Four prototypes (R3a, fieldcells, depth, core) found the same failure: derivatives of laws that move with a celled parent must enter the columns, and frozen nodes are admissible only with density reweighting on a fixed support.
* Rule-family wall times were taken under load and are inflated 3 to 8 times, while method ratios timed back to back hold. A JAX port of the full stack provider, JAX sharding of the jet and a jet over a tabulated `F`, `G` are not measured. Large lattices are sized but not built. G15 has 2 737 152 rows and 1 140 480 columns with depth order 3 and depth outside the total degree (Spec Sec. 8.7).
* The 76-day estimate has prototypes behind the rules only. Step 7 (role rules, routes and `AzimuthFourier`, 18 days) carries the most risk, and `AzimuthFourier` has no study prototype.
* Only one cell width (half-width 0.242) was tested in the study. The default cell count has no convergence study. `se.convergence` gives a measured comparison per population.
* Several terms stay unbounded by construction: `representation` for free fits, `representation:angular` for `mu` samples under `Beaming`, and every `specified:` term. `conditional_truncation` in R3a is bounded per unit parent mass, and its prediction-level value falls with order while `zeta_b <= 1`.
* With the continuum kernel, field cells and free field-law parameters add directions the data cannot separate from energy moments (PAPER:562, 833). The design reports this and does not refuse.
* Basis-parametric fits redo S1 by lookups per trial (about 0.2 s, estimate) and run on the host in 0.8. The effect of the S0 lookups on a fit is not measured.
* Harmonic cells, `FieldScaling` and `PointMasses` are deferred to 0.9. Stokes V with cells, and the harmonic regime, therefore wait for 0.9.
* The 17-node table misses its own peak in exponential-tail cells (1.2e2 of the P peak at `r = 1` in E1 cell 1), so it is checked only against the channel column maximum and serves as the representation reference only where it passes that check (Spec Sec. 8.3).
