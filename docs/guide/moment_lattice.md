# Spectral bases and moment fits

The 0.8 interface pairs a population with a lattice and uses the resulting
moments to predict channel Stokes values. The kernel and the observing
response define the spectral basis. The population supplies the physical
electron column and the other distribution laws.

```python
import syncem as se

channels = se.Channels.log(0.2, 8.0, 16, unit="GHz", half_width=0.2)
population = se.paper.e1()
lattice = se.Lattice(gamma=se.paper.cells(order=2))

plan = se.plan(population, channels, lattice=lattice)
basis = se.spectral_basis(population, channels, lattice=lattice)
spectrum = basis.predict(population).to("Jy/sr")
print(plan)
print(spectrum.table())
```

`basis.moments(population)` returns scaled lattice coefficients. Their weighted
order-zero sum is the electron column in the cells. `basis.direct(population)`
integrates on independent quadrature panels and can be restricted to the cell
boxes with `within="cells"`. Where the direct check runs, `representation`
compares the lattice prediction to the cell integral and `outside_lattice`
is the signed contribution of the omitted support. The error ledger records
the reference and kind for each term.

The direct integral of the continuum kernel evaluates each frequency node
at the local transverse field. When its synchrotron argument `x` falls below
`x_tail = min(x_min, 1e-9)`, the node takes the kernel value at `x_tail`
times `(x / x_tail)^(1/3)`. The channel average retains the response and
polarisation phase at each node's original frequency.

For a single node far below a join at `x_tail = 1e-9`, the leading small-`x`
correction gives an estimated relative intensity error of about 8e-7 against
the continuum kernel. The ledger leaves the integrated tail error unbounded.
When the integration support can reach the continuation, `basis.direct`
records `small_x:tail`, as does a prediction that completes the direct
comparison. If the check budget skips that comparison, the prediction
records `representation` as unbounded.

## Fitting observations

```python
data = se.StokesData(
    channels, I=spectrum.I, Q=spectrum.Q, U=spectrum.U,
    unit=spectrum.unit, sigma=0.01 * spectrum.I[:, None],
).withhold([1, 5, 9, 13])
unknown = population.replace(gamma=se.FreeMoments())
result = se.fit(unknown, data, basis=basis)
print(result)
prediction = result.predict(channels)
```

The fit estimates retained linear combinations of lattice moments. Its
minimum-norm completion depends on the lattice reference weights. The
`unconstrained` fraction describes how much a prediction row lies outside
the retained directions. Its units differ from flux and flux uncertainty.
The withheld rows stay out of the fit and rank selection. Use `predict=` when
setting up a fit with a default lattice to include the intended frequency
range in the energy coverage check.

Use a family with `Free` parameters when a density is required. A specified
family on a fixed spectral basis can also be fitted to `result.retained`. The
fixed-basis likelihood of `se.problem` supports JAX differentiation for free
moments and primary families. A non-negative fit represents the source by
point masses in the cells and provides `profile_max` and noise resampling.

When a resolved child has a specified conditional law on unknown parent
moments, the fit contracts the joint basis onto the parent lattice.
`conditional_truncation_bound()` gives the error envelope of that contraction
per unit positive parent column for finite cells and normalized conditional
probability laws. It uses the joint basis polynomial as the reference.
`conditional_truncation()` samples the residual on a parent grid. The fit's
ledger labels the result obtained with inferred parent cell masses as an
estimate. The source error still depends on the true cell masses.

## Saving and reusing a basis

`basis.save(path)` writes numerical arrays with a versioned JSON manifest.
`se.SpectralBasis.load(path)` restores the design for compatible `Moments` and
fits on the saved channels. Computing new channel rows requires the verified
live averaged kernel. Population pairing and the independent direct integral
use the original population and phase declarations. `result.save(path)`
saves fitted coordinates and diagnostics.

`syncem.model` continues to use the normalised moments and response conventions
of 0.7.10. `SpectralBasis.from_single_reference` converts an existing harmonic
basis to unit-area channels and physical-column conventions. Supply the
physical electron column when converting legacy normalised moments.
