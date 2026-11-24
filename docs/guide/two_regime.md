# Two-regime synchrotron spectra and fits

`syncem.paper.population` and `syncem.paper.regime_basis` reproduce the
two-regime setup for the classical synchrotron study. The default source
extends to Lorentz factor 100. The response evaluates finite-channel
harmonics below 300 and the leading continuum kernel from 300 upward.
A point mass at 300 belongs to the continuum branch. The prediction has
components $I$, $Q$ and $U$.

## Predict and fit on physical energy cells

```python
import syncem as se

channels = se.Channels.log(0.2, 8.0, 8, unit="GHz", half_width=0.2, n_nu=48)
source = se.paper.population("E2")
response = se.paper.regime_basis("E2", channels, check=False)
truth = response.predict(source, check=True)

physical = se.paper.energy_basis(response, source)
data = se.StokesData(
    channels, stokes=truth.stokes, sigma=0.01 * truth.I[:, None], unit="cgs"
)
fit = se.fit(source.replace(gamma=se.FreeMoments()), data, basis=physical)

print(truth.plan.source_columns)
print(fit.predict().to("Jy/sr").table())
```

The code uses noiseless observations and assigns a 1% uncertainty for the
fit. The spread reported by the fit depends on the supplied uncertainty and
the observed channels. The same physical basis can be used with measured
Stokes data. The default physical fit uses two energy cells below the switch
and the original 19 above it. The numerical response uses its own lattices
in physical variables below the switch and in log critical frequency and
optional depth above it.

`source_columns` gives the physical electron columns below and above the
switch when population pairing is checked. The source presets preserve the
amplitudes of the original electron laws when extending to lower energy.
The original unit column refers to the interval above 300. The continuum
response moments include a transverse-field weight, so their zeroth
moments have a different normalization.

## Channel coverage and accuracy

Use finite-width channels. The integer harmonic sum evaluates the response
at the line frequencies. The dense harmonic route uses smooth bump
channels. For paired continuum populations the support of a channel must
span at least 32 harmonic spacings on the checked physical nodes.

`regime_basis(..., check=True)` compares harmonic columns under a finer
quadrature and interpolation rule. `response.predict(source, check=True)`
checks a separate term by doubling the population quadrature nodes.
Both changes are numerical estimates. The ledger marks errors in the
physical approximation and finite-cell representation as unbounded until
there is a suitable reference comparison.

Use `RegimeBasis.build` for custom channels, populations and lattices.
The [spectral bases and moment fits guide](moment_lattice.md) covers the
single-kernel workflow.

## Use a conditional Faraday screen

A depth law may depend on physical energy, field strength, pitch or viewing
cosine. Declare it on the population passed to `RegimeBasis.build`. Both
branches apply it at the emission frequency before averaging over the band.
An independent foreground can be passed through `phase=`.

```python
import jax.numpy as jnp
import syncem as se

def screen(gamma):
    mean = 8.0 + 0.1 * jnp.log(gamma.value / 300)
    return se.CumulantScreen(jnp.array([mean, 0.25, 0.02, 0.0]), tau_max=1.0)

source = se.Population(
    gamma=se.PowerLaw(2.6, lo=200, hi=5000, column=1.0),
    B="5 uG", eta=0.25, mu=se.Isotropic(), phi=se.AzimuthFactor(0.4),
    depth=se.Conditional(screen),
)
channels = se.Channels.at([0.65, 1.2, 2.0], unit="GHz", half_width=0.08, n_nu=24)
harmonic = se.Lattice(gamma=se.Cells.log(200, 300, 2), phi=se.AzimuthHarmonic())
continuum = se.Lattice(nu_c=se.Cells.log(1e6, 1e9, 8, unit="Hz"), phi=se.AzimuthHarmonic())
response = se.RegimeBasis.build(
    source, channels, harmonic=harmonic, continuum=continuum, check=False
)
spectrum = response.predict(source, check=False)
```

The conditional screen must be a leaf, with no other law depending on its
depth. The reduced critical-frequency moments cannot determine the screen
when different parent points share the same critical frequency. For a linear
fit with a fixed screen law, pass `se.paper.energy_basis(response, source,
energy_cells)` to `se.fit`. The [Faraday guide](faraday.md) describes the
phase-domain and error-ledger conventions.
