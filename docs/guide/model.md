# Finite joint response and spectral fits

The `syncem.model` layer: the finite joint response of channel Stokes
spectra (manuscript Section 4), its error budget, declared population
assumptions and spectral fits. The design rationale, index layout and
acceptance record are in [the design document](../DESIGN.md).

```{include} ../../README.md
:start-after: <!-- docs:model:start -->
:end-before: <!-- docs:model:end -->
```

## Instrument bandpasses and user responses

The basis needs smooth responses of compact support above zero frequency.
A measured bandpass is written in a set of smooth reference responses, and
the representation residual is carried as a response error:

```python
import numpy as np
from syncem.model import Channels, build_basis, combine_basis, fit_bandpass, predict

# Overlapping bumps on a grid finer than the instrument channels. Together
# they must cover the measured support.
centres = np.linspace(1.0e9, 2.0e9, 41)
reference = Channels.bump(centres, np.full(41, 0.05e9))
fit = fit_bandpass(reference, nu_table, bandpass_table)  # (n_pts,), (n_ch, n_pts)

reference_basis = build_basis(kernel, reference, truncation, ref, support=support)
basis = combine_basis(reference_basis, fit.channels)  # C_j = sum_k c_jk C_k

prediction = predict(reference_basis, moments, amplitude=N_src, **inputs)
data_error = prediction.propagate(
    fit.operator(), response_uncertainty=fit.response_uncertainty()
)
```

`build_basis(kernel, fit.channels, ...)` gives the same `basis` and builds
the reference columns itself. `fit.relative_residual` and `fit.uncovered`
report how well the references represent the table. The response-error
term is an `estimate`, and it is `unbounded` when a residual lies where no
reference reaches.

A response given by a formula enters through `Channels.from_callable`:

```python
import jax.numpy as jnp

def response(nu):
    t = (nu - 1.4e9) / 0.1e9
    return jnp.exp(1.0 - 1.0 / jnp.maximum(1.0 - t * t, 1e-3))

channel = Channels.from_callable(
    response, (1.3e9, 1.5e9), smoothness=99, derivative_bound={3: D3}
)
```

The derivatives come from JAX autodiff. A line-kernel `basis_remainder` is
a `bound` only with `derivative_bound` for the order `N + 1`, a declared
value that the package checks against sampled derivatives but does not
verify. Without it the remainder is an `estimate`. The floor
`jnp.maximum(..., 1e-3)` keeps high-order derivatives finite near the
edges, as in the built-in `bump`.

The callable is stored as a static field. Parameters it closes over are
constants: a basis cannot be differentiated in them, and every new closure
makes `jax.jit` retrace and recompile.
