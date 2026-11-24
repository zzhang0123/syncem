# Faraday rotation and screens

## Statistical interfaces

```{include} ../../README.md
:start-after: <!-- docs:screens:start -->
:end-before: <!-- docs:screens:end -->
```

## Conditional depth and finite-band cumulants

Declare a depth law separately to model an independent screen. Use
`Conditional` when its parameters depend on emitting coordinates. For
example, the Gaussian below makes the mean depth depend on electron
energy. The first two Gaussian cumulants give its exact characteristic
function.

```python
import jax.numpy as jnp
import syncem as se

def screen(gamma):
    mean = 8.0 + 0.1 * jnp.log(gamma.value / 1e4)
    return se.GaussianScreen(mean, 0.5)

source = se.paper.population("E2").replace(depth=se.Conditional(screen))
```

`se.CumulantScreen([k1, k2, k3, k4], tau_max=...)` evaluates a finite-order
exponential of the depth cumulants. Supply a finite `tau_max` in m² when
the third or fourth cumulant is nonzero. The declared interval is
`[-tau_max, tau_max]` in the phase variable $\tau=2(c/\nu)^2$. The model
rejects intervals that allow amplification and evaluations outside the
interval. The magnitude check alone gives no estimate of the error from
omitted cumulants. Compare with a depth law such as a Gaussian, Laplace
or shifted Gamma when a physical reference is available.

A conditional characteristic screen enters the complex polarisation at
the parent point and emission frequency. Independent foreground factors
multiply it there. A depth axis or a law conditioned on depth needs a depth
distribution with quadrature nodes. See the [two-regime guide](two_regime.md)
for a conditional cumulant screen across both providers.

## Mixed emission and Faraday rotation

```{include} ../../README.md
:start-after: <!-- docs:mixed:start -->
:end-before: <!-- docs:mixed:end -->
```
