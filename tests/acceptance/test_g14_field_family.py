"""G14 field-width fits on averaged and resolved continuum bases."""

import jax.numpy as jnp
import numpy as np
import pytest

import syncem as se


@se.family(variable="B", unit="uG", support=(0.5, 50.0))
def field(B, median=5.0, width=0.3):
    return jnp.exp(-0.5 * (jnp.log(B / median) / width) ** 2) / B


@pytest.mark.slow
@pytest.mark.parametrize("resolved", [False, True])
def test_g14_fixed_and_free_energy_law(resolved):
    ch = se.Channels.log(0.2, 8, 12, unit="GHz", half_width=0.2, n_nu=16)
    lattice = se.Lattice(gamma=se.paper.cells())
    if resolved:
        lattice = lattice.replace(B=se.Cells.log("0.5 uG", "50 uG", 4, order=2))
    truth = se.paper.e2().replace(B=field(width=0.3))
    model = truth.replace(B=field(width=se.Free(0.28, bounds=(0.18, 0.46))))
    basis = se.spectral_basis(truth, ch, lattice=lattice, check=False)
    assert basis.columns.n == (456 if resolved else 57)
    physical = basis.predict(truth, check=False)
    data = se.StokesData(
        ch, stokes=physical.stokes, sigma=0.01 * physical.I[:, None], unit="cgs"
    )
    fitted = se.fit(model, data, lattice=lattice, check=False)
    assert fitted.kind == ("separable" if resolved else "basis-parametric")
    assert fitted.params["B.width"].value == pytest.approx(0.3, abs=1e-5)
    np.testing.assert_allclose(fitted.predict().stokes, physical.stokes, rtol=1e-7)
    free = model.replace(gamma=se.FreeMoments())
    weak = se.fit(free, data, lattice=lattice, check=False)
    assert weak.kind == (
        "bilinear (parametrised factor)" if resolved else "basis-parametric"
    )
    # The width is weakly constrained when the energy moments are free.
    relative = abs(weak.predict().stokes - physical.stokes) / physical.I[:, None]
    assert np.max(relative) < 0.02
