"""Effective critical-frequency column of an angle-dependent magnetic field."""

import numpy as np

import syncem as se
from syncem.base.window import A1_HZ_PER_G
from tests.v090.test_regime_review_regressions import _population, _regime


def test_field_transmits_pitch_support_to_the_continuum_diagonal():
    gamma, field = 3e4, 5e-6
    critical = A1_HZ_PER_G * field * gamma**2
    channel = se.Channels.at([0.43 * critical], unit="Hz", half_width=0.05, n_nu=8)

    def magnetic(eta):
        return se.Delta(field * (1 + 0.25 * eta.value), unit="G")

    def pitch(B):
        shift = B.value / field - 1
        return se.Uniform(0.239 + shift, 0.241 + shift)

    def inline(eta):
        return se.Uniform(0.239 + 0.25 * eta.value, 0.241 + 0.25 * eta.value)

    base = _population(se.Delta(gamma), eta=se.Isotropic()).replace(
        B=se.Conditional(magnetic)
    )
    rb = _regime(
        base,
        channel,
        se.Cells.linear(299, 300, 1, order=0),
        se.Cells.log(0.8 * critical, 1.25 * critical, 4, order=4),
        mu=se.Legendre(1),
        eta=se.Legendre(1),
        B=se.Cells.linear(2, 8, 1, unit="uG"),
        moments=12,
    )
    # Integrate (1+eta/4) sqrt(1-eta**2) / 0.002 from 0.239/0.75 to 0.241/0.75.
    # I folds the same measure against the channel response and the Bessel kernel.
    for conditional in (
        se.Conditional(pitch, units={"B": "G"}),
        se.Conditional(inline),
    ):
        moments = rb.moments(base.replace(mu=conditional))
        np.testing.assert_allclose(
            rb.continuum.lattice.column(moments[1]), 1.3642806928822016, rtol=2e-12
        )
        np.testing.assert_allclose(
            rb.predict(moments, check=False).I, 1.141211855932871e-28, rtol=2e-7, atol=0
        )
