"""Reference populations and observing channels for the classical synchrotron study.

The presets reproduce the declared distributions and total column used in
the v0.7.10 study. They construct public SyncEM objects and build no basis.

LABEL ``app: test ensembles``.

Field and frequency declarations use microgauss and GHz. Channel arrays
have shape (n_ch, n_nu) and populations are normalized in column units.
These are reference models with no certified error for observed sources.
"""

from __future__ import annotations

from cmath import exp
from math import pi, sqrt

from .base.channels import Channels
from .distributions import (
    AgedPowerLaw,
    AzimuthDensity,
    AzimuthFactor,
    BrokenPowerLaw,
    Conditional,
    GaussianScreen,
    Isotropic,
    Maxwellian,
    PowerLaw,
    PowerLawMixture,
    Uniform,
)
from .lattice.cells import Cells
from .population import Population

__all__ = [
    "cells",
    "channels",
    "e1",
    "e1m",
    "e2",
    "e3",
    "e4",
    "e4kp",
    "effective_kp_moments",
    "e5",
    "e6",
    "energy_basis",
    "extrapolation_channels",
    "population",
    "regime_basis",
    "specified",
]

_GAMMA_LO = 300.0
_GAMMA_HI = 2962642.632683605
_GAMMA_PIVOT = 1e4
_AGE = "19 Myr"
_RADIATION = "1 eV cm-3"
_FIELD_RMS = (sqrt(76 / 3), "uG")
_AZIMUTH = 0.4 * exp(0.5j)
# These are the study's coefficients for unit total column. The amplitudes
# passed to the ageing laws are coefficients of the uncut density at the pivot.
_E3_COEFFICIENT = 0.004912116485435222  # forward_results.json, E3 norm
_KP_COEFFICIENT = 0.006000543577583004  # KardashevPacholczyk.norm_unscaled_inv
_E6_COEFFICIENT = 0.006000558809723244  # cost_results.json, A_unit_column
_E5_STEEP_FRACTION = 0.9378848242226161  # forward_results.json, equal I at 1 GHz


def channels(*, n_nu: int = 400) -> Channels:
    """The 41 radio channels of the study, with response half-width 0.2 times the centre."""
    return Channels.log(0.1, 10, 41, unit="GHz", half_width=0.2, n_nu=n_nu)


def extrapolation_channels(*, n_nu: int = 400) -> tuple[Channels, Channels]:
    """Return the 32-channel band and the 121-channel prediction grid of gallery G6."""
    band = Channels.log(1, 40 ** (31 / 20), 32, unit="GHz", half_width=0.2, n_nu=n_nu)
    dense = Channels.log(1, 300, 121, unit="GHz", half_width=0.2, n_nu=n_nu)
    return band, dense


def cells(*, order: int = 2) -> Cells:
    """The 19 equal log-energy cells of the study, for use as the gamma axis."""
    return Cells.log(_GAMMA_LO, _GAMMA_HI, 19, order=order)


def specified() -> dict:
    """The common field, pitch, inclination, azimuth and screen declarations."""
    return {
        "B": Uniform(4, 6, unit="uG", nodes=16),
        "eta": Uniform(-0.5, 0.5, nodes=12),
        "mu": Isotropic(),
        "phi": AzimuthFactor(_AZIMUTH),
        "depth": GaussianScreen(8.0, 0.5),
    }


def _power_law(index: float, column: float = 1.0) -> PowerLaw:
    return PowerLaw(index, lo=_GAMMA_LO, hi=_GAMMA_HI, column=column)


def _aged(**parameters) -> AgedPowerLaw:
    return AgedPowerLaw(2.6, lo=_GAMMA_LO, age=_AGE, radiation=_RADIATION, **parameters)


def e1() -> Population:
    """E1, a power law of index 2.6 and unit total electron column."""
    return Population(gamma=_power_law(2.6), **specified())


def e1m() -> Population:
    """E1M, E1 with a Maxwellian field strength of RMS 5 microgauss."""
    return e1().replace(B=Maxwellian("5 uG", cut=8.0, nodes=96))


def e2() -> Population:
    """E2, the smoothly broken power law with break at gamma = 7800."""
    gamma = BrokenPowerLaw(
        2.0, 3.2, 7800, smoothness=2.0, lo=_GAMMA_LO, hi=_GAMMA_HI, column=1.0
    )
    return Population(gamma=gamma, **specified())


def e3() -> Population:
    """E3, a Gaussian index mixture with mean 2.6 and standard deviation 0.2."""
    gamma = PowerLawMixture.from_paper_coefficient(
        _E3_COEFFICIENT, k1=2.6, k2=0.2**2, lo=_GAMMA_LO, hi=_GAMMA_HI
    )
    return Population(gamma=gamma, **specified())


def e4() -> Population:
    """E4-JP at age 19 Myr, with losses set by the mean squared field."""
    return Population(gamma=_aged(loss_field=_FIELD_RMS, column=1.0), **specified())


def _kp(mu) -> AgedPowerLaw:
    return _aged(
        loss_field=_FIELD_RMS, pitch=mu, amplitude=_KP_COEFFICIENT / _GAMMA_PIVOT
    )


def e4kp() -> Population:
    """E4-KP with pitch-dependent radiative losses at age 19 Myr.

    One common uncut amplitude normalises the joint energy and pitch
    density to unit column over all pitch cosines, as in the study code.
    Use a pitch Legendre axis for the paper's projected basis.
    """
    gamma = Conditional(_kp, parity={"mu": "even"})
    return Population(gamma=gamma, symmetries=("pitch_reversal",), **specified())


def e5() -> Population:
    """E5, two power-law zones behind one screen with equal I at 1 GHz.

    Use an explicit azimuth harmonic axis for the paper's joint basis.
    """
    other = specified()
    del other["phi"]
    steep = Population(
        gamma=_power_law(3.1, _E5_STEEP_FRACTION),
        phi=AzimuthDensity(0.8, 0.25, unit="rad"),
        **other,
    )
    flat = Population(
        gamma=_power_law(2.1, 1 - _E5_STEEP_FRACTION),
        phi=AzimuthDensity(0.8, 0.25 + pi / 4, unit="rad"),
        **other,
    )
    return steep + flat


def _local_field_ageing(B) -> AgedPowerLaw:
    return _aged(loss_field=B, amplitude=_E6_COEFFICIENT / _GAMMA_PIVOT)


def e6() -> Population:
    """E6, JP ageing with the cutoff set by the local field strength.

    One common uncut amplitude gives unit total column after integrating
    over the field and energy. Use a field axis for the paper's joint basis.
    """
    gamma = Conditional(_local_field_ageing, units={"B": "G"})
    return Population(gamma=gamma, **specified())


def population(name, *, gamma_min=100.0):
    """Return the two-regime version of an existing study ensemble."""
    from .front._regime_paper import population as make

    return make(name, gamma_min=gamma_min)


def regime_basis(
    name,
    observation=None,
    *,
    gamma_min=100.0,
    gamma_switch=300.0,
    harmonic_cells=16,
    continuum_cells=512,
    order=8,
    **options,
):
    """Build both bases for a paper population on the requested channels."""
    from .front._regimes import RegimeBasis
    from .lattice.lattice import Lattice
    from .lattice.representations import AzimuthHarmonic, Legendre
    from .base.window import A1_HZ_PER_G

    if not 1 < gamma_min < gamma_switch < _GAMMA_HI:
        raise ValueError(
            "the paper response needs 1 < gamma_min < gamma_switch below the upper cutoff"
        )
    source = population(name, gamma_min=gamma_min)
    field_max = 40 if name.upper() == "E1M" else 6
    nu_min = A1_HZ_PER_G * 1e-9 * gamma_switch**2
    nu_max = A1_HZ_PER_G * field_max * 1e-6 * _GAMMA_HI**2
    low_order = min(order, 4)
    harmonic = {
        "gamma": Cells.log(gamma_min, gamma_switch, harmonic_cells, order=low_order)
    }
    if name.upper() == "E4-KP":
        harmonic["mu"] = Legendre(8)
    if name.upper() == "E6":
        harmonic["B"] = Cells.log(4, 6, 2, order=min(order, 2), unit="uG")
    low = Lattice(**harmonic, phi=AzimuthHarmonic(), order=low_order)
    high = Lattice(
        nu_c=Cells.log(nu_min, nu_max, continuum_cells, order=order, unit="Hz"),
        phi=AzimuthHarmonic(),
    )
    defaults = dict(
        field_nodes=4, inclination_nodes=4, angular_nodes=32, frequency_nodes=64
    )
    return RegimeBasis.build(
        source,
        channels() if observation is None else observation,
        harmonic=low,
        continuum=high,
        gamma_switch=gamma_switch,
        **{**defaults, **options},
    )


def energy_basis(regime, source, energy_cells=None, *, order=2, extra_nodes=None):
    """Project the paper's physical energy moments through both branch bases."""
    import numpy as np

    from .front._regime_physical import paper_energy_basis
    from .lattice.lattice import Lattice
    from .lattice.representations import AzimuthHarmonic
    from .population.declaration import parents

    if isinstance(source, str):
        axis = regime.harmonic.lattice.cells.variables.index("gamma")
        floor = min(regime.harmonic.lattice.cells.lo[:, axis])
        source = population(source, gamma_min=float(floor))
    if energy_cells is None:
        table = regime.harmonic.lattice.cells
        axis = table.variables.index("gamma")
        low = Cells.log(float(min(table.lo[:, axis])), regime.gamma_switch, 2)
        edges = np.unique(np.r_[low.edges, cells(order=order).edges])
        weights = (edges[:-1] ** -1.6 - edges[1:] ** -1.6) / (
            _GAMMA_LO**-1.6 - _GAMMA_HI**-1.6
        )
        axes = {"gamma": Cells.from_edges(edges, order=order, chart="log")}
        if source.components:
            axes["phi"] = AzimuthHarmonic()
        elif "B" in parents(source.factors.get("gamma")):
            axes["B"] = Cells.linear(4, 6, 1, order=order, unit="uG")
        energy_cells = Lattice(**axes, order=order, weights=weights)
    if extra_nodes is None:
        field_cells = (
            isinstance(energy_cells, Lattice) and "B" in energy_cells.cells.variables
        )
        extra_nodes = 11 if field_cells else 3
    return paper_energy_basis(regime, source, energy_cells, extra_nodes=extra_nodes)


def effective_kp_moments(lattice, *, gamma_min=100.0, nodes=24):
    """Return the paper's effective diagonal moments for its separable KP fit."""
    from .base.continuum import ContinuumKernel
    from .front._regime_measure import diagonal_energy_moments
    from .front._setup import placed

    kp, _ = placed(population("E4-KP", gamma_min=gamma_min), ContinuumKernel())
    return diagonal_energy_moments((kp,), lattice, nodes=nodes)
