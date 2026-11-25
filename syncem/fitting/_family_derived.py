"""Invariants of an electron family in the manuscript's conventions."""

import numpy as np

from ..base.variables import SYNCHROTRON_VARIABLES
from ..constants import C_CGS, E_ESU, M_E
from ..distributions.densities import Isotropic, Uniform
from ..population.declaration import ancestors, parents
from ..population.quadrature import panels
from ..values.errors import RoleError


def _mean_perpendicular_field(population):
    """Column expectation of B sqrt(1-eta^2) over its joint marginal."""
    pop = population.placed(tuple(SYNCHROTRON_VARIABLES.values()))
    factors = dict(pop.factors)
    block_names = {name for block in pop.blocks for name in block.variables}
    if "eta" not in factors and "eta" not in block_names:
        factors["eta"] = Isotropic()
    needed = {"B", "eta"}
    needed.update(ancestors(factors, "eta"), ancestors(factors, "B"))
    if needed - factors.keys() - block_names:
        raise RoleError(
            "break_frequency needs the joint field and inclination statistics"
        )
    marginal = {name: law for name, law in factors.items() if name in needed}
    blocks = tuple(
        block for block in pop.blocks if needed.intersection(block.variables)
    )
    angle = marginal.get("eta")
    # Integrate an independent uniform inclination exactly, including both poles.
    sine = None
    if isinstance(angle, Uniform) and all(
        "eta" not in parents(law) for law in marginal.values()
    ):
        lo, hi = map(float, angle.support)

        def primitive(x):
            return (x * np.sqrt(max(1 - x * x, 0.0)) + np.arcsin(x)) / 2

        sine = (primitive(hi) - primitive(lo)) / (hi - lo)
        marginal.pop("eta")
    mass = weighted = 0.0
    for point, weights in panels(
        marginal,
        {},
        blocks=blocks,
        units={k: v.unit for k, v in SYNCHROTRON_VARIABLES.items()},
        variables=SYNCHROTRON_VARIABLES,
        charts={"gamma": "log"},
        n=96,
    ):
        sin_theta = (
            sine
            if sine is not None
            else np.sqrt(np.maximum(1 - point["eta"] ** 2, 0.0))
        )
        weighted += weights @ (point["B"] * sin_theta)
        mass += np.sum(weights)
    if not mass > 0:
        raise RoleError("break_frequency needs a nonzero column measure")
    return weighted / mass


def family_quantity(population, name):
    law = population.factors.get("gamma")
    if name == "paper_coefficient" and hasattr(law, "paper_coefficient"):
        coefficient = law.paper_coefficient()
        return float(population.column_factor * coefficient.value), coefficient.label
    if name == "break_frequency":
        gamma = getattr(law, "gamma_cut", None)
        if gamma is None:
            gamma = getattr(law, "gamma_break", None)
        if gamma is not None:
            factor = 3 * E_ESU / (4 * np.pi * M_E * C_CGS)
            return (
                factor * _mean_perpendicular_field(population) * float(gamma) ** 2,
                "break frequency in Hz",
            )
    raise RoleError(f"{name!r} is undefined for this fitted electron population")
