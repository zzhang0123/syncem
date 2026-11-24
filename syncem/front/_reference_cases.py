"""Independent finite-population checks on one saved column geometry."""

from dataclasses import dataclass

import numpy as np

from ..base.continuum import ContinuumKernel
from ..distributions import AgedPowerLaw, BrokenPowerLaw, PowerLaw
from ..lattice._lattice_parts import REFERENCE_SLOPE


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    band_hz: tuple[float, float]
    max_fraction_of_intensity: float | None


def reference_cases(basis):
    """Check the three prescribed electron populations on first inspection."""
    primary = basis.averaged.kernel.primary
    table, population = basis.lattice.cells, basis.forward_population
    if (
        basis.forward_accuracy is None
        or population is None
        or type(basis.averaged.kernel) is not ContinuumKernel
        or table.variables != (primary,)
        or "I" not in basis.record.stokes
    ):
        return ()
    lo, hi = float(table.lo.min()), float(table.hi.max())
    cut = float(table.reference[table.n_cells // 2, 0])
    if not 0 < lo < cut < hi < np.inf:
        return ()
    limits = {"lo": lo, "hi": hi, "column": 1.0}
    models = (
        ("power law", PowerLaw(REFERENCE_SLOPE, **limits)),
        ("smoothly broken power law", BrokenPowerLaw(2.0, 3.2, cut, **limits)),
        ("aged power law", AgedPowerLaw(2.6, gamma_cut=cut, lo=lo, column=1.0)),
    )
    support = np.asarray(basis.channels.support)
    band = float(support.min()), float(support.max())
    checks = []
    for name, law in models:
        spectrum = basis.predict(
            population.replace(**{primary: law}), check=basis.forward_accuracy
        )
        ledger = {term.name: term for term in spectrum.ledger.terms}
        inside = ledger.get("representation")
        outside = ledger.get("outside_lattice")
        maximum = None
        if (
            inside is not None
            and outside is not None
            and inside.value is not None
            and outside.value is not None
        ):
            delta = inside.value
            angular = ledger.get("azimuth_truncation")
            if angular is not None and angular.value is not None:
                delta = delta + angular.value
            ref = spectrum.stokes - delta - outside.value
            i = np.abs(np.asarray(ref[..., spectrum.components.index("I")]))[:, None]
            if np.all(i > 0):
                maximum = float(np.max(np.abs(delta) / i))
        checks.append(ReferenceCase(name, band, maximum))
    return tuple(checks)
