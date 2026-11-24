"""Recognize a separate free azimuth as one shared complex factor."""

from ..distributions import AzimuthFactor
from ..lattice.lattice import Lattice
from ..lattice.representations import AzimuthHarmonic
from ..population import FreeMoments, Population
from ..population.declaration import parents
from ..values.errors import RoleError


def shared_azimuth(kernel, component):
    """Match one independent azimuth marker beside a free primary block."""
    phi, primary = kernel.azimuth, kernel.primary
    if phi is None or primary is None or component.components:
        return False
    free = [
        block.variables for block in component.blocks if isinstance(block, FreeMoments)
    ]
    free.extend(
        law.bind(name).variables
        for name, law in component.factors.items()
        if isinstance(law, FreeMoments)
    )
    return (
        set(free) == {(phi,), (primary,)}
        and len(free) == 2
        and not any(phi in parents(law) for law in component.factors.values())
    )


def shared_source(kernel, component, lattice):
    """Attach a placeholder that leaves the natural P spectra available."""
    name = kernel.azimuth
    blocks = [
        block
        for block in component.blocks
        if not (isinstance(block, FreeMoments) and block.variables == (name,))
    ]
    facts = {**component.factors, name: AzimuthFactor(1)}
    source = Population(*blocks, symmetries=component.symmetries, **facts)
    if lattice is None or name not in lattice.representations:
        return source, lattice
    reps = lattice.representations
    if not isinstance(reps.pop(name), AzimuthHarmonic):
        raise RoleError("the shared factor needs a spin-2 azimuth or no azimuth axis")
    if lattice.cell_axes is not None:
        return source, Lattice(
            **lattice._config(),
            **dict(lattice.cell_axes),
            **reps,
            order=lattice.order if lattice.explicit_order else None,
            weights=lattice.weights if lattice.explicit_weights else None,
            depth_in_total=lattice.depth_in_total,
        )
    return source, Lattice(
        **lattice._config(),
        _table=lattice.cells,
        _reps=reps,
        order=lattice.order if lattice.explicit_order else None,
        weights=lattice.weights if lattice.explicit_weights else None,
        depth_in_total=lattice.depth_in_total,
    )
