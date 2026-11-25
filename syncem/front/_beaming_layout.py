"""Geometry and direct quadrature for two resolved Beaming axes."""

from ..base.capabilities import Beaming
from ..population._diagonal import conditional_crossings
from ..population.declaration import ancestors, law_at, parents
from ..population.quadrature import _subtract
from ..records.ledger import ErrorTerm
from ..values.errors import RoleError


def crossing_terms(populations, kernel=None):
    """Record the finite resolution of a conditional diagonal boundary search."""
    pair = ("mu", "eta")
    if kernel is not None:
        cap = next((c for c in kernel.capabilities if isinstance(c, Beaming)), None)
        if cap is None:
            return ()
        pair = (cap.variable, cap.partner)
    if any(conditional_crossings(pop.factors, pair) for pop in populations):
        return (
            ErrorTerm.unbounded(
                "Conditional beaming boundaries can have unresolved crossings.",
                name="quadrature:beaming_crossings",
            ),
        )
    return ()


def merge_beaming(kernel, lattice):
    """Swapped diagonal projections are identical at matching Legendre degree."""
    capability = next((c for c in kernel.capabilities if isinstance(c, Beaming)), None)
    if capability is None:
        return None
    pair = (capability.variable, capability.partner)
    degrees = dict(lattice.index.legendre)
    if all(name in degrees for name in pair) and degrees[pair[0]] == degrees[pair[1]]:
        return pair
    return None


def _check_pitch_cover(table, name):
    # Every interior cosine face must meet cells covering that face in
    # the other coordinates. This also works on locally refined tables.
    i = table.variables.index(name)
    other = [j for j in range(len(table.variables)) if j != i]
    projected = [
        {table.variables[j]: (lo[j], hi[j]) for j in other}
        for lo, hi in zip(table.lo, table.hi)
    ]
    for faces, adjacent, pole in (
        (table.lo[:, i], table.hi[:, i], -1.0),
        (table.hi[:, i], table.lo[:, i], 1.0),
    ):
        by_face = {}
        for c, value in enumerate(adjacent):
            by_face.setdefault(value, []).append(c)
        for cell, face in enumerate(faces):
            if face == pole:
                continue
            uncovered = (projected[cell],)
            for neighbour in by_face.get(face, ()):
                uncovered = tuple(
                    part
                    for box in uncovered
                    for part in _subtract(box, projected[neighbour])
                )
                if not uncovered:
                    break
            if uncovered:
                raise RoleError(
                    f"{name} cells under Beaming with a Legendre partner must "
                    "cover [-1, 1] in each slice of the other cells."
                )


def resolved_beaming(kernel, population, lattice):
    beaming = next((c for c in kernel.capabilities if isinstance(c, Beaming)), None)
    if beaming is None:
        return ()
    pair = (beaming.variable, beaming.partner)
    degrees = dict(lattice.index.legendre)
    cells = lattice.cells.variables
    if beaming.variable in cells and any(
        set(pair).issubset(block.variables) for block in population.blocks
    ):
        raise RoleError("joint point samples under Beaming need a Legendre pitch axis")
    if not set(pair).issubset(set(cells) | degrees.keys()):
        return ()
    if all(name in cells for name in pair):
        raise RoleError(
            "Beaming cannot resolve both cosines by cells. Use a Legendre axis on one cosine."
        )
    if beaming.variable in cells:
        _check_pitch_cover(lattice.cells, beaming.variable)
    return pair


def diagonal_reference(pop, lattice, pair, units):
    factors = pop.factors
    directions = sorted(
        (pair, pair[::-1]), key=lambda d: d[0] in lattice.cells.variables
    )
    for source, target in directions:
        if source in ancestors(factors, target):
            continue
        law = factors.get(source)
        if law is None or (not hasattr(law, "density") and not parents(law)):
            continue

        def density(point):
            local = law_at(law, point, units)
            if not hasattr(local, "density"):
                raise RoleError(
                    f"Beaming reference needs a cosine density for {source}"
                )
            return 2 * local.density(point[target])

        return ((source, target),), (("all", density),)
    raise RoleError(
        "the direct Beaming diagonal needs one integrable cosine and a density for its partner."
    )
