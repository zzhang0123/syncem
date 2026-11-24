"""Starting-value convention for specified predictions."""

from ..fitting.family_fit import free_parameters, start_values
from ..population import FreeMoments, NonNegativeCells, Population
from ..values.errors import NothingToPredict, RoleError


def prediction_source(population, strict=False):
    """Fill declared starting values and refuse a prediction of unknown moment blocks."""
    if isinstance(population, (FreeMoments, NonNegativeCells)):
        raise NothingToPredict(
            "the population has unknown moments. Fit the observations first."
        )
    if not isinstance(population, Population):
        return population, ()
    parts, notes = [], []
    for part in population.parts:
        if any(
            isinstance(x, (FreeMoments, NonNegativeCells))
            for x in (*part.blocks, *part.factors.values())
        ):
            raise NothingToPredict(
                "the population has unknown moments. Fit the observations first."
            )
        names = free_parameters(part)
        if names:
            if strict:
                raise RoleError(f"strict prediction requires values for {tuple(names)}")
            part = part.at(start_values(part))
            notes.extend(names)
        parts.append(part)
    if not notes:
        return population, ()
    rebuilt = parts[0]
    for part in parts[1:]:
        rebuilt = rebuilt + part
    return rebuilt, tuple(dict.fromkeys(notes))
