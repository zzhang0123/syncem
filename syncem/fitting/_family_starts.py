"""Bounded multistart search in the family's optimization coordinates."""

import numpy as np

from ..values.errors import DataError


def _starts(initial, lower, upper, count):
    """Sample distinct deterministic interior starts around the user's point."""
    from scipy.stats import qmc

    scale = np.where(initial == 0, 1.0, np.abs(initial))
    left = np.maximum(lower, initial - scale)
    right = np.minimum(upper, initial + scale)
    yield initial
    if count == 1:
        return
    # The shifted Halton coordinates explore different directions in each dimension.
    sequence = qmc.Halton(initial.size, scramble=False)
    sequence.fast_forward(1)
    used = {tuple(initial)}
    for relative in sequence.random(4 * count):
        trial = left + relative * (right - left)
        key = tuple(trial)
        if key not in used:
            used.add(key)
            yield trial
        if len(used) == count:
            return


def solve_starts(residual, jacobian, initial, lower, upper, count):
    from scipy.optimize import least_squares

    costs = []
    attempts = 0
    best = None
    last_error = None
    for start in _starts(initial, lower, upper, count):
        attempts += 1
        try:
            result = least_squares(
                residual,
                start,
                jac=jacobian,
                bounds=(lower, upper),
                method="trf",
                max_nfev=60,
            )
        except (ValueError, FloatingPointError, DataError) as exc:
            last_error = exc
            continue
        if not result.success:
            last_error = result.message
            continue
        score = 2 * result.cost
        costs.append(score)
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        raise DataError(f"No family fit start converged. {last_error}")
    matches = np.isclose(costs, 2 * best.cost, rtol=1e-8, atol=1e-8)
    return best.x, attempts, int(np.count_nonzero(matches))
