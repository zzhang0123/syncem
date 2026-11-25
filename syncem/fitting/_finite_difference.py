"""Bound-aware finite differences for host fit diagnostics."""

import numpy as np


def _coordinate_scale(x, lo, hi):
    return np.minimum(np.maximum(abs(x), 1.0), hi - lo)


def _stencil(evaluate, x, center, i, target, plus, minus):
    """Return a local derivative, physical step and output-wise roundoff scale."""
    eps = 16 * np.finfo(float).eps
    direction = np.zeros_like(x)
    minimum = 2 * np.finfo(float).eps * max(abs(x[i]), 1e-300)
    for divisor in (1, 2, 4, 8):
        h = target / divisor
        if h <= minimum:
            break
        if h > min(plus, minus):
            continue
        direction[i] = h
        right, left = evaluate(x + direction), evaluate(x - direction)
        if right is not None and left is not None:
            return (right - left) / (2 * h), h, eps * (abs(right) + abs(left)) / (2 * h)
    # At a bound or fixed-rank boundary, stay on the admissible side.
    for sign, reach in sorted(((1, plus), (-1, minus)), key=lambda item: -item[1]):
        for divisor in (1, 2, 4, 8):
            h = min(target, reach / 2) / divisor
            if h <= minimum:
                break
            direction[i] = sign * h
            middle, end = evaluate(x + direction), evaluate(x + 2 * direction)
            if middle is not None and end is not None:
                delta = 4 * (middle - center) - (end - center)
                magnitude = 3 * abs(center) + 4 * abs(middle) + abs(end)
                return sign * delta / (2 * h), sign * h, eps * magnitude / (2 * h)
    return np.zeros_like(center), 0.0, np.zeros_like(center)


def _difference(evaluate, x, center, i, target, plus, minus):
    def stencil(factor):
        return _stencil(evaluate, x, center, i, factor * target, plus, minus)

    fine, step, error = stencil(1)
    unresolved = (abs(fine) <= error) & (error > 0)
    result = np.where(unresolved, 0.0, fine)
    last_step, reported = step, step
    # An invariant can vary by a few ulps between quadrature evaluations.
    # A weak slope can do the same at the initial step. Retain it when two
    # wider stencils resolve a common derivative above their own roundoff.
    for factor in (8, 64, 512, 4096):
        if not np.any(unresolved):
            break
        try:
            coarse, h, coarse_error = stencil(factor)
        except (ValueError, FloatingPointError):
            # A wider trial can leave the family's physical domain.
            break
        if abs(h) <= abs(last_step):
            break
        consistent = abs(coarse - fine) <= error + coarse_error
        recovered = (
            unresolved & (abs(coarse) > coarse_error) & (abs(fine) > error) & consistent
        )
        if np.any(recovered):
            result = np.where(recovered, coarse, result)
            unresolved &= ~recovered
            reported = h
        fine, last_step, error = coarse, h, coarse_error
    return result, reported


def _jacobian(evaluate, x, lo, hi, step):
    """Differentiate profiled residuals in transformed coordinates."""
    center = evaluate(x)
    derivative = [
        _difference(evaluate, x, center, i, target, plus, minus)
        for i, (target, plus, minus) in enumerate(
            zip(step * _coordinate_scale(x, lo, hi), hi - x, x - lo)
        )
    ]
    columns, steps = zip(*derivative)
    return np.column_stack(columns), tuple(map(float, steps))
