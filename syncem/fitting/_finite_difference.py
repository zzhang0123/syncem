"""Bound-aware finite differences for host fit diagnostics."""

import numpy as np


def _coordinate_scale(x, lo, hi):
    return np.minimum(np.maximum(abs(x), 1.0), hi - lo)


def _difference(evaluate, x, center, i, target, plus, minus):
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
            return (right - left) / (2 * h), h
    # At a bound or fixed-rank boundary, stay on the admissible side.
    for sign, reach in sorted(((1, plus), (-1, minus)), key=lambda item: -item[1]):
        for divisor in (1, 2, 4, 8):
            h = min(target, reach / 2) / divisor
            if h <= minimum:
                break
            direction[i] = sign * h
            middle, end = evaluate(x + direction), evaluate(x + 2 * direction)
            if middle is not None and end is not None:
                return sign * (-3 * center + 4 * middle - end) / (2 * h), sign * h
    return np.zeros_like(center), 0.0


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
