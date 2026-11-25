"""Local identifiability and noise propagation for a physical family."""

from dataclasses import dataclass

import numpy as np

from ._finite_difference import _jacobian
from .rank import RANK_RTOL


@dataclass(frozen=True)
class FamilyNoise:
    """Identified right singular directions of the whitened response."""

    directions: np.ndarray
    singular: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_response(cls, jacobian, physical_scale):
        # Equilibrate in two stages so squaring a derivative in physical
        # parameter units cannot underflow or overflow the column norm.
        peak = np.max(np.abs(jacobian), axis=0)
        peak = np.where(peak > 0, peak, 1.0)
        unit = jacobian / peak
        norm = np.linalg.norm(unit, axis=0)
        norm = np.where(norm > 0, norm, 1.0)
        _, singular, vectors = np.linalg.svd(unit / norm, full_matrices=False)
        keep = singular > RANK_RTOL * singular[0]
        return cls(vectors[keep], singular[keep], physical_scale / peak / norm)

    def response(self, jacobian):
        """Noise response and visibility of each physical observable."""
        rows = np.atleast_2d(jacobian) * self.scale
        peak = np.max(np.abs(rows), axis=1, keepdims=True)
        unit = rows / np.where(peak > 0, peak, 1.0)
        retained = unit @ self.directions.T
        missing = unit - retained @ self.directions
        visible = np.linalg.norm(missing, axis=1) <= RANK_RTOL * np.linalg.norm(
            unit, axis=1
        )
        return (retained / self.singular) * peak, visible

    def covariance(self):
        response, visible = self.response(np.eye(len(self.scale)))
        cov = response @ response.T
        cov[~visible, :] = np.nan
        cov[:, ~visible] = np.nan
        return cov, visible, len(self.singular)

    def propagate(self, jacobian, shape, *, cov=False):
        response, visible = self.response(jacobian)
        response[~visible, :] = np.nan
        sigma = np.where(visible, np.linalg.norm(response, axis=1), np.nan)
        matrix = response @ response.T if cov else None
        if matrix is not None:
            matrix[~visible, :] = np.nan
            matrix[:, ~visible] = np.nan
        return sigma.reshape(shape), matrix


def physical_jacobian(function, params):
    """Differentiate in relative physical coordinates, using one-sided steps at bounds."""
    center = np.array([v.value for v in params.values()])
    bounds = np.array([v.bounds for v in params.values()])
    # Scales follow physical units, including very small fields and amplitudes.
    scale = np.where(center != 0, abs(center), 1.0)
    scale = np.minimum(scale, bounds[:, 1] - bounds[:, 0])
    lo, hi = ((bounds - center[:, None]) / scale[:, None]).T

    def evaluate(offset):
        return np.asarray(function(center + scale * offset)).reshape(-1)

    jac, _ = _jacobian(
        evaluate, np.zeros_like(center), lo, hi, np.cbrt(np.finfo(float).eps)
    )
    return jac / scale
