"""Tensor moment coordinates and sparse lifts for independent factors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..values.errors import RoleError


@dataclass(frozen=True)
class _Group:
    names: tuple[str, ...]
    coordinates: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    row: np.ndarray

    @property
    def size(self):
        return len(self.coordinates)

    def mass_start(self):
        return np.array([not any(power) for _, power in self.coordinates], dtype=float)


def _groups(lattice, groups):
    axes = lattice.cells.variables
    if lattice.cells.tensor is None or lattice.index.blocks != ("M0",) or lattice.reps:
        raise RoleError("independent factors need an all-celled tensor lattice")
    declared = tuple(
        (group,) if isinstance(group, str) else tuple(group) for group in groups
    )
    names = tuple(name for group in declared for name in group)
    if (
        len(declared) < 2
        or not all(declared)
        or len(names) != len(axes)
        or set(names) != set(axes)
    ):
        raise RoleError("independent factors must partition all celled variables")
    arrays = lattice.index.arrays
    cells = np.column_stack(np.unravel_index(arrays.cell, lattice.cells.tensor))
    out = []
    for group in declared:
        ix = [axes.index(name) for name in group]
        keys, row = np.unique(
            np.column_stack((cells[:, ix], arrays.powers[:, ix])),
            axis=0,
            return_inverse=True,
        )
        labels = tuple(
            (tuple(map(int, k[: len(ix)])), tuple(map(int, k[len(ix) :]))) for k in keys
        )
        out.append(_Group(group, labels, row))
    return tuple(out)


def _unit(vector, fallback):
    norm = np.linalg.norm(vector)
    if norm == 0:
        vector = fallback
        norm = np.linalg.norm(vector)
    sign = np.copysign(1.0, vector[np.argmax(abs(vector))])
    return vector / (norm * sign)


def _leading_pair(groups, raw, rng):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import ArpackNoConvergence, svds

    first, second = groups
    shape = (first.size, second.size)
    tensor = csr_matrix((raw, (first.row, second.row)), shape=shape)
    if min(shape) == 1:
        return tensor.toarray().ravel() if first.size == 1 else np.ones(1)
    if not np.any(raw):
        return second.mass_start()
    if shape[0] * shape[1] <= 128**2:
        _, _, directions = np.linalg.svd(tensor.toarray(), full_matrices=False)
    else:
        try:
            _, _, directions = svds(tensor, k=1, v0=rng.standard_normal(min(shape)))
        except ArpackNoConvergence:
            # A bounded power iteration supplies an approximate leading start.
            factor = second.mass_start()
            for _ in range(30):
                factor = _unit(tensor.T @ (tensor @ factor), factor)
            return factor
    return directions[0]


def _initial(groups, raw, rng):
    if len(groups) == 2:
        return (
            groups[0].mass_start(),
            _unit(_leading_pair(groups, raw, rng), groups[1].mass_start()),
        )
    # The power iteration uses only lattice rows, including with a total-degree cap.
    factors = [g.mass_start() for g in groups]
    for _ in range(8):
        for i, group in enumerate(groups):
            values = raw * _other_rows(groups, factors, i)
            vector = np.bincount(group.row, weights=values, minlength=group.size)
            factors[i] = _unit(vector, factors[i])
    return tuple(factors)


def _other_rows(groups, factors, active):
    return np.prod(
        [f[g.row] for i, (g, f) in enumerate(zip(groups, factors)) if i != active],
        axis=0,
    )


@dataclass(frozen=True)
class _Lift:
    """A signed isometry into scaled joint rows and the raw coordinate norms."""

    group: _Group
    weight: np.ndarray
    norms: np.ndarray

    @classmethod
    def make(cls, groups, factors, weights, active):
        group = groups[active]
        raw = _other_rows(groups, factors, active) / weights
        norms = np.sqrt(np.bincount(group.row, weights=raw**2, minlength=group.size))
        norm = norms[group.row]
        weight = np.divide(raw, norm, out=np.zeros_like(raw), where=norm > 0)
        return cls(group, weight, norms)

    def design(self, joint):
        from scipy.sparse import csr_matrix

        rows = np.arange(self.weight.size)
        lift = csr_matrix(
            (self.weight, (rows, self.group.row)), shape=(rows.size, self.group.size)
        )
        return joint @ lift

    def joint(self, values):
        return values[..., self.group.row] * self.weight

    def raw(self, fitted):
        norms = self.norms
        return np.divide(fitted, norms, out=np.zeros_like(norms), where=norms > 0)
