"""Bounded cache of whitened reductions for fixed-rank basis trials."""

from __future__ import annotations

from collections import OrderedDict
from typing import NamedTuple

import equinox as eqx
import numpy as np

from ..engine.basis import SpectralBasis
from ..values.errors import DataError, RoleError
from .family_fit import _coordinate
from .free_fit import _design, _fit_basis, _pairs, _white
from .rank import reduce_svd

_CACHE_BYTES = 32 * 2**20
_CACHE_ENTRIES = 8


@eqx.filter_jit
def _trial_svd(design, pairs):
    svd = reduce_svd(design, complex_columns=pairs)
    return svd.left, svd.cut.numerical_rank, svd.paired


class _Trial(NamedTuple):
    left: np.ndarray
    scores: np.ndarray


class _Trials:
    """Retain small trial reductions and release each trial basis after evaluation."""

    def __init__(self, build_basis, data, names, log):
        self.build_basis = build_basis
        self.data = data
        self.names = names
        self.log = log
        self.signature = None
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.y = None
        self.evaluations = 0

    def basis(self, x):
        parameters = dict(zip(self.names, _coordinate(x, self.log)))
        basis = self.build_basis(parameters)
        self.evaluations += 1
        if not isinstance(basis, SpectralBasis):
            raise TypeError("build_basis must return a SpectralBasis")
        signature = (
            basis.lattice.key,
            basis.columns.key,
            basis.record.stokes,
            basis.channels.row_keys(),
        )
        if signature[0] is None:
            raise DataError("the basis-parametric fit needs a concrete lattice key")
        if self.signature is not None and self.signature != signature:
            raise RoleError(
                "basis parameters must preserve the moment coordinates, column map and channels"
            )
        self.signature = signature
        return basis

    def get(self, x):
        key = tuple(x)
        entry = self.cache.pop(key, None)
        if entry is None:
            entry = self._evaluate(x)
            size = entry.left.nbytes + entry.scores.nbytes
            if size > _CACHE_BYTES:
                return entry
            while self.cache and (
                len(self.cache) >= _CACHE_ENTRIES
                or self.cache_bytes + size > _CACHE_BYTES
            ):
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= old.left.nbytes + old.scores.nbytes
            self.cache_bytes += size
        self.cache[key] = entry
        return entry

    def _evaluate(self, x):
        basis = _fit_basis(self.basis(x), self.data.channels)
        white = _white(self.data, basis, _design(basis))
        if self.y is None:
            if int(white.count) == 0:
                raise DataError("there are no observations in the fitted rows")
            self.y = np.asarray(white.data)
            if not np.all(np.isfinite(self.y)):
                raise DataError("the whitened observations must be finite")
        left, numerical_rank, pairs = _trial_svd(white.design, _pairs(basis))
        left = np.asarray(left)
        projection = left.T @ self.y
        residual = self.y - left @ projection
        tail = np.r_[np.cumsum(projection[::-1] ** 2)[::-1], 0.0]
        ranks = np.arange(tail.size)
        valid = (ranks <= int(numerical_rank)) & np.r_[True, ~np.asarray(pairs), True]
        scores = np.where(valid, residual @ residual + tail, np.inf)
        scores[0] = self.y @ self.y
        return _Trial(left, scores)

    def residual(self, x, rank):
        trial = self.get(x)
        if not np.isfinite(trial.scores[rank]):
            return None
        active = trial.left[:, :rank]
        return self.y - active @ (active.T @ self.y)
