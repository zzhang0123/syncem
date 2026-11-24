"""Weighted joint push-forward for harmonic shape prediction."""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from ..distributions.screens import (
    CumulantScreen,
    EmpiricalScreen,
    GammaScreen,
    GaussianScreen,
    LaplaceScreen,
)
from ..values.errors import DataError
from ._basis_archive import read_archive, write_archive
from ._basis_io_codec import pack, unpack

_KIND = "SyncEM HarmonicShapeStatistics"
_ARRAYS = ("coordinates", "power_weights", "spin_weights", "depths", "screen_indices")


def _array(values, kinds, dtype):
    values = np.asarray(values)
    if values.dtype.kind not in kinds:
        raise DataError("invalid harmonic shape statistics array dtype")
    if (
        dtype is np.int64
        and values.dtype.kind == "u"
        and np.any(values > np.iinfo(np.int64).max)
    ):
        raise DataError("invalid depth screen index")
    array = np.array(values, dtype=dtype, copy=True)
    if not np.all(np.isfinite(array)):
        raise DataError("harmonic shape statistics arrays must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False, slots=True)
class DepthRule:
    """A finite depth characteristic with the declared rule weights."""

    depths: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        depths = _array(self.depths, "iuf", float)
        weights = _array(self.weights, "iuf", float)
        if depths.ndim != 1 or not depths.size or depths.shape != weights.shape:
            raise DataError("depth rule needs nonempty matching vectors")
        object.__setattr__(self, "depths", depths)
        object.__setattr__(self, "weights", weights)

    def characteristic(self, tau):
        return jnp.sum(
            jnp.exp(1j * jnp.asarray(tau)[..., None] * self.depths) * self.weights,
            axis=-1,
        )


_SCREEN_FIELDS = {
    DepthRule: ("depths", "weights"),
    GaussianScreen: ("mean", "sigma", "nodes"),
    LaplaceScreen: ("mean", "sigma", "nodes"),
    GammaScreen: ("mean", "sigma", "shape", "nodes"),
    EmpiricalScreen: ("depths", "weights"),
    CumulantScreen: ("cumulants", "tau_max"),
}
_SCREEN_TYPES = {cls.__name__: cls for cls in _SCREEN_FIELDS}


def _screen_record(screen):
    cls = type(screen)
    if cls not in _SCREEN_FIELDS:
        raise DataError(
            f"cannot archive the custom depth characteristic {cls.__name__}"
        )
    return cls.__name__, {name: getattr(screen, name) for name in _SCREEN_FIELDS[cls]}


def _load_screen(record):
    name, attrs = record
    cls = _SCREEN_TYPES[name]
    if set(attrs) != set(_SCREEN_FIELDS[cls]):
        raise ValueError("invalid depth characteristic parameters")
    return cls(**attrs)


@dataclass(frozen=True, eq=False)
class HarmonicShapeStatistics:
    """Empirical joint measure in `(nu1, z, s)` with absolute cgs weights.

    Rows of `coordinates` contain spacing in Hz and two dimensionless
    coordinates. Each row has real `power_weights` for I and V and complex
    `spin_weights` for Q+iU before propagation. Its depth in rad/m² can be
    accompanied by a characteristic from `screens`, selected by
    `screen_indices`. An index of -1 omits the characteristic.
    """

    coordinates: np.ndarray
    power_weights: np.ndarray
    spin_weights: np.ndarray
    depths: np.ndarray
    screen_indices: np.ndarray
    screens: tuple = field(default=(), repr=False)
    defaults: tuple = field(default=(), repr=False)

    def __post_init__(self):
        specs = (
            ("iuf", float),
            ("iuf", float),
            ("iufc", complex),
            ("iuf", float),
            ("iu", np.int64),
        )
        for name, (kinds, dtype) in zip(_ARRAYS, specs, strict=True):
            object.__setattr__(self, name, _array(getattr(self, name), kinds, dtype))
        screens = tuple(self.screens)
        object.__setattr__(self, "screens", screens)
        defaults = tuple(self.defaults)
        try:
            hash(defaults)
        except TypeError as exc:
            raise DataError(
                "harmonic shape statistics defaults must be hashable"
            ) from exc
        object.__setattr__(self, "defaults", defaults)
        coordinates = self.coordinates
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise DataError("coordinates must have shape (n_atoms, 3)")
        n = coordinates.shape[0]
        if any(getattr(self, name).shape != (n,) for name in _ARRAYS[1:]):
            raise DataError(
                "one set of weights and depth marks is required for each atom"
            )
        spacing, z, s = coordinates.T
        eps = 8 * np.finfo(float).eps
        if np.any(
            (spacing <= 0)
            | (z < 0)
            | (z > 1)
            | (np.abs(s) > 1)
            | (z**2 + s**2 > 1 + eps)
        ):
            raise DataError("invalid harmonic shape coordinates")
        if np.any(self.power_weights < 0) or np.any(
            np.abs(self.spin_weights) - self.power_weights > eps * self.power_weights
        ):
            raise DataError(
                "spin weight modulus must be at most the nonnegative power weight"
            )
        if any(
            not callable(getattr(screen, "characteristic", None)) for screen in screens
        ):
            raise DataError("a depth screen must supply a characteristic")
        if np.any((self.screen_indices < -1) | (self.screen_indices >= len(screens))):
            raise DataError("invalid depth screen index")

    def save(self, path):
        """Write the weighted rule as an atomic numeric archive."""
        arrays = {name: getattr(self, name) for name in _ARRAYS}
        records = tuple(_screen_record(screen) for screen in self.screens)
        payload = pack((records, self.defaults), arrays)
        write_archive(path, payload, arrays, kind=_KIND)

    @classmethod
    def load(cls, path) -> "HarmonicShapeStatistics":
        """Load a weighted rule without loading executable objects."""
        payload, arrays = read_archive(path, kind=_KIND)
        try:
            records, defaults = unpack(payload, arrays)
            screens = tuple(_load_screen(record) for record in records)
            return cls(
                coordinates=arrays["coordinates"],
                power_weights=arrays["power_weights"],
                spin_weights=arrays["spin_weights"],
                depths=arrays["depths"],
                screen_indices=arrays["screen_indices"],
                screens=screens,
                defaults=defaults,
            )
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise DataError("invalid harmonic shape statistics archive") from exc
