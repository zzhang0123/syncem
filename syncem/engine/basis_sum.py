"""The sum of spectral bases on one lattice (design/0.8/SPEC.md Sec. 3.4, rule R7, decision 152).

LABEL ``eq: channel basis expansion``. A ``BasisSum`` holds one ``SpectralBasis`` per distinct set of specified
statistics of a sum of components. Every part is built on the same lattice and channels with the same Stokes
request, and the build checks equal lattice keys, equal channel row keys, equal Stokes requests and distinct
component keys. ``offsets`` gives the column offset of each part. ``design`` is the horizontal concatenation of
the parts' designs at these offsets. ``predict`` takes a tuple of ``Moments``, one per part. It matches them by
``Moments.component`` when every moment is named and by position when none is, and a partly named tuple raises.
The prediction is the sum of the parts' predictions in the order of the parts. ``at`` maps over the parts. In
step 5 prediction returns arrays, and ``Spectrum`` arrives in step 6.

Units: those of the parts, the kernel per unit scaled moment for the design and the kernel times the moments for
a prediction. Array shapes: ``design()`` ``(n_ch, n_stokes, n)`` with ``n`` the sum of the parts' columns, a prediction
``(..., n_ch, n_stokes)``. Not certified: the parts' spectra, which the providers compute and the tests compare
with the study oracles. Concatenation and summation are exact.
"""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..lattice.moments import Moments
from .assemble import predict_values
from .basis import SpectralBasis

__all__ = ["BasisSum"]


class BasisSum(eqx.Module):
    """One ``SpectralBasis`` per component key on one lattice, with the column offset of each part (decision 152).

    ``parts`` are the bases, ``components`` (static) the distinct component keys, and ``offsets`` (static) the
    column offset of each part in the design. Not certified: the parts' spectra, compared with oracles by the
    tests.
    """

    LABEL: ClassVar[str] = "eq: channel basis expansion"
    parts: tuple
    components: tuple[str, ...] = eqx.field(static=True)
    offsets: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, parts: Any, components: Any):
        parts, components = _checked_parts(parts), tuple(components)
        _check_components(components, len(parts))
        _check_shared(parts)
        sizes = [p.columns.n for p in parts]
        self.parts = parts
        self.components = components
        self.offsets = tuple(int(o) for o in np.cumsum([0] + sizes[:-1]))

    @property
    def n(self) -> int:
        """Columns of the design, the sum over the parts."""
        return int(sum(p.columns.n for p in self.parts))

    def design(self, data: Any = None, *, stokes: Any = None) -> Any:
        """``(n_ch, n_stokes, n)``: the parts' designs side by side at ``offsets``."""
        return jnp.concatenate(
            [p.design(data, stokes=stokes) for p in self.parts], axis=-1
        )

    def predict(self, moments: Any) -> Any:
        """``(..., n_ch, n_stokes)``: the sum of the parts' predictions of ``moments``, one ``Moments`` per part."""
        ordered = _match(moments, self.components)
        out = None
        for part, m in zip(self.parts, ordered):
            value = predict_values(part, m)
            out = value if out is None else out + value
        return out

    def at(self, channels: Any) -> "BasisSum":
        """The same parts on ``channels``, each through ``SpectralBasis.at``."""
        return BasisSum(tuple(p.at(channels) for p in self.parts), self.components)


def _checked_parts(parts: Any) -> tuple:
    parts = tuple(parts)
    if not parts:
        raise ValueError("a BasisSum needs at least one part")
    for p in parts:
        if not isinstance(p, SpectralBasis):
            raise TypeError(
                f"BasisSum parts must be SpectralBasis, not {type(p).__name__}"
            )
    return parts


def _check_components(components: tuple, n_parts: int) -> None:
    if len(components) != n_parts:
        raise ValueError(
            f"a BasisSum needs one component key per part: {n_parts} parts, {len(components)} components"
        )
    if any(not isinstance(c, str) or not c for c in components):
        raise ValueError(f"component keys must be non-empty str, not {components!r}")
    if len(set(components)) != len(components):
        raise ValueError(f"component keys must be distinct, not {components!r}")


def _check_shared(parts: tuple) -> None:
    """Equal lattice keys, channel row keys and Stokes requests across the parts (decision 152)."""
    keys = [p.lattice.key for p in parts]
    if any(k is None for k in keys):
        raise ValueError(
            "a part has the lattice key None, so the parts cannot be shown to share one lattice. Build the "
            "lattice outside the transformation, or carry its key with Lattice.with_key"
        )
    if len(set(keys)) != 1:
        raise ValueError(
            f"BasisSum parts must share one lattice, and the lattice keys are {[k[:12] for k in keys]}"
        )
    rows = parts[0].channels.row_keys()
    if any(p.channels.row_keys() != rows for p in parts[1:]):
        raise ValueError(
            "BasisSum parts must share their channels, and the channel row keys differ"
        )
    stokes = [p.columns.stokes for p in parts]
    if any(s != stokes[0] for s in stokes[1:]):
        raise ValueError(f"BasisSum parts must share one Stokes request, not {stokes}")


def _match(moments: Any, components: tuple[str, ...]) -> tuple:
    """``moments`` in the order of ``components``: by name when every one is named, by position when none is."""
    if not isinstance(moments, (tuple, list)):
        raise TypeError(
            f"BasisSum.predict takes a tuple of Moments, one per part, not {type(moments).__name__}"
        )
    moments = tuple(moments)
    for m in moments:
        if not isinstance(m, Moments):
            raise TypeError(f"BasisSum.predict takes Moments, not {type(m).__name__}")
    if len(moments) != len(components):
        raise ValueError(
            f"the sum has {len(components)} parts {components}, and {len(moments)} Moments were given"
        )
    names = [m.component for m in moments]
    if all(n is None for n in names):
        return moments
    if any(n is None for n in names):
        raise ValueError(
            f"a partly named tuple of Moments (components {names}) cannot be matched. Name every Moments "
            "or none"
        )
    if sorted(names) != sorted(components):
        raise ValueError(
            f"the Moments name the components {names}, and the sum has the components {components}"
        )
    by_name = dict(zip(names, moments))
    return tuple(by_name[c] for c in components)
