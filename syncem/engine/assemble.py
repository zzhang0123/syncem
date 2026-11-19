"""Design assembly, the Stokes layout and the key rule of prediction (design/0.8/SPEC.md Sec. 3.4, 8.2, decision 151).

LABEL ``eq: channel basis expansion``. The stored spectra of a basis become design columns at S3. Column ``j``
repeats the spin-0 spectrum ``spin0_from[j]`` in I and V where ``uses`` lists it, and the spin-2 spectrum
``spin2_from[j]`` in Q and U. A spin-2 spectrum ``p = a + ib`` maps to columns as ``ST/lib/basis.py:241-263``
does. Without ``AzimuthHarmonic`` the M0 columns take ``c p`` with Q the real part and U the imaginary part, and
``c = 1`` when the kernel has no azimuth variable. ``Re M2`` columns take Q = a and U = b, and ``Im M2`` columns
take Q = -b and U = a. Every column is multiplied by ``lattice.weights[column_cell]``, so a change of ``w``
re-runs this stage only. The Stokes axis follows "IQUV" filtered by the request, and P gives Q and U together.

``predict_values`` is ``einsum('csn,...n->...cs', design, columns.apply(values))``. It refuses moments whose
lattice key differs from the basis lattice key, and a ``None`` key on either side, and it has no argument that
bypasses the check (SPEC Sec. 3.3).

Units: those of the kernel per unit scaled moment for the design, and of the kernel times the moments for a
prediction. Array shapes: ``spectra0`` ``(n_ch, n_s0, n0)``, ``spectra2`` ``(n_ch, n2)``, design ``(n_ch, n_stokes,
n)``, prediction ``(..., n_ch, n_stokes)``. Not certified: the spectra, which the providers compute and the
tests compare with the study oracles and the jet. Assembly itself is exact selection and scaling.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from ..lattice.moments import Moments

__all__ = [
    "AXES",
    "check_key",
    "design_array",
    "gather_spectra",
    "predict_values",
    "select_stokes",
    "spin0_blocks",
    "stokes_axes",
]

AXES = ("I", "Q", "U", "V")
_AXES_OF = {"I": ("I",), "P": ("Q", "U"), "V": ("V",)}


def stokes_axes(blocks: tuple[str, ...]) -> tuple[str, ...]:
    """The Stokes axes of the design for blocks in "IPV" order: I, Q, U, V filtered by the request."""
    wanted = {a for b in blocks for a in _AXES_OF[b]}
    return tuple(a for a in AXES if a in wanted)


def spin0_blocks(blocks: tuple[str, ...]) -> tuple[str, ...]:
    """The spin-0 blocks stored in ``spectra0`` along its middle axis, I before V."""
    return tuple(b for b in ("I", "V") if b in blocks)


def gather_spectra(I: Any, P: Any, V: Any, columns: Any) -> tuple[Any, Any]:
    """``spectra0 (n_ch, n_s0, n0)`` and ``spectra2 (n_ch, n2)`` or ``None`` from block-sized provider rows.

    ``I``, ``P`` and ``V`` have the shape ``(block_size, n_ch)``, as ``route_cells`` returns them, and ``V`` is
    ``None`` when V is not requested.
    """
    blocks = columns.stokes
    rows = {"I": I, "V": V}
    found = [np.asarray(rows[b])[columns.spin0_base].T for b in spin0_blocks(blocks)]
    n_ch = np.shape(I)[1]
    spectra0 = np.stack(found, axis=1) if found else np.zeros((n_ch, 0, columns.n0))
    spectra2 = np.asarray(P)[columns.spin2_base].T if "P" in blocks else None
    return spectra0, spectra2


def _spin0_rows(spectra0: Any, columns: Any, k: int, block: str, n_ch: int) -> Any:
    cols = columns.uses[block]
    out = jnp.zeros((n_ch, columns.n), dtype=jnp.float64)
    values = jnp.asarray(spectra0)[:, k, :][:, columns.spin0_from[cols]]
    return out.at[:, cols].set(values)


def _spin2_rows(
    spectra2: Any, columns: Any, block_names: tuple[str, ...], factor: Any, n_ch: int
) -> tuple[Any, Any]:
    """Q and U rows ``(n_ch, n)`` of the columns P uses, by the block of each column."""
    cols = columns.uses["P"]
    names = np.asarray(block_names)[columns.column_block[cols]]
    p = jnp.asarray(spectra2)[:, columns.spin2_from[cols]]
    q = jnp.zeros((n_ch, columns.n), dtype=jnp.float64)
    u = jnp.zeros((n_ch, columns.n), dtype=jnp.float64)
    m0, re, im = (np.flatnonzero(names == b) for b in ("M0", "Re M2", "Im M2"))
    c = p[:, m0] if factor is None else factor * p[:, m0]
    q = q.at[:, cols[m0]].set(c.real).at[:, cols[re]].set(p[:, re].real)
    u = u.at[:, cols[m0]].set(c.imag).at[:, cols[re]].set(p[:, re].imag)
    q = q.at[:, cols[im]].set(-p[:, im].imag)
    u = u.at[:, cols[im]].set(p[:, im].real)
    return q, u


def design_array(
    spectra0: Any,
    spectra2: Any,
    columns: Any,
    weights: Any,
    *,
    block_names: tuple[str, ...],
    azimuth_factor: Any = None,
) -> Any:
    """The design ``(n_ch, n_stokes, n)`` with ``w`` applied, Stokes axes as ``stokes_axes(columns.stokes)``.

    ``block_names`` are the lattice blocks ("M0", "Re M2", "Im M2"). ``azimuth_factor`` multiplies the M0
    columns of P, and ``None`` stands for 1.
    """
    blocks = columns.stokes
    n_ch = int(np.shape(spectra0)[0])
    rows: dict[str, Any] = {}
    for k, block in enumerate(spin0_blocks(blocks)):
        rows[block] = _spin0_rows(spectra0, columns, k, block, n_ch)
    if "P" in blocks:
        rows["Q"], rows["U"] = _spin2_rows(
            spectra2, columns, block_names, azimuth_factor, n_ch
        )
    stacked = jnp.stack([rows[a] for a in stokes_axes(blocks)], axis=1)
    w = jnp.asarray(np.asarray(weights, dtype=float)[columns.column_cell])
    return stacked * w


def select_stokes(design: Any, axes: tuple[str, ...], stokes: Any) -> Any:
    """Rows of the built Stokes ``axes`` named by ``stokes`` ("I", "P" for Q and U, "QU", or a sequence)."""
    if stokes is None:
        return design
    names = tuple(stokes)
    wanted = [a for n in names for a in _AXES_OF.get(n, (n,))]
    if not wanted or len(set(wanted)) != len(wanted) or not set(wanted) <= set(axes):
        raise ValueError(
            f"stokes={stokes!r} must name distinct built Stokes axes among {axes} (P for Q and U)"
        )
    return design[:, [axes.index(a) for a in wanted]]


def check_key(lattice_key: str | None, moments: Any) -> None:
    """Refuse moments of another lattice, and a ``None`` key on either side (SPEC Sec. 3.3, 8.6)."""
    if not isinstance(moments, Moments):
        raise TypeError(f"moments must be Moments, not {type(moments).__name__}")
    if moments.key is None or lattice_key is None:
        raise ValueError(
            "a moment key or the basis lattice key is None, so the moments cannot be matched to the lattice. "
            "Build the lattice outside the transformation, or carry its key with Lattice.with_key"
        )
    if moments.key != lattice_key:
        raise ValueError(
            f"the moments carry the lattice key {moments.key[:12]}, and the basis lattice key is "
            f"{lattice_key[:12]}"
        )


def predict_values(basis: Any, moments: Any) -> Any:
    """``(..., n_ch, n_stokes)`` prediction of ``moments`` in lattice rows. Moments of another lattice raise."""
    check_key(basis.lattice.key, moments)
    columns = basis.columns.apply(moments.values)
    return jnp.einsum("csn,...n->...cs", basis.design(), columns)
