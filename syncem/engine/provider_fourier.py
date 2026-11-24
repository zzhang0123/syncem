"""Azimuthal Fourier projection of the column jets (SPEC Sec. 5.2 R4a, decision 179).

LABEL ``eq: channel basis expansion``. Evaluate the natural-basis kernel at
equispaced angles and keep coefficients for ``-m_max .. m_max``. Rotate
polarization through the signed moment indices ``m + 2``. Both spins use the
same real and imaginary moments of the measure.

Units are those of the kernel. Spectra have the usual jet axes with a final
Fourier-mode axis. The design is real, with shape ``(n_ch, n_stokes, n)``.
Not certified: the finite-grid projection, its quadrature, or omitted orders.

Fourier factors are dimensionless. The projected spectra carry the units of
the averaged kernel.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..lattice.representations import AzimuthFourier
from ..records.ledger import ErrorTerm, Ledger
from ..records.records import RouteRecord
from ..values.errors import RoleError
from ._basis_ledger import channel_ledger
from ._basis_update import build_record, capped_lattice, order_indices, stokes_blocks
from .assemble import select_stokes, spin0_blocks, stokes_axes
from .basis import SpectralBasis, specified_records
from .columns import column_map
from .direct import _azimuth_independent
from .provider_jet import jet_lattice, make_routine


def fourier_axis(lattice, azimuth):
    """Fourier representation of the kernel's azimuth, if present."""
    rep = lattice.representations.get(azimuth)
    return rep if isinstance(rep, AzimuthFourier) else None


def sample_count(m_max):
    """The full-turn projection uses 32 nodes or the number of model modes."""
    return max(32, 2 * int(m_max) + 1)


class _AtAngle(eqx.Module):
    inner: Any
    angle: Any

    @property
    def averaged(self):
        return self.inner.averaged

    @property
    def fixed(self):
        return self.inner.fixed

    @property
    def legendre(self):
        return self.inner.legendre

    def __call__(self, channels, point):
        return self.inner(channels, {**point, self.inner.kernel.azimuth: self.angle})


def _project(averaged, channels, lattice, columns, m_max, counts):
    """Coarse spectra and the fine-minus-coarse spectra on the same modes."""
    n = sample_count(m_max)
    modes = np.arange(-m_max, m_max + 1)
    spin0 = spin0_blocks(columns.stokes)
    with_v = "V" in spin0
    with_p = "P" in columns.stokes
    routines = {}

    def routine(names, charts, exponents):
        key = names, charts, exponents
        if key not in routines:
            routines[key] = make_routine(*key)
        return routines[key]

    shapes = ((channels.n_ch, len(spin0), columns.n0, modes.size),)
    if with_p:
        shapes += ((channels.n_ch, columns.n2, modes.size),)
    coarse = tuple(np.zeros(shape, dtype=complex) for shape in shapes)
    fine = tuple(np.zeros_like(a) for a in coarse)
    for h in range(2 * n):
        angle = np.pi * h / n
        out = jet_lattice(
            _AtAngle(averaged, jnp.asarray(angle)),
            channels,
            lattice,
            with_v=with_v,
            routine=routine,
            events=counts,
        )
        natural = {"I": out[0], "V": out[2] if with_v else None}
        spectra0 = (
            np.stack([natural[s][columns.spin0_base].T for s in spin0], axis=1)
            if spin0
            else np.zeros(shapes[0][:-1])
        )
        values = (spectra0,) + ((out[1][columns.spin2_base].T,) if with_p else ())
        phase = np.exp(-1j * modes * angle) / (2 * n)
        for low, high, value in zip(coarse, fine, values, strict=True):
            weighted = value[..., None] * phase
            high += weighted
            if h % 2 == 0:
                low += 2 * weighted
    return coarse, tuple(b - a for a, b in zip(coarse, fine, strict=True))


def _fold(spectra, groups, order, k, imag, spin):
    """Coefficients of real and imaginary M_k, with M_-k = conjugate(M_k)."""
    plus = k - spin + order
    minus = -k - spin + order
    valid_plus = (0 <= plus) & (plus <= 2 * order)
    valid_minus = (k > 0) & (0 <= minus) & (minus <= 2 * order)
    p = jnp.where(valid_plus, spectra[:, groups, np.clip(plus, 0, 2 * order)], 0)
    m = jnp.where(valid_minus, spectra[:, groups, np.clip(minus, 0, 2 * order)], 0)
    return jnp.where(imag, 1j * (p - m), p + m)


def _design(s0, s2, columns, weights, order):
    rows = {}
    k = (columns.column_block + 1) // 2
    imag = (k > 0) & (columns.column_block % 2 == 0)
    for i, block in enumerate(spin0_blocks(columns.stokes)):
        use = columns.uses[block]
        values = _fold(
            jnp.asarray(s0)[:, i], columns.spin0_from[use], order, k[use], imag[use], 0
        )
        rows[block] = jnp.zeros((s0.shape[0], columns.n)).at[:, use].set(values.real)
    if "P" in columns.stokes:
        use = columns.uses["P"]
        values = _fold(
            jnp.asarray(s2), columns.spin2_from[use], order, k[use], imag[use], 2
        )
        shape = (s0.shape[0], columns.n)
        rows["Q"] = jnp.zeros(shape).at[:, use].set(values.real)
        rows["U"] = jnp.zeros(shape).at[:, use].set(values.imag)
    out = jnp.stack([rows[name] for name in stokes_axes(columns.stokes)], axis=1)
    return out * np.asarray(weights)[columns.column_cell]


class FourierBasis(SpectralBasis):
    """Selected Fourier coefficients of the natural-basis column jets."""

    m_max: int = eqx.field(static=True, kw_only=True)
    refined_delta: tuple = eqx.field(kw_only=True)

    def design(self, data=None, *, stokes=None):
        """Real design with the cell weights and shared angular rows."""
        if data is not None:
            return self._observed_design(data, stokes)
        out = _design(
            self.spectra0, self.spectra2, self.columns, self.lattice.weights, self.m_max
        )
        return select_stokes(out, self.record.stokes, stokes)

    def sampling_error(self, moments):
        """Magnitude of the doubled-grid projection difference on supplied moments."""
        s0, *s2 = self.refined_delta
        delta = _design(
            s0, s2[0] if s2 else None, self.columns, self.lattice.weights, self.m_max
        )
        return jnp.abs(
            jnp.einsum("csn,...n->...cs", delta, self.columns.apply(moments.values))
        )

    def _selection_spectra(self, idx0, idx2):
        spectra = super()._selection_spectra(idx0, idx2)
        s0, *s2 = self.refined_delta
        spectra["refined_delta"] = (
            np.asarray(s0)[:, :, idx0],
            *(np.asarray(part)[:, idx2] for part in s2),
        )
        return spectra

    def at(self, channels):
        """Gather stored rows or rebuild the Fourier projection on new channels."""
        lookup = {key: i for i, key in enumerate(self.channels.row_keys())}
        keys = channels.row_keys()
        if any(key not in lookup for key in keys):
            return self._rebuild(channels, self.lattice)
        rows = np.array([lookup[key] for key in keys], dtype=np.int64)
        return dataclasses.replace(
            self,
            channels=channels,
            spectra0=np.asarray(self.spectra0)[rows],
            spectra2=None if self.spectra2 is None else np.asarray(self.spectra2)[rows],
            refined_delta=tuple(np.asarray(x)[rows] for x in self.refined_delta),
            ledger=channel_ledger(self, channels, rows),
        )

    def with_order(self, order):
        """Gather the retained orders or extend the Fourier basis on a new lattice."""
        if (
            isinstance(order, (bool, np.bool_))
            or not isinstance(order, (int, np.integer))
            or order < 0
        ):
            raise ValueError("order must be a non-negative integer")
        if self.lattice.order is None:
            raise ValueError("the lattice has no total order")
        if order > self.lattice.order:
            return self._rebuild(self.channels, self.lattice.with_order(order))
        lattice = capped_lattice(self.lattice, int(order))
        options = dict(self.record.defaults)
        columns = column_map(
            lattice,
            self.averaged.kernel,
            self.rules,
            max_columns=options["max_columns"],
        )
        idx0, idx2 = order_indices(self, lattice, columns)
        s0, *s2 = self.refined_delta
        return dataclasses.replace(
            self,
            lattice=lattice,
            columns=columns,
            spectra0=self.spectra0[:, :, idx0],
            spectra2=None if self.spectra2 is None else self.spectra2[:, idx2],
            refined_delta=(s0[:, :, idx0],) + ((s2[0][:, idx2],) if s2 else ()),
            record=build_record(
                self.record.routes,
                columns.stokes,
                self.record.s1,
                columns,
                lattice,
                defaults=options,
                notes=self.record.notes
                + ("Lower order from stored Fourier coefficients.",),
            ),
        )

    def _rebuild(self, channels, lattice):
        ledger = channel_ledger(self, channels, np.full(channels.n_ch, -1))
        maximum = dict(self.record.defaults)["max_columns"]
        result = build_fourier(
            self.averaged, channels, lattice, rules=self.rules, max_columns=maximum
        )
        return dataclasses.replace(
            self,
            channels=channels,
            lattice=lattice,
            columns=result.columns,
            spectra0=result.spectra0,
            spectra2=result.spectra2,
            refined_delta=result.refined_delta,
            record=result.record,
            specified=result.specified,
            key=result.key,
            ledger=ledger,
        )


def build_fourier(averaged, channels, lattice, *, rules, max_columns=20_000):
    """Build compact column coefficients of an averaged kernel with resolved azimuth."""
    start = time.perf_counter()
    kernel = averaged.kernel
    rep = fourier_axis(lattice, kernel.azimuth)
    if rep is None:
        raise RoleError(
            "give the kernel azimuth an AzimuthFourier(m_max) representation"
        )
    if not kernel.traceable:
        raise RoleError("azimuth Fourier column jets need a traceable kernel")
    if kernel.azimuth in dict(averaged.fixed) or kernel.azimuth in dict(
        averaged.averaged
    ):
        raise RoleError(
            "the Fourier azimuth must be resolved, without an averaged or fixed value"
        )
    m_max = int(rep.m_max)
    request = stokes_blocks(kernel, rules, None)
    rules = dataclasses.replace(rules, stokes=request)
    columns = column_map(lattice, kernel, rules, max_columns=max_columns)
    counts = {"jet cells": lattice.n_cells}
    spectra, delta = _project(averaged, channels, lattice, columns, m_max, counts)
    routes = tuple(
        RouteRecord(c, "jet", mode="azimuth_fourier") for c in range(lattice.n_cells)
    )
    record = build_record(
        routes,
        columns.stokes,
        "quadrature",
        columns,
        lattice,
        events=counts,
        timings={"build": time.perf_counter() - start},
        defaults={
            "max_columns": int(max_columns),
            "cache": False,
            "azimuth_fourier_order": m_max,
            "azimuth_fourier_nodes": sample_count(m_max),
        },
        notes=(
            "The Fourier projection used ordinary column jets at fixed angles.",
            "The Fourier projection does not use the basis memo or S2 store.",
        ),
    )
    truncation = (
        ErrorTerm.not_applicable(
            "The natural-basis kernel and every averaged law are independent of azimuth.",
            name="azimuth_truncation",
        )
        if _azimuth_independent(averaged, kernel.azimuth)
        else ErrorTerm.unbounded(
            "Omitted azimuth orders need a specified population and a direct check.",
            name="azimuth_truncation",
        )
    )
    return FourierBasis(
        lattice=lattice,
        channels=channels,
        columns=columns,
        spectra0=spectra[0],
        spectra2=spectra[1] if len(spectra) > 1 else None,
        azimuth_factor=None,
        specified=specified_records(averaged),
        record=record,
        ledger=Ledger((truncation,)),
        key=None,
        averaged=averaged,
        rules=rules,
        m_max=m_max,
        refined_delta=delta,
    )
