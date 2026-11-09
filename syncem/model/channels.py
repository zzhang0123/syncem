"""Compactly supported channel responses ``R_j(nu)`` for the finite joint response.

LABEL ``eq: channel kernel``: the Dirac-line harmonic spectrum is evaluated
only after pairing with a smooth response of compact support away from zero
frequency. The ``bump`` family ``R = exp[1 - 1/(1 - t^2)]``, ``t = (nu - c)/w``,
``|t| < 1``, used in the acceptance benchmark, has the shape of main.tex
``eq: forward response families`` with ``k = 1``; the package parametrises it
by the support half-width ``w`` and scales it to unit peak by default
(``"unit_integral"`` gives the unit area of main.tex).

Units: frequencies in Hz. ``R`` is dimensionless for ``normalisation="unit_peak"``
(height-normalised, as in the acceptance benchmark; outputs are
band-integrated powers) and has units 1/Hz for ``"unit_integral"`` (``int R dnu = 1``). No
``2 pi`` enters anywhere: ``from_table(..., argument="omega")`` converts by
``R_nu(nu) = R_omega(2 pi nu)``.

Shapes: ``n_ch`` channels; ``__call__(nu)`` returns ``(n_ch, *nu.shape)``.
``nodes``/``weights`` are an ``n_nu``-point Gauss-Legendre rule on each support
(weights include the Jacobian ``(nu_hi - nu_lo)/2``) for the continuum route.
For ``bump``, the stored support is padded outward by one floating-point step
after checking that the band is resolvable. This contains the exact-binary
zeros ``c±w`` even when the rounded sum/subtraction falls inside them.

Smoothness classes (``smoothness``): 99 for the C-infinity families (``bump``,
``planck_taper``), 1 for ``raised_cosine`` (continuous first derivative, jump
in the second), 0 for ``tophat`` and the truncated ``gaussian`` (edge jump
``exp(-support_sigma^2/2)``, recorded by ``describe``) and the declared value
for ``from_table`` (linear interpolation is at most C^0). ``build_basis``
needs ``smoothness >= N + 1`` for a ``basis_remainder`` of kind ``bound``.

Composed responses (``_channel_compose``). ``from_basis(reference,
coefficients)`` is ``R_j = sum_k c_jk phi_k`` in the responses ``phi_k`` of
another ``Channels`` (``coefficients`` ``(n_ch, n_ref)``): its smoothness is
the reference's, its support per row the hull of the references with
``c_jk != 0``, and ``build_basis`` computes the reference columns once and
returns ``C_j = sum_k c_jk C_k`` (the kernel columns are linear in ``R_j``).
``from_callable(fn, support, smoothness, derivative_bound=None)`` takes a
JAX-differentiable ``fn(nu)`` on a declared support in ``(0, inf)``; the
derivatives ``R^(r)`` the kernels need come from JAX autodiff. A line-kernel
``basis_remainder`` on a callable response is a ``bound`` only when
``derivative_bound`` declares ``sup|R^(N+1)|`` (``derivative_bound_reason``);
otherwise it is reported as an ``estimate``. Both families use the response
as supplied: ``normalisation`` names its units (dimensionless for
``"unit_peak"``, 1/Hz for ``"unit_integral"``) and no rescaling is applied.
``bandpass.fit_bandpass`` writes a measured bandpass table in a reference
set and reports the representation residual as a response error.

Not certified: the physical adequacy of any response as a model of a real
bandpass; the quadrature accuracy of ``nodes``/``weights`` for ``planck_taper``
with a small ``taper`` (the rule converges slowly across the taper, increase
``n_nu``); interpolation error of ``from_table`` between its samples; the declared
smoothness and ``derivative_bound`` of ``from_callable`` (the eager checks on
sampled points can refute them, not verify them). The shape functions live in
``syncem.model._channel_shapes``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ._channel_compose import (
    basis_parts,
    callable_parts,
    callable_response,
    check_callable,
    derivative_bounds,
    describe_fn,
)
from ._channel_shapes import (
    COMPOSED,
    FAMILIES,
    NORMALISATIONS,
    SMOOTH,
    legendre_rule,
    table_edge_jump,
    table_support,
    unit_peak_area,
    unit_peak_shape,
)


def _as_positive_1d(value, name):
    value = jnp.asarray(value)
    if jnp.iscomplexobj(value) or value.ndim != 1 or value.size == 0:
        raise ValueError(f"{name} must be a nonempty real 1D array")
    value = value.astype(jnp.result_type(value, 1.0))
    return eqx.error_if(
        value,
        jnp.any(~jnp.isfinite(value)) | jnp.any(value <= 0),
        f"{name} must be finite and positive",
    )


def _static_int(value, name, minimum):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


class Channels(eqx.Module):
    """Frequency responses of ``n_ch`` channels; see the module docstring.

    ``widths_hz`` is the half-width of the support for ``bump``, ``planck_taper``,
    ``raised_cosine`` and ``tophat``, the standard deviation for ``gaussian`` and
    the support half-width for ``table``. ``support`` is ``(n_ch, 2)`` with
    ``0 < nu_lo < nu_hi``. Arrays are traced leaves; ``family``, ``normalisation``,
    ``smoothness``, ``taper``, ``n_nu``, ``table`` and ``support_sigma`` are static.
    ``from_basis`` adds the leaves ``reference_channels`` (a ``Channels``) and
    ``coefficients`` ``(n_ch, n_ref)``; ``from_callable`` adds the static
    ``response_fn`` (one callable per channel), ``derivative_bound`` and
    ``edge`` (the sampled relative edge value).

    Units: all frequencies in Hz; ``__call__`` is dimensionless for
    ``unit_peak`` and per Hz for ``unit_integral``. Assumes the response is
    exactly the declared function of frequency (an instrument's measured
    response enters through ``from_table`` with its declared smoothness).
    Not certified: the fidelity of that response to the instrument, and the
    Gauss-Legendre ``nodes``/``weights`` accuracy for the continuum route
    (``n_nu`` is a declared count; ``planck_taper`` converges slowly).
    """

    family: str = eqx.field(static=True)
    centres_hz: jax.Array
    widths_hz: jax.Array
    support: jax.Array
    nodes: jax.Array
    weights: jax.Array
    normalisation: str = eqx.field(static=True)
    smoothness: int = eqx.field(static=True)
    taper: float = eqx.field(static=True, default=0.5)
    n_nu: int = eqx.field(static=True, default=64)
    table: tuple | None = eqx.field(static=True, default=None)
    support_sigma: float = eqx.field(static=True, default=6.0)
    reference_channels: "Channels | None" = None
    coefficients: jax.Array | None = None
    response_fn: tuple | None = eqx.field(static=True, default=None)
    derivative_bound: tuple | None = eqx.field(static=True, default=None)
    edge: float = eqx.field(static=True, default=0.0)
    LABEL: ClassVar[str] = "eq: channel kernel"

    # ------------------------------------------------------------ constructors
    @classmethod
    def _build(cls, family, centres, widths, half_widths, *, bounds=None, **static):
        normalisation, n_nu = static["normalisation"], static["n_nu"]
        if family not in FAMILIES:
            raise ValueError(f"unknown channel family {family!r}")
        if normalisation not in NORMALISATIONS:
            raise ValueError(f"normalisation must be one of {NORMALISATIONS}")
        _static_int(n_nu, "n_nu", 1)
        _static_int(static["smoothness"], "smoothness", 0)
        centres = _as_positive_1d(centres, "centres_hz")
        widths = _as_positive_1d(widths, "widths_hz")
        if widths.shape != centres.shape:
            raise ValueError("centres_hz and widths_hz must have the same shape")
        if bounds is None:
            lo, hi = centres - half_widths, centres + half_widths
        else:  # composed families store their support exactly as computed
            lo, hi = (jnp.asarray(edge, dtype=centres.dtype) for edge in bounds)
        raw_invalid = (
            jnp.any(~jnp.isfinite(lo))
            | jnp.any(~jnp.isfinite(hi))
            | jnp.any(lo <= 0)
            | jnp.any(hi <= lo)
        )
        lo = eqx.error_if(
            lo,
            raw_invalid,
            "channel support must be finite with 0 < lo < hi at working precision",
        )
        if family == "bump":
            # Exact binary c±w can lie beyond a rounded floating-point sum.
            # Include both flat zeros so the stored support never clips a
            # mathematically smooth bump into a moving-edge jump. The small
            # support adjustment has zero tangent: response derivatives are
            # governed by the smooth bump profile, not by float ULP steps.
            lo_fixed = jax.lax.stop_gradient(lo)
            hi_fixed = jax.lax.stop_gradient(hi)
            lo += (
                jnp.nextafter(lo_fixed, jnp.asarray(-jnp.inf, dtype=lo.dtype))
                - lo_fixed
            )
            hi += (
                jnp.nextafter(hi_fixed, jnp.asarray(jnp.inf, dtype=hi.dtype)) - hi_fixed
            )
        lo = eqx.error_if(
            lo,
            jnp.any(~jnp.isfinite(lo))
            | jnp.any(~jnp.isfinite(hi))
            | jnp.any(lo <= 0)
            | jnp.any(hi <= lo),
            "channel support must be finite with 0 < lo < hi at working precision",
        )
        x, w = legendre_rule(n_nu)
        half = (hi - lo) / 2
        mid = lo + half
        return cls(
            family=family,
            centres_hz=centres,
            widths_hz=widths,
            support=jnp.stack([lo, hi], axis=-1),
            nodes=mid[:, None] + half[:, None] * jnp.asarray(x)[None, :],
            weights=half[:, None] * jnp.asarray(w)[None, :],
            normalisation=normalisation,
            smoothness=static["smoothness"],
            taper=float(static.get("taper", 0.5)),
            n_nu=n_nu,
            table=static.get("table"),
            support_sigma=float(static.get("support_sigma", 6.0)),
            reference_channels=static.get("reference_channels"),
            coefficients=static.get("coefficients"),
            response_fn=static.get("response_fn"),
            derivative_bound=static.get("derivative_bound"),
            edge=float(static.get("edge", 0.0)),
        )

    @classmethod
    def bump(
        cls, centres_hz, widths_hz, *, normalisation="unit_peak", n_nu=64
    ) -> "Channels":
        """``R = exp(1 - 1/(1 - t^2))``, ``t = (nu - c)/w``; C-infinity.

        The stored support includes one outward ULP on each side of ``c±w``
        so exact-binary input rounding cannot clip the flat zeros.
        """
        w = jnp.asarray(widths_hz)
        return cls._build(
            "bump",
            centres_hz,
            w,
            w,
            normalisation=normalisation,
            n_nu=n_nu,
            smoothness=SMOOTH,
        )

    @classmethod
    def planck_taper(
        cls, centres_hz, widths_hz, *, taper=0.5, normalisation="unit_peak", n_nu=64
    ) -> "Channels":
        """Planck-taper window on ``[c - w, c + w]``: ``u = (nu - c + w)/(2w)``,
        ``v = min(u, 1 - u)``; ``R = 1`` for ``v >= taper`` and
        ``R = 1/(1 + exp[taper/v - taper/(taper - v)])`` for ``0 < v < taper``.
        ``taper`` in ``(0, 0.5]``; ``0.5`` has no flat top. C-infinity."""
        if not (0.0 < float(taper) <= 0.5):
            raise ValueError("taper must lie in (0, 0.5]")
        w = jnp.asarray(widths_hz)
        return cls._build(
            "planck_taper",
            centres_hz,
            w,
            w,
            normalisation=normalisation,
            n_nu=n_nu,
            smoothness=SMOOTH,
            taper=taper,
        )

    @classmethod
    def raised_cosine(
        cls, centres_hz, widths_hz, *, normalisation="unit_peak", n_nu=64
    ) -> "Channels":
        """``R = [1 + cos(pi t)]/2`` for ``|t| < 1``; C^1 (second derivative jumps)."""
        w = jnp.asarray(widths_hz)
        return cls._build(
            "raised_cosine",
            centres_hz,
            w,
            w,
            normalisation=normalisation,
            n_nu=n_nu,
            smoothness=1,
        )

    @classmethod
    def tophat(
        cls, centres_hz, widths_hz, *, normalisation="unit_peak", n_nu=64
    ) -> "Channels":
        """``R = 1`` for ``|t| < 1``, else 0; discontinuous (smoothness 0)."""
        w = jnp.asarray(widths_hz)
        return cls._build(
            "tophat",
            centres_hz,
            w,
            w,
            normalisation=normalisation,
            n_nu=n_nu,
            smoothness=0,
        )

    @classmethod
    def gaussian(
        cls,
        centres_hz,
        sigma_hz,
        *,
        support_sigma=6.0,
        normalisation="unit_peak",
        n_nu=64,
    ) -> "Channels":
        """``R = exp(-t^2/2)``, ``t = (nu - c)/sigma``, truncated at ``|t| < support_sigma``.
        The edge jump ``exp(-support_sigma^2/2)`` makes this smoothness 0; it is
        recorded in ``describe()`` as ``edge_jump``. ``widths_hz`` stores sigma."""
        if not float(support_sigma) > 0:
            raise ValueError("support_sigma must be positive")
        sigma = jnp.asarray(sigma_hz)
        return cls._build(
            "gaussian",
            centres_hz,
            sigma,
            float(support_sigma) * sigma,
            normalisation=normalisation,
            n_nu=n_nu,
            smoothness=0,
            support_sigma=support_sigma,
        )

    @classmethod
    def from_table(
        cls,
        nu_hz,
        response,
        *,
        smoothness: int,
        argument="nu",
        normalisation="unit_peak",
        n_nu=64,
    ) -> "Channels":
        """Linear interpolation of a sampled response on one increasing grid.

        ``nu_hz`` is ``(n_pts,)``; ``response`` is ``(n_pts,)`` for one channel or
        ``(n_ch, n_pts)``. ``argument="omega"`` means the grid is angular frequency
        and the table is ``R_omega``: ``R_nu(nu) = R_omega(2 pi nu)``, no ``2 pi``
        factor on the values. Each channel's support is the grid interval that
        contains its nonzero samples extended to the adjacent zero samples (the
        interpolant is nonzero there); tables are rescaled to unit peak. The
        grid and rescaled table are concrete NumPy inputs stored as static
        tuples. ``smoothness`` is declared by the caller and not verified.
        """
        if argument not in ("nu", "omega"):
            raise ValueError("argument must be 'nu' or 'omega'")
        grid = np.asarray(nu_hz, dtype=float)
        if argument == "omega":
            grid = grid / (2 * np.pi)
        table = np.atleast_2d(np.asarray(response, dtype=float))
        if (
            grid.ndim != 1
            or grid.size < 2
            or table.ndim != 2
            or table.shape[1] != grid.size
        ):
            raise ValueError(
                "nu_hz must be (n_pts,) and response (n_pts,) or (n_ch, n_pts)"
            )
        finite = np.all(np.isfinite(grid)) and np.all(np.isfinite(table))
        if not (finite and np.all(np.diff(grid) > 0)):
            raise ValueError(
                "table grid must be finite and strictly increasing; response finite"
            )
        if np.any(table < 0) or not np.all(np.any(table != 0, axis=1)):
            raise ValueError(
                "each table response must be nonnegative with a nonzero sample"
            )
        table = table / table.max(axis=1, keepdims=True)
        lo, hi = table_support(grid, table)
        if np.any(lo <= 0):
            raise ValueError("table support must lie at positive frequency")
        static = (tuple(grid.tolist()), tuple(tuple(row.tolist()) for row in table))
        return cls._build(
            "table",
            (lo + hi) / 2,
            (hi - lo) / 2,
            (hi - lo) / 2,
            normalisation=normalisation,
            n_nu=n_nu,
            smoothness=smoothness,
            table=static,
        )

    @classmethod
    def from_basis(cls, reference, coefficients, *, n_nu=None) -> "Channels":
        """``R_j = sum_k c_jk phi_k`` in the reference responses ``phi_k``.

        ``reference`` is a ``Channels`` (a reference that is itself a basis
        is flattened); ``coefficients`` is ``(n_ch, n_ref)`` or ``(n_ref,)``,
        real, finite and concrete, each row with a nonzero entry. The support
        of row ``j`` is the hull of the supports of the references with
        ``c_jk != 0``, fixed at construction; smoothness and normalisation
        (the units) are the reference's, and the response is not rescaled.
        ``n_nu`` (default: the reference's) sets the nodes on that support,
        used by direct routes. ``build_basis`` builds the reference columns
        once and returns their combination (``basis.combine_basis``). The
        coefficients are array leaves, but the support, ``nodes`` and
        ``weights`` follow their nonzero pattern at construction: a
        replaced coefficient array (for example through ``eqx.tree_at``)
        must keep that pattern, or the direct routes integrate the wrong
        interval; build a new ``from_basis`` instead. ``describe()`` and
        ``edge_jump()`` need concrete coefficients. Not certified:
        how well the combination represents an instrument
        (``bandpass.fit_bandpass`` reports that residual).
        """
        if not isinstance(reference, Channels):
            raise ValueError("reference must be a Channels")
        reference, c, lo, hi = basis_parts(reference, coefficients)
        return cls._build(
            "basis",
            (lo + hi) / 2,
            (hi - lo) / 2,
            (hi - lo) / 2,
            bounds=(lo, hi),
            normalisation=reference.normalisation,
            n_nu=reference.n_nu if n_nu is None else n_nu,
            smoothness=reference.smoothness,
            reference_channels=reference,
            coefficients=jnp.asarray(c),
        )

    @classmethod
    def from_callable(
        cls,
        fn,
        support,
        smoothness: int,
        derivative_bound=None,
        *,
        normalisation="unit_peak",
        n_nu=64,
    ) -> "Channels":
        """A user response ``fn(nu)`` on a declared support (module docstring).

        ``fn`` maps an array of frequencies [Hz] to the response of the same
        shape, elementwise, and must be JAX-traceable and differentiable to
        the orders the kernel uses; a sequence gives one callable per
        channel. ``support`` is ``(2,)`` or ``(n_ch, 2)`` with
        ``0 < nu_lo < nu_hi`` finite (``ValueError`` otherwise), stored
        exactly; ``__call__`` is ``fn`` inside the open support and exactly
        zero outside it. ``smoothness`` is the declared class.
        ``derivative_bound`` is ``None`` or ``{order: sup|R^(order)|}`` with
        scalar or ``(n_ch,)`` values (Hz^-order times the response units); a
        ``basis_remainder`` of kind ``bound`` on a line kernel needs the
        order ``N + 1`` (``derivative_bound_reason``). The response is used
        as supplied; ``normalisation`` names its units. Eager checks that
        can refute the declarations (``_channel_compose``): finite real
        values of the input's shape, a JAX trace of ``fn'``, vanishing
        ``R^(k)`` one floating-point step inside each edge for
        ``k = 0 .. min(smoothness, 4)`` when ``smoothness >= 1``, and every
        declared bound at least the sampled ``|R^(order)|``. Passing them
        does not verify the smoothness or the bounds. ``derivative_bound``
        decides the kind only; its value enters no computed envelope, so
        supplied ``RemainderInputs`` must be derived from it. ``response_fn``
        is a static field: parameters closed over by ``fn`` are constants,
        so a basis cannot be differentiated in them, and every new closure is
        a new static value that ``jax.jit`` retraces and recompiles. The
        constructor is eager: ``fn`` must not close over traced values.
        ``describe()`` names each callable by its ``__qualname__``, which
        does not identify the function.
        """
        if normalisation not in NORMALISATIONS:
            raise ValueError(f"normalisation must be one of {NORMALISATIONS}")
        _static_int(smoothness, "smoothness", 0)
        _static_int(n_nu, "n_nu", 1)
        fns, lo, hi = callable_parts(fn, support)
        bounds = derivative_bounds(derivative_bound, lo.size)
        edge = check_callable(fns, lo, hi, smoothness, bounds, n_nu)
        return cls._build(
            "callable",
            (lo + hi) / 2,
            (hi - lo) / 2,
            (hi - lo) / 2,
            bounds=(lo, hi),
            normalisation=normalisation,
            n_nu=n_nu,
            smoothness=smoothness,
            response_fn=fns,
            derivative_bound=bounds,
            edge=edge,
        )

    # --------------------------------------------------------------- queries
    @property
    def n_ch(self) -> int:
        return self.centres_hz.shape[0]

    def __call__(self, nu_hz) -> jax.Array:
        """Response ``(n_ch, *nu.shape)``: exactly zero outside ``(nu_lo, nu_hi)``.

        Safe under ``jit``, ``jacfwd``/``grad`` and nested derivatives at and
        beyond the support edges (no ``0 * inf``); ``nu_hz`` must be real.
        """
        nu = jnp.asarray(nu_hz)
        if jnp.iscomplexobj(nu):
            raise ValueError("frequencies must be real")
        nu = nu.astype(jnp.result_type(nu, 1.0))
        if self.family == "basis":  # zero outside the hull by construction
            values = self.reference_channels(nu)  # (n_ref, *nu.shape)
            return jnp.tensordot(self.coefficients, values, axes=1)
        nu = jnp.broadcast_to(nu, (self.n_ch, *nu.shape))
        expand = (self.n_ch,) + (1,) * (nu.ndim - 1)
        lo, hi = self.support[:, 0].reshape(expand), self.support[:, 1].reshape(expand)
        inside = (nu > lo) & (nu < hi)
        if self.family == "callable":
            return callable_response(self.response_fn, nu, inside, self.centres_hz)
        shape = unit_peak_shape(
            self.family,
            nu,
            inside,
            centres=self.centres_hz,
            widths=self.widths_hz,
            support=self.support,
            taper=self.taper,
            table=self.table,
        )
        return jnp.where(inside, shape, 0.0) * self.peak_scale().reshape(expand)

    def node_values(self) -> jax.Array:
        """``R_j`` at that channel's own quadrature nodes, ``(n_ch, n_nu)``."""
        return jax.vmap(lambda j, x: self(x)[j])(jnp.arange(self.n_ch), self.nodes)

    def peak_scale(self) -> jax.Array:
        """``(n_ch,)`` factor from the unit-peak shape to the declared normalisation.

        ``unit_integral`` uses closed-form band areas (``bump``: a cached
        512-node constant; ``planck_taper``: ``2 w (1 - taper)``;
        ``raised_cosine``: ``w``; ``tophat``: ``2 w``; ``gaussian``: the
        truncated erf area; ``table``: the trapezoid rule, exact for the
        interpolant), differentiable in ``widths_hz``. ``basis`` and
        ``callable`` responses are used as supplied (ones).
        """
        if self.normalisation == "unit_peak" or self.family in COMPOSED:
            return jnp.ones_like(self.centres_hz)
        if self.family == "table":
            assert self.table is not None
            grid, rows = (np.asarray(part) for part in self.table)
            return 1.0 / jnp.asarray(np.trapezoid(rows, grid, axis=1))
        return 1.0 / (
            self.widths_hz * unit_peak_area(self.family, self.taper, self.support_sigma)
        )

    def width_ratio(self) -> np.ndarray:
        """Concrete ``(nu_hi - nu_lo)/centre`` per channel (eager use only)."""
        support, centres = np.asarray(self.support), np.asarray(self.centres_hz)
        return (support[:, 1] - support[:, 0]) / centres

    def edge_jump(self) -> float:
        """Largest unit-peak response value at a support edge (0 when continuous).

        ``basis``: the reference's value times the largest row sum of
        ``|c_jk|`` (a bound on any jump inside the hull); ``callable``: the
        sampled edge value relative to the sampled peak.
        """
        if self.family == "basis":
            rows = np.sum(np.abs(np.asarray(self.coefficients)), axis=1)
            return self.reference_channels.edge_jump() * float(np.max(rows))
        if self.family == "callable":
            return self.edge
        if self.family == "tophat":
            return 1.0
        if self.family == "gaussian":
            return math.exp(-0.5 * self.support_sigma**2)
        if self.family == "table":
            assert self.table is not None
            grid, rows = (np.asarray(part) for part in self.table)
            return table_edge_jump(grid, rows, np.asarray(self.support))
        return 0.0

    def describe(self) -> tuple:
        """Hashable provenance tuple of ``(key, value)`` pairs (eager use only)."""
        support = np.asarray(self.support)
        items = [
            ("family", self.family),
            ("normalisation", self.normalisation),
            ("smoothness", self.smoothness),
            ("n_ch", self.n_ch),
            ("n_nu", self.n_nu),
            ("centres_hz", tuple(float(v) for v in np.asarray(self.centres_hz))),
            ("widths_hz", tuple(float(v) for v in np.asarray(self.widths_hz))),
            ("support_hz", tuple((float(lo), float(hi)) for lo, hi in support)),
            ("width_ratio", tuple(float(v) for v in self.width_ratio())),
            ("edge_jump", self.edge_jump()),
        ]
        if self.family == "planck_taper":
            items.append(("taper", self.taper))
        if self.family == "gaussian":
            items.append(("support_sigma", self.support_sigma))
        if self.family == "table":
            assert self.table is not None
            items.append(("n_table", len(self.table[0])))
        if self.family == "basis":
            coefficients = np.asarray(self.coefficients)
            items += [
                ("n_reference", self.reference_channels.n_ch),
                ("reference", self.reference_channels.describe()),
                (
                    "coefficients",
                    tuple(tuple(float(v) for v in row) for row in coefficients),
                ),
            ]
        if self.family == "callable":
            items += [
                ("response_fn", tuple(describe_fn(f) for f in self.response_fn)),
                ("derivative_bound", self.derivative_bound),
            ]
        return tuple(items)

    def derivative_bound_reason(self, order: int) -> str | None:
        """Why ``sup|R^(order)|`` is not declared, or ``None`` (eager, static).

        ``None`` for the built-in families, whose formulas fix the
        derivatives, and for a ``callable`` whose ``derivative_bound`` has
        ``order``; a ``basis`` response asks its reference. A line-kernel
        ``basis_remainder`` with a reason is an ``estimate``, never a
        ``bound``. A declared value is a caller statement, not verified here.
        """
        if self.family == "basis":
            return self.reference_channels.derivative_bound_reason(order)
        if self.family != "callable":
            return None
        if order in dict(self.derivative_bound or ()):
            return None
        return (
            f"callable channel response: sup|R^({order})| not declared "
            f"(derivative_bound={{{order}: ...}}), so the remainder is an "
            "estimate, not a bound"
        )

    def select(self, j: int) -> "Channels":
        """Channel ``j`` alone (a static int), same family and settings."""
        if not isinstance(j, int) or isinstance(j, bool) or not 0 <= j < self.n_ch:
            raise ValueError(f"channel index must be an int in [0, {self.n_ch})")
        rows = slice(j, j + 1)
        table = None if self.table is None else (self.table[0], (self.table[1][j],))
        bounds = self.derivative_bound
        return dataclasses.replace(
            self,
            centres_hz=self.centres_hz[rows],
            widths_hz=self.widths_hz[rows],
            support=self.support[rows],
            nodes=self.nodes[rows],
            weights=self.weights[rows],
            table=table,
            coefficients=None if self.coefficients is None else self.coefficients[rows],
            response_fn=None if self.response_fn is None else (self.response_fn[j],),
            derivative_bound=(
                None
                if bounds is None
                else tuple((order, (values[j],)) for order, values in bounds)
            ),
        )


__all__ = ["Channels", "FAMILIES", "NORMALISATIONS"]
