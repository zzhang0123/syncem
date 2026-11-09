"""Measured bandpasses written in smooth reference responses (``syncem.model.bandpass``).

LABEL: ``eq: chromatic beam response`` (the response error ``Delta W_aj``:
a datum changes by at most the integral of ``|Delta W_aj|`` against the
total variation of the observed Stokes spectrum) and ``eq: channel kernel``
(the smooth responses the basis needs).

:func:`fit_bandpass` fits ``R_j ~ sum_k c_jk phi_k`` to a measured table by
least squares on the table samples, with ``phi_k`` the responses of a
reference ``Channels`` (for example overlapping ``bump`` channels on a finer
grid than the instrument's). The measured response is the linear
interpolant of the table (zero outside its grid), as in
``Channels.from_table``. The residual ``Delta R_j = R_j - sum_k c_jk phi_k``
is evaluated on a check grid: the table grid with ``oversample`` points per
interval, plus ``4 n_nu`` Gauss-Legendre nodes on every reference support.

Response-error term. With nonnegative references and ``Phi = sum_k phi_k``,
``d_jk = max |Delta R_j| / Phi`` over the check points where ``phi_k > 0``
gives ``|Delta R_j| <= sum_k d_jk phi_k`` at every check point where
``Phi > 0``. A Stokes datum of channel ``j`` is ``X_j = sum_m X_m R_j(nu_m)``
over the lines ``m``, and the physical emission of every line has
``|Q_m|, |U_m|, |V_m| <= I_m`` (polarisation degree at most one, kept by
Faraday rotation). The residual therefore changes ``X_j`` by at most
``sum_m I_m |Delta R_j(nu_m)| <= sum_k d_jk I_k``, with ``I_k`` the
reference-channel intensities, when every ``nu_m`` is a check point.
:meth:`BandpassFit.response_uncertainty` returns these ``d_jk`` as a
``(4 n_ch, 4 n_ref)`` ``ErrorTerm`` for
``Prediction.propagate(fit.operator(), response_uncertainty=...)`` on a
prediction in the reference channels, where ``operator()`` is
``c (x) 1_4``, ``(4 n_ch, 4 n_ref)``. ``propagate`` multiplies the term by
``|stokes| + envelope``, which covers the error of the predicted ``I_k``.

Units: frequencies in Hz, and the table in the units of the reference responses
(dimensionless for ``unit_peak`` references), so the coefficients are
dimensionless. Shapes: ``nu_hz`` ``(n_pts,)``, ``response`` ``(n_pts,)`` or
``(n_ch, n_pts)``.

Kind: the covering is an ``estimate``. Its maximum is taken on finitely many
check points, so it is not a supremum, and the measured response between
table samples is the interpolant, not the instrument. A residual where no
reference reaches (``uncovered > 0``), or a table row that does not start
and end at zero (the response beyond the grid is then not given), makes the
term ``unbounded``. A large ``d_jk`` on an outer reference means that the
residual near the edge of the reference set is covered only by that
reference's tail. The covering is then valid but loose, and a wider
reference set tightens it. Not certified: the measured table itself, the
interpolation model, and the adequacy of the reference set. The beam,
Stokes mixing and calibration parts of ``W_aj`` are outside this helper.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ._channel_shapes import legendre_rule
from .channels import Channels
from .errors import ErrorTerm

MANUSCRIPT_TERM = "Delta W_aj of eq: chromatic beam response"
COEFFICIENT_FLOOR = 1e-13  # relative to the row's largest coefficient


class BandpassFit(eqx.Module):
    """A measured bandpass written in reference responses (see the module docstring).

    ``channels`` is ``Channels.from_basis(reference, coefficients)``, with
    ``coefficients`` ``(n_ch, n_ref)``. ``residual`` ``(n_ch, n_grid)`` is
    ``Delta R`` on ``check_grid`` ``(n_grid,)`` [Hz]. ``covering``
    ``(n_ch, n_ref)`` holds the ``d_jk``. ``uncovered`` ``(n_ch,)`` is the
    largest ``|Delta R_j|`` where no reference reaches, and
    ``relative_residual`` ``(n_ch,)`` the largest ``|Delta R_j|`` over the
    largest measured ``|R_j|``. ``open_edges`` lists the channels whose
    table does not start and end at zero. Units: the response units
    (``residual``, ``covering``, ``uncovered``) and dimensionless
    coefficients. Not certified: the covering between check points and the
    measured response between table samples (an ``estimate``).
    """

    reference: Channels
    channels: Channels
    coefficients: jax.Array
    check_grid: jax.Array
    residual: jax.Array
    covering: jax.Array
    uncovered: jax.Array
    relative_residual: jax.Array
    n_table: int = eqx.field(static=True)
    open_edges: tuple[int, ...] = eqx.field(static=True, default=())
    LABEL: ClassVar[str] = "eq: chromatic beam response"

    @property
    def n_ch(self) -> int:
        return self.coefficients.shape[0]

    def operator(self) -> jax.Array:
        """``c (x) 1_4``, ``(4 n_ch, 4 n_ref)``: reference to instrument channel Stokes."""
        return jnp.kron(self.coefficients, jnp.eye(4))

    def response_uncertainty(self) -> ErrorTerm:
        """``(4 n_ch, 4 n_ref)`` covering ``d_jk`` on the ``I`` column of reference ``k``.

        Row ``4 j + X`` (``X`` in ``I, Q, U, V``) has ``d_jk`` in column
        ``4 k``, so ``|dR| @ |stokes_ref|`` is ``sum_k d_jk |I_k|`` for every
        Stokes datum of channel ``j``. Kind ``estimate``, or ``unbounded`` when
        some residual is not covered by any reference (module docstring).
        Not certified: the covering between check points.
        """
        if self.open_edges:
            return ErrorTerm.unbounded(
                f"bandpass tables of channels {list(self.open_edges)} do not start "
                "and end at zero: the response beyond the table grid is not given",
                MANUSCRIPT_TERM,
            )
        uncovered = np.asarray(self.uncovered)
        if np.any(uncovered > 0):
            channels = [int(j) for j in np.flatnonzero(uncovered > 0)]
            return ErrorTerm.unbounded(
                f"bandpass residual not covered by any reference response in "
                f"channels {channels} (max |Delta R| = {float(uncovered.max()):.3g}); "
                "widen the reference set",
                MANUSCRIPT_TERM,
            )
        on_I = jnp.zeros((4, 4)).at[:, 0].set(1.0)
        return ErrorTerm(
            jnp.kron(self.covering, on_I),  # (4 n_ch, 4 n_ref)
            "estimate",
            "|Delta R_j| <= sum_k d_jk phi_k on the check grid "
            f"({self.check_grid.shape[0]} points; d_jk = max |Delta R_j| / sum_k "
            "phi_k on the support of phi_k); |Q_m|, |U_m|, |V_m| <= I_m bound "
            "every Stokes datum of channel j by sum_k d_jk I_k; estimate: a "
            "maximum on finitely many points, and the measured response is "
            "linear between table samples",
            MANUSCRIPT_TERM,
        )

    def describe(self) -> tuple:
        """Hashable ``(key, value)`` record (eager use only)."""
        return (
            ("label", self.LABEL),
            ("n_ch", self.n_ch),
            ("n_reference", int(self.coefficients.shape[1])),
            ("n_table", self.n_table),
            ("n_check", int(self.check_grid.shape[0])),
            ("reference", self.reference.describe()),
            (
                "relative_residual",
                tuple(float(v) for v in np.asarray(self.relative_residual)),
            ),
            ("uncovered", tuple(float(v) for v in np.asarray(self.uncovered))),
            ("open_edges", self.open_edges),
        )


def _table(nu_hz, response, argument):
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
    if not (np.all(np.isfinite(grid)) and np.all(np.isfinite(table))):
        raise ValueError("table grid and response must be finite")
    if not (np.all(np.diff(grid) > 0) and grid[0] > 0):
        raise ValueError("table grid must be strictly increasing and positive")
    if not np.all(np.any(table != 0, axis=1)):
        raise ValueError("each table response needs a nonzero sample")
    return grid, table


def _check_grid(grid, reference, oversample):
    t = np.arange(oversample) / oversample
    fine = np.append((grid[:-1, None] + np.diff(grid)[:, None] * t).ravel(), grid[-1])
    x, _ = legendre_rule(4 * reference.n_nu)
    support = np.asarray(reference.support, dtype=float)
    mid, half = support.mean(axis=1), (support[:, 1] - support[:, 0]) / 2
    nodes = mid[:, None] + half[:, None] * np.asarray(x)[None, :]
    return np.unique(np.concatenate([fine, nodes.ravel()]))


def _nonnegative(phi, where):
    if np.any(phi < 0):
        raise ValueError(
            f"reference responses must be nonnegative for the total-variation "
            f"covering (negative values on the {where})"
        )


def fit_bandpass(reference, nu_hz, response, *, argument="nu", oversample=8):
    """Least-squares ``c_jk`` of a measured bandpass table (see the module docstring).

    ``reference``: a ``Channels`` with nonnegative responses (``ValueError``
    otherwise). ``nu_hz`` ``(n_pts,)`` strictly increasing and positive
    (``argument="omega"`` for angular frequency). ``response`` ``(n_pts,)``
    or ``(n_ch, n_pts)``, finite, each row with a nonzero sample, in the
    units of the reference responses (used as supplied, not rescaled).
    ``oversample`` (int >= 1) check points per table interval. References
    that vanish at every table sample get ``c_jk = 0``, as do coefficients
    below ``1e-13`` of the row's largest (the residual is computed after
    that, so it includes them). Returns a
    :class:`BandpassFit`. Eager (NumPy). Not certified: the measured table
    and the covering between check points (an ``estimate``).
    """
    if not isinstance(reference, Channels):
        raise ValueError("reference must be a Channels")
    if (
        not isinstance(oversample, int)
        or isinstance(oversample, bool)
        or oversample < 1
    ):
        raise ValueError("oversample must be an integer >= 1")
    grid, table = _table(nu_hz, response, argument)
    phi_table = np.asarray(reference(grid), dtype=float)  # (n_ref, n_pts)
    _nonnegative(phi_table, "table grid")
    active = np.any(phi_table != 0, axis=1)
    if not np.any(active):
        raise ValueError("no reference response is nonzero at a table sample")
    solution, *_ = np.linalg.lstsq(phi_table[active].T, table.T, rcond=None)
    coefficients = np.zeros((table.shape[0], reference.n_ch))
    coefficients[:, active] = solution.T
    row_max = np.max(np.abs(coefficients), axis=1, keepdims=True)
    coefficients[np.abs(coefficients) < COEFFICIENT_FLOOR * row_max] = 0.0
    if not np.all(np.any(coefficients != 0, axis=1)):
        raise ValueError("a table row has no least-squares component on the references")
    check = _check_grid(grid, reference, oversample)
    phi = np.asarray(reference(check), dtype=float)  # (n_ref, n_grid)
    _nonnegative(phi, "check grid")
    measured = np.stack(
        [np.interp(check, grid, row, left=0.0, right=0.0) for row in table]
    )
    residual = measured - coefficients @ phi
    total = phi.sum(axis=0)
    covered = total > 0
    ratio = np.abs(residual) / np.where(covered, total, 1.0) * covered
    covering = np.max(np.where(phi[None] > 0, ratio[:, None, :], 0.0), axis=-1)
    uncovered = np.max(np.where(covered, 0.0, np.abs(residual)), axis=-1)
    peak = np.max(np.abs(measured), axis=-1)
    return BandpassFit(
        reference=reference,
        channels=Channels.from_basis(reference, coefficients),
        coefficients=jnp.asarray(coefficients),
        check_grid=jnp.asarray(check),
        residual=jnp.asarray(residual),
        covering=jnp.asarray(covering),
        uncovered=jnp.asarray(uncovered),
        relative_residual=jnp.asarray(np.max(np.abs(residual), axis=-1) / peak),
        n_table=int(grid.size),
        open_edges=tuple(
            int(j) for j in np.flatnonzero((table[:, 0] != 0) | (table[:, -1] != 0))
        ),
    )


__all__ = ["BandpassFit", "fit_bandpass", "MANUSCRIPT_TERM"]
