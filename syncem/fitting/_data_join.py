"""Join observations on the same pixels while keeping each bandpass (SPEC Sec. 3.7).

LABEL sec: linear fit. The joined noise model assumes independence between
instruments. Within-instrument covariance and the original row quadratures
are retained. The response is block diagonal after selecting the instrument's
source Stokes components.

Not certified. Noise independence is an input convention of this operation.
"""

from __future__ import annotations

from itertools import groupby

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..base.channels import Channels
from ..model.errors import WEAKEST_ORDER
from ..records.ledger import ErrorTerm
from ..values.errors import DataError
from ._response_bias import _has_discrepancy, _model_discrepancy
from .data import _AXES, _checked, _rows


class _ChannelUnion(Channels):
    """An ordered union of rows that preserves the channel-weight measure."""

    members: tuple[Channels, ...] = eqx.field(kw_only=True)

    def __getitem__(self, rows):
        selected = _rows(rows, self.n_ch)
        offsets = np.cumsum([0, *(p.n_ch for p in self.members)])
        owners = np.searchsorted(offsets[1:], selected, side="right")
        parts = [
            self.members[owner][[i - offsets[owner] for _, i in entries]]
            for owner, entries in groupby(
                zip(owners, selected), key=lambda pair: pair[0]
            )
        ]
        return parts[0] if len(parts) == 1 else _join_channels(parts)

    def select(self, j):
        """Return one channel without its padding."""
        return self[j]

    def _double_nodes(self):
        return _join_channels(part._double_nodes() for part in self.members)

    def __call__(self, nu_hz):
        return jnp.concatenate(tuple(p(nu_hz) for p in self.members), axis=0)

    def node_values(self):
        """Return each response at its own nodes with zero in the padding."""
        return jnp.concatenate(
            tuple(
                jnp.pad(p.node_values(), ((0, 0), (0, self.n_nu - p.n_nu)))
                for p in self.members
            )
        )

    def row_keys(self):
        """Use the per-row keys of the underlying channel quadratures."""
        return tuple(key for p in self.members for key in p.row_keys())

    def peak_scale(self):
        """Return the original normalisation factors."""
        return jnp.concatenate(tuple(p.peak_scale() for p in self.members))

    def edge_jump(self):
        """Return the largest original edge jump."""
        return max(p.edge_jump() for p in self.members)

    def derivative_bound_reason(self, order):
        """Report an absent derivative bound across any member."""
        return next(
            (why for p in self.members if (why := p.derivative_bound_reason(order))),
            None,
        )


def _join_channels(groups):
    parts = tuple(
        member
        for part in groups
        for member in (part.members if isinstance(part, _ChannelUnion) else (part,))
    )
    normalisations = {p.normalisation for p in parts}
    if len(normalisations) != 1:
        raise DataError("joined channels must use the same response normalisation.")
    width = max(p.n_nu for p in parts)

    def array(name):
        return jnp.concatenate(tuple(getattr(p, name) for p in parts))

    nodes = [
        jnp.concatenate(
            (p.nodes, jnp.repeat(p.nodes[:, -1:], width - p.n_nu, axis=1)), axis=1
        )
        for p in parts
    ]
    weights = [jnp.pad(p.weights, ((0, 0), (0, width - p.n_nu))) for p in parts]
    units = {p.unit for p in parts}
    return _ChannelUnion(
        family="joined",
        centres_hz=array("centres_hz"),
        widths_hz=array("widths_hz"),
        support=array("support"),
        nodes=jnp.concatenate(nodes),
        weights=jnp.concatenate(weights),
        normalisation=normalisations.pop(),
        smoothness=min(p.smoothness for p in parts),
        n_nu=width,
        unit=units.pop() if len(units) == 1 else "Hz",
        members=parts,
    )


def _block_diag(a, b):
    batch = jnp.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = jnp.broadcast_to(a, (*batch, *a.shape[-2:]))
    b = jnp.broadcast_to(b, (*batch, *b.shape[-2:]))
    zero = ((0, 0),) * len(batch)
    return jnp.pad(a, (*zero, *((0, d) for d in b.shape[-2:]))) + jnp.pad(
        b, (*zero, *((d, 0) for d in a.shape[-2:]))
    )


def _flat_sigma(data):
    return data.sigma.reshape(*data.sigma.shape[: -len(data._layout)], data.n_data)


def _covariance(data):
    if data.cov is not None:
        return data.cov
    noise = _flat_sigma(data)
    invalid = (noise <= 0) | ~jnp.isfinite(noise)
    noise = _checked(
        noise,
        jnp.any(data.mask & invalid),
        "unmasked standard deviations must be finite and positive.",
    )
    return jnp.where(jnp.eye(data.n_data, dtype=bool), noise[..., :, None] ** 2, 0.0)


def _response_part(data, labels, *, uncertainty=False):
    n = data.channels.n_ch * len(data.components)
    if uncertainty:
        value = data.response_uncertainty
        if value is None:
            value = jnp.zeros((data.n_data, n))
    else:
        value = jnp.eye(n) if data.response is None else data.response
    columns = np.arange(data.channels.n_ch * len(labels)).reshape(
        data.channels.n_ch, len(labels)
    )
    columns = columns[:, [labels.index(x) for x in data.components]].reshape(-1)
    return (
        jnp.zeros((data.n_data, data.channels.n_ch * len(labels)))
        .at[:, columns]
        .set(value)
    )


def _combine_discrepancy(a, b):
    terms = (a.discrepancy, b.discrepancy)
    if all(term is None for term in terms):
        return None
    if any(term is None or term.kind == "unbounded" for term in terms):
        return ErrorTerm.unbounded(
            "an instrument has no finite discrepancy envelope", name="discrepancy"
        )
    valued = [data.discrepancy for data in (a, b) if _has_discrepancy(data)]
    if not valued:
        return ErrorTerm.not_applicable(name="discrepancy")
    value = jnp.concatenate(
        tuple(
            (
                jnp.zeros(data.n_data)
                if term.value is None
                else _model_discrepancy(data, term.value)
            )
            for data, term in zip((a, b), terms)
        )
    )
    kind = max((term.kind for term in valued), key=WEAKEST_ORDER.index)
    return ErrorTerm(
        value,
        kind,
        "joined instrument envelopes in observation space",
        name="discrepancy",
        declared=True,
    )


def join_data(a, b):
    """Join channel rows with independent cross-instrument noise."""
    if not isinstance(b, type(a)):
        return NotImplemented
    if a.batch_shape != b.batch_shape or a.unit != b.unit:
        raise DataError("joined observations must have the same pixel shape and unit.")
    same = a.components == b.components
    labels = (
        a.components
        if same
        else tuple(x for x in _AXES if x in a.components or x in b.components)
    )
    stokes_layout = same and len(a._layout) == len(b._layout) == 2
    values = (
        jnp.concatenate((a.stokes, b.stokes), axis=-2)
        if stokes_layout
        else jnp.concatenate((a.data_vector(), b.data_vector()), axis=-1)
    )
    response = None
    uncertainty = None
    if not same or a.response is not None or b.response is not None:
        response = _block_diag(_response_part(a, labels), _response_part(b, labels))
        parts = (a, b)
        known = all(
            d.response is None or d.response_uncertainty is not None for d in parts
        )
        if known:
            uncertainty = _block_diag(
                *(_response_part(d, labels, uncertainty=True) for d in parts)
            )
    sigma, cov = None, None
    if a.sigma is not None and b.sigma is not None:
        sa, sb = _flat_sigma(a), _flat_sigma(b)
        batch = jnp.broadcast_shapes(sa.shape[:-1], sb.shape[:-1])
        sigma = jnp.concatenate(
            (
                jnp.broadcast_to(sa, (*batch, a.n_data)),
                jnp.broadcast_to(sb, (*batch, b.n_data)),
            ),
            axis=-1,
        )
    else:
        cov = _block_diag(_covariance(a), _covariance(b))
    batch = jnp.broadcast_shapes(a.mask.shape[:-1], b.mask.shape[:-1])
    mask = jnp.concatenate(
        (
            jnp.broadcast_to(a.mask, (*batch, a.n_data)),
            jnp.broadcast_to(b.mask, (*batch, b.n_data)),
        ),
        axis=-1,
    )
    output = type(a)(
        _join_channels((a.channels, b.channels)),
        stokes=values,
        unit=a.unit,
        sigma=sigma,
        cov=cov,
        mask=mask,
        components=labels,
        response=response,
        response_uncertainty=uncertainty,
        discrepancy=_combine_discrepancy(a, b),
    )
    withheld = jnp.concatenate((a.withheld, b.withheld))
    return eqx.tree_at(lambda d: d.withheld, output, withheld)
