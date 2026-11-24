"""Sampled one-variable Taylor remainder diagnostics in channel units."""

import jax.numpy as jnp
from jax.core import Tracer
import numpy as np

from ..records.ledger import ErrorTerm
from .provider_table import TABLE_NODES, TABLE_TOL, TableRefused, build_cell_table


def _cell_order(lattice, i):
    index = lattice.index
    return max(power[0] for power in index.patterns[index.cell_pattern[i]])


def sampled_remainder(basis):
    """Largest sampled next coefficient on each bounded cell."""
    lat, ch = basis.lattice, basis.channels
    policy = basis.forward_accuracy
    agreement = dict(basis.record.agreement).get("reference_table")
    if (
        policy is None
        or agreement is None
        or agreement > TABLE_TOL
        or len(lat.cells.variables) != 1
        or lat.reps
        or lat.cells.excluded
        or lat.order is None
        or lat.order + 1 >= TABLE_NODES
    ):
        return None
    name = lat.cells.variables[0]
    point = {**dict(basis.averaged.fixed), name: lat.cells.reference[0, 0]}
    operations = lat.n_cells * TABLE_NODES * ch.n_ch * ch.n_nu
    for variable, rule in basis.averaged.averaged:
        nodes, _ = rule(point)
        operations *= len(nodes)
        point[variable] = nodes[len(nodes) // 2]
    if operations * 1e-8 > policy.max_seconds:
        return None
    terms = []
    try:
        for cell in range(lat.n_cells):
            table = build_cell_table(
                basis.averaged,
                ch,
                lat,
                cell,
                cache=dict(basis.record.defaults).get("cache", True),
            )
            modes = table.sampled_derivative(_cell_order(lat, cell) + 1)
            p = modes.P[:, 0, :]
            if basis.azimuth_factor is not None:
                p = p * basis.azimuth_factor
            values = {"I": modes.I, "Q": p.real, "U": p.imag, "V": modes.V}
            terms.append(
                np.stack(
                    tuple(
                        np.max(np.abs(values[axis]), axis=0)
                        for axis in basis.record.stokes
                    ),
                    axis=-1,
                )
            )
    except TableRefused:
        return None
    return np.stack(terms, axis=-1)


def remainder_term(basis, moments):
    """Propagate the sampled coefficient with known even moments only."""
    measure = moments.origin
    if isinstance(moments.values, Tracer):
        return ErrorTerm.unbounded(
            "the remainder table is evaluated on the host", name="remainder"
        )
    if measure not in ("specified", "family", "nonnegative_penalised"):
        return ErrorTerm.unbounded(
            "the remainder needs the true even moments in each cell", name="remainder"
        )
    table = basis.remainder
    if table is None:
        return ErrorTerm.unbounded(
            "a checked single-axis remainder table is unavailable", name="remainder"
        )
    lattice = basis.lattice
    row_ids, ratio = [], []
    for i in range(lattice.n_cells):
        order = _cell_order(lattice, i)
        even = 2 * (order // 2)
        base = lattice.index.cell_rows(i).start
        row_ids.append(base + even)
        cells = lattice.cells
        lo, hi, ref = (getattr(cells, item)[i, 0] for item in ("lo", "hi", "reference"))
        if cells.chart[i][0] == "log":
            lo, hi, ref = np.log([lo, hi, ref])
        extent = max(abs(lo - ref), abs(hi - ref)) / cells.half_width[i, 0]
        ratio.append(extent ** (order + 1 - even))
    moments_even = jnp.abs(moments.values[..., np.asarray(row_ids)])
    moments_even *= lattice.weights * np.asarray(ratio)
    value = jnp.einsum("csk,...k->...cs", table, moments_even)
    note = "Sampled Taylor remainder for the fitted point-mass measure."
    if measure != "nonnegative_penalised":
        note = "Sampled Taylor remainder of the declared measure."
    return ErrorTerm(value, "estimate", note, name="remainder")
