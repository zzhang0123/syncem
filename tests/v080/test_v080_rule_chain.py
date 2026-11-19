"""Rule chains and ``TabulatedRule`` of design/0.8/SPEC.md decisions 157 and 160 (step 5b-2).

The chain ``d_0, ..., d_N`` of an averaged ``NodeRule`` at one cell changes with nodes, probabilities and
density values at its point sets (``d_0``) and with the order-``n`` Taylor coefficients at the reference point
(``d_n``). Every assertion is an equality or inequality of sha256 digests, an exact identity, so no tolerance
enters. A change that must not reach ``d_0`` is masked to ``abs(u) < 0.05`` in the chart coordinate ``u``. The
point sets of ``d_0`` have ``abs(u) = 1`` at the edges of a centred cell and ``abs(u) >= 0.1834`` at the 8
Gauss-Legendre points, so the masked term is 0 there.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.variables import Variable
from syncem.engine._rule_fingerprint import (
    _cell_values,
    TabulatedRule,
    chain_with_reason,
    fixed_fingerprint,
    rule_chain,
    with_tabulated_rules,
)
from syncem.engine.average import AveragedKernel, NodeRule, fixed_rule
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import Legendre

T12, W12 = (jnp.asarray(a) for a in np.polynomial.legendre.leggauss(12))
MASK = 0.05


class _Kernel:
    """Variables only: the chain never evaluates the kernel."""

    name = "stub"

    def __init__(self, names, cosines=()):
        self.variables = tuple(
            Variable(n, kind="cosine" if n in cosines else "real") for n in names
        )


def _lattice(order=4, weights=None, **axes):
    axes = axes or {"x": Cells.log(1.0, 100.0, 3, order=order)}
    return Lattice(weights=weights, **axes)


def _u(point, lat, cell=0, var="x"):
    """Chart coordinate of ``var`` in ``cell`` (log chart), as the rule sees it."""
    table = lat.cells
    i = table.variables.index(var)
    return jnp.log(point[var] / table.reference[cell, i]) / table.half_width[cell, i]


def _masked(u, value):
    return jnp.where(jnp.abs(u) < MASK, value, 0.0)


def _law(shift=None, density=None, parents=("x",), slope=0.3):
    """``y`` given ``x``: 12 nodes around ``slope * ln x`` with an optional added ``shift(point)``."""

    def fn(point):
        centre = slope * jnp.log(point["x"])
        if shift is not None:
            centre = centre + shift(point)
        return centre + 0.5 * T12, 0.5 * W12

    def dens(nodes, point):
        return jnp.where(jnp.abs(nodes - slope * jnp.log(point["x"])) <= 0.5, 1.0, 0.0)

    return NodeRule(fn, parents=parents, density=density or dens)


def _chain(rule, name="y", extra=(), lat=None, cell=0, order=4, fixed=None):
    lat = lat or _lattice()
    names = ["x", name, *[n for n, _ in extra], *(fixed or {})]
    averaged = AveragedKernel(_Kernel(names), (*extra, (name, rule)), fixed=fixed or {})
    return rule_chain(rule, name, averaged, lat, cell, order)


# ------------------------------------------------------------------ d_0: values at the point sets


def test_chain_has_one_digest_per_order_and_is_deterministic():
    a, b = _chain(_law()), _chain(_law())
    assert len(a) == 5 and a == b and len(set(a)) == 5
    assert all(isinstance(d, str) and len(d) == 64 for d in a)


@pytest.mark.parametrize(
    "other",
    [
        pytest.param(_law(slope=0.31), id="nodes"),
        pytest.param(
            NodeRule(
                lambda p: (_law().fn(p)[0], 0.5 * W12 * (1 + 0.01 * T12)),
                parents=("x",),
                density=_law().density,
            ),
            id="probabilities",
        ),
        pytest.param(
            _law(density=lambda n, p: 2.0 * _law().density(n, p)), id="density-values"
        ),
    ],
)
def test_d0_changes_with_values(other):
    assert _chain(other)[0] != _chain(_law())[0]


def test_d0_sees_the_edges_of_a_declared_parent():
    """A change that is 0 at the reference and not at the edges reaches ``d_0``."""
    lat = _lattice()
    edge = _law(shift=lambda p: 0.1 * _u(p, lat) ** 2)
    assert _chain(edge, lat=lat)[0] != _chain(_law(), lat=lat)[0]


@pytest.mark.parametrize(
    "chart, ref, hw, lo, hi, edges",
    [
        ("log", 10.0, 0.5, 1.0, 100.0, (1.0, 100.0)),
        ("log", 30.0, 0.4, 0.0, np.inf, (30.0 * np.exp(-0.4), 30.0 * np.exp(0.4))),
        ("affine", 2.0, 0.5, -np.inf, 2.25, (1.5, 2.25)),
    ],
)
def test_cell_points_take_the_edges_or_chart_coordinate_one(
    chart, ref, hw, lo, hi, edges
):
    """A bounded edge is taken as it is. An unbounded one is taken at chart coordinate -1 or 1."""
    values = _cell_values(chart, ref, hw, lo, hi)
    assert values.shape == (10,) and np.all(np.isfinite(values))
    assert tuple(values[:2]) == edges
    assert np.all((values[2:] > edges[0]) & (values[2:] < edges[1]))


def test_cells_without_edges_have_a_chain():
    lat = Lattice(x=Cells.at_points([3.0, 30.0], [0.4, 0.4], chart="log", order=2))
    a, b = _chain(_law(), lat=lat, cell=1, order=2), _chain(
        _law(), lat=lat, cell=0, order=2
    )
    assert len(a) == 3 and a[0] != b[0]


# ------------------------------------------------------------------ d_n: Taylor coefficients at the reference


def test_density_slope_at_frozen_nodes_reaches_d1():
    """Equal nodes, probabilities and density values at every point, other gamma slope of the density."""
    lat = _lattice()
    base = _law()
    sloped = _law(
        density=lambda n, p: base.density(n, p)
        * (1.0 + _masked(_u(p, lat), _u(p, lat)))
    )
    a, b = _chain(base, lat=lat), _chain(sloped, lat=lat)
    assert a[0] == b[0] and a[1] != b[1]


def test_node_slope_reaches_d1():
    lat = _lattice()
    sloped = _law(shift=lambda p: _masked(_u(p, lat), 0.2 * _u(p, lat)))
    a, b = _chain(_law(), lat=lat), _chain(sloped, lat=lat)
    assert a[0] == b[0] and a[1] != b[1]


def test_undeclared_read_of_a_celled_variable_reaches_d1():
    """A law with ``parents=()`` that reads ``x``: equal values at the reference, other slope."""

    def fn(scale):
        return lambda p: (1.0 + scale * jnp.log(p["x"] / 10.0) + 0.5 * T12, 0.5 * W12)

    lat = _lattice(x=Cells.log(1.0, 100.0, 1, order=4))  # reference 10
    still = NodeRule(fn(0.0), parents=())
    reads = NodeRule(fn(0.4), parents=())
    a, b = _chain(still, lat=lat), _chain(reads, lat=lat)
    assert a[0] == b[0] and a[1] != b[1]


@pytest.mark.parametrize("n", [0, 1, 3])
def test_change_at_order_n_plus_one_keeps_d0_to_dn(n):
    lat = _lattice()
    late = _law(shift=lambda p: _masked(_u(p, lat), 0.7 * _u(p, lat) ** (n + 1)))
    a, b = _chain(_law(), lat=lat), _chain(late, lat=lat)
    assert a[: n + 1] == b[: n + 1] and a[n + 1] != b[n + 1]
    assert all(x != y for x, y in zip(a[n + 1 :], b[n + 1 :]))


def test_averaged_parent_moving_with_a_celled_variable_enters_the_slope():
    """``y`` reads ``e``, whose node moves with ``x``. A slope of ``y`` in ``e`` reaches ``d_1``.

    The term vanishes at every ``e`` value of the point sets (``abs(e - 0.5) >= 0.04`` off the
    reference in the middle cell, half-width 0.768), so only the chain rule through ``e(x)`` sees it.
    """
    lat = _lattice()
    ref = float(lat.cells.reference[1, 0])
    e_rule = NodeRule(
        lambda p: (jnp.atleast_1d(0.5 * p["x"] / ref), jnp.ones(1)), parents=("x",)
    )

    def y_rule(k):
        def fn(p):
            t = p["e"] - 0.5
            return jnp.atleast_1d(1.0 + p["e"] + _masked(t / 0.4, k * t)), jnp.ones(1)

        return NodeRule(fn, parents=("e",))

    extra = (("e", e_rule),)
    a = _chain(y_rule(0.0), extra=extra, lat=lat, cell=1)
    b = _chain(y_rule(3.0), extra=extra, lat=lat, cell=1)
    assert a[0] == b[0] and a[1] != b[1]


def test_chain_is_unchanged_by_weights_and_unrelated_fixed_values():
    base = _chain(_law(), fixed={"m": 0.3})
    other = _chain(_law(), lat=_lattice(weights=[1.0, 2.0, 5.0]), fixed={"m": 0.7})
    assert base == other
    assert fixed_fingerprint(0.3) != fixed_fingerprint(0.7)
    assert fixed_fingerprint(5) == fixed_fingerprint(5.0)
    assert fixed_fingerprint(-0.0) == fixed_fingerprint(0.0)


def test_a_read_fixed_value_changes_the_chain():
    rule = NodeRule(lambda p: (p["m"] + 0.5 * T12, 0.5 * W12), parents=("m",))
    assert _chain(rule, fixed={"m": 0.3}) != _chain(rule, fixed={"m": 0.7})


def test_legendre_parent_is_tabulated_at_its_gauss_nodes():
    lat = _lattice(x=Cells.log(1.0, 100.0, 2, order=2), mu=Legendre(2))

    def averaged(r):
        kernel = _Kernel(["x", "y", "mu"], cosines=("mu",))
        return AveragedKernel(kernel, (("y", r),), legendre={"mu": (2, 4)})

    def rule(k):
        return NodeRule(
            lambda p: (k * p["mu"] ** 2 + 0.5 * T12, 0.5 * W12), parents=("mu",)
        )

    a, b = rule(1.0), rule(2.0)
    assert rule_chain(a, "y", averaged(a), lat, 0, 2) != rule_chain(
        b, "y", averaged(b), lat, 0, 2
    )


@pytest.mark.parametrize("names", [("x", "z"), ("x", "z", "s")])
def test_chain_in_several_celled_variables(names):
    """``d`` = 2 and 3: a mixed slope in ``x`` and ``z`` reaches ``d_2`` only."""
    axes = {n: Cells.log(1.0, 100.0, 2, order=2) for n in names}
    lat = Lattice(**axes)

    def averaged(r):
        return AveragedKernel(_Kernel([*names, "y"]), (("y", r),))

    def rule(k):
        def fn(p):
            mixed = _masked(_u(p, lat), k * _u(p, lat) * _u(p, lat, var="z"))
            return 0.1 * jnp.log(p["x"] * p["z"]) + mixed + 0.5 * T12, 0.5 * W12

        return NodeRule(fn, parents=names)

    a = rule_chain(rule(0.0), "y", averaged(rule(0.0)), lat, 0, 3)
    b = rule_chain(rule(1.0), "y", averaged(rule(1.0)), lat, 0, 3)
    assert a[:2] == b[:2] and a[2] != b[2]


# ------------------------------------------------------------------ not cacheable: both sides of the dispatcher


@pytest.mark.parametrize(
    "rule, reason",
    [
        pytest.param(lambda p: (T12, W12), "bare callable", id="bare-callable"),
        pytest.param(
            NodeRule(lambda p: (T12, W12), parents=None),
            "parents=None",
            id="unknown-parents",
        ),
        pytest.param(
            NodeRule(lambda p: (p["q"] + T12, W12), parents=("q",)),
            "q",
            id="unknown-parent",
        ),
        pytest.param(
            NodeRule(lambda p: (np.asarray(p["x"]) + T12, W12), parents=("x",)),
            "differentiated",
            id="not-traceable",
        ),
    ],
)
def test_not_cacheable_gives_none_and_a_reason(rule, reason):
    lat = _lattice()
    averaged = AveragedKernel(_Kernel(["x", "y"]), (("y", rule),))
    chain, why = chain_with_reason(rule, "y", averaged, lat, 0, 2)
    assert chain is None and why.startswith("not cached: ") and reason in why
    assert rule_chain(rule, "y", averaged, lat, 0, 2) is None


def test_cacheable_side_has_no_reason():
    lat = _lattice()
    rule = fixed_rule(T12, 0.5 * W12)
    averaged = AveragedKernel(_Kernel(["x", "y"]), (("y", rule),))
    chain, why = chain_with_reason(rule, "y", averaged, lat, 0, 2)
    assert why is None and len(chain) == 3


@pytest.mark.parametrize(
    "kwargs, match",
    [({"cell": 3}, "cell"), ({"order": -1}, "order"), ({"name": "x"}, "averaged")],
)
def test_arguments_are_checked(kwargs, match):
    lat, rule = _lattice(), _law()
    averaged = AveragedKernel(_Kernel(["x", "y"]), (("y", rule),))
    args = {"name": "y", "cell": 0, "order": 2, **kwargs}
    with pytest.raises(ValueError, match=match):
        rule_chain(rule, args["name"], averaged, lat, args["cell"], args["order"])


# ------------------------------------------------------------------ TabulatedRule (decision 160)


def test_tabulated_rule_equality():
    rule = _law()
    a, b = TabulatedRule(rule, "f" * 64), TabulatedRule(rule, "f" * 64)
    assert a == b and hash(a) == hash(b)
    assert a != TabulatedRule(rule, "e" * 64)
    assert a != TabulatedRule(_law(), "f" * 64)  # another function object
    none = TabulatedRule(rule, None)
    assert none == none and none != TabulatedRule(rule, None)
    assert a({"x": 2.0})[0].tobytes() == rule({"x": 2.0})[0].tobytes()
    assert a.parents == rule.parents and a.density is rule.density


class _Holder(eqx.Module):
    rule: object = eqx.field(static=True)


def test_traces_are_reused_by_fingerprint_and_never_without_one():
    """``eqx.filter_jit`` keys a static field on equality, as the jet routine does."""
    traces = []

    @eqx.filter_jit
    def run(holder, x):
        traces.append(1)
        return holder.rule({"x": x})[0]

    rule = _law()
    run(_Holder(TabulatedRule(rule, "a" * 64)), 2.0)
    run(_Holder(TabulatedRule(rule, "a" * 64)), 2.0)
    assert len(traces) == 1
    run(_Holder(TabulatedRule(rule, None)), 2.0)
    run(_Holder(TabulatedRule(rule, None)), 2.0)
    assert len(traces) == 3


TABLE = np.array([0.0])


def _global_law(point):
    return point["x"] * 0.0 + TABLE + 0.5 * T12, 0.5 * W12


def test_a_global_array_edited_in_place_changes_the_chain_and_retraces():
    """The stale-trace probe of decision 160: the edit changes the chain, so the trace is rebuilt."""
    rule = NodeRule(_global_law, parents=("x",))
    lat = _lattice()
    averaged = AveragedKernel(_Kernel(["x", "y"]), (("y", rule),))

    @eqx.filter_jit
    def run(holder, x):
        return holder.rule({"x": x})[0]

    before = rule_chain(rule, "y", averaged, lat, 0, 2)
    old = run(_Holder(TabulatedRule(rule, before[-1])), 2.0)
    TABLE[0] = 10.0
    try:
        after = rule_chain(rule, "y", averaged, lat, 0, 2)
        new = run(_Holder(TabulatedRule(rule, after[-1])), 2.0)
    finally:
        TABLE[0] = 0.0
    assert before[0] != after[0]
    assert new.tobytes() == (old + 10.0).tobytes()


def test_with_tabulated_rules_wraps_each_averaged_rule():
    rule = _law()
    averaged = AveragedKernel(
        _Kernel(["x", "y", "m"]), (("y", rule),), fixed={"m": 0.3}, outer_batch=2
    )
    wrapped = with_tabulated_rules(averaged, {"y": "c" * 64})
    ((name, tab),) = wrapped.averaged
    assert name == "y" and tab == TabulatedRule(rule, "c" * 64)
    assert wrapped.fixed == averaged.fixed and wrapped.outer_batch == 2
    again = with_tabulated_rules(wrapped, {})
    assert (
        again.averaged[0][1].rule is rule and again.averaged[0][1].fingerprint is None
    )
