"""Cache keys against a kernel subclass, an undeclared ``.get`` read and edited globals (decisions 156 to 160).

* A kernel class outside syncem that inherits ``describe()`` from a syncem class is not cached, and the class
  of every kernel enters its record. A class that defines its own ``describe()`` is keyed by it.
* A rule that reads an undeclared averaged or Legendre variable through ``.get`` or ``in`` has no chain, as a
  read through ``[]`` has none. The same read with the variable declared has a chain.
* The kernel record and the phase fingerprints enter the trace token of the jet routines, so an in-place edit
  of a global array that the kernel function or ``tau`` reads gives a new trace.
* A provider call outside a basis build takes its routine under the token "direct", which ignores such an
  edit. The test pins that documented limit (decision 160).

Every comparison with ``cache=False`` is by ``tobytes()``. The factor 2 of the subclass (against the plain
kernel on the jet, since the stack refuses the subclass), the factor 10 of the edited arrays and the ratio 4/3
of the ``.get`` law follow from the code of the cases. Their tolerance 1e-15 is that of the stale-trace test of
``test_v080_cache.py`` (plan row 160).
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.continuum import ContinuumKernel
from syncem.base.phase import PhaseLaw
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.engine import _keys as keys
from syncem.engine._rule_fingerprint import chain_with_reason
from syncem.engine.average import AveragedKernel, NodeRule, uniform_rule
from syncem.engine.basis import build_from_averaged
from syncem.engine.provider_jet import jet_lattice, make_routine
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.model.phase import GaussianScreen
from tests.v080.test_v080_cache import memo_key, same_bits
from tests.v080.test_v080_stack import small_channels

TOL = 1e-15
AB = [Variable("a", kind="positive"), Variable("b", kind="positive")]
ABE = [*AB, Variable("eta", kind="cosine")]
GAMMA2 = Cells.log(300, 3e5, 2, order=2)
A2 = Cells.log(1.0, 30.0, 2, order=2)


def events(basis):
    return dict(basis.record.events)


def not_cached(basis, word):
    assert basis.key is None and memo_key(basis) is None
    assert any(n.startswith("not cached: ") and word in n for n in basis.record.notes)


# ------------------------------------------------------------------ the kernel class


class Doubled(ContinuumKernel):
    def _node_values(self, nodes, point, phase):
        k_i, k_p = super()._node_values(nodes, point, phase)
        return 2.0 * k_i, 2.0 * k_p


class _Own:
    def describe(self):
        return (("name", "own"),)


class _OtherOwn(_Own):
    pass


def test_the_kernel_record_carries_the_class_of_the_kernel():
    record = keys.kernel_record(ContinuumKernel(x_min=1e-12))
    assert record[0] == ("class", "syncem.base.continuum.ContinuumKernel")
    assert keys.kernel_record(Doubled(x_min=1e-12)) is None
    own, other = keys.kernel_record(_Own()), keys.kernel_record(_OtherOwn())
    assert own is not None and other is not None and own != other
    assert own[1:] == other[1:] == (("name", "own"),)


def _eta_avg(kern):
    return AveragedKernel(kern, [("eta", uniform_rule(0.0, 0.5, 3))], fixed={"B": 5e-6})


@pytest.mark.parametrize("routes", ["jet", "auto"])
def test_a_subclass_that_inherits_describe_is_not_cached(routes):
    ch, lat, kw = (
        small_channels(3, 8),
        Lattice(gamma=GAMMA2),
        {"routes": routes, "azimuth_factor": 0.4},
    )
    base = build_from_averaged(_eta_avg(ContinuumKernel(x_min=1e-12)), ch, lat, **kw)
    assert base.key is not None
    doubled = _eta_avg(Doubled(x_min=1e-12))
    got = build_from_averaged(doubled, ch, lat, **kw)
    not_cached(got, "describe")
    same_bits(got, build_from_averaged(doubled, ch, lat, cache=False, **kw))
    # the stack refuses a subclass, so the reference of the factor 2 is the plain kernel on the jet
    jet = {**kw, "routes": "jet"}
    plain = build_from_averaged(
        _eta_avg(ContinuumKernel(x_min=1e-12)), ch, lat, cache=False, **jet
    )
    np.testing.assert_allclose(
        np.asarray(got.spectra0), 2.0 * np.asarray(plain.spectra0), rtol=TOL, atol=0
    )


# ------------------------------------------------------------------ an undeclared read through .get


@kernel(variables=ABE, stokes=("I",))
def ab_kernel(nu, a, b, eta):
    return {"I": a * b * jnp.ones_like(nu)}


def _get_avg(c, parents, read):
    def law(p):
        scale = 1 + c * read(p)
        return scale * jnp.asarray([1.25, 1.75]), jnp.asarray([0.5, 0.5])

    levels = [("eta", uniform_rule(0.2, 0.8, 3)), ("b", NodeRule(law, parents=parents))]
    return AveragedKernel(ab_kernel, levels)


READS = {
    "get": lambda p: p.get("eta", 0.0),
    "in": lambda p: p["eta"] if "eta" in p else 0.0,
}


@pytest.mark.parametrize("how", sorted(READS))
def test_an_undeclared_read_has_no_chain_and_a_declared_one_has(how):
    lat = Lattice(a=A2)
    hidden = _get_avg(1.0, (), READS[how])
    chain, why = chain_with_reason(hidden.averaged[1][1], "b", hidden, lat, 0, 2)
    assert chain is None and "eta" in why and why.startswith("not cached: ")
    shown = _get_avg(1.0, ("eta",), READS[how])
    chain, why = chain_with_reason(shown.averaged[1][1], "b", shown, lat, 0, 2)
    assert why is None and len(chain) == 3


def test_a_get_read_of_an_undeclared_variable_is_not_cached():
    ch, lat = small_channels(2, 8), Lattice(a=A2)
    one = build_from_averaged(_get_avg(1.0, (), READS["get"]), ch, lat)
    two_avg = _get_avg(2.0, (), READS["get"])
    two = build_from_averaged(two_avg, ch, lat)
    not_cached(two, "eta")
    same_bits(two, build_from_averaged(two_avg, ch, lat, cache=False))
    ratio = np.asarray(two.spectra0)[:, 0, 0] / np.asarray(one.spectra0)[:, 0, 0]
    np.testing.assert_allclose(ratio, 4.0 / 3.0, rtol=TOL)


def test_a_declared_get_read_is_cached_and_keyed_by_the_law():
    ch, lat = small_channels(2, 8), Lattice(a=A2)
    one = build_from_averaged(_get_avg(1.0, ("eta",), READS["get"]), ch, lat)
    two_avg = _get_avg(2.0, ("eta",), READS["get"])
    two = build_from_averaged(two_avg, ch, lat)
    assert None not in (memo_key(one), memo_key(two))
    assert memo_key(one) != memo_key(two) and events(two)["S2 columns reused"] == 0
    same_bits(two, build_from_averaged(two_avg, ch, lat, cache=False))


# ------------------------------------------------------------------ edited globals and the trace token

SCALE = np.array([1.0])
TAU = np.array([2.0])


@kernel(variables=AB, stokes=("I",))
def scaled_kernel(nu, a, b):
    return {"I": SCALE[0] * a * b * jnp.ones_like(nu)}


def _times_ten(array, run):
    np.multiply(array, 10.0, out=array)
    try:
        return run()
    finally:
        np.divide(array, 10.0, out=array)


def test_an_edited_global_array_read_by_the_kernel_function_gives_the_new_numbers():
    avg = AveragedKernel(
        scaled_kernel,
        [("b", NodeRule(lambda p: (jnp.ones(1), jnp.ones(1)), parents=()))],
    )
    lattice = Lattice(a=Cells.log(math.exp(-0.2), math.exp(0.2), 1, order=1))
    ch = small_channels(2, 8)
    before = build_from_averaged(avg, ch, lattice)
    after, fresh = _times_ten(
        SCALE,
        lambda: (
            build_from_averaged(avg, ch, lattice),
            build_from_averaged(avg, ch, lattice, cache=False),
        ),
    )
    assert memo_key(after) not in (None, memo_key(before))
    np.testing.assert_allclose(
        np.asarray(after.spectra0), 10.0 * np.asarray(before.spectra0), rtol=TOL, atol=0
    )
    same_bits(after, fresh)


def _tau(nu):
    return TAU[0] * (2.99792458e8 / nu) ** 2


def test_an_edited_global_array_read_by_tau_gives_the_new_numbers():
    phase = PhaseLaw(_tau, screen=GaussianScreen(8.0, 0.5))
    avg = AveragedKernel(
        ContinuumKernel(x_min=1e-12), [], fixed={"B": 5e-6, "eta": 0.3}, phase=phase
    )
    lat, ch = Lattice(gamma=GAMMA2), small_channels(3, 8)
    before = build_from_averaged(avg, ch, lat, routes="jet", azimuth_factor=0.4)
    after, fresh = _times_ten(
        TAU,
        lambda: (
            build_from_averaged(avg, ch, lat, routes="jet", azimuth_factor=0.4),
            build_from_averaged(
                avg, ch, lat, routes="jet", azimuth_factor=0.4, cache=False
            ),
        ),
    )
    assert memo_key(after) not in (None, memo_key(before))
    assert np.asarray(after.spectra2).tobytes() != np.asarray(before.spectra2).tobytes()
    same_bits(after, fresh)


# ------------------------------------------------------------------ direct provider calls (decision 160)

NODES = np.array([1.5])


@kernel(variables=AB, stokes=("I",))
def plain_kernel(nu, a, b):
    return {"I": a * b * jnp.ones_like(nu)}


def test_a_direct_jet_call_reuses_its_trace_and_a_fresh_routine_does_not():
    """The documented limit of decision 160: the token "direct" keys a trace by rule identity."""
    rule = NodeRule(lambda p: (jnp.asarray(NODES), jnp.ones(1)), parents=())
    avg = AveragedKernel(plain_kernel, [("b", rule)])
    lat = Lattice(a=Cells.log(math.exp(-0.2), math.exp(0.2), 1, order=1))
    ch = small_channels(2, 8)
    before = np.asarray(jet_lattice(avg, ch, lat)[0])
    direct, fresh = _times_ten(
        NODES,
        lambda: (
            np.asarray(jet_lattice(avg, ch, lat)[0]),
            np.asarray(jet_lattice(avg, ch, lat, routine=make_routine)[0]),
        ),
    )
    assert direct.tobytes() == before.tobytes()
    np.testing.assert_allclose(fresh, 10.0 * before, rtol=TOL, atol=0)
