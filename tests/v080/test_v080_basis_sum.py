"""``BasisSum`` of step 5b-1 (design/0.8/SPEC.md Sec. 3.4, decision 152).

Two parts share one lattice of 19 energy cells at order 2 (57 columns each) and differ in the averaged field
strength: B ~ U[4, 6] uG and a Maxwellian of b_rms 5 uG, each on 16 nodes. The design of the sum is the
horizontal concatenation of the parts' designs, a prediction is the sum of the parts' predictions in the order of
the parts, and ``at`` maps over the parts. Every comparison is an exact identity, so no tolerance enters.
"""

import dataclasses

import equinox as eqx
import jax
import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.engine.assemble import predict_values
from syncem.engine.average import AveragedKernel, fixed_rule, uniform_rule
from syncem.engine.basis import build_from_averaged
from syncem.engine.basis_sum import BasisSum
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.moments import Moments
from tests.v080.test_v080_stack import screen, small_channels

K = ContinuumKernel(x_min=1e-12)
NAMES = ("uniform", "maxwell")
LATTICE = Lattice(gamma=Cells.log(300, 3e5, 19, order=2))


def maxwell_rule(rms=5e-6, n=16, cut=8.0):
    """Gauss-Legendre on [0, cut rms] times the Maxwellian density of the field strength, as ``ST/lib/config.py``."""
    t, w = np.polynomial.legendre.leggauss(n)
    half = 0.5 * cut * rms
    b, w = half * (t + 1.0), half * w
    sig = rms / np.sqrt(3.0)
    pdf = np.sqrt(2.0 / np.pi) * b**2 / sig**3 * np.exp(-(b**2) / (2.0 * sig**2))
    return fixed_rule(b, w * pdf)


def averaged(law):
    return AveragedKernel(
        K, [("B", law), ("eta", uniform_rule(0.0, 0.5, 4))], phase=screen()
    )


def part(law, channels=None, **kwargs):
    ch = small_channels() if channels is None else channels
    return build_from_averaged(averaged(law), ch, LATTICE, azimuth_factor=0.4, **kwargs)


@pytest.fixture(scope="module")
def parts():
    return part(uniform_rule(4e-6, 6e-6, 16)), part(maxwell_rule())


@pytest.fixture(scope="module")
def total(parts):
    return BasisSum(parts, NAMES)


def moments(seed, component=None, batch=()):
    values = np.random.default_rng(seed).uniform(0.5, 2.0, batch + (LATTICE.n,))
    return Moments(LATTICE, values, component=component)


def test_design_is_the_horizontal_concatenation(parts, total):
    a, b = parts
    assert total.offsets == (0, 57) and total.n == 114
    assert total.components == NAMES and total.parts == parts
    assert np.shape(total.design()) == (5, 3, 114)
    np.testing.assert_array_equal(
        np.asarray(total.design()),
        np.concatenate([np.asarray(a.design()), np.asarray(b.design())], axis=-1),
    )
    np.testing.assert_array_equal(
        np.asarray(total.design(stokes="I")),
        np.concatenate(
            [np.asarray(a.design(stokes="I")), np.asarray(b.design(stokes="I"))],
            axis=-1,
        ),
    )
    with pytest.raises(NotImplementedError, match="step 8"):
        total.design(np.ones((5, 3)))


def test_the_parts_differ(parts):
    """The two field laws give different designs, so the sum checks below are not trivial."""
    a, b = parts
    assert np.max(np.abs(np.asarray(a.design()) - np.asarray(b.design()))) > 0


@pytest.mark.parametrize("batch", [(), (3,)])
def test_predict_is_the_sum_of_the_parts_in_part_order(parts, total, batch):
    a, b = parts
    ma, mb = moments(1, batch=batch), moments(2, batch=batch)
    explicit = np.asarray(predict_values(a, ma)) + np.asarray(predict_values(b, mb))
    assert explicit.shape == batch + (5, 3)
    np.testing.assert_array_equal(np.asarray(total.predict((ma, mb))), explicit)
    na, nb = (moments(s, c, batch) for s, c in zip((1, 2), NAMES))
    for given in ((na, nb), (nb, na), [nb, na]):
        np.testing.assert_array_equal(np.asarray(total.predict(given)), explicit)


def test_predict_refusals(total):
    ma, mb = moments(1), moments(2)
    named = moments(1, "uniform")
    with pytest.raises(ValueError, match="partly named"):
        total.predict((named, mb))
    with pytest.raises(ValueError, match="components"):
        total.predict((named, moments(2, "other")))
    with pytest.raises(ValueError, match="components"):
        total.predict((named, moments(2, "uniform")))
    with pytest.raises(ValueError, match="2 parts"):
        total.predict((ma,))
    with pytest.raises(TypeError, match="tuple"):
        total.predict(ma)
    other = Lattice(gamma=Cells.log(300, 3e5, 19, order=1))
    with pytest.raises(ValueError, match="key"):
        total.predict((ma, Moments(other, np.ones(other.n))))


def test_build_refusals(parts):
    a, b = parts
    with pytest.raises(ValueError, match="distinct"):
        BasisSum(parts, ("same", "same"))
    with pytest.raises(ValueError, match="one component"):
        BasisSum(parts, ("only",))
    with pytest.raises(ValueError, match="at least one"):
        BasisSum((), ())
    with pytest.raises(TypeError, match="SpectralBasis"):
        BasisSum((a, "b"), NAMES)
    with pytest.raises(ValueError, match="non-empty str"):
        BasisSum(parts, ("uniform", ""))
    moved = dataclasses.replace(b, lattice=LATTICE.with_key("another lattice"))
    with pytest.raises(ValueError, match="lattice keys"):
        BasisSum((a, moved), NAMES)
    unkeyed = dataclasses.replace(b, lattice=LATTICE.with_key(None))
    with pytest.raises(ValueError, match="None"):
        BasisSum((a, unkeyed), NAMES)
    with pytest.raises(ValueError, match="channel row keys"):
        BasisSum((a, b.at(b.channels[[0, 1, 2, 3]])), NAMES)


def test_different_stokes_requests_raise(parts):
    a, _ = parts
    only_i = build_from_averaged(
        averaged(maxwell_rule()), small_channels(), LATTICE, stokes="I"
    )
    with pytest.raises(ValueError, match="Stokes"):
        BasisSum((a, only_i), NAMES)


def test_at_maps_over_the_parts(parts, total):
    """A subset with repeats (gathered rows) and a wider channel set (new rows computed by the providers)."""
    wide = Channels.log(0.1, 10, 9, unit="GHz", half_width=0.2, n_nu=32)
    for channels in (total.parts[0].channels[[3, 0, 0, 4]], wide):
        moved = total.at(channels)
        assert isinstance(moved, BasisSum)
        assert moved.components == NAMES and moved.offsets == (0, 57)
        for got, p in zip(moved.parts, parts):
            np.testing.assert_array_equal(
                np.asarray(got.design()), np.asarray(p.at(channels).design())
            )
        np.testing.assert_array_equal(
            np.asarray(moved.design()),
            np.concatenate([np.asarray(p.design()) for p in moved.parts], axis=-1),
        )


def test_static_fields_and_pytree(total):
    """``components`` and ``offsets`` are static, so the parts' arrays are the only leaves."""
    assert isinstance(total, eqx.Module)
    leaves, parts_leaves = (jax.tree_util.tree_leaves(t) for t in (total, total.parts))
    assert len(leaves) == len(parts_leaves) > 0
    assert all(x is y for x, y in zip(leaves, parts_leaves))
    names = {f.name: f.metadata.get("static", False) for f in dataclasses.fields(total)}
    assert names == {"parts": False, "components": True, "offsets": True}
