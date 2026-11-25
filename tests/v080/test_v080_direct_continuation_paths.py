"""Continuation of the physical integrand and its declared support (SPEC Sec. 8.3)."""

import numpy as np
import pytest

import syncem as se
from syncem.base.capabilities import Beaming

_CHANNELS = se.Channels.at([0.01], unit="GHz", half_width=0.01, n_nu=4)
_CHECK = se.AccuracyCheck(max_seconds=1e9)
_REFERENCE = se.ContinuumKernel(x_min=1e-15)


def _population(gamma, field, eta=0.0):
    return se.Population(
        gamma=se.Delta(gamma),
        B=field,
        eta=eta,
        mu=se.Isotropic(),
        phi=se.AzimuthFactor(0.4),
        depth=se.NoRotation(),
    )


def _basis(population, kernel=None):
    lattice = se.Lattice(gamma=se.Cells.linear(1000, 3000, 2, order=1))
    return se.spectral_basis(
        population,
        _CHANNELS,
        lattice=lattice,
        kernel=kernel,
        check=False,
        cache=False,
    )


def _has_tail(spectrum):
    return any(term.name == "small_x:tail" for term in spectrum.ledger.terms)


@pytest.mark.parametrize(
    "target_field", ["1e5 G", lambda eta: se.Delta(1e5 * (1 + eta), unit="G")]
)
def test_candidate_in_the_tail_is_recorded(target_field):
    base = _population(3000, "5 uG")
    basis = _basis(base)
    target = base.replace(B=target_field)
    direct = basis.direct(target)
    reference = _REFERENCE.channel(
        _CHANNELS, {"gamma": 3000, "eta": 0, "B": 1e5}, phase=None
    )
    np.testing.assert_allclose(direct.I, reference.I, rtol=1e-6, atol=0)
    checked = basis.predict(target, check=_CHECK, allow_mismatch=True)
    assert checked.ledger["specified:B"].kind == "measured"
    assert (
        direct.ledger["small_x:tail"].kind
        == checked.ledger["small_x:tail"].kind
        == "unbounded"
    )


def test_only_an_intermediate_replacement_uses_the_tail():
    base = _population(3000, "5 uG")
    target = base.replace(B="1e5 G", eta=1.0)
    basis = _basis(base)
    assert not _has_tail(basis.direct(base))
    assert not _has_tail(basis.direct(target))
    checked = basis.predict(target, check=_CHECK, allow_mismatch=True)
    assert checked.ledger["small_x:tail"].kind == "unbounded"
    assert checked.ledger["specified:B"].kind == "measured"
    assert checked.ledger["specified:eta"].kind == "measured"
    assert not _has_tail(basis.predict(target, allow_mismatch=True, check=False))


@pytest.mark.parametrize("eta", [-1.0, 1.0])
def test_conditional_field_cannot_reach_the_tail_at_either_pole(eta):
    target = _population(1e7, lambda eta: se.Delta(5 + eta, unit="uG"), eta)
    basis = _basis(target)
    for spectrum in (basis.direct(target), basis.predict(target, check=_CHECK)):
        np.testing.assert_array_equal(spectrum.stokes, 0)
        assert not _has_tail(spectrum)


@pytest.mark.parametrize("mu", [-1.0, 1.0])
def test_reverse_beaming_integral_obeys_the_pitch_atom_at_a_pole(mu):
    target = _population(
        1e7, lambda eta: se.Delta(5 + eta, unit="uG"), se.Isotropic()
    ).replace(mu=se.Delta(mu))
    basis = _basis(target)
    for spectrum in (basis.direct(target), basis.predict(target, check=_CHECK)):
        np.testing.assert_array_equal(spectrum.stokes, 0)
        assert not _has_tail(spectrum)


def test_empirical_pitch_measure_on_both_poles_has_zero_transverse_field():
    population = se.Population(
        se.Samples(mu=[-1.0, 0.0, 1.0], weights=[0.5, 0.0, 0.5]),
        gamma=se.Delta(1e7),
        B=lambda eta: se.Delta(5 + eta, unit="uG"),
        eta=se.Isotropic(),
        phi=se.AzimuthFactor(0.4),
        depth=se.NoRotation(),
    )
    basis = _basis(population)
    for spectrum in (basis.direct(population), basis.predict(population, check=_CHECK)):
        np.testing.assert_array_equal(spectrum.stokes, 0)
        assert not _has_tail(spectrum)


def test_hybrid_switch_continues_each_continuum_child_outside_the_lattice():
    target = _population(1e8, lambda eta: se.Delta(5 + eta, unit="uG"))
    child = se.ContinuumKernel()
    basis = _basis(target, se.HybridKernel(child, child, switch={"gamma": 2000.0}))
    plain = _basis(target).direct(target)
    direct = basis.direct(target)
    np.testing.assert_allclose(direct.stokes, plain.stokes, rtol=1e-12, atol=0)
    reference = _REFERENCE.channel(
        _CHANNELS, {"gamma": 1e8, "eta": 0, "B": 5e-6}, phase=None
    )
    np.testing.assert_allclose(direct.I, reference.I, rtol=1e-6, atol=0)
    checked = basis.predict(target, check=_CHECK)
    np.testing.assert_allclose(
        checked.ledger["outside_lattice"].value, -direct.stokes, rtol=1e-13, atol=0
    )
    assert (
        checked.ledger["small_x:tail"].kind
        == direct.ledger["small_x:tail"].kind
        == "unbounded"
    )
    inside = basis.direct(target, within="cells")
    np.testing.assert_array_equal(inside.stokes, 0.0)
    assert not _has_tail(inside)


@se.kernel(
    variables=_REFERENCE.variables,
    stokes=("I", "P"),
    azimuth="phi",
    depth="depth",
    capabilities=tuple(c for c in _REFERENCE.capabilities if isinstance(c, Beaming)),
)
def _custom_continuum(nu, gamma, B, eta, mu):
    modes = _REFERENCE.spectral(nu, {"gamma": gamma, "B": B, "eta": eta}, phase=None)
    return {"I": modes.I, "P": modes.P[0]}


@pytest.mark.parametrize("assignment", ["switch", "assign"])
def test_mixed_continuation_uses_the_assigned_child(assignment):
    route = (
        {"switch": {"gamma": 2000.0}}
        if assignment == "switch"
        else {"assign": lambda box: int(box["gamma"][0] >= 2000.0)}
    )
    baseline = _population(2000.0, "5 uG")
    basis = _basis(
        baseline, se.HybridKernel(_custom_continuum, se.ContinuumKernel(), **route)
    )
    # Points on the shared face use cell zero. Its custom kernel integrates
    # the uncontinued spectrum across the full observing band.
    for gamma, continued in [(2000.0, False), (2500.0, True)]:
        target = _population(gamma, "1e5 G")
        direct = basis.direct(target)
        reference = _REFERENCE.channel(
            _CHANNELS, {"gamma": gamma, "eta": 0.0, "B": 1e5}, phase=None
        )
        np.testing.assert_allclose(
            direct.I, reference.I, rtol=1e-6 if continued else 1e-13, atol=0
        )
        assert _has_tail(direct) == continued
        assert _has_tail(basis.direct(target, within="cells")) == continued
        assert (
            _has_tail(basis.predict(target, allow_mismatch=True, check=_CHECK))
            == continued
        )


def _field_assignment(box):
    # The public callback receives only lattice coordinates.
    assert set(box) == {"B"}
    return int(box["B"][0] >= 5e-6)


@pytest.mark.parametrize(
    "route", [{"switch": {"B": "5 uG"}}, {"assign": _field_assignment}]
)
@pytest.mark.parametrize("energy", [se.Delta(3000.0), se.Uniform(2800.0, 3200.0)])
def test_field_partition_validates_continuum_on_declared_energy_support(
    route, energy, tmp_path
):
    field = se.Lattice(
        B=se.Cells.from_edges([2, 5, 8], chart="affine", order=1, unit="uG")
    )
    pop = _population(3000, "8 uG").replace(gamma=energy)
    child = se.ContinuumKernel()
    hybrid = se.HybridKernel(child, child, **route)
    plain = se.spectral_basis(pop, _CHANNELS, lattice=field, kernel=child, check=False)
    mixed = se.spectral_basis(pop, _CHANNELS, lattice=field, kernel=hybrid, check=False)
    other = se.Channels.at([0.015], unit="GHz", half_width=0.01, n_nu=4)
    for grid, hbasis in [(_CHANNELS, mixed), (other, mixed.at(other))]:
        reference = plain if grid is _CHANNELS else plain.at(other)
        assert dict(hbasis.record.agreement)["hybrid comparisons"] == 3
        np.testing.assert_allclose(
            hbasis.direct(pop).stokes, reference.direct(pop).stokes, rtol=1e-13, atol=0
        )
        far = pop.replace(gamma=se.Delta(1e9))
        tail = hbasis.direct(far)
        np.testing.assert_allclose(
            tail.stokes, reference.direct(far).stokes, rtol=1e-13, atol=0
        )
        assert _has_tail(tail)
    if "switch" in route and isinstance(energy, se.Uniform):
        path = tmp_path / "energy-averaged-hybrid.npz"
        mixed.save(path)
        restored = se.SpectralBasis.load(path, averaged=mixed.averaged)
        np.testing.assert_array_equal(
            restored.at(other).design(), mixed.at(other).design()
        )
    with pytest.raises(se.RegimeError, match="gamma"):
        se.spectral_basis(
            pop.replace(gamma=se.Uniform(250.0, 3100.0)),
            _CHANNELS,
            lattice=field,
            kernel=hybrid,
            check=False,
        )
