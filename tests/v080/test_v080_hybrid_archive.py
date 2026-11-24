"""Stored hybrid routes and independent child guards (SPEC Sec. 8.4–8.5)."""

from dataclasses import replace

import equinox as eqx
import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.hybrid import HybridKernel
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.engine._basis_archive import read_archive, write_archive
from syncem.engine._basis_io_codec import pack, unpack
from syncem.engine._hybrid_basis import HybridRouteRecord
from syncem.engine.average import AveragedKernel
from syncem.engine.basis import SpectralBasis, build_from_averaged
from syncem.engine.basis_io import ArchivedHybridBasis, basis_source_key, load_basis
from syncem.engine.direct import node_modes
from syncem.fitting.data import StokesData
from syncem.fitting.free_fit import fit_free
from syncem.fitting.rank import FullRank
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.moments import Moments
from syncem.values.errors import DataError

_X = (Variable("x"),)


@kernel(variables=_X, stokes=("I", "P"))
def low(nu, x):
    i = 1 + x * nu / 1e9 + 0.01 * x**2
    return {"I": i, "P": 0.2j * i}


def high(factor=1.0002):
    @kernel(variables=_X, stokes=("I", "P"))
    def child(nu, x):
        i = factor * (1 + x * nu / 1e9 + 0.01 * x**2)
        return {"I": i, "P": 0.2j * i}

    return child


@pytest.fixture(scope="module")
def model():
    source = AveragedKernel(HybridKernel(low, high(), switch={"x": 2}))
    channels = Channels.at([1.0, 2.0, 4.0], unit="GHz")
    lattice = Lattice(x=Cells.linear(1.0, 3.0, 2, order=2), order=1)
    return build_from_averaged(source, channels, lattice), source


def test_stored_hybrid_round_trip_and_fit(model, tmp_path):
    basis, _ = model
    path = tmp_path / "hybrid.npz"
    basis.save(path)
    restored = SpectralBasis.load(path)
    assert isinstance(restored, ArchivedHybridBasis)
    assert restored.key == basis.key
    assert restored.record == basis.record
    assert restored.ledger.to_dict() == basis.ledger.to_dict()
    assert tuple(r.kernel_index for r in restored.record.routes) == (0, 1)
    assert all(isinstance(r, HybridRouteRecord) for r in restored.record.routes)
    moment = Moments(basis.lattice, np.array([0.7, 0.1, 0.3, 0.08]))
    expected = basis.predict(moment).stokes
    np.testing.assert_array_equal(restored.predict(moment).stokes, expected)
    subset = restored.at(restored.channels[[2, 0]])
    assert isinstance(subset, ArchivedHybridBasis)
    np.testing.assert_array_equal(subset.design(), np.asarray(basis.design())[[2, 0]])
    data = StokesData(restored.channels, stokes=expected, sigma=0.01, unit="cgs")
    fit = fit_free(restored, data, rank=FullRank())
    np.testing.assert_allclose(fit.predict().stokes, expected, atol=1e-13)
    with pytest.raises(DataError, match="compatible AveragedKernel"):
        restored.at(Channels.at([3.0], unit="GHz"))
    with pytest.raises(DataError, match="compatible AveragedKernel"):
        restored.with_order(0)
    copied = tmp_path / "re-saved.npz"
    subset.save(copied)
    np.testing.assert_array_equal(SpectralBasis.load(copied).design(), subset.design())


def test_live_source_verifies_children_and_keeps_archive_subtype(model, tmp_path):
    basis, source = model
    path = tmp_path / "live.npz"
    basis.save(path)
    key = basis_source_key(basis.averaged)
    assert key is not None
    live = load_basis(path, registry={key: source})
    assert live.averaged.kernel.assignment == (0, 1)
    channels = Channels.at([4.0, 7.0, 1.0], unit="GHz")
    selected = live.at(channels)
    assert isinstance(selected, ArchivedHybridBasis)
    np.testing.assert_array_equal(selected.design(), basis.at(channels).design())
    for order in (0, 2):
        changed = selected.with_order(order)
        assert isinstance(changed, ArchivedHybridBasis)
        np.testing.assert_array_equal(
            changed.design(), basis.at(channels).with_order(order).design()
        )
        with pytest.raises(DataError, match="original population and phase"):
            changed.direct(None)
    candidates = (
        HybridKernel(low, high(1.0004), switch={"x": 2}),
        HybridKernel(low, high(), assign=lambda box: int(box["x"][1] <= 2)),
        HybridKernel(low, high(), switch={"x": 2}, switch_tol=0.0015),
    )
    for child in candidates:
        with pytest.raises(DataError, match="not compatible"):
            load_basis(path, averaged=AveragedKernel(child))


def test_structural_validation_refuses_rehashed_wrong_routes(model, tmp_path):
    basis, source = model
    path = tmp_path / "correct.npz"
    basis.save(path)
    payload, arrays = read_archive(path)
    body = unpack(payload, arrays)
    routes = body["record"].routes
    body["record"] = replace(
        body["record"], routes=(replace(routes[0], kernel_index=1), routes[1])
    )
    defect = tmp_path / "wrong.npz"
    changed_arrays = {}
    write_archive(defect, pack(body, changed_arrays), changed_arrays)
    for declaration in (None, source):
        with pytest.raises(DataError, match="stored hybrid routes disagree"):
            load_basis(defect, averaged=declaration)


@kernel(variables=_X, stokes=("I",))
def guarded_low(nu, x):
    safe = eqx.error_if(x, x > 2, "low child is outside its domain")
    return {"I": safe + np.ones(nu.shape)}


@kernel(variables=_X, stokes=("I",))
def guarded_high(nu, x):
    safe = eqx.error_if(x, x < 2, "high child is outside its domain")
    return {"I": 10 * safe + np.ones(nu.shape)}


def test_direct_runner_only_executes_the_selected_child():
    channels = Channels.at([1.0, 2.0], unit="GHz")
    avg = AveragedKernel(HybridKernel(guarded_low, guarded_high, switch={"x": 2}))
    x = np.array([1.0, 2.4, 1.5, 2.8, 2.0])
    result = node_modes(avg, channels, {"x": x}, batch=4)
    expected = np.where(x <= 2, x, 10 * x) + 1
    np.testing.assert_array_equal(result.I, np.repeat(expected[:, None], 2, axis=1))


def test_direct_runner_retains_azimuth_when_it_selects_the_child():
    variables = (*_X, Variable("phi", kind="angle", unit="rad"))

    @kernel(variables=variables, stokes=("I",), azimuth="phi")
    def first(nu, x):
        return {"I": x * np.ones(nu.shape)}

    @kernel(variables=variables, stokes=("I",), azimuth="phi")
    def second(nu, x):
        return {"I": 2 * x * np.ones(nu.shape)}

    model = HybridKernel(first, second, switch={"phi": "1 rad"})
    channels = Channels.at([1.0, 2.0], unit="GHz")
    coordinates = {"x": np.ones(4), "phi": np.array([0.5, 1.5, 1.0, 0.8])}
    result = node_modes(AveragedKernel(model), channels, coordinates)
    np.testing.assert_array_equal(result.I, [[1, 1], [2, 2], [1, 1], [1, 1]])
