"""Archive and replay a sparse R3a basis without serializing its callable."""

import numpy as np
import pytest

import syncem as se
from syncem.engine._basis_archive import read_archive, write_archive
from syncem.engine._basis_io_codec import pack, unpack
from syncem.engine._conditional_archive import ArchivedConditionalBasis
from syncem.engine.average import AveragedKernel
from syncem.engine.basis import SpectralBasis, build_from_averaged
from syncem.fitting.conditional_basis import conditional_basis
from syncem.fitting.free_fit import fit_free
from syncem.lattice.moments import Moments
from syncem.values.errors import DataError
from tests.v080.test_v080_conditional_legendre import skew, template
from tests.v080.test_v080_hybrid_archive import high, low


@pytest.fixture(scope="module")
def basis():
    lattice = se.Lattice(
        x=se.Cells.linear(1, 3, 1, order=1),
        eta=se.Legendre(2),
        y=se.Cells.linear(-1, 1, 1, order=1),
        order=1,
        weights=[0.41],
    )
    channels = se.Channels.at([1, 2, 4], unit="GHz")
    average = AveragedKernel(template, legendre={"eta": (2, 15)})
    joint = build_from_averaged(average, channels, lattice)

    def declared_child(x, eta):
        return skew(beta=0.2 * x * eta)

    return conditional_basis(joint, "y", declared_child, parent_weights=[0.8], nodes=8)


def test_stored_conditional_preserves_sparse_map_prediction_and_fit(tmp_path, basis):
    output = tmp_path / "R3a.npz"
    basis.save(output)
    restored = SpectralBasis.load(output)
    assert isinstance(restored, ArchivedConditionalBasis)
    assert restored.child == basis.child and restored.nodes == basis.nodes
    assert restored.record == basis.record and restored.key == basis.key
    assert restored.ledger.to_dict() == basis.ledger.to_dict()
    assert restored.lattice.key == basis.lattice.key
    assert restored.joint.lattice.key == basis.joint.lattice.key
    assert restored.mapping.key == basis.mapping.key
    for name in ("joint_rows", "parent_rows", "coefficients"):
        np.testing.assert_array_equal(
            getattr(restored.mapping, name), getattr(basis.mapping, name)
        )
    np.testing.assert_array_equal(restored.design(), basis.design())
    np.testing.assert_array_equal(
        restored.conditional_truncation(), basis.conditional_truncation()
    )

    values = np.array([1.2, 0.15, -0.1, 0.02, 0.05, -0.07])
    moments = Moments(basis.lattice, values)
    expected = se.predict(moments, basis.channels, basis=basis, check=False)
    prediction = se.predict(moments, basis.channels, basis=restored, check=False)
    np.testing.assert_array_equal(prediction.stokes, expected.stokes)
    independent = basis.joint.predict(
        Moments(basis.joint.lattice, basis.mapping.apply(values))
    )
    np.testing.assert_allclose(expected.stokes, independent.stokes, rtol=1e-14)
    selection = basis.channels[[2, 0, 2]]
    subset = restored.at(selection)
    assert isinstance(subset, ArchivedConditionalBasis)
    np.testing.assert_array_equal(
        subset.predict(moments).stokes, np.asarray(expected.stokes)[[2, 0, 2]]
    )
    data = se.StokesData(restored.channels, I=prediction.I, sigma=0.05, unit="cgs")
    np.testing.assert_array_equal(restored.design(data), basis.design(data))
    live_fit = fit_free(basis, data, rank=se.FullRank())
    saved_fit = fit_free(restored, data, rank=se.FullRank())
    np.testing.assert_array_equal(saved_fit.moments.values, live_fit.moments.values)
    np.testing.assert_allclose(
        saved_fit.predict(selection).I, np.asarray(expected.I)[[2, 0, 2]], rtol=1e-14
    )
    resave = tmp_path / "R3a-selection.npz"
    subset.save(resave)
    np.testing.assert_array_equal(SpectralBasis.load(resave).design(), subset.design())


def test_stored_conditional_refuses_unavailable_rebuilds(tmp_path, basis):
    path = tmp_path / "conditional.npz"
    basis.save(path)
    with pytest.raises(DataError, match="supports only stored"):
        SpectralBasis.load(path, averaged=basis.joint.averaged)
    loaded = SpectralBasis.load(path)
    with pytest.raises(DataError, match="supports only stored"):
        loaded.with_order(0)
    with pytest.raises(DataError, match="supports only stored"):
        loaded.at(se.Channels.at([5.0], unit="GHz"))
    with pytest.raises(DataError, match="original population and phase"):
        loaded.direct(None)


def test_structural_map_check_refuses_rehashed_coefficients(tmp_path, basis):
    path = tmp_path / "clean.npz"
    basis.save(path)
    payload, arrays = read_archive(path)
    body = unpack(payload, arrays)
    altered = body["conditional"]
    altered["coefficients"] = altered["coefficients"] + 0.2
    mutable_arrays = {}
    defect = tmp_path / "wrong-coefficients.npz"
    write_archive(defect, pack(body, mutable_arrays), mutable_arrays)
    with pytest.raises(DataError, match="map does not match its key"):
        SpectralBasis.load(defect)


def test_archived_hybrid_refreshes_metadata_after_select(tmp_path):
    averaged = AveragedKernel(se.HybridKernel(low, high(), switch={"x": 2}))
    channels = se.Channels.at([1.0, 2.0, 4.0], unit="GHz")
    lattice = se.Lattice(x=se.Cells.linear(1.0, 3.0, 2, order=1))
    basis = build_from_averaged(averaged, channels, lattice)
    path = tmp_path / "hybrid.npz"
    basis.save(path)
    live = SpectralBasis.load(path, averaged=averaged)
    selected = live.select(x=(2, 3))
    assert selected.averaged.kernel.assignment == (1,)
    assert dict(selected.hybrid_metadata)["assignment"] == (1,)
    stored = tmp_path / "one-cell.npz"
    selected.save(stored)
    restored = SpectralBasis.load(stored)
    assert dict(restored.hybrid_metadata)["assignment"] == (1,)
    np.testing.assert_array_equal(restored.design(), selected.design())
