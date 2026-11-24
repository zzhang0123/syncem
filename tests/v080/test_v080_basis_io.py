"""Portable spectra, integrity checks and declared rebuild paths."""

import hashlib
import json
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.capabilities import ReferenceDensity
from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.phase import FaradayRotation
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.engine.assemble import predict_values
from syncem.engine.average import AveragedKernel, fixed_rule
from syncem.engine.basis import SpectralBasis, build_from_averaged
from syncem.engine.basis_io import (
    basis_source_key,
    load_basis,
    require_rebuild,
    save_basis,
)
from syncem.engine.single_reference import SingleReferenceBasis, from_single_reference
from syncem.distributions import Isotropic, NoRotation, UniformAzimuth
from syncem.fitting.data import StokesData
from syncem.fitting.free_fit import fit_free
from syncem.fitting.rank import FullRank
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.moments import Moments
from syncem.lattice.representations import AzimuthHarmonic
from syncem.model.phase import GaussianScreen
from syncem.records.ledger import ErrorTerm, Ledger
from syncem.values.errors import DataError
from tests.v080.test_v080_single_reference import _frozen, _moments


@kernel(variables=(Variable("x", kind="real"),), stokes=("I",))
def _test_kernel(nu, x):
    return {"I": 1 + (nu / 1e9) * x}


@pytest.fixture(scope="module")
def sample():
    channels = Channels.at([1.0, 2.0, 4.0], unit="GHz", half_width=0.2, n_nu=8)
    lat = Lattice(x=Cells.linear(1.0, 3.0, 1, order=1))
    averaged = AveragedKernel(_test_kernel)
    return build_from_averaged(averaged, channels, lat), averaged


def test_npz_round_trip_preserves_design_records_and_fit(tmp_path, sample):
    basis, _ = sample
    ledger = Ledger(
        (
            ErrorTerm.unbounded(name="physical_kernel"),
            ErrorTerm(
                [[-0.1], [0.0], [0.2]], "measured", "comparison", name="representation"
            ),
        )
    ).allow("physical_kernel", ErrorTerm(0.1, "bound"), note="external comparison")
    basis = replace(basis, ledger=ledger)
    path = tmp_path / "basis.npz"
    basis.save(path)
    restored = SpectralBasis.load(path)
    assert restored.key == basis.key
    assert restored.lattice.key == basis.lattice.key
    assert (
        restored.lattice.cell_axes[0][1].spelling
        == basis.lattice.cell_axes[0][1].spelling
    )
    assert restored.lattice.explicit_weights == basis.lattice.explicit_weights
    assert restored.columns.key == basis.columns.key
    assert restored.record == basis.record
    assert restored.ledger.to_dict() == basis.ledger.to_dict()
    assert restored.channels.row_keys() == basis.channels.row_keys()
    np.testing.assert_array_equal(
        restored.channels(basis.channels.nodes[0]),
        basis.channels(basis.channels.nodes[0]),
    )
    np.testing.assert_array_equal(restored.design(), basis.design())
    subset = restored.at(restored.channels[[2, 0]])
    np.testing.assert_array_equal(subset.design(), np.asarray(basis.design())[[2, 0]])
    moments = Moments(basis.lattice, np.array([1.5, 0.2]))
    observed = predict_values(basis, moments)
    np.testing.assert_array_equal(predict_values(restored, moments), observed)
    data = StokesData(restored.channels, stokes=observed, sigma=0.2, unit="cgs")
    fit = fit_free(restored, data, rank=FullRank())
    np.testing.assert_allclose(fit.moments.values, moments.values, rtol=1e-13)
    np.testing.assert_allclose(fit.predict().I, observed[:, 0], rtol=1e-13)
    with pytest.raises(DataError, match="compatible AveragedKernel"):
        require_rebuild(restored)
    with pytest.raises(DataError, match="compatible AveragedKernel"):
        restored.at(Channels.at([13.0], unit="GHz"))
    with pytest.raises(DataError, match="compatible AveragedKernel"):
        restored.with_order(0)
    np.testing.assert_array_equal(restored.predict(moments).I, observed[:, 0])
    saved_again = tmp_path / "again.npz"
    save_basis(restored, saved_again)
    np.testing.assert_array_equal(load_basis(saved_again).design(), basis.design())


def test_live_source_and_registry_verified_before_rebuilding(tmp_path, sample):
    basis, source = sample
    key = basis_source_key(source)
    assert key is not None
    path = tmp_path / "basis.npz"
    save_basis(basis, path)
    with pytest.raises(DataError, match="not compatible"):
        load_basis(path, averaged=AveragedKernel(source.kernel, outer_batch=None))
    loaded = load_basis(path, registry={key: source})
    require_rebuild(loaded)
    with pytest.raises(DataError, match="original population and phase"):
        loaded.direct(None)
    with pytest.raises(DataError, match="supplied Moments"):
        loaded.moments(None)
    new_channel = Channels.at([7.0], unit="GHz", half_width=0.2, n_nu=8)
    np.testing.assert_array_equal(
        loaded.at(new_channel).design(), basis.at(new_channel).design()
    )


def test_spin2_and_single_field_axis_survive_archive(tmp_path):
    ch = Channels.at([1.0, 3.0], unit="GHz")
    avg = AveragedKernel(
        ContinuumKernel(x_min=1e-14),
        [("eta", fixed_rule(jnp.array([0.2]), jnp.array([1.0])))],
        phase=FaradayRotation(screen=GaussianScreen(4.0, 0.3)),
    )
    lattice = Lattice(
        gamma=Cells.log(300.0, 2e5, 3, order=1),
        B=Cells.single("5 uG", scale="1 uG", order=1),
        phi=AzimuthHarmonic(),
        order=1,
    )
    basis = build_from_averaged(avg, ch, lattice, cache=False)
    path = tmp_path / "spin2.npz"
    save_basis(basis, path)
    assert basis_source_key(avg) is not None
    loaded = load_basis(path, averaged=avg)
    assert loaded.lattice.index.blocks == ("M0", "Re M2", "Im M2")
    np.testing.assert_array_equal(loaded.lattice.cells.hi, basis.lattice.cells.hi)
    np.testing.assert_array_equal(loaded.spectra2, basis.spectra2)
    np.testing.assert_array_equal(loaded.design(), basis.design())
    assert [term.variable for term in loaded.specified] == ["eta"]


def test_callable_channel_snapshot_keeps_exact_weights_and_row_identity(tmp_path):
    ch = Channels.from_callable(
        lambda nu: jnp.ones_like(nu) / 1e8,
        [[0.9e9, 1.0e9], [2.1e9, 2.2e9]],
        smoothness=0,
        normalisation="unit_integral",
        n_nu=8,
    )
    basis = build_from_averaged(
        AveragedKernel(_test_kernel),
        ch,
        Lattice(x=Cells.linear(1.0, 3.0, 1, order=1)),
        cache=False,
    )
    path = tmp_path / "callable.npz"
    save_basis(basis, path)
    restored = load_basis(path)
    assert restored.channels.row_keys() == ch.row_keys()
    assert restored.channels[[1, 0]].row_keys() == ch[[1, 0]].row_keys()
    np.testing.assert_array_equal(
        restored.channels.response_weights(), ch.response_weights()
    )
    with pytest.raises(DataError, match="only their stored quadrature response"):
        restored.channels(ch.centres_hz)


def _rewrite(path, edit):
    with np.load(path, allow_pickle=False) as npz:
        values = {k: npz[k] for k in npz.files}
    edit(values)
    np.savez(path, **values)


def test_corrupt_array_and_incompatible_package_version_refused(tmp_path, sample):
    basis, _ = sample
    path = tmp_path / "bad.npz"
    save_basis(basis, path)

    def corrupt(values):
        name = next(
            k
            for k, v in values.items()
            if k.startswith("a") and v.size and v.dtype.kind == "f"
        )
        values[name].flat[0] += 0.5

    _rewrite(path, corrupt)
    with pytest.raises(DataError, match="array.*integrity"):
        load_basis(path)
    save_basis(basis, path)

    def wrong_version(values):
        config = json.loads(values["metadata_json"].tobytes())
        config["manifest"]["package_version"] = "unsupported-version"

        def canon(value):
            return json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")

        config["sha256"] = hashlib.sha256(canon(config["manifest"])).hexdigest()
        values["metadata_json"] = np.frombuffer(canon(config), dtype=np.uint8)

    _rewrite(path, wrong_version)
    with pytest.raises(DataError, match="package version"):
        load_basis(path)


def test_default_law_records_and_legacy_subtype_round_trip(tmp_path, sample):
    basis, _ = sample
    defaults = (
        ("eta", Isotropic(nodes=9)),
        ("phi", UniformAzimuth(nodes=7)),
        ("depth", NoRotation()),
    )
    path = tmp_path / "defaults.npz"
    with pytest.warns(UserWarning, match="A JAX array is being set as static"):
        original = replace(basis, forward_defaults=defaults)
    save_basis(original, path)
    loaded = load_basis(path)
    assert loaded.forward_defaults == (
        ("eta", ("Isotropic", ("nodes", 9))),
        ("phi", ("UniformAzimuth", ("nodes", 7))),
        ("depth", ("NoRotation", ("depth", 0.0))),
    )

    old = _frozen()
    bridge = from_single_reference(old)
    legacy_path = tmp_path / "legacy.npz"
    save_basis(bridge, legacy_path)
    legacy = load_basis(legacy_path)
    assert isinstance(legacy, SingleReferenceBasis)
    assert legacy.legacy_provenance == bridge.legacy_provenance
    assert legacy.legacy_reference == bridge.legacy_reference
    physical = legacy.from_legacy_moments(_moments(old), column=7.0)
    np.testing.assert_array_equal(
        legacy.predict(physical).stokes, bridge.predict(physical).stokes
    )


def test_resaved_snapshot_retains_custom_reference_density_identity(tmp_path):
    def density(x):
        return 1.0 + x**2

    @kernel(
        variables=(Variable("x", kind="real"),),
        stokes=("I",),
        capabilities=(ReferenceDensity("x", density),),
    )
    def local_kernel(nu, x):
        return {"I": 1.0 + nu / 1e9 * x}

    lat = Lattice(
        x=Cells.linear(1.0, 3.0, 2, order=1),
        _variables=local_kernel.variables,
        _references=(("x", density),),
    )
    averaged = AveragedKernel(local_kernel)
    ch = Channels.at([1.0, 3.0], unit="GHz")
    basis = build_from_averaged(averaged, ch, lat, cache=False)
    assert basis_source_key(averaged) is not None
    path = tmp_path / "reference.npz"
    save_basis(basis, path)
    saved_again = tmp_path / "reference_copy.npz"
    save_basis(load_basis(path), saved_again)
    linked = load_basis(saved_again, averaged=averaged)
    assert linked.reference_names == ("x",)
    assert [x for x, _ in linked.lattice._references] == ["x"]
    np.testing.assert_array_equal(linked.with_order(0).lattice.weights, lat.weights)
