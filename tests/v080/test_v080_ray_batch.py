"""Streaming direct and resolved spectra of batched Ray profiles."""

import weakref

import numpy as np
import pytest

import syncem as se
from syncem.front._rays import predict_ray_batch
from syncem.records.ledger import ErrorTerm, Ledger
from syncem.values.spectrum import Spectrum


def _ray():
    fields = np.array(
        [
            [[[3, 1, 1], [2, -3, 1]], [[5, 4, -1], [-3, 2, 1]]],
            [[[3, 2, -1], [4, 0, 2]], [[1, 2, 1], [2, 3, -1]]],
        ]
    )
    density = [[(1.0, 0.8), (0.0, 0.0)], [(2.0, 1.0), (1.4, 1.5)]]
    laws = tuple(
        tuple(
            tuple(
                se.PowerLaw(2.1 + 0.04 * (4 * i + 2 * j + at), lo=1000, hi=5000)
                for at in range(2)
            )
            for j in range(2)
        )
        for i in range(2)
    )
    return se.Ray(
        step=([1.0, 1.5], "pc"),
        B_vector=(fields, "uG"),
        thermal_density=(0.02, "cm^-3"),
        electron_density=(density, "cm^-3"),
        electrons=laws,
    )


def test_batched_direct_ray_matches_individual_law_profiles():
    ray = _ray()
    ch = se.Channels.at([1.0, 4.0], unit="GHz")
    result = se.predict(ray, ch, stokes=("I", "P"))

    assert result.stokes.shape == (2, 2, 2, 3)
    assert result.components == ("I", "Q", "U") and result.moments is None
    assert result.ledger["quadrature"].kind == "unbounded"
    for index in ((0, 0), (1, 0), (1, 1)):
        individual = se.predict(ray[index], ch, stokes=("I", "P"))
        np.testing.assert_allclose(result.stokes[index], individual.stokes, rtol=1e-13)
    assert not np.any(result.stokes[0, 1])
    scaled = se.predict(2.5 * se.Population(ray), ch, stokes=("I", "P"))
    np.testing.assert_allclose(scaled.stokes, 2.5 * result.stokes, rtol=1e-13)


def test_resolved_batched_ray_keeps_sightline_error_values():
    @se.kernel(
        variables=se.ContinuumKernel().variables,
        azimuth="phi",
        depth="depth",
        stokes=("I",),
    )
    def simple_ray(nu, gamma, B, mu, eta):
        return {"I": (nu / 1e9) * (gamma / 1000 + 1e6 * B + 0.1 * eta + 0.1 * mu)}

    ray = _ray()
    ch = se.Channels.at([1.0, 4.0], unit="GHz")
    lattice = se.Lattice(
        gamma=se.Cells.log(1100, 5500, 1, order=0),
        B=se.Cells.linear(0.5, 10, 1, unit="uG", order=0),
        eta=se.Legendre(1),
        phi=se.AzimuthHarmonic(),
        depth=se.Cells.linear(-10, 10, 1, order=0),
    )
    options = {"kernel": simple_ray, "lattice": lattice, "cache": False}
    result = se.predict(ray, ch, **options)

    assert result.moments is None
    for name in ("representation", "outside_lattice"):
        assert result.ledger[name].signed
        assert result.ledger[name].value.shape == result.stokes.shape
        assert not np.any(result.ledger[name].value[0, 1])
    for index in ((0, 0), (1, 0), (1, 1)):
        individual = se.predict(ray[index], ch, **options)
        np.testing.assert_allclose(result.stokes[index], individual.stokes, rtol=1e-13)
        for name in ("representation", "outside_lattice"):
            np.testing.assert_allclose(
                result.ledger[name].value[index],
                individual.ledger[name].value,
                rtol=1e-13,
            )


def test_streaming_releases_each_spectrum_and_combines_incomplete_ledgers():
    ray = _ray()
    ch = se.Channels.at([1.0, 4.0], unit="GHz")
    previous = None
    seen = 0

    def evaluate(line):
        nonlocal previous, seen
        assert previous is None or previous() is None
        seen += 1
        terms = [
            ErrorTerm(-float(seen), "measured", name="representation", signed=True),
            ErrorTerm(float(seen), "bound", name="depth_truncation", relative_to="I"),
        ]
        if seen != 2:
            terms.append(ErrorTerm(0.2, "estimate", name="quadrature"))
        member = Spectrum(
            np.full((2, 1), float(seen)),
            ch,
            ledger=Ledger(tuple(terms)),
            moments=object(),
        )
        previous = weakref.ref(member)
        return member

    result = predict_ray_batch(ray, ch, evaluate)
    assert seen == 3 and previous() is None
    assert result.moments is None
    np.testing.assert_array_equal(result.I[..., 0], [[1, 0], [2, 3]])
    np.testing.assert_array_equal(result.ledger["representation"].value, -result.stokes)
    np.testing.assert_array_equal(
        result.ledger["depth_truncation"].value, [[1, 0], [2, 3]]
    )
    assert result.ledger["quadrature"].kind == "unbounded"
    converted = result.to("Jy/sr")
    np.testing.assert_array_equal(
        converted.ledger["depth_truncation"].value,
        result.ledger["depth_truncation"].value,
    )
    np.testing.assert_allclose(
        converted.ledger["representation"].value, -converted.stokes
    )


def test_batched_ray_requires_a_lattice_to_resolve_variables():
    ray = _ray()
    ch = se.Channels.at([1.0], unit="GHz")
    with pytest.raises(se.RoleError, match="lattice"):
        se.predict(ray, ch, resolve=("gamma",))
    with pytest.raises(se.RoleError, match="average"):
        se.predict(ray, ch, average=("B",))


def test_direct_ray_preserves_component_order_and_single_output():
    ray = _ray()[0, 0]
    channels = se.Channels.at([1, 4], unit="GHz")
    full = se.predict(ray, channels)
    for request, cols in (("UQI", [2, 1, 0]), ("Q", [1])):
        selected = se.predict(ray, channels, stokes=request)
        assert selected.components == tuple(request)
        np.testing.assert_array_equal(selected.stokes, np.asarray(full.stokes)[:, cols])
