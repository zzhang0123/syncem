"""Seeded observational fits on the stored v0.7.10 study realisations."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import syncem as se
from tests.v080 import _study


@pytest.fixture(scope="module")
def paper_fits():
    data = _study.ACCEPTANCE_DATA
    _study.verify_sums(data / "SHA256SUMS")
    with np.load(data / "fits_products.npz") as archive:
        cases = ("E1", "E2", "E3", "E4-JP", "E4-KP", "E5", "E1M")
        noise = {
            case: {
                field: archive[f"{case.replace('-', '_')}_{field}"]
                for field in ("data_fit", "data_held", "sigma")
            }
            for case in cases
        }
    pinned = json.loads((data / "fits_results.json").read_text())["ensembles"]
    return noise, {case: pinned[case]["seeded"] for case in cases}


@pytest.fixture(scope="module", params=[False, True], ids=["separable", "joint"])
def basis(request):
    cells, channels = se.paper.cells(), se.paper.channels()
    if request.param:
        lattice = se.Lattice(gamma=cells, phi=se.AzimuthHarmonic())
        specified = se.paper.specified()
        del specified["phi"]
        source = se.Population(se.FreeMoments("gamma", "phi"), **specified)
        return (
            source,
            se.spectral_basis(se.paper.e5(), channels, lattice=lattice, check=False),
            ("E5",),
        )
    # The E4-KP and Maxwellian spectra in the seeded study use the common
    # separable fit design and full direct spectra for the observations.
    source = se.paper.e1().replace(gamma=se.FreeMoments())
    return (
        source,
        se.spectral_basis(
            se.paper.e1(), channels, lattice=se.Lattice(gamma=cells), check=False
        ),
        ("E1", "E2", "E3", "E4-JP", "E4-KP", "E1M"),
    )


def test_seeded_rank_and_chi2(basis, paper_fits):
    source, columns, cases = basis
    noise, expected = paper_fits
    fitted = np.arange(0, 41, 2)
    held = np.arange(1, 41, 2)
    for case in cases:
        reference, draws = expected[case], noise[case]
        y = np.empty((41, 3))
        y[fitted] = draws["data_fit"]
        y[held] = draws["data_held"]
        data = se.StokesData(
            columns.channels,
            I=y[:, 0],
            Q=y[:, 1],
            U=y[:, 2],
            sigma=draws["sigma"],
            unit="cgs",
        ).withhold(held)
        fit = se.fit(source, data, basis=columns, check=False)
        assert int(fit.rank) == reference["R_star"], case
        assert int(fit.dof) == reference["dof_fit"], case
        assert float(fit.chi2) == pytest.approx(reference["chi2_fit"], abs=1e-6), case
        if case == "E1":
            report = str(fit)
            assert "Rank 7 of numerical rank 26" in report
            assert "Stokes I Q U" in report
            assert "Withheld χ²" in report


def test_full_and_compressed_parameter_covariance(paper_fits):
    numbers, _ = paper_fits
    path = _study.ACCEPTANCE_DATA / "verify_params_results.json"
    reference = json.loads(path.read_text())
    pop = se.paper.e1()
    cfg = _study.paper()
    band = se.paper.channels()
    lattice = se.Lattice(gamma=se.paper.cells())
    columns = se.spectral_basis(pop, band, lattice=lattice, check=False)
    p = se.Free(2.6, bounds=(1.1, 5.0))
    limits = {"lo": cfg.gamma_lo, "hi": cfg.gamma_hi}
    laws = {
        "E1": se.PowerLaw(p, **limits),
        "E2": se.BrokenPowerLaw(
            se.Free(2.0, bounds=(0, 7)),
            se.Free(3.2, bounds=(0, 7)),
            se.Free(7800, bounds=(100, 3e5), log=True),
            **limits,
        ),
        "E3": se.PowerLawMixture.cumulant_form(
            p,
            se.Free(0.04, bounds=(-0.5, 0.5)),
            **limits,
        ),
        "E4-JP": se.AgedPowerLaw(
            p,
            gamma_cut=se.Free(19000, bounds=(400, 1e6), log=True),
            lo=cfg.gamma_lo,
        ),
    }
    rows = cfg.fit_idx
    for name, law in laws.items():
        draws = numbers[name]
        data = se.StokesData(
            band[rows],
            stokes=draws["data_fit"],
            sigma=draws["sigma"][rows],
            unit="cgs",
        )
        model = pop.replace(gamma=law)
        unknown = pop.replace(gamma=se.FreeMoments())
        free = se.fit(unknown, data, basis=columns, keep_directions=True, check=False)
        result = reference["E2_s2" if name == "E2" else name]
        for tag, target in (("seeded_full", data), ("seeded_comp", free.retained)):
            fit = se.fit(model, target, basis=columns, check=False)
            fitted_law = fit.electron_spectrum()
            names = tuple(key.split(".", 1)[1] for key in fit.params)
            physical = jnp.asarray([item.value for item in fit.params.values()])

            def manuscript(x):
                new = fitted_law._at(dict(zip(names, x)))
                amplitude = jnp.log(new.paper_coefficient().value)
                other = []
                for field, value in zip(names, x):
                    if field == "amplitude":
                        continue
                    if field == "gamma_break":
                        value = jnp.log(value)
                    if field == "explicit_cut":
                        value = reference["E4JP_truth"]["ln(a1<B_perp>)"] + 2 * jnp.log(
                            value
                        )
                    other.append(value)
                return jnp.asarray((amplitude, *other))

            pin = result[tag]
            jac = jax.jacfwd(manuscript)(physical)
            sigma = np.sqrt(np.diag(jac @ fit.parameter_covariance @ jac.T))
            np.testing.assert_allclose(
                sigma, pin["sd"], rtol=0.003, err_msg=f"{name} {tag}"
            )
            np.testing.assert_array_less(
                abs(np.asarray(manuscript(physical)) - np.asarray(pin["phi"])),
                sigma,
                err_msg=f"{name} {tag}",
            )
