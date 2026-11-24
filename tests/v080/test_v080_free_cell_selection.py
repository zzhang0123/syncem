"""FreeMoments(cells=slice(...)) as in SPEC G6 (review P2-7)."""

import numpy as np

import syncem as se


def test_free_moment_selection_equals_a_selected_basis():
    channels = se.Channels.log(0.5, 8.0, 12, unit="GHz", half_width=0.2, n_nu=8)
    population = se.paper.e1()
    lattice = se.Lattice(gamma=se.paper.cells(order=1))
    basis = se.spectral_basis(population, channels, lattice=lattice, check=False)
    spectrum = basis.predict(population, check=False)
    data = se.StokesData(
        channels, I=spectrum.I, unit=spectrum.unit, sigma=0.01 * spectrum.I
    )
    high = slice(3, None)
    marked = se.fit(
        population.replace(gamma=se.FreeMoments(cells=high)),
        data,
        basis=basis,
        check=False,
    )
    manual = se.fit(
        population.replace(gamma=se.FreeMoments()),
        data,
        basis=basis.select(gamma=high),
        check=False,
    )
    assert marked.moments.key == manual.moments.key
    assert marked.moments.index.table.n_cells == lattice.n_cells - 3
    np.testing.assert_allclose(
        np.asarray(marked.moments.values), np.asarray(manual.moments.values)
    )
    dense = se.Channels.log(0.5, 30.0, 20, unit="GHz", half_width=0.2, n_nu=8)
    np.testing.assert_allclose(marked.predict(dense).I, manual.predict(dense).I)


def test_problem_and_fit_use_identical_selected_rows():
    channels = se.Channels.log(0.5, 15, 8, unit="GHz", half_width=0.2, n_nu=4)
    source = se.paper.e1()
    lattice = se.Lattice(gamma=se.Cells.log(1e3, 1e5, 6, order=1))
    basis = se.spectral_basis(source, channels, lattice=lattice, check=False)
    truth = basis.predict(source, check=False)
    data = se.StokesData(channels, I=truth.I, sigma=0.01 * truth.I, unit=truth.unit)
    cut = slice(3, None)
    free = source.replace(gamma=se.FreeMoments(cells=cut))
    manual = se.problem(
        source.replace(gamma=se.FreeMoments()), data, basis=basis.select(gamma=cut)
    )
    for options in (dict(lattice=lattice), dict(basis=basis)):
        fitted = se.fit(free, data, **options, check=False)
        compiled = se.problem(free, data, **options, check=False)
        assert compiled.lattice.key == fitted.moments.key == manual.lattice.key
        np.testing.assert_allclose(compiled.design, manual.design, rtol=1e-12, atol=0)
        np.testing.assert_allclose(compiled.solve().chi2, fitted.chi2, rtol=1e-11)


def test_sum_uses_common_free_cell_selection_and_rejects_incompatible_supports():
    import pytest
    from tests.v080.test_v080_r7_fit import _kernel, _setup

    first, _, _, data = _setup()
    lattice = se.Lattice(a=se.Cells.linear(1, 2, 4, order=1))
    selected = first.replace(a=se.FreeMoments(cells=slice(2, None)))
    options = dict(kernel=_kernel, lattice=lattice, check=False)
    fit = se.fit(selected + selected, data, **options)
    problem = se.problem(selected + selected, data, **options)
    assert fit.moments[0].index.table.n_cells == 2
    assert problem.basis.parts[0].lattice.n_cells == 2
    with pytest.raises(se.RoleError, match="cell selection"):
        se.problem(selected + first, data, **options)
